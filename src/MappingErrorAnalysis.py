import sys

sys.dont_write_bytecode = True

from logger         import Logger
from adapter        import labelClass, writeHugeCSV
from MappingUtils   import (
    loadInputs, keepHPOConcepts, buildAttributeLookup,
    build_child_map, build_parent_map, make_descendant_lookup, pairSeries, inGoldSeries,
    make_ancestor_distance_lookup, make_hierarchy_distance,
    ALL_METHODS, SIM_METHODS, VALIDITY_LABELS, REVIEW_REASON,
    llm_mappings, llm_filtered_path, llm_review_path, evaluation_path, no_gold_path,
    llm_disagreements_path,
    validity_domain_mismatch, validity_accepted, validity_hierarchy_reversals,
    validity_best_candidate, validity_last_resort, validity_reviewable,
    similarity_steps, confidence_column, validity_column,
    hpo_id_column, sct_id_column,
    getRepresentativeLabel,
)
import os
import pandas as pd
import numpy  as np
from collections import Counter, defaultdict
from itertools   import combinations

# ==========================================================================
# MappingErrorAnalysis.py
#
# Evaluates the output of the whole pipeline (llm_filtered.csv) against the
# Graham gold standard. Everything is computed on unique (hpo_id, sct_id)
# pairs.
#
# Accepted sets:
#   auto-accepted : original validity code 10 (before the LLM review)
#   LLM-accepted  : pairs with an "llm" row
#   final         : auto-accepted + LLM-accepted (the default "accepted")
# The original validity code comes from the generation rows only, i.e. the
# code before the LLM review.
# ==========================================================================

ERROR_EXAMPLE_LIMIT = 20

# Thresholds in the precision-recall sweep (section 8).
PR_STEPS = 50

# Section 16: distances of MAX_DISTANCE or more share one bucket.
MAX_DISTANCE           = 5
DISTANCE_EXAMPLE_LIMIT = 5

# Bucket of pairs whose gold targets are all missing from sct.csv.
GOLD_NOT_IN_SCT = "gold target not in SCT"

# "llm" only re-accepts generated pairs, so it is left out wherever methods
# are compared.
GENERATION_METHODS = [method for method in ALL_METHODS if method != llm_mappings]

def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )


def _metrics(tp: int, pred_total: int, gold_total: int, universe_total: int) -> dict:
    """
    Precision, recall, F1 and universe share (tp / all generated pairs; not
    accuracy, as true negatives are not counted).
    """
    precision      = _safe_div(tp, pred_total)
    recall         = _safe_div(tp, gold_total)
    f1             = _f1(precision, recall)
    universe_share = _safe_div(tp, universe_total)
    return dict(precision=precision, recall=recall, f1=f1, universe_share=universe_share)


def _pair_set(df: pd.DataFrame, id_col: str, val_col: str) -> set:
    """Set of unique (id_col, val_col) pairs in df."""
    return set(zip(df[id_col], df[val_col]))


def _hpo_depth(hpo_id: str, parent_map: dict) -> int:
    """
    Depth of hpo_id (a root = 0): the largest shortest-path distance to any
    ancestor, which can differ from the shortest distance to the root.
    parent_map: child -> set of parent ids. Cycle-safe.
    """
    visited  = {hpo_id}
    frontier = {hpo_id}
    depth    = 0
    while True:
        next_frontier = set()
        for node in frontier:
            for parent in parent_map.get(node, set()):
                if parent not in visited:
                    visited.add(parent)
                    next_frontier.add(parent)
        if not next_frontier:
            break
        frontier = next_frontier
        depth   += 1
    return depth


if __name__ == "__main__":

    l = Logger()
    l.printHeader("Evaluation of HPO → SNOMED CT mapping pipeline")

    l.printHeader("1. Loading data")

    hpo, sct, umls, hpo_ontology, sct_ontology, umls_ontology, graham = loadInputs(";")
    del umls, umls_ontology  # not used in this script; free the memory

    id_col   = hpo_ontology.config.id_column       # HPO concept ID
    val_col  = hpo_ontology.config.value_column     # SCT concept ID
    attr_col = hpo_ontology.config.attribute_column # mapping method

    l.log("Loading LLM-reviewed mappings (llm_filtered.csv)...")
    final_df = pd.read_csv(llm_filtered_path, sep=";")
    final_df = keepHPOConcepts(final_df, id_col)
    l.log(f"LLM-reviewed mappings: {len(final_df)} rows.")
    l.log("Loading LLM-reviewed mappings (llm_filtered.csv) completed.")

    gold_pairs = set(zip(graham[hpo_id_column], graham[sct_id_column]))
    gold_hpo   = set(graham[hpo_id_column].unique())
    gold_total = len(gold_pairs)
    l.log(f"Gold standard: {gold_total} unique pairs for {len(gold_hpo)} HPO concepts.")

    l.printHeader("2. Building lookup tables")
    l.log("Building label, hierarchy and pair lookup tables...")

    sct_labels_dict = buildAttributeLookup(sct, sct_ontology, labelClass)
    hpo_labels_dict = buildAttributeLookup(hpo, hpo_ontology, labelClass)

    def sct_label(sct_id: str) -> str:
        return getRepresentativeLabel(sct_labels_dict.get(sct_id, [])) or sct_id

    def hpo_label(hpo_id: str) -> str:
        terms = hpo_labels_dict.get(hpo_id, [])
        return terms[0] if terms else hpo_id

    # hpo_parent_map holds direct parents (for the depth, section 13).
    hpo_parent_map  = build_parent_map(hpo, hpo_ontology)
    hpo_ancestors   = make_descendant_lookup(hpo_parent_map)
    hpo_descendants = make_descendant_lookup(build_child_map(hpo, hpo_ontology))
    sct_descendants = make_descendant_lookup(build_child_map(sct, sct_ontology))

    # Generation rows carry MappingSetBuilder's code, shared by all rows of a
    # pair, so "first" gives the original validity (or NA).
    gen_rows = final_df[final_df[attr_col] != llm_mappings]

    pair_table = (
        gen_rows.groupby([id_col, val_col])
        .agg(**{
            validity_column: (validity_column,   "first"),
            "n_methods":     (attr_col,          "nunique"),
            "best_conf":     (confidence_column, "max"),
        })
        .reset_index()
    )
    pair_keys      = list(zip(pair_table[id_col], pair_table[val_col]))
    pair_validity  = dict(zip(pair_keys, pair_table[validity_column]))
    pair_n_methods = dict(zip(pair_keys, pair_table["n_methods"]))
    pair_best_conf = dict(zip(pair_keys, pair_table["best_conf"]))

    universe_pairs = set(pair_keys)
    universe_total = len(universe_pairs)

    def pairs_with(codes) -> set:
        """Pairs whose original validity code is in codes (never NA pairs)."""
        codes = set(codes)
        return {p for p, v in pair_validity.items() if pd.notna(v) and v in codes}

    auto_pairs  = pairs_with([validity_accepted])
    llm_pairs   = _pair_set(final_df[final_df[attr_col] == llm_mappings], id_col, val_col)
    final_pairs = auto_pairs | llm_pairs

    l.log(f"Generated pairs: {universe_total}.")
    l.log(f"Accepted pairs: {len(auto_pairs)} auto, {len(llm_pairs)} by the LLM, "
          f"{len(final_pairs)} final.")
    l.log("Building label, hierarchy and pair lookup tables completed.")

    def log_pairs(label: str, pairs: set, share: bool = False, indent: str = "  ") -> None:
        """
        One metric line for a set of pairs. Precision counts all pairs, also
        of HPO concepts without gold; share=True adds the TP share.
        """
        tp = len(pairs & gold_pairs)
        m  = _metrics(tp, len(pairs), gold_total, universe_total)
        line = (f"{indent}{label + ':':<48} pairs {len(pairs):7} | in gold {tp:6} | "
                f"P {m['precision']:.3f} | R {m['recall']:.3f} | F1 {m['f1']:.3f}")
        if share:
            line += f" | TP share {m['universe_share']:.4f}"
        l.log(line)

    def log_examples(pairs, limit: int = ERROR_EXAMPLE_LIMIT, indent: str = "    ") -> None:
        """Log up to limit (HPO, SCT) pairs with labels, sorted for reproducibility."""
        pairs = sorted(pairs)
        for hid, sid in pairs[:limit]:
            l.log(f"{indent}{hid} [{hpo_label(hid)}]  →  {sid} [{sct_label(sid)}]")
        if len(pairs) > limit:
            l.log(f"{indent}... and {len(pairs) - limit} more.")

    l.printHeader("3. Stage-by-stage comparison")

    log_pairs("Stage 1 — all generated pairs",                   universe_pairs,                 share=True)
    log_pairs("Stage 2 — after filtering (validity=1–10)",       pairs_with(validity_reviewable), share=True)
    log_pairs("Stage 3a — auto-accepted (before LLM review)",    auto_pairs,                     share=True)
    log_pairs("Stage 3b — LLM-accepted only",                    llm_pairs,                      share=True)
    log_pairs("Stage 3c — final (auto-accepted + LLM-accepted)", final_pairs,                    share=True)
    l.log(f"Auto-accepted pairs not accepted by the LLM (rejected or no valid "
          f"answer): {len(auto_pairs - llm_pairs)}.")
    l.log("These auto-accepted pairs stay in the final set.")

    l.printHeader("4. Validity-level breakdown")

    l.log("Pairs by original validity code (before LLM review):")
    for v_code, v_label in sorted(VALIDITY_LABELS.items()):
        log_pairs(f"validity={v_code} ({v_label})", pairs_with([v_code]))
    not_reached = {p for p, v in pair_validity.items() if pd.isna(v)}
    log_pairs("validity=NA (not reached by any step)", not_reached)

    l.printHeader("5. Method-level breakdown")

    for method in ALL_METHODS:
        method_pairs = _pair_set(final_df[final_df[attr_col] == method], id_col, val_col)
        if not method_pairs:
            l.log(f"Method '{method}': no pairs.")
            continue
        l.log(f"Method '{method}':")
        log_pairs("all pairs",          method_pairs)
        log_pairs("in the final set",   method_pairs & final_pairs)

    l.printHeader("6. Method-agreement analysis")

    final_n_methods = {p: pair_n_methods.get(p, 0) for p in final_pairs}
    max_agree = max(final_n_methods.values(), default=0)

    l.log("Final pairs found by exactly k generation methods:")
    for k in range(1, max_agree + 1):
        bucket = {p for p, n in final_n_methods.items() if n == k}
        if bucket:
            log_pairs(f"k = {k}", bucket)

    l.log("Final pairs found by at least k generation methods:")
    for k in range(1, max_agree + 1):
        bucket = {p for p, n in final_n_methods.items() if n >= k}
        if bucket:
            log_pairs(f"k ≥ {k}", bucket)

    # Sections 7 and 8 cover all generated pairs of a method, i.e. what a
    # pure threshold would achieve. "Jaccard or embedding" keeps the higher
    # of both per pair: read its thresholds as "either method reaches t".
    l.printHeader("7. Confidence threshold analysis")

    def similarity_pools():
        """(label, rows) for each similarity method and their combination."""
        for method in SIM_METHODS:
            yield method, gen_rows[gen_rows[attr_col] == method]
        yield "Jaccard or embedding (max per pair)", gen_rows[gen_rows[attr_col].isin(SIM_METHODS)]

    def best_confidence(pool: pd.DataFrame) -> pd.DataFrame:
        """One row per pair, with its highest confidence."""
        return (
            pool.sort_values(confidence_column, ascending=False)
            .drop_duplicates(subset=[id_col, val_col])
        )

    for method_label, pool in similarity_pools():
        if pool.empty:
            continue
        pool_deduped = best_confidence(pool)
        l.log(f"Method '{method_label}':")
        for thresh in similarity_steps:
            subset = pool_deduped[pool_deduped[confidence_column] >= thresh]
            log_pairs(f"confidence ≥ {thresh:.2f}", _pair_set(subset, id_col, val_col))

    l.printHeader("8. Precision–recall curve (fine-grained threshold sweep)")

    for method_label, pool in similarity_pools():
        if pool.empty:
            continue
        pool_deduped = best_confidence(pool)
        conf_vals    = pool_deduped[confidence_column].values
        thresholds   = np.linspace(float(np.min(conf_vals)), float(np.max(conf_vals)), PR_STEPS)

        l.log(f"Method '{method_label}':")
        best_f1_thresh = None
        best_f1_val    = -1.0

        for thresh in thresholds:
            subset = pool_deduped[pool_deduped[confidence_column] >= thresh]
            if subset.empty:
                continue
            pairs = _pair_set(subset, id_col, val_col)
            log_pairs(f"confidence ≥ {thresh:.4f}", pairs)
            m = _metrics(len(pairs & gold_pairs), len(pairs), gold_total, universe_total)
            if m["f1"] > best_f1_val:
                best_f1_val    = m["f1"]
                best_f1_thresh = thresh

        if best_f1_thresh is not None:
            l.log(f"  Best F1 {best_f1_val:.3f} at confidence ≥ {best_f1_thresh:.4f}.")

    l.printHeader("9. LLM-specific analysis")

    reviewed_pairs = pairs_with(REVIEW_REASON.keys())
    n_reviewed     = len(reviewed_pairs)
    n_accepted     = len(llm_pairs & reviewed_pairs)

    l.log("A. Overall LLM review:")
    l.log(f"  Pairs reviewed: {n_reviewed}, accepted: {n_accepted} "
          f"({_safe_div(n_accepted, n_reviewed):.2%}).")
    l.log(f"  Pairs rejected or without a valid answer: {n_reviewed - n_accepted} "
          f"({_safe_div(n_reviewed - n_accepted, n_reviewed):.2%}).")
    log_pairs("LLM-accepted", llm_pairs)

    # None for runs from before MappingLLM wrote the review file.
    review = pd.read_csv(llm_review_path, sep=";") if os.path.exists(llm_review_path) else None

    if review is not None:
        l.log(f"LLM decisions per pass (pairs, from {os.path.basename(llm_review_path)}):")
        for llm_pass, decisions in review.groupby("llm_pass"):
            n_acc   = int((decisions["llm_decision"] == "ACCEPT").sum())
            n_rej   = int((decisions["llm_decision"] == "REJECT").sum())
            n_empty = int(decisions["llm_rationale"].isna().sum())
            l.log(f"  Pass {llm_pass}: {len(decisions)} reviewed, {n_acc} ACCEPT, "
                  f"{n_rej} REJECT, {len(decisions) - n_acc - n_rej} other decision.")
            l.log(f"  Pass {llm_pass}: {n_empty} without rationale (incl. unparsable or "
                  f"missing answers).")
    else:
        l.log(f"  {os.path.basename(llm_review_path)} not found; decisions per "
              f"pass are not available.")
        l.log("  The file is written by MappingLLM since the rationale field was added.")

    l.log("B. Per review reason (original validity code):")
    for v_code, reason_label in sorted(REVIEW_REASON.items()):
        reason_pairs = pairs_with([v_code])
        if not reason_pairs:
            l.log(f"  validity={v_code} ({reason_label}): no pairs reviewed.")
            continue
        accepted_from_reason = reason_pairs & llm_pairs
        l.log(f"  validity={v_code} ({reason_label}): {len(reason_pairs)} reviewed, "
              f"{len(accepted_from_reason)} accepted "
              f"({_safe_div(len(accepted_from_reason), len(reason_pairs)):.2%}).")
        log_pairs("accepted", accepted_from_reason, indent="    ")

    llm_fp = llm_pairs - gold_pairs
    l.log(f"C. LLM false positives (accepted but NOT in gold): {len(llm_fp)} pairs.")
    log_examples(llm_fp, indent="  ")

    l.printHeader("10. Mapping completeness")

    all_hpo_concepts = set(hpo[hpo_ontology.config.id_column].unique())
    generated_hpo    = {h for h, _ in universe_pairs}
    accepted_hpo     = {h for h, _ in final_pairs}

    n_total_hpo = len(all_hpo_concepts)
    n_generated = len(generated_hpo & all_hpo_concepts)
    n_accepted  = len(accepted_hpo & all_hpo_concepts)

    l.log("A. HPO concept coverage:")
    l.log(f"  {'HPO concepts (HP: namespace):':<38}{n_total_hpo:7}")
    l.log(f"  {'With ≥1 generated pair:':<38}{n_generated:7} "
          f"({_safe_div(n_generated, n_total_hpo):.2%})")
    l.log(f"  {'With ≥1 final mapping:':<38}{n_accepted:7} "
          f"({_safe_div(n_accepted, n_total_hpo):.2%})")

    accepted_sct_per_hpo: dict = defaultdict(set)
    accepted_hpo_per_sct: dict = defaultdict(set)
    for hid, sid in final_pairs:
        accepted_sct_per_hpo[hid].add(sid)
        accepted_hpo_per_sct[sid].add(hid)

    gold_sct_per_hpo: dict = defaultdict(set)
    for hid, sid in gold_pairs:
        gold_sct_per_hpo[hid].add(sid)

    multi_mapped = {h: s for h, s in accepted_sct_per_hpo.items() if len(s) > 1}
    l.log(f"B. HPO concepts with more than one final SCT target: {len(multi_mapped)} "
          f"({_safe_div(len(multi_mapped), len(accepted_sct_per_hpo)):.2%} of mapped "
          f"concepts).")
    l.log("Examples with the most targets:")

    for hpo_id, our_scts in sorted(multi_mapped.items(), key=lambda x: (-len(x[1]), x[0]))[:10]:
        gold_scts = gold_sct_per_hpo.get(hpo_id, set())
        spurious  = our_scts - gold_scts
        l.log(f"  {hpo_id} [{hpo_label(hpo_id)}]: {len(our_scts)} SCT targets, "
              f"{len(our_scts & gold_scts)} in gold, {len(spurious)} not in gold")
        for sid in sorted(spurious):
            l.log(f"    not in gold: {sid} [{sct_label(sid)}]")

    l.printHeader("11. Missed gold-pair diagnosis")

    missed_gold = gold_pairs - final_pairs
    l.log(f"Missed gold pairs: {len(missed_gold)} of {gold_total}.")

    def diagnose(pair) -> str:
        """Why a gold pair is missing, from its original validity code."""
        if pair not in pair_validity:
            return "NEVER_GENERATED"
        code = pair_validity[pair]
        if pd.isna(code):
            return "NOT_REACHED"
        if code == validity_domain_mismatch:
            return "DOMAIN_REJECTED"
        if code in validity_hierarchy_reversals:
            return "HIERARCHY_REVERSAL_LLM_REJECTED"
        if code in (validity_best_candidate, validity_last_resort):
            return "FALLBACK_LLM_REJECTED"
        # Validity 10 is always in the final set.
        return "INCONSISTENCY"

    diagnosis: dict = defaultdict(list)
    for pair in missed_gold:
        diagnosis[diagnose(pair)].append(pair)

    for category, pairs in sorted(diagnosis.items(), key=lambda x: -len(x[1])):
        l.log(f"{category}: {len(pairs)} pairs")
        log_examples(pairs, limit=5, indent="  ")

    generated_missed = [p for p in missed_gold if p in pair_best_conf]
    if generated_missed:
        conf_vals = pd.Series([pair_best_conf[p] for p in generated_missed]).dropna()
        l.log("Cross-reference rows always have confidence 1.0.")
        l.log(f"Best confidence of the {len(generated_missed)} generated-but-missed "
              f"gold pairs (any generation method):")
        for thresh in similarity_steps:
            n_above = int((conf_vals >= thresh).sum())
            l.log(f"  confidence ≥ {thresh:.2f}: {n_above:6} pairs "
                  f"({_safe_div(n_above, len(conf_vals)):.2%})")
    else:
        l.log("All missed gold pairs were never generated.")

    l.printHeader("12. Pairwise method overlap and unique contributions")

    method_pair_sets = {
        method: _pair_set(gen_rows[gen_rows[attr_col] == method], id_col, val_col)
        for method in GENERATION_METHODS
    }

    l.log("Pairwise overlap (pairs generated by both methods):")
    for m1, m2 in combinations(GENERATION_METHODS, 2):
        log_pairs(f"{m1} ∩ {m2}", method_pair_sets[m1] & method_pair_sets[m2])

    l.log("Unique gold contributions (gold pairs found by only this method):")
    for method in GENERATION_METHODS:
        other_pairs = set().union(*(
            s for m, s in method_pair_sets.items() if m != method
        ))
        unique_gold = (method_pair_sets[method] - other_pairs) & gold_pairs
        l.log(f"  {method}: {len(unique_gold)} gold pairs")
        log_examples(unique_gold, limit=5)

    l.printHeader("13. Hierarchy-depth analysis")

    def depth_stats(hpo_ids: set, label: str) -> None:
        if not hpo_ids:
            l.log(f"{label}: (empty)")
            return
        arr = np.array([_hpo_depth(h, hpo_parent_map) for h in hpo_ids])
        l.log(f"{label}: n={len(arr)}, mean={arr.mean():.2f}, "
              f"median={np.median(arr):.1f}, min={arr.min()}, max={arr.max()}")
        for lo, hi in [(0, 2), (3, 5), (6, 9), (10, 14), (15, 999)]:
            count    = int(((arr >= lo) & (arr <= hi)).sum())
            label_hi = str(hi) if hi < 999 else "+"
            l.log(f"  depth {lo}–{label_hi}: {count}")

    depth_stats(all_hpo_concepts,                        "All HPO concepts")
    depth_stats(gold_hpo,                                "Gold-standard HPO concepts")
    depth_stats({h for h, _ in final_pairs & gold_pairs}, "HPO concepts with ≥1 correct final mapping")
    depth_stats({h for h, _ in missed_gold},             "HPO concepts with ≥1 missed gold pair")

    # Section 14: does MappingSetBuilder's narrower-wins rule over-prune?
    l.printHeader("14. Hierarchy-reversal quality check")

    reversal_pairs = pairs_with(validity_hierarchy_reversals)
    log_pairs("hierarchy reversals (validity=1–7)", reversal_pairs)
    log_pairs("of which accepted by the LLM",       reversal_pairs & llm_pairs)

    lost_gold = (reversal_pairs - llm_pairs) & gold_pairs
    l.log(f"Gold pairs among the hierarchy reversals that are not in the final "
          f"set: {len(lost_gold)}.")
    log_examples(lost_gold, indent="  ")

    l.printHeader("15. Error analysis")

    def log_concepts(hpo_ids: list, describe=None) -> None:
        for hid in hpo_ids[:ERROR_EXAMPLE_LIMIT]:
            extra = f"  {describe(hid)}" if describe else ""
            l.log(f"  {hid} [{hpo_label(hid)}]{extra}")
        if len(hpo_ids) > ERROR_EXAMPLE_LIMIT:
            l.log(f"  ... and {len(hpo_ids) - ERROR_EXAMPLE_LIMIT} more.")

    cat_a = sorted(gold_hpo - generated_hpo)
    l.log(f"A. Gold HPO concepts with no generated pair: {len(cat_a)}.")
    log_concepts(cat_a)

    cat_b = sorted((gold_hpo & generated_hpo) - accepted_hpo)
    l.log(f"B. Gold HPO concepts with generated pairs but no final mapping: {len(cat_b)}.")
    log_concepts(cat_b, lambda hid: "gold SCT(s): " + ", ".join(
        sct_label(s) for s in sorted(gold_sct_per_hpo.get(hid, set()))))

    cat_c = sorted(accepted_hpo - gold_hpo)
    l.log(f"C. HPO concepts with a final mapping but not in the gold standard: {len(cat_c)}.")
    l.log("These mappings may be correct but cannot be verified.")
    log_concepts(cat_c, lambda hid: "mapped to: " + ", ".join(
        sct_label(s) for s in sorted(accepted_sct_per_hpo.get(hid, set()))))

    cat_d_exact_wrong   = []
    cat_d_partial_wrong = []
    for hid in sorted(gold_hpo & accepted_hpo):
        our_scts  = accepted_sct_per_hpo.get(hid, set())
        gold_scts = gold_sct_per_hpo.get(hid, set())
        correct   = our_scts & gold_scts
        if not correct:
            cat_d_exact_wrong.append((hid, our_scts, gold_scts))
        elif correct != gold_scts:
            cat_d_partial_wrong.append((hid, our_scts, gold_scts, correct))

    l.log(f"D. HPO concepts in gold and the final set with entirely wrong SCT "
          f"target(s): {len(cat_d_exact_wrong)}.")
    for hid, our, gold in cat_d_exact_wrong[:10]:
        l.log(f"  {hid} [{hpo_label(hid)}]")
        l.log(f"    ours: {', '.join(sct_label(s) for s in sorted(our))}")
        l.log(f"    gold: {', '.join(sct_label(s) for s in sorted(gold))}")
    if len(cat_d_exact_wrong) > 10:
        l.log(f"  ... and {len(cat_d_exact_wrong) - 10} more.")

    l.log(f"D. HPO concepts in gold and the final set with some gold SCT "
          f"target(s) missing: {len(cat_d_partial_wrong)}.")
    for hid, our, gold, correct in cat_d_partial_wrong[:10]:
        l.log(f"  {hid} [{hpo_label(hid)}]")
        l.log(f"    correct: {', '.join(sct_label(s) for s in sorted(correct))}")
        l.log(f"    missing: {', '.join(sct_label(s) for s in sorted(gold - our))}")
        if our - gold:
            l.log(f"    extra:   {', '.join(sct_label(s) for s in sorted(our - gold))}")
    if len(cat_d_partial_wrong) > 10:
        l.log(f"  ... and {len(cat_d_partial_wrong) - 10} more.")

    # Section 16: is_a distance of a pair's SCT target to the nearest gold
    # target of the same HPO concept (see make_hierarchy_distance), only for
    # gold HPO concepts. "cumulative" is the precision if near misses up to
    # that distance counted as correct. C measures how close the missed gold
    # targets came to the nearest final target.
    l.printHeader("16. Hierarchy distance to the gold standard")

    l.log("Building SNOMED CT ancestor lookup...")
    sct_concepts       = set(sct[sct_ontology.config.id_column].unique())
    sct_up             = make_ancestor_distance_lookup(build_parent_map(sct, sct_ontology))
    hierarchy_distance = make_hierarchy_distance(sct_up)
    l.log(f"SNOMED CT concepts: {len(sct_concepts)}.")
    l.log("Building SNOMED CT ancestor lookup completed.")

    def sct_relation(sct_id: str, other_id: str) -> str:
        """Relation of sct_id to other_id: equal, broader, narrower or lateral."""
        if sct_id == other_id:
            return "equal"
        if sct_id in sct_up(other_id):
            return "broader"
        if other_id in sct_up(sct_id):
            return "narrower"
        return "lateral"

    def nearest_target(sct_id: str, targets) -> tuple:
        """
        (distance, target) of the target nearest to sct_id, or (None, None)
        if sct_id or all targets are missing from sct.csv. Targets are
        visited in sorted order, so ties go to the smallest SCT ID.
        """
        best_distance, best_target = None, None
        if sct_id not in sct_concepts:
            return best_distance, best_target
        for target in sorted(targets):
            if target not in sct_concepts:
                continue
            distance = hierarchy_distance(sct_id, target)
            if distance is not None and (best_distance is None or distance < best_distance):
                best_distance, best_target = distance, target
        return best_distance, best_target

    def distance_bucket(distance) -> str:
        """Histogram bucket; None means the gold targets are not in sct.csv."""
        if distance is None:
            return GOLD_NOT_IN_SCT
        return f"{MAX_DISTANCE}+" if distance >= MAX_DISTANCE else str(distance)

    distance_buckets = [str(d) for d in range(MAX_DISTANCE)] + [f"{MAX_DISTANCE}+", GOLD_NOT_IN_SCT]

    # (hpo, sct) -> (distance, nearest gold target, relation of sct to it).
    # auto_pairs and llm_pairs are subsets of final_pairs.
    pair_gold_distance: dict = {}
    for hid, sid in final_pairs:
        if hid in gold_hpo:
            distance, target = nearest_target(sid, gold_sct_per_hpo[hid])
            pair_gold_distance[(hid, sid)] = (
                distance, target, sct_relation(sid, target) if target is not None else ""
            )

    def log_distances(label: str, info: dict) -> None:
        """
        Distance histogram of info (pair -> (distance, other, relation)):
        count and share per bucket, the cumulative share up to the bucket,
        and the relations within each bucket.
        """
        total = len(info)
        l.log(f"{label}: {total} pairs")
        if not total:
            return
        counts    = Counter(distance_bucket(d) for d, _, _ in info.values())
        relations = defaultdict(Counter)
        for distance, _, relation in info.values():
            relations[distance_bucket(distance)][relation] += 1

        cumulative = 0
        for bucket in distance_buckets:
            count = counts.get(bucket, 0)
            name  = {"0": "distance 0 (correct)", GOLD_NOT_IN_SCT: GOLD_NOT_IN_SCT}.get(
                bucket, f"distance {bucket}"
            )
            line  = f"    {name + ':':<33}{count:7} ({_safe_div(count, total):7.2%})"
            if bucket != GOLD_NOT_IN_SCT:
                cumulative += count
                line += f" | cumulative {_safe_div(cumulative, total):7.2%}"
            if bucket not in ("0", GOLD_NOT_IN_SCT) and count:
                rel = relations[bucket]
                line += (f" | broader {rel['broader']:5} | narrower {rel['narrower']:5} "
                         f"| lateral {rel['lateral']:5}")
            l.log(line)

        wrong = [d for d, _, _ in info.values() if d]
        if wrong:
            l.log(f"    mean distance of the {len(wrong)} wrong pairs with a distance: "
                  f"{np.mean(wrong):.2f}, median {np.median(wrong):.1f}")

    l.log("A. Distance to the nearest gold target (pairs of gold HPO concepts):")
    for label, pairs in [
        ("auto-accepted (before LLM review)",    auto_pairs),
        ("LLM-accepted only",                    llm_pairs),
        ("final (auto-accepted + LLM-accepted)", final_pairs),
    ]:
        log_distances(f"  {label}", {p: v for p, v in pair_gold_distance.items() if p in pairs})

    l.log("B. Examples of final pairs per distance:")
    for bucket in distance_buckets[1:]:
        examples = sorted(
            (p, v) for p, v in pair_gold_distance.items() if distance_bucket(v[0]) == bucket
        )
        if not examples:
            continue
        name = GOLD_NOT_IN_SCT if bucket == GOLD_NOT_IN_SCT else f"distance {bucket}"
        l.log(f"  {name}: {len(examples)} pairs")
        for (hid, sid), (_, target, relation) in examples[:DISTANCE_EXAMPLE_LIMIT]:
            l.log(f"    {hid} [{hpo_label(hid)}]  →  {sid} [{sct_label(sid)}]")
            if target is not None:
                phrase = f"{relation} than" if relation in ("broader", "narrower") else f"{relation} to"
                l.log(f"      {phrase} gold: {target} [{sct_label(target)}]")
            else:
                l.log("      gold: " + ", ".join(
                    f"{g} [{sct_label(g)}]" for g in sorted(gold_sct_per_hpo[hid])))
        if len(examples) > DISTANCE_EXAMPLE_LIMIT:
            l.log(f"    ... and {len(examples) - DISTANCE_EXAMPLE_LIMIT} more.")

    # (hpo, gold sct) -> (distance, nearest final target, its relation to gold).
    missed_gold_distance: dict = {}
    for hid, gid in missed_gold:
        if hid in accepted_hpo:
            distance, target = nearest_target(gid, accepted_sct_per_hpo[hid])
            missed_gold_distance[(hid, gid)] = (
                distance, target, sct_relation(target, gid) if target is not None else ""
            )
    l.log(f"C. Missed gold pairs: {len(missed_gold)}.")
    l.log(f"{len(missed_gold) - len(missed_gold_distance)} of them belong to HPO concepts "
          f"without a final mapping and are not measured.")
    log_distances("  missed gold pairs of HPO concepts with a final mapping "
                  "(relation = final target to gold target)", missed_gold_distance)

    # Section 17: MappingSetBuilder removes source collapses (A) only among
    # auto-accepted pairs, so LLM-accepted pairs can still cause them; B and
    # C are not checked by any earlier step.
    l.printHeader("17. Structural quality warnings")

    def related_pairs(concepts: set, descendants) -> list:
        """
        (broader, narrower) for every two concepts of the set in an
        ancestor/descendant relation, in reproducible order.
        """
        found = []
        for c1, c2 in combinations(sorted(concepts), 2):
            if c2 in descendants(c1):
                found.append((c1, c2))
            elif c1 in descendants(c2):
                found.append((c2, c1))
        return found

    source_collapse = {
        hid: related_pairs(scts, sct_descendants)
        for hid, scts in accepted_sct_per_hpo.items() if len(scts) > 1
    }
    source_collapse = {hid: pairs for hid, pairs in source_collapse.items() if pairs}
    l.log(f"WARNING A — Source collapse: {sum(len(p) for p in source_collapse.values())} "
          f"related SCT pairs for {len(source_collapse)} HPO concepts.")
    l.log("Source collapse: an HPO concept is mapped to an SCT concept and to one of "
          "its ancestors.")
    for hid in sorted(source_collapse)[:ERROR_EXAMPLE_LIMIT]:
        l.log(f"  {hid} [{hpo_label(hid)}]")
        for broader, narrower in source_collapse[hid]:
            l.log(f"    broader:  {broader} [{sct_label(broader)}]")
            l.log(f"    narrower: {narrower} [{sct_label(narrower)}]")
    if len(source_collapse) > ERROR_EXAMPLE_LIMIT:
        l.log(f"  ... and {len(source_collapse) - ERROR_EXAMPLE_LIMIT} more HPO concepts.")

    target_collapse = {
        sid: related_pairs(hids, hpo_descendants)
        for sid, hids in accepted_hpo_per_sct.items() if len(hids) > 1
    }
    target_collapse = {sid: pairs for sid, pairs in target_collapse.items() if pairs}
    l.log(f"WARNING B — Target collapse: {sum(len(p) for p in target_collapse.values())} "
          f"related HPO pairs for {len(target_collapse)} SCT concepts.")
    l.log("Target collapse: related HPO concepts are mapped to the same SCT concept.")
    for sid in sorted(target_collapse)[:ERROR_EXAMPLE_LIMIT]:
        l.log(f"  {sid} [{sct_label(sid)}]")
        for broader, narrower in target_collapse[sid]:
            l.log(f"    broader HPO:  {broader} [{hpo_label(broader)}]")
            l.log(f"    narrower HPO: {narrower} [{hpo_label(narrower)}]")
    if len(target_collapse) > ERROR_EXAMPLE_LIMIT:
        l.log(f"  ... and {len(target_collapse) - ERROR_EXAMPLE_LIMIT} more SCT concepts.")

    # C, e.g. "Focal seizure" (child) -> "Seizure" and "Seizure" (parent) ->
    # "Focal seizure". Only HPO ancestors are compared, not all concept pairs.
    reversal_examples = []
    for hpo_child in sorted(accepted_sct_per_hpo):
        for hpo_parent in sorted(set(hpo_ancestors(hpo_child)) & set(accepted_sct_per_hpo)):
            for sct_of_child in sorted(accepted_sct_per_hpo[hpo_child]):
                for sct_of_parent in sorted(accepted_sct_per_hpo[hpo_parent]):
                    if sct_of_parent in sct_descendants(sct_of_child):
                        reversal_examples.append((hpo_parent, hpo_child, sct_of_parent, sct_of_child))

    l.log(f"WARNING C — Cross-pair hierarchy reversal: {len(reversal_examples)} cases.")
    l.log("Cross-pair hierarchy reversal: the HPO hierarchy direction is inverted in "
          "the SCT targets.")
    for hpo_parent, hpo_child, sct_of_parent, sct_of_child in reversal_examples[:ERROR_EXAMPLE_LIMIT]:
        l.log(f"  HPO parent: {hpo_parent} [{hpo_label(hpo_parent)}]  →  "
              f"SCT: {sct_of_parent} [{sct_label(sct_of_parent)}]")
        l.log(f"  HPO child:  {hpo_child} [{hpo_label(hpo_child)}]  →  "
              f"SCT: {sct_of_child} [{sct_label(sct_of_child)}]")
        l.log("    The child's SCT target is an ancestor of the parent's SCT target.")
    if len(reversal_examples) > ERROR_EXAMPLE_LIMIT:
        l.log(f"  ... and {len(reversal_examples) - ERROR_EXAMPLE_LIMIT} more.")

    l.printHeader("18. Writing output CSVs")

    # Pair-level columns of every final pair, used in all three files.
    # gold_distance is empty if the HPO concept has no gold mapping or its
    # gold targets are not in sct.csv.
    final_list = sorted(final_pairs)

    def gold_relation(pair) -> str:
        if pair not in pair_gold_distance:
            return "no_gold"
        return pair_gold_distance[pair][2] or "gold_not_in_sct"

    final_gen_rows = gen_rows[pairSeries(gen_rows, id_col, val_col).isin(final_pairs)]
    methods_per_pair = (
        final_gen_rows.groupby([id_col, val_col])[attr_col]
        .agg(lambda methods: ",".join(sorted(set(methods))))
        .to_dict()
    )
    pair_info = pd.DataFrame({
        id_col:              [h for h, _ in final_list],
        "hpo_label":         [hpo_label(h) for h, _ in final_list],
        val_col:             [s for _, s in final_list],
        "sct_label":         [sct_label(s) for _, s in final_list],
        "auto_accepted":     [p in auto_pairs for p in final_list],
        "llm_accepted":      [p in llm_pairs  for p in final_list],
        "original_validity": [pair_validity.get(p) for p in final_list],
        "methods":           [methods_per_pair.get(p, "") for p in final_list],
        "best_conf":         [pair_best_conf.get(p) for p in final_list],
        "gold_distance":     pd.array(
            [pair_gold_distance.get(p, (None,))[0] for p in final_list], dtype="Int64"
        ),
        "nearest_gold_sct":  [(pair_gold_distance.get(p, (None, None))[1] or "") for p in final_list],
        "gold_relation":     [gold_relation(p) for p in final_list],
    })

    # evaluation_final.csv: every row (method and "llm") of each final pair.
    # The validity column holds 10 on "llm" rows; original_validity has the
    # code before the LLM review.
    final_accepted = final_df[
        pairSeries(final_df, id_col, val_col).isin(final_pairs)
    ].copy().reset_index(drop=True)
    final_accepted["in_gold"] = inGoldSeries(final_accepted, gold_pairs, id_col, val_col)
    final_accepted = final_accepted.merge(pair_info, on=[id_col, val_col], how="left")

    # Empty without the review file, so the columns are always the same.
    if review is not None:
        final_accepted = final_accepted.merge(
            review[[id_col, val_col, "llm_rationale"]].drop_duplicates(subset=[id_col, val_col]),
            on=[id_col, val_col], how="left",
        )
    else:
        final_accepted["llm_rationale"] = pd.NA
    final_accepted = final_accepted.rename(columns={id_col: hpo_id_column, val_col: sct_id_column})

    l.log(f"Rows to write to {os.path.basename(evaluation_path)}: {len(final_accepted)} "
          f"({int(final_accepted['in_gold'].sum())} in gold) for {len(final_pairs)} "
          f"final pairs.")
    writeHugeCSV(final_accepted, evaluation_path)

    # accepted_without_gold.csv: final pairs that cannot be checked against
    # the gold standard (section 15 C), e.g. for manual review. Concepts
    # whose gold rows loadGold removed count as without gold mapping.
    no_gold_df = (
        pair_info[~pair_info[id_col].isin(gold_hpo)]
        .rename(columns={id_col: hpo_id_column, val_col: sct_id_column})
        .reset_index(drop=True)
    )

    l.log(f"Rows to write to {os.path.basename(no_gold_path)}: {len(no_gold_df)} "
          f"final pairs of {no_gold_df[hpo_id_column].nunique()} HPO concepts "
          f"without a gold mapping.")
    l.log(f"Of these, {int(no_gold_df['auto_accepted'].sum())} are auto-accepted and "
          f"{int(no_gold_df['llm_accepted'].sum())} LLM-accepted; a pair can be both.")
    writeHugeCSV(no_gold_df, no_gold_path)

    # llm_disagreements.csv: pairs on which MappingSetBuilder and the LLM
    # disagree (auto-accepted pairs the LLM rejected stay in the final set).
    disagreement_pairs = (auto_pairs - llm_pairs) | (llm_pairs - auto_pairs)
    disagreements = pair_info[
        pairSeries(pair_info, id_col, val_col).isin(disagreement_pairs)
    ].copy()
    disagreements.insert(0, "disagreement", np.where(
        disagreements["auto_accepted"], "auto_accepted_llm_rejected", "candidate_llm_accepted"
    ))
    disagreements["in_gold"] = inGoldSeries(disagreements, gold_pairs, id_col, val_col)
    disagreements["review_reason"] = disagreements["original_validity"].map(
        lambda v: REVIEW_REASON.get(int(v), "") if pd.notna(v) else ""
    )

    # LLM confidence normalised to 0.2-1.0; empty columns without review file.
    llm_cols = ["llm_pass", "llm_decision", "llm_confidence", "llm_rationale"]
    if review is not None:
        disagreements = disagreements.merge(
            review[[id_col, val_col, "llm_pass", "llm_decision",
                    confidence_column, "llm_rationale"]]
            .rename(columns={confidence_column: "llm_confidence"}),
            on=[id_col, val_col], how="left",
        )
    else:
        for col in llm_cols:
            disagreements[col] = pd.NA

    disagreements = (
        disagreements
        .rename(columns={id_col: hpo_id_column, val_col: sct_id_column})
        .sort_values(["disagreement", hpo_id_column, sct_id_column])
        .reset_index(drop=True)
    )

    for kind, rows in disagreements.groupby("disagreement"):
        l.log(f"Rows to write to {os.path.basename(llm_disagreements_path)} ({kind}): "
              f"{len(rows)} pairs, {int(rows['in_gold'].sum())} in gold.")
    if review is None:
        l.log(f"{os.path.basename(llm_review_path)} not found; the LLM columns are empty.")
    writeHugeCSV(disagreements, llm_disagreements_path)

    l.printHeader("19. Overall summary")

    l.log(f"Gold standard: {gold_total} pairs.")
    l.log(f"Universe (generated pairs): {universe_total}.")
    log_pairs("auto-accepted (before LLM review)",    auto_pairs,  share=True)
    log_pairs("LLM-accepted only",                    llm_pairs,   share=True)
    log_pairs("final (auto-accepted + LLM-accepted)", final_pairs, share=True)

    # Near misses (section 16).
    final_distances = [d for p, (d, _, _) in pair_gold_distance.items() if p in final_pairs]
    within = {k: _safe_div(sum(1 for d in final_distances if d is not None and d <= k),
                           len(final_distances)) for k in (0, 1, 2)}
    l.log(f"  Final pairs of gold HPO concepts: {len(final_distances)}.")
    l.log(f"  Share within is_a distance 0 / ≤1 / ≤2 of a gold target: "
          f"{within[0]:.3f} / {within[1]:.3f} / {within[2]:.3f}.")

    l.printHeader("Evaluation completed.")
