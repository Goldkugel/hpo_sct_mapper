import sys

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

from logger         import Logger
from adapter        import labelClass, writeHugeCSV
from MappingUtils   import (
    loadInputs, keepHPOConcepts, buildAttributeLookup,
    build_parent_map, make_descendant_lookup, pairSeries, inGoldSeries,
    ALL_METHODS, SIM_METHODS, VALIDITY_LABELS, REVIEW_REASON,
    llm_mappings, llm_filtered_path, evaluation_path,
    validity_accepted, validity_hierarchy_reversals, validity_reviewable,
    evaluation, printCounts, similarity_steps,
    confidence_column, validity_column,
    hpo_id_column, sct_id_column,
    threshold_cosine_similarity,
    getRepresentativeLabel,
)
import pandas as pd
import numpy  as np
from collections import defaultdict
from itertools   import combinations

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How many examples to print per error-analysis category
ERROR_EXAMPLE_LIMIT = 20

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )


def _metrics(tp: int, pred_total: int, gold_total: int, universe_total: int) -> dict:
    """
    Returns precision, recall, F1, and universe share given:
      tp             — true positives (correct predictions)
      pred_total     — total predictions made
      gold_total     — total gold-standard pairs
      universe_total — total unique (HPO, SCT) pairs ever considered
                       (denominator for universe share)

    universe_share is tp / universe_total, i.e. the fraction of all
    considered pairs that are correct predictions. It is not accuracy:
    true negatives are not counted.
    """
    precision      = _safe_div(tp, pred_total)
    recall         = _safe_div(tp, gold_total)
    f1             = _f1(precision, recall)
    universe_share = _safe_div(tp, universe_total)
    return dict(precision=precision, recall=recall, f1=f1, universe_share=universe_share)


def _log_metrics(l: Logger, m: dict, prefix: str = "") -> None:
    p = f"{prefix}: " if prefix else ""
    l.log(f"{p}Precision: {m['precision']:.4f}  |  Recall: {m['recall']:.4f}  "
          f"|  F1: {m['f1']:.4f}  |  TP share of universe: {m['universe_share']:.4f}")


def _dedupe_by_pair(df: pd.DataFrame, id_col: str, val_col: str) -> pd.DataFrame:
    """One row per unique (hpo_id, sct_id) pair."""
    return df.drop_duplicates(subset=[id_col, val_col]).reset_index(drop=True)


def _tp_from_df(df: pd.DataFrame, gold_set: set, id_col: str, val_col: str) -> int:
    """Count rows in df whose (id_col, val_col) pair is in gold_set."""
    return int(inGoldSeries(df, gold_set, id_col, val_col).sum())


def _hpo_depth(hpo_id: str, parent_map: dict) -> int:
    """
    Returns the depth of hpo_id in the HPO hierarchy (root = 0).
    parent_map: child -> set of parent ids.
    Uses BFS upward; cycle-safe.
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    l = Logger()
    l.printHeader("Evaluation of HPO → SNOMED CT mapping pipeline")

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    l.printHeader("1. Loading data")

    hpo, sct, umls, hpo_ontology, sct_ontology, umls_ontology, graham = loadInputs(";")

    id_col   = hpo_ontology.config.id_column       # e.g. "hpo_id"
    val_col  = hpo_ontology.config.value_column     # e.g. "sct_id"
    attr_col = hpo_ontology.config.attribute_column

    l.log("Loading llm_filtered.csv (final mappings)...")
    final_df = pd.read_csv(llm_filtered_path, sep=";")
    final_df = keepHPOConcepts(final_df, id_col)
    l.log(f"  Rows in llm_filtered.csv: {len(final_df)}")

    # Gold-standard pair set
    gold_pairs = set(zip(graham[hpo_id_column], graham[sct_id_column]))
    gold_hpo   = set(graham[hpo_id_column].unique())
    gold_total = len(gold_pairs)
    l.log(f"  Gold standard: {gold_total} unique (HPO, SCT) pairs "
          f"across {len(gold_hpo)} HPO concepts.")

    # Universe = all unique (HPO, SCT) pairs ever generated (universe-share denominator).
    all_pairs_df   = _dedupe_by_pair(final_df, id_col, val_col)
    universe_total = len(all_pairs_df)
    l.log(f"  Universe (unique pairs in llm_filtered.csv): {universe_total}")

    # -----------------------------------------------------------------------
    # 2. Build lookup tables
    # -----------------------------------------------------------------------
    l.printHeader("2. Building lookup tables")
    l.log("Building label and hierarchy lookup tables...")

    sct_labels_dict = buildAttributeLookup(sct, sct_ontology, labelClass)
    hpo_labels_dict = buildAttributeLookup(hpo, hpo_ontology, labelClass)

    # hpo_parent_map[child] = set of direct parents, used for depth calculation.
    hpo_parent_map: dict[str, set] = build_parent_map(hpo, hpo_ontology)

    def sct_label(sct_id: str) -> str:
        return getRepresentativeLabel(sct_labels_dict.get(sct_id, [])) or sct_id

    def hpo_label(hpo_id: str) -> str:
        terms = hpo_labels_dict.get(hpo_id, [])
        return terms[0] if terms else hpo_id

    l.log("Lookup tables built.")

    # -----------------------------------------------------------------------
    # 3. Stage-by-stage comparison
    #
    #   Stage 1 — All generated pairs (entire llm_filtered.csv, deduplicated)
    #   Stage 2 — After filtering: validity in {1..9,10} (domain-rejected
    #             pairs excluded; these were never seen by the LLM)
    #   Stage 3 — Final accepted: validity = 10, deduplicated
    # -----------------------------------------------------------------------
    l.printHeader("3. Stage-by-stage comparison")

    stages = {
        "Stage 1 — All generated pairs":
            final_df,
        "Stage 2 — After filtering (validity ≠ 0, not NA)":
            final_df[final_df[validity_column].isin(validity_reviewable)],
        "Stage 3 — Final accepted (validity = 10)":
            final_df[final_df[validity_column] == validity_accepted],
    }

    for stage_name, stage_df in stages.items():
        deduped = _dedupe_by_pair(stage_df, id_col, val_col)
        tp      = _tp_from_df(deduped, gold_pairs, id_col, val_col)
        m       = _metrics(tp, len(deduped), gold_total, universe_total)
        l.log(f"\n  {stage_name}")
        l.log(f"    Unique pairs: {len(deduped)}  |  Correct (in gold): {tp}")
        _log_metrics(l, m, "    Metrics")

    # -----------------------------------------------------------------------
    # 4. Validity-level breakdown
    # -----------------------------------------------------------------------
    l.printHeader("4. Validity-level breakdown")

    for v_code, v_label in sorted(VALIDITY_LABELS.items()):
        subset = _dedupe_by_pair(
            final_df[final_df[validity_column] == v_code], id_col, val_col
        )
        if subset.empty:
            l.log(f"  validity={v_code} ({v_label}): no pairs.")
            continue
        tp = _tp_from_df(subset, gold_pairs, id_col, val_col)
        m  = _metrics(tp, len(subset), gold_total, universe_total)
        l.log(f"\n  validity={v_code} ({v_label}): {len(subset)} pairs, {tp} in gold.")
        _log_metrics(l, m, "    Metrics")

    # -----------------------------------------------------------------------
    # 5. Method-level breakdown
    # -----------------------------------------------------------------------
    l.printHeader("5. Method-level breakdown")

    for method in ALL_METHODS:
        method_df = _dedupe_by_pair(
            final_df[final_df[attr_col] == method], id_col, val_col
        )
        if method_df.empty:
            l.log(f"  Method '{method}': no pairs found.")
            continue

        tp_all = _tp_from_df(method_df, gold_pairs, id_col, val_col)
        m_all  = _metrics(tp_all, len(method_df), gold_total, universe_total)

        accepted_df = _dedupe_by_pair(
            final_df[(final_df[attr_col] == method) & (final_df[validity_column] == validity_accepted)],
            id_col, val_col,
        )
        tp_acc = _tp_from_df(accepted_df, gold_pairs, id_col, val_col)
        m_acc  = _metrics(tp_acc, len(accepted_df), gold_total, universe_total)

        l.log(f"\n  Method: '{method}'")
        l.log(f"    All pairs ({len(method_df)}):             in gold={tp_all}")
        _log_metrics(l, m_all, "      All pairs ")
        l.log(f"    Accepted (validity=10, {len(accepted_df)}): in gold={tp_acc}")
        _log_metrics(l, m_acc, "      Accepted  ")

    # -----------------------------------------------------------------------
    # 6. Method-agreement analysis
    #
    # For each unique (HPO, SCT) pair, count how many distinct mapping methods
    # produced it. Bucket accepted (validity=10) pairs by that count and report
    # metrics — both exact ("exactly k methods") and cumulative ("≥ k methods").
    # -----------------------------------------------------------------------
    l.printHeader("6. Method-agreement analysis")

    method_agreement = (
        final_df.groupby([id_col, val_col])[attr_col]
        .nunique()
        .reset_index()
        .rename(columns={attr_col: "method_count"})
    )

    accepted_all = _dedupe_by_pair(
        final_df[final_df[validity_column] == validity_accepted], id_col, val_col
    )
    accepted_with_agree = accepted_all.merge(
        method_agreement, on=[id_col, val_col], how="left"
    )
    accepted_with_agree["method_count"] = (
        accepted_with_agree["method_count"].fillna(1).astype(int)
    )

    max_agree = int(accepted_with_agree["method_count"].max()) if not accepted_with_agree.empty else 0

    l.log("  Accepted pairs (validity=10) — exact bucket (exactly k methods agree):")
    for k in range(1, max_agree + 1):
        bucket = accepted_with_agree[accepted_with_agree["method_count"] == k]
        if bucket.empty:
            continue
        tp = _tp_from_df(bucket, gold_pairs, id_col, val_col)
        m  = _metrics(tp, len(bucket), gold_total, universe_total)
        l.log(f"\n    {k} method(s) agree — {len(bucket)} pairs, {tp} in gold")
        _log_metrics(l, m, f"      k={k}")

    l.log("\n  Cumulative: pairs agreed upon by AT LEAST k methods:")
    for k in range(1, max_agree + 1):
        bucket = accepted_with_agree[accepted_with_agree["method_count"] >= k]
        if bucket.empty:
            continue
        tp = _tp_from_df(bucket, gold_pairs, id_col, val_col)
        m  = _metrics(tp, len(bucket), gold_total, universe_total)
        l.log(f"\n    ≥ {k} method(s) — {len(bucket)} pairs, {tp} in gold")
        _log_metrics(l, m, f"      k≥{k}")

    # -----------------------------------------------------------------------
    # 7. Confidence threshold analysis
    #
    # Sweep confidence thresholds for similarity-based methods (Jaccard,
    # embedding, and their combination). Reports all four metrics at each
    # threshold, deduplicating by keeping the highest confidence per pair.
    # -----------------------------------------------------------------------
    l.printHeader("7. Confidence threshold analysis")

    for method in SIM_METHODS + ["combined"]:
        if method == "combined":
            pool = final_df[final_df[attr_col].isin(SIM_METHODS)].copy()
            method_label = "Jaccard + Embedding (combined)"
        else:
            pool = final_df[final_df[attr_col] == method].copy()
            method_label = method

        if pool.empty:
            continue

        pool_deduped = (
            pool.sort_values(confidence_column, ascending=False)
            .drop_duplicates(subset=[id_col, val_col])
        )

        l.log(f"\n  Method: {method_label}")

        for thresh in similarity_steps:
            subset = pool_deduped[pool_deduped[confidence_column] >= thresh]
            if subset.empty:
                l.log(f"    conf ≥ {thresh:.2f}: no pairs.")
                continue
            tp = _tp_from_df(subset, gold_pairs, id_col, val_col)
            m  = _metrics(tp, len(subset), gold_total, universe_total)
            l.log(
                f"    conf ≥ {thresh:.2f}: {len(subset):6} pairs, {tp:5} in gold | "
                f"P={m['precision']:.4f}  R={m['recall']:.4f}  "
                f"F1={m['f1']:.4f}  TPshare={m['universe_share']:.4f}"
            )

    # -----------------------------------------------------------------------
    # 8. Precision–recall curve (simulation across thresholds)
    #
    # For each similarity method and their combination, sweep a finer grid of
    # confidence thresholds and log (threshold, precision, recall, F1) so the
    # caller can plot or inspect the operating-point trade-off.
    # -----------------------------------------------------------------------
    l.printHeader("8. Precision–recall curve (fine-grained threshold sweep)")

    PR_STEPS = 50   # number of evenly-spaced threshold steps

    for method in SIM_METHODS + ["combined"]:
        if method == "combined":
            pool = final_df[final_df[attr_col].isin(SIM_METHODS)].copy()
            method_label = "Jaccard + Embedding (combined)"
        else:
            pool = final_df[final_df[attr_col] == method].copy()
            method_label = method

        if pool.empty:
            continue

        pool_deduped = (
            pool.sort_values(confidence_column, ascending=False)
            .drop_duplicates(subset=[id_col, val_col])
        )

        conf_vals  = pool_deduped[confidence_column].values
        min_conf   = float(np.min(conf_vals)) if len(conf_vals) else 0.0
        max_conf   = float(np.max(conf_vals)) if len(conf_vals) else 1.0
        thresholds = np.linspace(min_conf, max_conf, PR_STEPS)

        l.log(f"\n  PR curve — {method_label}  (threshold : precision / recall / F1):")
        best_f1_thresh = None
        best_f1_val    = -1.0

        for thresh in thresholds:
            subset = pool_deduped[pool_deduped[confidence_column] >= thresh]
            if subset.empty:
                continue
            tp = _tp_from_df(subset, gold_pairs, id_col, val_col)
            m  = _metrics(tp, len(subset), gold_total, universe_total)
            l.log(
                f"    {thresh:.4f} : P={m['precision']:.4f}  "
                f"R={m['recall']:.4f}  F1={m['f1']:.4f}  ({len(subset)} pairs)"
            )
            if m["f1"] > best_f1_val:
                best_f1_val    = m["f1"]
                best_f1_thresh = thresh

        if best_f1_thresh is not None:
            l.log(f"  → Best F1={best_f1_val:.4f} at threshold={best_f1_thresh:.4f}")

    # -----------------------------------------------------------------------
    # 9. LLM-specific analysis
    #
    #   A. Overall LLM accept/reject counts and precision.
    #   B. Per review-reason breakdown: acceptance rate and precision per
    #      validity code (1–10) that was sent to LLM review.
    #   C. LLM false-positive rate: of accepted pairs, what fraction are
    #      NOT in the gold standard?
    # -----------------------------------------------------------------------
    l.printHeader("9. LLM-specific analysis")

    llm_rows    = final_df[final_df[attr_col] == llm_mappings].copy()
    llm_deduped = _dedupe_by_pair(llm_rows, id_col, val_col)

    # The LLM reviewed all validity 1–10 pairs (including auto-accepted).
    reviewed_pool = _dedupe_by_pair(
        final_df[final_df[validity_column].isin(list(REVIEW_REASON.keys()))],
        id_col, val_col,
    )

    llm_accepted_pairs = set(zip(llm_deduped[id_col], llm_deduped[val_col]))
    reviewed_pairs     = set(zip(reviewed_pool[id_col], reviewed_pool[val_col]))

    n_reviewed = len(reviewed_pairs)
    n_accepted = len(llm_accepted_pairs)
    n_rejected = n_reviewed - n_accepted

    tp_llm = _tp_from_df(llm_deduped, gold_pairs, id_col, val_col)
    fp_llm = n_accepted - tp_llm
    m_llm  = _metrics(tp_llm, n_accepted, gold_total, universe_total)

    l.log(f"\n  A. Overall LLM pass:")
    l.log(f"    Pairs reviewed:  {n_reviewed}")
    l.log(f"    Accepted:        {n_accepted}  ({_safe_div(n_accepted, n_reviewed):.2%})")
    l.log(f"    Rejected:        {n_rejected}  ({_safe_div(n_rejected, n_reviewed):.2%})")
    l.log(f"    True positives:  {tp_llm}")
    l.log(f"    False positives: {fp_llm}  "
          f"(FP rate among accepted: {_safe_div(fp_llm, n_accepted):.2%})")
    _log_metrics(l, m_llm, "    Metrics")

    l.log(f"\n  B. Per review-reason breakdown:")
    for v_code, reason_label in sorted(REVIEW_REASON.items()):
        # Pairs that entered LLM with this validity code.
        reason_pool = _dedupe_by_pair(
            final_df[final_df[validity_column] == v_code], id_col, val_col
        )
        reason_pairs = set(zip(reason_pool[id_col], reason_pool[val_col]))

        # LLM-accepted pairs that came from this validity code.
        accepted_from_reason = llm_deduped[
            inGoldSeries(llm_deduped, reason_pairs, id_col, val_col)
        ]

        n_pool = len(reason_pairs)
        n_acc  = len(accepted_from_reason)
        if n_pool == 0:
            l.log(f"    validity={v_code} ({reason_label}): no pairs reviewed.")
            continue

        tp_r = _tp_from_df(accepted_from_reason, gold_pairs, id_col, val_col)
        m_r  = _metrics(tp_r, n_acc, gold_total, universe_total)
        l.log(
            f"\n    validity={v_code} ({reason_label}): "
            f"{n_pool} reviewed, {n_acc} accepted "
            f"({_safe_div(n_acc, n_pool):.2%}), {tp_r} in gold"
        )
        _log_metrics(l, m_r, "      Metrics")

    l.log(f"\n  C. LLM false positives (accepted but NOT in gold):")
    llm_fp = llm_deduped[~inGoldSeries(llm_deduped, gold_pairs, id_col, val_col)]
    l.log(f"    Count: {len(llm_fp)}")
    for _, row in llm_fp.head(ERROR_EXAMPLE_LIMIT).iterrows():
        l.log(
            f"    {row[id_col]} [{hpo_label(row[id_col])}]  →  "
            f"{row[val_col]} [{sct_label(row[val_col])}]"
        )
    if len(llm_fp) > ERROR_EXAMPLE_LIMIT:
        l.log(f"    ... and {len(llm_fp) - ERROR_EXAMPLE_LIMIT} more.")

    # -----------------------------------------------------------------------
    # 10. Mapping completeness
    #
    #   A. Coverage: fraction of all HPO concepts that received ≥1 accepted
    #      mapping, vs. fraction that received ≥1 generated (any validity).
    #   B. Multi-mapping rate: how many HPO concepts have more than one
    #      accepted SCT target, and are those multi-mappings correct?
    # -----------------------------------------------------------------------
    l.printHeader("10. Mapping completeness")

    all_hpo_concepts = set(hpo[hpo_ontology.config.id_column].unique())
    generated_hpo    = set(final_df[id_col].unique())
    accepted_v10     = _dedupe_by_pair(
        final_df[final_df[validity_column] == validity_accepted], id_col, val_col
    )
    accepted_hpo     = set(accepted_v10[id_col].unique())

    n_total_hpo = len(all_hpo_concepts)
    n_generated = len(generated_hpo & all_hpo_concepts)
    n_accepted  = len(accepted_hpo & all_hpo_concepts)

    l.log(f"\n  A. HPO concept coverage:")
    l.log(f"    Total HPO concepts (HP: namespace): {n_total_hpo}")
    l.log(
        f"    With ≥1 generated mapping:  {n_generated:6}  "
        f"({_safe_div(n_generated, n_total_hpo):.2%})"
    )
    l.log(
        f"    With ≥1 accepted mapping:   {n_accepted:6}  "
        f"({_safe_div(n_accepted, n_total_hpo):.2%})"
    )

    l.log(f"\n  B. Multi-mapping rate (accepted, validity=10):")
    sct_count_per_hpo = accepted_v10.groupby(id_col)[val_col].nunique()
    multi_mapped      = sct_count_per_hpo[sct_count_per_hpo > 1]
    l.log(
        f"    HPO concepts with >1 accepted SCT target: {len(multi_mapped)}  "
        f"({_safe_div(len(multi_mapped), len(sct_count_per_hpo)):.2%} of mapped concepts)"
    )

    for hpo_id, n_sct in multi_mapped.sort_values(ascending=False).head(10).items():
        our_scts  = set(accepted_v10.loc[accepted_v10[id_col] == hpo_id, val_col])
        gold_scts = set(graham.loc[graham[hpo_id_column] == hpo_id, sct_id_column])
        correct   = our_scts & gold_scts
        spurious  = our_scts - gold_scts
        l.log(
            f"    {hpo_id} [{hpo_label(hpo_id)}]: {n_sct} SCT targets — "
            f"{len(correct)} in gold, {len(spurious)} not in gold"
        )
        for sid in spurious:
            l.log(f"      spurious: {sid} [{sct_label(sid)}]")

    # -----------------------------------------------------------------------
    # 11. Missed gold-pair diagnosis
    #
    # For every gold pair NOT in the final accepted set, determine why:
    #   - NEVER_GENERATED       : pair absent from llm_filtered.csv entirely
    #   - DOMAIN_REJECTED       : present but validity=0 (domain mismatch)
    #   - FILTERED_OUT_HIERARCHY: present, validity in {1..7} (hierarchy
    #                             reversal steps)
    #   - LLM_REJECTED          : present, validity in {8,9} but not accepted
    #   - OTHER                 : any other reason
    # -----------------------------------------------------------------------
    l.printHeader("11. Missed gold-pair diagnosis")

    accepted_pairs_set = set(zip(accepted_v10[id_col], accepted_v10[val_col]))
    missed_gold        = gold_pairs - accepted_pairs_set

    l.log(f"  Total missed gold pairs: {len(missed_gold)} / {gold_total}")

    # For fast lookup: pair -> set of validity codes it appears under
    pair_validities: dict[tuple, set] = defaultdict(set)
    for _, row in final_df[[id_col, val_col, validity_column]].iterrows():
        pair_validities[(row[id_col], row[val_col])].add(row[validity_column])

    diagnosis_counts   = defaultdict(int)
    diagnosis_examples: dict[str, list] = defaultdict(list)

    for pair in missed_gold:
        if pair not in pair_validities:
            diagnosis_counts["NEVER_GENERATED"] += 1
            diagnosis_examples["NEVER_GENERATED"].append(pair)
        else:
            validities = pair_validities[pair]
            if 10 in validities:
                # Should not happen — accepted pairs are excluded from missed.
                diagnosis_counts["INCONSISTENCY"] += 1
            elif 0 in validities and len(validities) == 1:
                diagnosis_counts["DOMAIN_REJECTED"] += 1
                diagnosis_examples["DOMAIN_REJECTED"].append(pair)
            elif validities <= {1, 2, 3, 4, 5, 6, 7}:
                diagnosis_counts["FILTERED_OUT_HIERARCHY"] += 1
                diagnosis_examples["FILTERED_OUT_HIERARCHY"].append(pair)
            elif validities & {8, 9}:
                diagnosis_counts["LLM_REJECTED"] += 1
                diagnosis_examples["LLM_REJECTED"].append(pair)
            else:
                diagnosis_counts["OTHER"] += 1
                diagnosis_examples["OTHER"].append(pair)

    for category, count in sorted(diagnosis_counts.items(), key=lambda x: -x[1]):
        l.log(f"\n  {category}: {count} pairs")
        for pair in diagnosis_examples[category][:5]:
            hid, sid = pair
            l.log(f"    {hid} [{hpo_label(hid)}]  →  {sid} [{sct_label(sid)}]")
        if count > 5:
            l.log(f"    ... and {count - 5} more.")

    # Confidence distribution for generated-but-missed pairs
    l.log("\n  Confidence distribution for generated-but-missed gold pairs:")
    never_generated_set  = set(diagnosis_examples.get("NEVER_GENERATED", []))
    generated_missed     = missed_gold - never_generated_set
    if generated_missed:
        gm_mask = inGoldSeries(final_df, generated_missed, id_col, val_col)
        gm_df   = final_df[gm_mask]
        gm_best = (
            gm_df.sort_values(confidence_column, ascending=False)
            .drop_duplicates(subset=[id_col, val_col])
        )
        conf_vals = gm_best[confidence_column].dropna()
        if not conf_vals.empty:
            for thresh in similarity_steps:
                n_above = int((conf_vals >= thresh).sum())
                l.log(
                    f"    {n_above:5} missed gold pairs had best confidence ≥ {thresh:.2f}  "
                    f"({_safe_div(n_above, len(conf_vals)):.2%} of generated-but-missed)"
                )
    else:
        l.log("    All missed gold pairs were never generated.")

    # -----------------------------------------------------------------------
    # 12. Pairwise method overlap and unique contributions
    #
    #   For every pair of methods: how many (HPO, SCT) pairs did BOTH produce,
    #   and what fraction of those overlapping pairs are in the gold standard?
    #
    #   For each method individually: how many gold pairs does it find that NO
    #   other method finds? (unique contribution — methods you cannot remove.)
    # -----------------------------------------------------------------------
    l.printHeader("12. Pairwise method overlap and unique contributions")

    # Build pair set per method (all pairs generated, regardless of validity).
    method_pair_sets: dict[str, set] = {}
    for method in ALL_METHODS:
        rows = final_df[final_df[attr_col] == method]
        method_pair_sets[method] = set(zip(rows[id_col], rows[val_col]))

    l.log("  Pairwise overlap (pairs generated by both methods):")
    for m1, m2 in combinations(ALL_METHODS, 2):
        s1, s2  = method_pair_sets[m1], method_pair_sets[m2]
        overlap = s1 & s2
        if not overlap:
            l.log(f"    {m1} ∩ {m2}: 0 pairs")
            continue
        in_g  = len(overlap & gold_pairs)
        prec  = _safe_div(in_g, len(overlap))
        l.log(
            f"    {m1} ∩ {m2}: {len(overlap):6} pairs, "
            f"{in_g} in gold ({prec:.2%} precision)"
        )

    l.log("\n  Unique gold contributions (gold pairs found by ONLY this method):")
    for method in ALL_METHODS:
        own_pairs   = method_pair_sets[method]
        other_pairs = set().union(*(
            s for m, s in method_pair_sets.items() if m != method
        ))
        unique_gold = (own_pairs - other_pairs) & gold_pairs
        l.log(
            f"    {method}: {len(unique_gold)} gold pairs found by no other method"
        )
        for pair in list(unique_gold)[:5]:
            hid, sid = pair
            l.log(f"      {hid} [{hpo_label(hid)}]  →  {sid} [{sct_label(sid)}]")
        if len(unique_gold) > 5:
            l.log(f"      ... and {len(unique_gold) - 5} more.")

    # -----------------------------------------------------------------------
    # 13. Hierarchy-depth analysis
    #
    # Are accepted/missed mappings concentrated at particular depths in HPO?
    # Reports mean depth and a histogram for: all HPO concepts, gold-covered
    # concepts, correctly accepted concepts, and missed gold concepts.
    # -----------------------------------------------------------------------
    l.printHeader("13. Hierarchy-depth analysis")

    def depth_stats(hpo_ids: set, label: str) -> None:
        if not hpo_ids:
            l.log(f"  {label}: (empty)")
            return
        depths = [_hpo_depth(h, hpo_parent_map) for h in hpo_ids]
        arr    = np.array(depths)
        l.log(
            f"  {label}: n={len(arr)}  "
            f"mean={arr.mean():.2f}  median={np.median(arr):.1f}  "
            f"min={arr.min()}  max={arr.max()}"
        )
        buckets = [(0, 2), (3, 5), (6, 9), (10, 14), (15, 999)]
        for lo, hi in buckets:
            count    = int(((arr >= lo) & (arr <= hi)).sum())
            label_hi = str(hi) if hi < 999 else "+"
            l.log(f"    depth {lo}–{label_hi}: {count}")

    correctly_accepted_hpo = set(
        accepted_v10.loc[
            inGoldSeries(accepted_v10, gold_pairs, id_col, val_col), id_col
        ].unique()
    )
    missed_gold_hpo = set(h for h, _ in missed_gold)

    depth_stats(all_hpo_concepts,       "All HPO concepts")
    depth_stats(gold_hpo,               "Gold-standard HPO concepts")
    depth_stats(correctly_accepted_hpo, "Correctly accepted HPO concepts")
    depth_stats(missed_gold_hpo,        "HPO concepts with ≥1 missed gold pair")

    # -----------------------------------------------------------------------
    # 14. Hierarchy-reversal quality check
    #
    # Validity codes 1–7 were rejected by the filter as hierarchy reversals.
    # How many of those rejected pairs are actually correct gold pairs?
    # This diagnoses whether the narrower-wins rule over-prunes.
    # -----------------------------------------------------------------------
    l.printHeader("14. Hierarchy-reversal quality check")

    reversal_df = _dedupe_by_pair(
        final_df[final_df[validity_column].isin(validity_hierarchy_reversals)],
        id_col, val_col,
    )
    tp_rev = _tp_from_df(reversal_df, gold_pairs, id_col, val_col)
    m_rev  = _metrics(tp_rev, len(reversal_df), gold_total, universe_total)

    l.log(f"  Pairs rejected as hierarchy reversals (validity=1–7): {len(reversal_df)}")
    l.log(f"  Of those, in gold standard (false negatives from filter): {tp_rev}")
    _log_metrics(l, m_rev, "  Metrics")

    if tp_rev > 0:
        l.log(f"\n  Examples of incorrectly demoted gold pairs:")
        false_neg = reversal_df[
            inGoldSeries(reversal_df, gold_pairs, id_col, val_col)
        ]
        for _, row in false_neg.head(ERROR_EXAMPLE_LIMIT).iterrows():
            l.log(
                f"    {row[id_col]} [{hpo_label(row[id_col])}]  →  "
                f"{row[val_col]} [{sct_label(row[val_col])}]"
            )
        if len(false_neg) > ERROR_EXAMPLE_LIMIT:
            l.log(f"    ... and {len(false_neg) - ERROR_EXAMPLE_LIMIT} more.")

    # -----------------------------------------------------------------------
    # 15. Error analysis
    #
    #   A. Gold HPO concepts with no generated mapping at all.
    #   B. Gold HPO concepts generated but never accepted.
    #   C. Accepted mappings for HPO concepts absent from gold.
    #   D. HPO concepts in both gold and accepted, but wrong SCT target
    #      (split into entirely wrong and partially wrong).
    # -----------------------------------------------------------------------
    l.printHeader("15. Error analysis")

    accepted_sct_per_hpo: dict[str, set] = defaultdict(set)
    for _, row in accepted_v10[[id_col, val_col]].iterrows():
        accepted_sct_per_hpo[row[id_col]].add(row[val_col])

    gold_sct_per_hpo: dict[str, set] = defaultdict(set)
    for _, row in graham[[hpo_id_column, sct_id_column]].iterrows():
        gold_sct_per_hpo[row[hpo_id_column]].add(row[sct_id_column])

    # A: Gold HPO concepts entirely absent from generated mappings
    l.log("\n  A. Gold HPO concepts with NO generated mapping:")
    cat_a = sorted(gold_hpo - generated_hpo)
    l.log(f"    Count: {len(cat_a)}")
    for hid in cat_a[:ERROR_EXAMPLE_LIMIT]:
        l.log(f"    {hid}  [{hpo_label(hid)}]")
    if len(cat_a) > ERROR_EXAMPLE_LIMIT:
        l.log(f"    ... and {len(cat_a) - ERROR_EXAMPLE_LIMIT} more.")

    # B: Gold HPO concepts generated but not accepted
    l.log("\n  B. Gold HPO concepts present in generated mappings but with NO accepted mapping:")
    cat_b = sorted((gold_hpo & generated_hpo) - accepted_hpo)
    l.log(f"    Count: {len(cat_b)}")
    for hid in cat_b[:ERROR_EXAMPLE_LIMIT]:
        gold_scts = gold_sct_per_hpo.get(hid, set())
        l.log(
            f"    {hid}  [{hpo_label(hid)}]  "
            f"gold SCT(s): {', '.join(sct_label(s) for s in gold_scts)}"
        )
    if len(cat_b) > ERROR_EXAMPLE_LIMIT:
        l.log(f"    ... and {len(cat_b) - ERROR_EXAMPLE_LIMIT} more.")

    # C: Accepted mappings for HPO concepts absent from gold
    l.log("\n  C. Accepted mappings for HPO concepts NOT in the gold standard:")
    cat_c = sorted(accepted_hpo - gold_hpo)
    l.log(f"    Count: {len(cat_c)} HPO concepts (may be correct but unverifiable)")
    for hid in cat_c[:ERROR_EXAMPLE_LIMIT]:
        our_scts = accepted_sct_per_hpo.get(hid, set())
        l.log(
            f"    {hid}  [{hpo_label(hid)}]  "
            f"mapped to: {', '.join(sct_label(s) for s in our_scts)}"
        )
    if len(cat_c) > ERROR_EXAMPLE_LIMIT:
        l.log(f"    ... and {len(cat_c) - ERROR_EXAMPLE_LIMIT} more.")

    # D: HPO concepts in both, but SCT target differs
    l.log("\n  D. HPO concepts in both gold and accepted, with WRONG SCT concept(s):")
    shared_hpo          = gold_hpo & accepted_hpo
    cat_d_exact_wrong   = []
    cat_d_partial_wrong = []

    for hid in sorted(shared_hpo):
        our_scts  = accepted_sct_per_hpo.get(hid, set())
        gold_scts = gold_sct_per_hpo.get(hid, set())
        correct   = our_scts & gold_scts
        if not correct:
            cat_d_exact_wrong.append((hid, our_scts, gold_scts))
        elif correct != gold_scts:
            cat_d_partial_wrong.append((hid, our_scts, gold_scts, correct))

    l.log(f"    Entirely wrong (no overlap with gold SCT): {len(cat_d_exact_wrong)}")
    for hid, our, gold in cat_d_exact_wrong[:10]:
        l.log(f"    {hid}  [{hpo_label(hid)}]")
        l.log(f"      Our:  {', '.join(sct_label(s) for s in our)}")
        l.log(f"      Gold: {', '.join(sct_label(s) for s in gold)}")
    if len(cat_d_exact_wrong) > 10:
        l.log(f"    ... and {len(cat_d_exact_wrong) - 10} more.")

    l.log(f"    Partially wrong (some gold SCTs missing): {len(cat_d_partial_wrong)}")
    for hid, our, gold, correct in cat_d_partial_wrong[:10]:
        missing = gold - our
        extra   = our - gold
        l.log(f"    {hid}  [{hpo_label(hid)}]")
        l.log(f"      Correct:  {', '.join(sct_label(s) for s in correct)}")
        l.log(f"      Missing:  {', '.join(sct_label(s) for s in missing)}")
        if extra:
            l.log(f"      Extra:    {', '.join(sct_label(s) for s in extra)}")
    if len(cat_d_partial_wrong) > 10:
        l.log(f"    ... and {len(cat_d_partial_wrong) - 10} more.")

    # -----------------------------------------------------------------------
    # 16. Structural quality warnings
    #
    #   A. Source collapse: one HPO concept mapped to two SCT concepts where
    #      one is a transitive ancestor of the other.
    #   B. Target collapse: two HPO concepts in an ancestor/descendant
    #      relationship both mapped to the same SCT concept.
    #   C. Cross-pair hierarchy reversal: HPO_1 is a transitive descendant of
    #      HPO_2, but the accepted SCT target of HPO_1 is a transitive
    #      ancestor of the accepted SCT target of HPO_2.
    # -----------------------------------------------------------------------
    l.printHeader("16. Structural quality warnings")

    # Build transitive closure lookups for both ontologies.
    # KNOWN ISSUE (kept on purpose for now): these walk the child -> parents
    # map, so *_descendants_eval actually return ANCESTORS. Warnings A/B
    # detect related pairs correctly but swap the broader/narrower labels;
    # warning C still needs to be checked. Switching to build_child_map
    # (parent -> children), as MappingSetBuilder does, fixes the direction.
    hpo_descendants_eval = make_descendant_lookup(build_parent_map(hpo, hpo_ontology))
    sct_descendants_eval = make_descendant_lookup(build_parent_map(sct, sct_ontology))

    # Work from the deduplicated accepted set.
    accepted_deduped = _dedupe_by_pair(
        final_df[final_df[validity_column] == validity_accepted], id_col, val_col
    )

    # Per-HPO accepted SCT set and per-SCT accepted HPO set for fast lookup.
    accepted_sct_for_struct: dict[str, set] = defaultdict(set)
    accepted_hpo_for_struct: dict[str, set] = defaultdict(set)
    for _, row in accepted_deduped[[id_col, val_col]].iterrows():
        accepted_sct_for_struct[row[id_col]].add(row[val_col])
        accepted_hpo_for_struct[row[val_col]].add(row[id_col])

    accepted_hpo_ids = list(accepted_sct_for_struct.keys())

    # -------------------------------------------------------------------
    # A. Source collapse
    # -------------------------------------------------------------------
    l.log("\n  WARNING A — Source collapse "
          "(HPO concept mapped to SCT concepts in ancestor/descendant relationship):")

    source_collapse_count = 0
    source_collapse_hpo   = []

    for hpo_id, sct_targets in accepted_sct_for_struct.items():
        if len(sct_targets) < 2:
            continue
        targets       = list(sct_targets)
        collapse_pairs = []
        for i in range(len(targets)):
            for j in range(i + 1, len(targets)):
                s1, s2 = targets[i], targets[j]
                if s2 in sct_descendants_eval(s1) or s1 in sct_descendants_eval(s2):
                    broader  = s1 if s2 in sct_descendants_eval(s1) else s2
                    narrower = s2 if s2 in sct_descendants_eval(s1) else s1
                    collapse_pairs.append((broader, narrower))
        if collapse_pairs:
            source_collapse_count += len(collapse_pairs)
            source_collapse_hpo.append((hpo_id, collapse_pairs))

    l.log(f"    Total source-collapse pairs found: {source_collapse_count} "
          f"across {len(source_collapse_hpo)} HPO concepts.")
    for hpo_id, pairs in source_collapse_hpo[:ERROR_EXAMPLE_LIMIT]:
        l.log(f"    {hpo_id} [{hpo_label(hpo_id)}]")
        for broader, narrower in pairs:
            l.log(f"      broader:  {broader} [{sct_label(broader)}]")
            l.log(f"      narrower: {narrower} [{sct_label(narrower)}]")
    if len(source_collapse_hpo) > ERROR_EXAMPLE_LIMIT:
        l.log(f"    ... and {len(source_collapse_hpo) - ERROR_EXAMPLE_LIMIT} more HPO concepts.")

    # -------------------------------------------------------------------
    # B. Target collapse
    # -------------------------------------------------------------------
    l.log("\n  WARNING B — Target collapse "
          "(related HPO concepts mapped to the same SCT concept):")

    target_collapse_count = 0
    target_collapse_sct   = []

    for sct_id, hpo_sources in accepted_hpo_for_struct.items():
        if len(hpo_sources) < 2:
            continue
        sources        = list(hpo_sources)
        collapse_pairs = []
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                h1, h2 = sources[i], sources[j]
                if h2 in hpo_descendants_eval(h1) or h1 in hpo_descendants_eval(h2):
                    broader  = h1 if h2 in hpo_descendants_eval(h1) else h2
                    narrower = h2 if h2 in hpo_descendants_eval(h1) else h1
                    collapse_pairs.append((broader, narrower))
        if collapse_pairs:
            target_collapse_count += len(collapse_pairs)
            target_collapse_sct.append((sct_id, collapse_pairs))

    l.log(f"    Total target-collapse pairs found: {target_collapse_count} "
          f"across {len(target_collapse_sct)} SCT concepts.")
    for sct_id, pairs in target_collapse_sct[:ERROR_EXAMPLE_LIMIT]:
        l.log(f"    {sct_id} [{sct_label(sct_id)}]")
        for broader, narrower in pairs:
            l.log(f"      broader HPO:  {broader} [{hpo_label(broader)}]")
            l.log(f"      narrower HPO: {narrower} [{hpo_label(narrower)}]")
    if len(target_collapse_sct) > ERROR_EXAMPLE_LIMIT:
        l.log(f"    ... and {len(target_collapse_sct) - ERROR_EXAMPLE_LIMIT} more SCT concepts.")

    # -------------------------------------------------------------------
    # C. Cross-pair hierarchy reversal
    # For each accepted HPO concept (child), find all accepted HPO concepts
    # that are transitive ancestors of it. For each such (child, ancestor)
    # pair, check whether the SCT target of the child is a transitive
    # ancestor of the SCT target of the ancestor — indicating an inverted
    # mapping direction.
    #
    # Efficiency: rather than checking all O(n^2) HPO pairs, we use the
    # pre-built transitive closure. For each child, the ancestor set is
    # derived by checking which accepted HPO concepts have the child in
    # their descendant set, using the memoized closure.
    # -------------------------------------------------------------------
    l.log("\n  WARNING C — Cross-pair hierarchy reversal "
          "(HPO hierarchy direction inverted in SCT mapping):")

    reversal_count    = 0
    reversal_examples = []
    accepted_hpo_set  = set(accepted_hpo_ids)

    for hpo_child_id in accepted_hpo_ids:
        # All accepted HPO concepts that are transitive ancestors of hpo_child_id.
        # An HPO concept h is an ancestor of hpo_child_id iff
        # hpo_child_id is in hpo_descendants_eval(h).
        hpo_ancestors_in_accepted = {
            h for h in accepted_hpo_set
            if h != hpo_child_id and hpo_child_id in hpo_descendants_eval(h)
        }

        if not hpo_ancestors_in_accepted:
            continue

        child_sct_targets = accepted_sct_for_struct[hpo_child_id]

        for hpo_parent_id in hpo_ancestors_in_accepted:
            parent_sct_targets = accepted_sct_for_struct[hpo_parent_id]

            for sct_child_target in child_sct_targets:
                for sct_parent_target in parent_sct_targets:
                    if sct_child_target == sct_parent_target:
                        continue
                    # Reversal: the child HPO maps to an SCT concept that is
                    # a transitive ancestor of the SCT concept the parent HPO
                    # maps to — the hierarchy direction is inverted.
                    if sct_parent_target in sct_descendants_eval(sct_child_target):
                        reversal_count += 1
                        reversal_examples.append((
                            hpo_parent_id, hpo_child_id,
                            sct_child_target, sct_parent_target,
                        ))

    l.log(f"    Total cross-pair hierarchy reversals found: {reversal_count}")
    for hpo_parent_id, hpo_child_id, sct_child_target, sct_parent_target \
            in reversal_examples[:ERROR_EXAMPLE_LIMIT]:
        l.log(
            f"    HPO parent: {hpo_parent_id} [{hpo_label(hpo_parent_id)}]  "
            f"→  SCT: {sct_parent_target} [{sct_label(sct_parent_target)}]"
        )
        l.log(
            f"    HPO child:  {hpo_child_id} [{hpo_label(hpo_child_id)}]  "
            f"→  SCT: {sct_child_target} [{sct_label(sct_child_target)}]"
        )
        l.log(
            f"      (SCT child target [{sct_label(sct_child_target)}] is an ancestor "
            f"of SCT parent target [{sct_label(sct_parent_target)}])"
        )
    if reversal_count > ERROR_EXAMPLE_LIMIT:
        l.log(f"    ... and {reversal_count - ERROR_EXAMPLE_LIMIT} more.")

    # -----------------------------------------------------------------------
    # 17. Writing final evaluation CSV
    #
    # All rows whose (HPO, SCT) pair was accepted (validity=10) or approved
    # by the LLM (attribute="llm") are kept — one row per contributing
    # method — so every mapping strategy that produced a given pair is
    # visible in the output. The in_gold flag is attached to every row.
    # -----------------------------------------------------------------------
    l.printHeader("17. Writing final evaluation CSV")

    accepted_pairs_for_csv = set(zip(
        final_df.loc[final_df[validity_column] == validity_accepted, id_col],
        final_df.loc[final_df[validity_column] == validity_accepted, val_col],
    ))
    llm_pairs_for_csv = set(zip(
        final_df.loc[final_df[attr_col] == llm_mappings, id_col],
        final_df.loc[final_df[attr_col] == llm_mappings, val_col],
    ))
    qualifying_pairs_for_csv = accepted_pairs_for_csv | llm_pairs_for_csv

    # Vectorised filter — no apply() row loop.
    pair_series  = pairSeries(final_df, id_col, val_col)
    final_accepted = final_df[pair_series.isin(qualifying_pairs_for_csv)].copy().reset_index(drop=True)
    final_accepted["in_gold"] = inGoldSeries(final_accepted, gold_pairs, id_col, val_col)

    output_path = evaluation_path
    writeHugeCSV(final_accepted, output_path)
    l.log(f"Final evaluation CSV written to: {output_path}")
    l.log(f"  Rows: {len(final_accepted)}  |  In gold: {final_accepted['in_gold'].sum()}")

    # -----------------------------------------------------------------------
    # 18. Overall summary
    # -----------------------------------------------------------------------
    l.printHeader("18. Overall summary — final accepted mappings (validity=10, deduplicated)")

    final_deduped = _dedupe_by_pair(
        final_df[final_df[validity_column] == validity_accepted], id_col, val_col
    )
    tp_final = _tp_from_df(final_deduped, gold_pairs, id_col, val_col)
    m_final  = _metrics(tp_final, len(final_deduped), gold_total, universe_total)

    l.log(f"  Accepted pairs:          {len(final_deduped)}")
    l.log(f"  Pairs in gold standard:  {tp_final}")
    l.log(f"  Gold standard size:      {gold_total}")
    l.log(f"  Universe size:           {universe_total}")
    _log_metrics(l, m_final, "  Final metrics")

    printCounts(
        final_df[final_df[validity_column] == validity_accepted],
        hpo_ontology,
        similarity_steps,
        graham,
        diff=True,
    )

    l.printHeader("Evaluation completed.")