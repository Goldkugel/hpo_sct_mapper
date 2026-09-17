import sys

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

from logger                 import Logger
from adapter                import BaseAdapter, writeHugeCSV
from MappingUtils           import loadOntologies, loadGold, isSCTDomainValid, evaluation, confidence_column, validity_column, printCounts, threshold_cosine_similarity, hpo_sct_mappings, hpo_umls_sct_mappings, umls_hpo_sct_mappings, embedding_mappings, jaccard_mappings
from adapter                import labelClass, childrenClass
from collections            import defaultdict
import pandas               as pd

EXACT_SIMILARITY = 1.0
_EPS             = 1e-9


def is_exact(value: float) -> bool:
    """Floating point safe check for 'similarity == 1.0'."""
    return abs(value - EXACT_SIMILARITY) < _EPS


def build_child_map(data: pd.DataFrame, adapter: BaseAdapter) -> dict:
    """
    Direct hierarchy lookup: for every row tagged as a "children" relation,
    records the direct hierarchy edge.
    """
    rows = data[data[adapter.config.attribute_column] == childrenClass]
    child_map = defaultdict(set)
    for parent, child in zip(rows[adapter.config.id_column], rows[adapter.config.value_column]):
        child_map[child].add(parent)
    return child_map


def make_descendant_lookup(child_map: dict):
    """
    Returns a memoized function descendants(node) -> frozenset of ALL nodes
    transitively reachable from `node` through child_map. Cycle-safe.
    Computed lazily per node and cached.
    """
    cache: dict = {}

    def descendants(node):
        if node in cache:
            return cache[node]
        seen = set()
        stack = [node]
        while stack:
            current = stack.pop()
            for child in child_map.get(current, ()):
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        cache[node] = frozenset(seen)
        return cache[node]

    return descendants


if __name__ == "__main__":

    l = Logger()

    l.printHeader("Filtering generated mappings (incremental anchor-based pipeline)")

    hpo, sct, umls, hpo_ontology, sct_ontology, umls_ontology = loadOntologies(";")
    hpo = hpo[hpo[hpo_ontology.config.id_column].str.startswith("HP:")].reset_index(drop=True)

    mask = sct[sct_ontology.config.attribute_column] == childrenClass
    val_col_sct = sct_ontology.config.value_column
    sct_vals = sct.loc[mask, val_col_sct].astype(str)
    already_prefixed = sct_vals.str.startswith("SNOMEDCT_US:")
    sct.loc[mask, val_col_sct] = "SNOMEDCT_US:" + sct_vals.where(already_prefixed, sct_vals)

    graham = loadGold("./data/input/graham/mapping_graham.csv", ";")

    l.log("Loading generated mappings...")
    mappings = pd.read_csv("./data/output/mapped/mapping.csv", sep=";")
    mappings = mappings[mappings[hpo_ontology.config.id_column].str.startswith("HP:")].reset_index(drop=True)
    l.log("Loading generated mappings completed.")

    # Initialise validity column to NaN; filled in per-row as the pipeline runs.
    mappings[validity_column] = pd.NA

    # Build a pair -> [row indices] lookup once, so mark_validity never has
    # to scan the full dataframe — O(k) per call instead of O(n).
    pair_to_indices: dict[tuple, list] = defaultdict(list)
    for idx, (hpo_id, sct_id) in enumerate(zip(mappings[hpo_ontology.config.id_column], mappings[hpo_ontology.config.value_column])):
        pair_to_indices[(hpo_id, sct_id)].append(idx)

    id_col   = hpo_ontology.config.id_column
    val_col  = hpo_ontology.config.value_column
    attr_col = hpo_ontology.config.attribute_column

    # ------------------------------------------------------------------
    # Hierarchy closures, built once and reused by all hierarchy checks.
    # ------------------------------------------------------------------
    hpo_child_map   = build_child_map(hpo, hpo_ontology)
    sct_child_map   = build_child_map(sct, sct_ontology)
    sct_descendants = make_descendant_lookup(sct_child_map)

    # ------------------------------------------------------------------
    # Incremental accepted-mapping state, built up step by step below.
    # accepted_pairs  : set of (hpo_id, sct_id) pairs accepted so far.
    # accepted_by_hpo : hpo_id -> {sct_id, ...} for fast per-concept lookup.
    # domain_invalid  : set of (hpo_id, sct_id) pairs marked validity=0.
    # ------------------------------------------------------------------
    accepted_pairs   = set()
    accepted_by_hpo  = defaultdict(set)
    domain_invalid   = set()
    anchor_pairs     = set()               # (hpo_id, sct_id) accepted as anchors; never demoted

    def mark_validity(hpo_id: str, sct_id: str, value: int) -> None:
        """Set validity_column on ALL rows sharing this pair via index lookup."""
        indices = pair_to_indices.get((hpo_id, sct_id))
        if indices:
            mappings.iloc[indices, mappings.columns.get_loc(validity_column)] = value

    def accept(hpo_id: str, sct_id: str, is_anchor: bool = False) -> None:
        """
        Register a pair as accepted (validity=5) and propagate to all rows
        sharing the same (hpo_id, sct_id), so all mapping strategies for
        this pair are marked accepted.
        """
        accepted_pairs.add((hpo_id, sct_id))
        accepted_by_hpo[hpo_id].add(sct_id)
        if is_anchor:
            anchor_pairs.add((hpo_id, sct_id))
        mark_validity(hpo_id, sct_id, 5)

    def demote(hpo_id: str, sct_id: str) -> None:
        """
        Remove a non-anchor accepted pair from the accepted set and mark it
        validity=1 (hierarchy reversal, steps 3-5). Only called when a narrower
        candidate displaces a broader already-accepted non-anchor mapping.
        """
        accepted_pairs.discard((hpo_id, sct_id))
        accepted_by_hpo[hpo_id].discard(sct_id)
        mark_validity(hpo_id, sct_id, 1)

    # Pre-build a Series of (hpo_id, sct_id) tuples aligned to mappings.index,
    # reused by evaluate_rejected for fast pair membership tests.
    mappings_pairs = pd.Series(
        list(zip(mappings[id_col], mappings[val_col])),
        index=mappings.index,
    )

    def evaluate_rejected(rejected_pairs: set, label: str, include_printcounts: bool = False) -> None:
        """
        Evaluate mappings rows whose pair appears in rejected_pairs.

        include_printcounts=True: additionally calls printCounts filtered to
        Jaccard and embedding rows only, where confidence varies meaningfully.
        Should not be used for UMLS-based or hierarchy reversal rejections
        where confidence is always 1.0.
        """
        if not rejected_pairs:
            return
        mask = mappings_pairs.isin(rejected_pairs)
        rejected = mappings[mask]
        evaluation(rejected, graham, label, hpo_ontology)
        if include_printcounts:
            rejected_sim = rejected[rejected[attr_col].isin([jaccard_mappings, embedding_mappings])]
            printCounts(rejected_sim, hpo_ontology, [1.0, 0.99, 0.95, 0.9, 0.8, threshold_cosine_similarity], graham, True)

    # ------------------------------------------------------------------
    # Hierarchy check: transitive within-concept, narrower-wins.
    # For the SAME HPO concept, checks whether the candidate's SCT target
    # is in an ancestor/descendant relationship with any other already-
    # accepted SCT target for that concept.
    #   - Candidate narrower than a non-anchor accepted pair -> "demote"
    #   - Candidate narrower than an anchor                  -> "reject"
    #     (anchors are immutable; candidate loses regardless)
    #   - Candidate broader than any accepted pair           -> "reject"
    #   - No relationship                                    -> "ok"
    # Cross-concept comparisons are omitted to avoid false positives.
    # Returns (action, conflicting_sct_id_or_None).
    # ------------------------------------------------------------------
    def check_hierarchy(hpo_id: str, sct_id: str):
        candidate_desc_sct = sct_descendants(sct_id)
        for other_sct in accepted_by_hpo.get(hpo_id, ()):
            if other_sct == sct_id:
                continue
            # Candidate is a descendant of other_sct -> candidate is narrower.
            if sct_id in sct_descendants(other_sct):
                if (hpo_id, other_sct) in anchor_pairs:
                    return "reject", other_sct   # anchor is immutable
                return "demote", other_sct
            # Other_sct is a descendant of candidate -> candidate is broader.
            if other_sct in candidate_desc_sct:
                return "reject", other_sct
        return "ok", None

    # ------------------------------------------------------------------
    # Similarity corroboration lookups, shared by steps 3-5.
    # A pair counts as "exact" if AT LEAST ONE of its rows hits 1.0.
    # ------------------------------------------------------------------
    jaccard_rows   = mappings[mappings[attr_col] == jaccard_mappings]
    embedding_rows = mappings[mappings[attr_col] == embedding_mappings]

    jaccard_exact_pairs = set(
        zip(
            jaccard_rows.loc[jaccard_rows[confidence_column].apply(is_exact), id_col],
            jaccard_rows.loc[jaccard_rows[confidence_column].apply(is_exact), val_col],
        )
    )
    embedding_exact_pairs = set(
        zip(
            embedding_rows.loc[embedding_rows[confidence_column].apply(is_exact), id_col],
            embedding_rows.loc[embedding_rows[confidence_column].apply(is_exact), val_col],
        )
    )

    # ------------------------------------------------------------------
    # Step 1: anchor set = all mappings already present in HPO.
    # Anchors with Jaccard=1 OR cosine=1 are accepted immediately and
    # marked as immutable. Anchors that fail the similarity criterion
    # are checked against the hierarchy and accepted (protected) if no
    # reversal is found, or marked validity=1 otherwise.
    # ------------------------------------------------------------------
    l.printHeader("Step 1: Building anchor set from existing HPO mappings")

    anchors = mappings[mappings[attr_col] == hpo_sct_mappings].drop_duplicates(subset=[id_col, val_col])
    l.log(f"Anchor mappings found: {len(anchors.index)}")

    qualifying_count     = 0
    non_qualifying_accepted = 0
    non_qualifying_reversal = 0

    for hpo_id, sct_id in zip(anchors[id_col], anchors[val_col]):
        pair = (hpo_id, sct_id)
        has_jaccard = pair in jaccard_exact_pairs
        has_cosine  = pair in embedding_exact_pairs

        if has_jaccard or has_cosine:
            # Full anchor: accepted and immutable.
            accept(hpo_id, sct_id, is_anchor=True)
            qualifying_count += 1
        else:
            # Does not meet similarity criterion: check hierarchy against
            # already-accepted anchors, then accept as protected if clear.
            action, _ = check_hierarchy(hpo_id, sct_id)
            if action in ("ok", "demote"):
                # Accept as protected (is_anchor=True) so later steps cannot
                # demote it, but it was not a full similarity-qualifying anchor.
                accept(hpo_id, sct_id, is_anchor=True)
                non_qualifying_accepted += 1
            else:
                mark_validity(hpo_id, sct_id, 2)
                non_qualifying_reversal += 1

    l.log(f"Qualifying anchors (Jaccard=1 OR cosine=1): {qualifying_count}")
    l.log(f"Non-qualifying anchors accepted (no hierarchy reversal): {non_qualifying_accepted}")
    l.log(f"Non-qualifying anchors rejected (hierarchy reversal): {non_qualifying_reversal}")
    l.log(f"Total accepted after Step 1: {len(accepted_pairs)}")
    evaluation(mappings[mappings[validity_column] == 5], graham, "After Step 1 — accepted:", hpo_ontology)

    # ------------------------------------------------------------------
    # Step 2: mark non-anchor mappings with a domain mismatch as
    # validity=0. They are excluded from all subsequent steps.
    # ------------------------------------------------------------------
    l.printHeader("Step 2: Marking non-anchor mappings with domain mismatch as invalid (validity=0)")

    sct_labels = (
        sct[sct[sct_ontology.config.attribute_column] == labelClass]
        [[sct_ontology.config.id_column, sct_ontology.config.value_column]]
        .groupby(sct_ontology.config.id_column)[sct_ontology.config.value_column]
        .apply(list)
        .to_dict()
    )

    non_anchors = mappings[mappings[attr_col] != hpo_sct_mappings].drop_duplicates(subset=[id_col, val_col])
    domain_mismatch_count = 0
    domain_rejected_pairs = set()

    for hpo_id, sct_id in zip(non_anchors[id_col], non_anchors[val_col]):
        pair = (hpo_id, sct_id)
        if pair in accepted_pairs:
            continue
        pts = sct_labels.get(sct_id, [])
        if not isSCTDomainValid(pts):
            mark_validity(hpo_id, sct_id, 0)
            domain_invalid.add(pair)
            domain_rejected_pairs.add(pair)
            domain_mismatch_count += 1

    l.log(f"{domain_mismatch_count} non-anchor pairs marked invalid due to domain mismatch.")
    evaluate_rejected(domain_rejected_pairs, "Step 2 — domain mismatch rejections:")

    # ------------------------------------------------------------------
    # Shared implementation for steps 3 and 4.
    # Accepts a pair if:
    #   - not already accepted or domain-invalid
    #   - Jaccard=1 OR cosine=1
    #   - no hierarchy reversal  -> validity=2
    # If Jaccard=1 OR cosine=1 but hierarchy reversal -> validity=1
    # ------------------------------------------------------------------
    def process_corroborated_step(step_name: str, attribute_value: str) -> None:
        l.printHeader(step_name)
        candidates = mappings[mappings[attr_col] == attribute_value].drop_duplicates(subset=[id_col, val_col])

        accepted_count       = 0
        reversal_count       = 0
        rejected_count       = 0
        demoted_count        = 0
        low_similarity_pairs = set()
        reversal_pairs       = set()

        for hpo_id, sct_id in zip(candidates[id_col], candidates[val_col]):
            pair = (hpo_id, sct_id)

            if pair in accepted_pairs:
                continue

            if pair in domain_invalid:
                continue

            has_jaccard = pair in jaccard_exact_pairs
            has_cosine  = pair in embedding_exact_pairs

            if not has_jaccard and not has_cosine:
                low_similarity_pairs.add(pair)
                rejected_count += 1
                continue

            action, conflict = check_hierarchy(hpo_id, sct_id)

            if action == "ok":
                accept(hpo_id, sct_id)
                accepted_count += 1

            elif action == "demote":
                # Candidate is narrower than the accepted non-anchor conflict.
                # Demote the broader accepted mapping and accept the candidate.
                other_sct = conflict
                demote(hpo_id, other_sct)
                reversal_pairs.add((hpo_id, other_sct))
                demoted_count += 1
                accept(hpo_id, sct_id)
                accepted_count += 1

            elif action == "reject":
                # Candidate is broader than an accepted mapping -> reject it.
                mark_validity(hpo_id, sct_id, 1)
                reversal_pairs.add(pair)
                reversal_count += 1

        l.log(f"{step_name}: {accepted_count} accepted, {reversal_count} rejected (hierarchy reversal), "
              f"{demoted_count} demoted (broader displaced), {rejected_count} rejected (low similarity).")
        evaluation(mappings[mappings[validity_column] == 5], graham, f"After {step_name} — accepted:", hpo_ontology)
        evaluate_rejected(low_similarity_pairs, f"{step_name} — low similarity rejections:", include_printcounts=True)
        evaluate_rejected(reversal_pairs,       f"{step_name} — hierarchy reversal rejections:")

    # ------------------------------------------------------------------
    # Step 3: HPO -> UMLS -> SNOMED CT mappings.
    # ------------------------------------------------------------------
    process_corroborated_step("Step 3: HPO -> UMLS -> SNOMED CT mappings", hpo_umls_sct_mappings)

    # ------------------------------------------------------------------
    # Step 4: mappings connecting HPO and SNOMED CT via a shared UMLS CUI.
    # ------------------------------------------------------------------
    process_corroborated_step("Step 4: Shared-UMLS-CUI mappings", umls_hpo_sct_mappings)

    # ------------------------------------------------------------------
    # Step 5: remaining candidates, requiring BOTH Jaccard=1.0 AND
    # cosine=1.0 (stricter than steps 3 and 4).
    # If both are exact but hierarchy reversal -> validity=1.
    # ------------------------------------------------------------------
    l.printHeader("Step 5: Adding remaining Jaccard/cosine mappings (both required)")

    step5_pool = mappings[mappings[attr_col].isin([jaccard_mappings, embedding_mappings])]
    seen_pairs = set()
    step5_candidate_pairs = []
    for hpo_id, sct_id in zip(step5_pool[id_col], step5_pool[val_col]):
        pair = (hpo_id, sct_id)
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            step5_candidate_pairs.append(pair)

    step5_accepted           = 0
    step5_reversal           = 0
    step5_rejected           = 0
    step5_demoted            = 0
    step5_low_similarity_pairs = set()
    step5_reversal_pairs       = set()

    for hpo_id, sct_id in step5_candidate_pairs:
        pair = (hpo_id, sct_id)

        if pair in accepted_pairs:
            continue

        if pair in domain_invalid:
            continue

        if pair not in jaccard_exact_pairs or pair not in embedding_exact_pairs:
            step5_low_similarity_pairs.add(pair)
            step5_rejected += 1
            continue

        action, conflict = check_hierarchy(hpo_id, sct_id)

        if action == "ok":
            accept(hpo_id, sct_id)
            step5_accepted += 1

        elif action == "demote":
            other_sct = conflict
            demote(hpo_id, other_sct)
            step5_reversal_pairs.add((hpo_id, other_sct))
            step5_demoted += 1
            accept(hpo_id, sct_id)
            step5_accepted += 1

        elif action == "reject":
            mark_validity(hpo_id, sct_id, 1)
            step5_reversal_pairs.add(pair)
            step5_reversal += 1

    l.log(f"Step 5: {step5_accepted} accepted, {step5_reversal} rejected (hierarchy reversal), "
          f"{step5_demoted} demoted (broader displaced), {step5_rejected} rejected (low similarity).")
    evaluation(mappings[mappings[validity_column] == 5], graham, "After Step 5 — accepted:", hpo_ontology)
    evaluate_rejected(step5_low_similarity_pairs, "Step 5 — low similarity rejections:", include_printcounts=True)
    evaluate_rejected(step5_reversal_pairs,       "Step 5 — hierarchy reversal rejections:")

    # ------------------------------------------------------------------
    # Step 6: best-candidate fallback for unmapped HPO concepts.
    # For every HPO concept that has neither an accepted mapping (validity=2)
    # nor a hierarchy-reversal candidate (validity=1) already, find the
    # highest-cosine and highest-Jaccard candidates (domain-valid only).
    # If both point to the same SCT concept, run the hierarchy check:
    #   - passes  -> mark validity=1 (LLM review)
    #   - fails   -> skip
    # ------------------------------------------------------------------
    l.printHeader("Step 6: Best-candidate fallback for unmapped HPO concepts")

    # Determine which HPO concepts already have at least one validity=1 or
    # validity=2 row — those are skipped entirely.
    handled_hpo = set(
        mappings.loc[mappings[validity_column].isin([1, 2, 3, 5]), id_col].unique()
    )

    # All HPO concepts that appear in the mappings dataframe.
    all_hpo = set(mappings[id_col].unique())
    unmapped_hpo = all_hpo - handled_hpo
    l.log(f"HPO concepts with no accepted or reversal mapping: {len(unmapped_hpo)}")

    # Build per-concept best-cosine and best-Jaccard lookups from the full
    # mappings pool, excluding domain-invalid pairs.
    # For each concept we want the SCT concept with the highest confidence
    # under each method, so group by (hpo_id) and pick argmax.
    valid_mask = ~mappings_pairs.isin(domain_invalid)
    valid_mappings = mappings[valid_mask]

    embedding_pool = valid_mappings[valid_mappings[attr_col] == embedding_mappings]
    jaccard_pool   = valid_mappings[valid_mappings[attr_col] == jaccard_mappings]

    # For each HPO concept, find the SCT with the highest confidence.
    def best_per_concept(pool: pd.DataFrame) -> dict:
        """Returns hpo_id -> (sct_id, confidence) for the top-scoring pair."""
        if pool.empty:
            return {}
        idx = pool.groupby(id_col)[confidence_column].idxmax()
        top = pool.loc[idx]
        return {
            row[id_col]: (row[val_col], row[confidence_column])
            for _, row in top.iterrows()
        }

    best_cosine  = best_per_concept(embedding_pool)
    best_jaccard = best_per_concept(jaccard_pool)

    step6_count   = 0
    step6_skipped = 0
    step6_pairs   = set()

    for hpo_id in unmapped_hpo:
        cosine_result  = best_cosine.get(hpo_id)
        jaccard_result = best_jaccard.get(hpo_id)

        if cosine_result is None or jaccard_result is None:
            step6_skipped += 1
            continue

        cosine_sct,  cosine_conf  = cosine_result
        jaccard_sct, jaccard_conf = jaccard_result

        # Both must agree on the same SCT concept.
        if cosine_sct != jaccard_sct:
            step6_skipped += 1
            continue

        sct_id = cosine_sct
        pair   = (hpo_id, sct_id)

        # Skip if already accepted or domain-invalid.
        if pair in accepted_pairs or pair in domain_invalid:
            step6_skipped += 1
            continue

        action, _ = check_hierarchy(hpo_id, sct_id)

        if action in ("ok", "demote"):
            # Hierarchy check passes (or candidate is narrower than a non-
            # anchor) — mark for LLM review rather than auto-accepting, since
            # similarity is below 1.
            mark_validity(hpo_id, sct_id, 3)
            step6_pairs.add(pair)
            step6_count += 1
        # "reject" -> skip, no validity assigned.

    l.log(f"Step 6: {step6_count} candidates marked for LLM review (validity=3), "
          f"{step6_skipped} concepts skipped (no agreeing candidate or domain-invalid).")
    evaluate_rejected(step6_pairs, "Step 6 — best-candidate fallback for LLM review:")
    evaluation(
        mappings[mappings[validity_column].isin([1, 2, 3, 5])],
        graham,
        "After Step 6 — accepted + LLM candidates (validity=1,2,3 and validity=5):",
        hpo_ontology,
    )

    # ------------------------------------------------------------------
    # Step 7: last-resort fallback for still-unmapped HPO concepts.
    # For every HPO concept that still has no validity=1 or validity=2
    # mapping after step 6, add the single highest-confidence candidate
    # (across all methods, excluding domain-invalid pairs) and mark it
    # validity=3.
    # ------------------------------------------------------------------
    l.printHeader("Step 7: Last-resort fallback for still-unmapped HPO concepts")

    # Recompute handled concepts after step 6 (validity=1 now includes
    # step 6 candidates).
    handled_hpo_after6 = set(
        mappings.loc[mappings[validity_column].isin([1, 2, 3, 5]), id_col].unique()
    )
    still_unmapped = all_hpo - handled_hpo_after6
    l.log(f"HPO concepts still without any mapping after step 6: {len(still_unmapped)}")

    # From the domain-valid pool, find the highest-confidence candidate
    # across all methods for each still-unmapped concept.
    step7_pool = valid_mappings[valid_mappings[id_col].isin(still_unmapped)]

    step7_count   = 0
    step7_skipped = 0

    if not step7_pool.empty:
        idx = step7_pool.groupby(id_col)[confidence_column].idxmax()
        top = step7_pool.loc[idx]

        for _, row in top.iterrows():
            hpo_id = row[id_col]
            sct_id = row[val_col]
            pair   = (hpo_id, sct_id)

            if pair in accepted_pairs or pair in domain_invalid:
                step7_skipped += 1
                continue

            indices = pair_to_indices.get(pair)
            if indices:
                mappings.iloc[indices, mappings.columns.get_loc(validity_column)] = 4
                step7_count += 1
            else:
                step7_skipped += 1

    l.log(f"Step 7: {step7_count} candidates marked as last-resort (validity=4), "
          f"{step7_skipped} concepts skipped.")

    step7_pairs = set(
        zip(
            mappings.loc[mappings[validity_column] == 4, id_col],
            mappings.loc[mappings[validity_column] == 4, val_col],
        )
    )
    evaluate_rejected(step7_pairs, "Step 7 — last-resort fallback:")

    # ------------------------------------------------------------------
    # Final output.
    # Write the full mappings dataframe (same format as input) with the
    # validity column filled in:
    #   0  = domain mismatch
    #   1  = hierarchy reversal (steps 3-5)
    #   2  = anchor hierarchy reversal (step 1, non-qualifying anchors)
    #   3  = step 6 LOW_SIMILARITY fallback
    #   4  = step 7 LAST_RESORT fallback
    #   5  = accepted
    # Rows with validity=NA were never reached by any step.
    # Only rows with validity in {0,1,2,3,4,5} are written to the output CSV.
    # ------------------------------------------------------------------
    final = mappings[mappings[validity_column].isin([0, 1, 2, 3, 4, 5])].copy()

    l.log(f"Total rows written: {len(final.index)} "
          f"(accepted={len(final[final[validity_column]==5])}, "
          f"reversal={len(final[final[validity_column]==1])}, "
          f"anchor_reversal={len(final[final[validity_column]==2])}, "
          f"low_similarity={len(final[final[validity_column]==3])}, "
          f"last_resort={len(final[final[validity_column]==4])}, "
          f"domain_mismatch={len(final[final[validity_column]==0])})")

    evaluation(final[final[validity_column] == 5], graham, "Final accepted mapping set:", hpo_ontology)
    printCounts(final[final[validity_column] == 5], hpo_ontology, [1.0, 0.99, 0.95, 0.9, 0.8, threshold_cosine_similarity], graham, True)

    writeHugeCSV(final, "./data/output/filtered/filtered.csv")

    # ------------------------------------------------------------------
    # Mapping constellation analysis.
    # For LLM review candidates (validity=1 and validity=3), count how
    # many unique (hpo_id, sct_id) pairs exist and classify them into
    # 1:1, 1:n, n:1, and m:n constellations based on how many SCT
    # concepts each HPO concept maps to and vice versa.
    # ------------------------------------------------------------------
    l.printHeader("Mapping constellation analysis (validity=1,2,3,4)")

    llm_pairs = (
        final[final[validity_column].isin([1, 2, 3, 4])]
        .drop_duplicates(subset=[id_col, val_col])[[id_col, val_col]]
    )

    # Count unique SCT targets per HPO concept and unique HPO sources per
    # SCT concept, considering only the LLM review pairs.
    sct_per_hpo = llm_pairs.groupby(id_col)[val_col].nunique()
    hpo_per_sct = llm_pairs.groupby(val_col)[id_col].nunique()

    # Classify each unique pair by its constellation.
    llm_pairs = llm_pairs.copy()
    llm_pairs["n_sct"] = llm_pairs[id_col].map(sct_per_hpo)
    llm_pairs["n_hpo"] = llm_pairs[val_col].map(hpo_per_sct)

    def classify(row):
        if row["n_sct"] == 1 and row["n_hpo"] == 1:
            return "1:1"
        elif row["n_sct"] > 1 and row["n_hpo"] == 1:
            return "1:n"
        elif row["n_sct"] == 1 and row["n_hpo"] > 1:
            return "n:1"
        else:
            return "m:n"

    llm_pairs["constellation"] = llm_pairs.apply(classify, axis=1)
    counts = llm_pairs["constellation"].value_counts()

    total_pairs = len(llm_pairs)
    l.log(f"Total unique pairs for LLM review: {total_pairs}")
    for constellation in ["1:1", "1:n", "n:1", "m:n"]:
        count = counts.get(constellation, 0)
        l.log(f"  {constellation}: {count:6} pairs ({count / total_pairs:.1%})")

    l.printHeader("Filtering generated mappings completed")