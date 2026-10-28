import sys

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

from logger                 import Logger
from adapter                import writeHugeCSV
from MappingUtils           import (
    similarity_steps, exact_lexical_mappings, loadInputs, isSCTDomainValid, evaluation,
    confidence_column, validity_column, printCounts, threshold_cosine_similarity,
    hpo_sct_mappings, hpo_umls_sct_mappings, umls_hpo_sct_mappings, embedding_mappings,
    jaccard_mappings, extract_snomed_domain, build_child_map, make_descendant_lookup,
    buildAttributeLookup, pairSeries, keepHPOConcepts, mapping_path, filtered_path,
    validity_domain_mismatch, validity_anchor_reversal, validity_exact_lexical_reversal,
    validity_hpo_xref_reversal, validity_umls_bridge_reversal, validity_umls_cui_reversal,
    validity_jaccard_cosine_reversal, validity_jaccard_only_reversal,
    validity_best_candidate, validity_last_resort, validity_accepted,
    validity_hierarchy_reversals, validity_reviewable, validity_all,
)
from adapter                import labelClass
from collections            import defaultdict
from typing                 import Optional
import pandas               as pd
import numpy                as np

EXACT_SIMILARITY = 1.0
_EPS             = 1e-9


def is_exact(value: float) -> bool:
    """Floating point safe check for 'similarity == 1.0'."""
    return abs(value - EXACT_SIMILARITY) < _EPS


if __name__ == "__main__":

    l = Logger()

    l.printHeader("Filtering generated mappings")

    hpo, sct, umls, hpo_ontology, sct_ontology, umls_ontology, graham = loadInputs(";")

    l.log("Loading generated mappings...")
    mappings = pd.read_csv(mapping_path, sep=";")
    mappings = keepHPOConcepts(mappings, hpo_ontology.config.id_column)
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
    # hpo_child_map is not used by the current hierarchy check (which
    # operates only within the SCT hierarchy) but is retained for the
    # TODO: accept both candidates when there is no overlap within three
    # generations in either hierarchy.
    # ------------------------------------------------------------------
    hpo_child_map   = build_child_map(hpo, hpo_ontology)
    sct_child_map   = build_child_map(sct, sct_ontology)
    l.log(f"SCT child map: {len(sct_child_map.keys())} parent nodes, "
          f"{sum(len(v) for v in sct_child_map.values())} edges.")
    l.log(f"HPO child map: {len(hpo_child_map.keys())} parent nodes, "
          f"{sum(len(v) for v in hpo_child_map.values())} edges.")
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
    classified_pairs = set()               # (hpo_id, sct_id) assigned any validity code so far

    def mark_validity(hpo_id: str, sct_id: str, value: int, replace: Optional[int] = None) -> int:
        """
        Set validity_column on rows sharing this pair via index lookup.
        Only rows that still have validity=NA are updated — once a row has
        been assigned any validity code it is never overwritten, ensuring
        that each step's marks are permanent and independent. The single
        exception is `replace`: rows currently holding that code are
        overwritten too (used by demote() to turn an accepted 10 into a
        reversal code).

        Returns the number of rows that were changed.
        """
        changed = 0
        indices = pair_to_indices.get((hpo_id, sct_id))
        if indices:
            loc = mappings.columns.get_loc(validity_column)
            for idx in indices:
                current = mappings.iloc[idx, loc]
                if pd.isna(current) or (replace is not None and current == replace):
                    mappings.iloc[idx, loc] = value
                    changed += 1
            classified_pairs.add((hpo_id, sct_id))
        return changed

    def accept(hpo_id: str, sct_id: str, is_anchor: bool = False) -> None:
        """
        Register a pair as accepted (validity=10) and propagate to all rows
        sharing the same (hpo_id, sct_id), so all mapping strategies for
        this pair are marked accepted.
        """
        accepted_pairs.add((hpo_id, sct_id))
        accepted_by_hpo[hpo_id].add(sct_id)
        if is_anchor:
            anchor_pairs.add((hpo_id, sct_id))
        mark_validity(hpo_id, sct_id, validity_accepted)

    def demote(hpo_id: str, sct_id: str, reversal_validity: int = 1) -> None:
        """
        Remove a non-anchor accepted pair from the accepted set and mark it
        with the given reversal_validity code. Only called when a narrower
        candidate displaces a broader already-accepted non-anchor mapping.
        The reversal_validity should match the step that triggered the demotion.
        """
        accepted_pairs.discard((hpo_id, sct_id))
        accepted_by_hpo[hpo_id].discard(sct_id)
        # The pair's rows already hold validity=10, so they must be replaced
        # explicitly; the NA-only guard would otherwise leave them accepted.
        mark_validity(hpo_id, sct_id, reversal_validity, replace=10)

    # Pre-build a Series of (hpo_id, sct_id) tuples aligned to mappings.index,
    # reused by evaluate_rejected for fast pair membership tests.
    mappings_pairs = pairSeries(mappings, id_col, val_col)

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
            printCounts(rejected_sim, hpo_ontology, similarity_steps, graham, True)

    # ------------------------------------------------------------------
    # Pre-compute the highest confidence seen for each (hpo_id, sct_id)
    # pair across all methods. Used by check_hierarchy to compare
    # evidence quality before applying the narrower-wins rule.
    # ------------------------------------------------------------------
    best_confidence_per_pair: dict[tuple, float] = (
        mappings.groupby([id_col, val_col])[confidence_column]
        .max()
        .to_dict()
    )

    # ------------------------------------------------------------------
    # Hierarchy check: transitive within-concept, confidence-first.
    # For the SAME HPO concept, checks whether the candidate's SCT target
    # is in an ancestor/descendant relationship with any other already-
    # accepted SCT target for that concept.
    #
    # All accepted targets are inspected before deciding, so the result
    # does not depend on which conflict happens to be found first.
    #
    # Priority rules, applied in this order:
    #   1. Candidate broader than any accepted target -> "reject" (narrower
    #      accepted mapping always wins, regardless of confidence)
    #   2. Candidate narrower than an anchor           -> "reject" (anchors
    #      are immutable)
    #   3. Candidate narrower than non-anchor targets:
    #        any target more confident than candidate -> "reject"
    #        otherwise                                -> "demote" ALL of them
    #   4. No relationship                             -> "ok"
    #
    # Rule 3 is all-or-nothing: accepting the candidate while one broader
    # target stays accepted would leave a source collapse.
    #
    # Returns (action, conflicting_sct_ids): for "demote" the targets to
    # demote, for "reject" the targets that caused the rejection, for "ok"
    # an empty list.
    # TODO: if there is no overlap in hierarchy in three generations above or
    # below, accept both candidates.
    # ------------------------------------------------------------------
    def check_hierarchy(hpo_id: str, sct_id: str, candidate_confidence: float):
        candidate_desc_sct = sct_descendants(sct_id)
        broader            = []    # accepted targets the candidate is narrower than
        narrower           = []    # accepted targets the candidate is broader than

        # sorted() makes the result independent of set iteration order.
        for other_sct in sorted(accepted_by_hpo.get(hpo_id, ())):
            if other_sct == sct_id:
                continue
            if sct_id in sct_descendants(other_sct):
                broader.append(other_sct)
            elif other_sct in candidate_desc_sct:
                narrower.append(other_sct)

        if narrower:
            return "reject", narrower

        if not broader:
            return "ok", []

        anchors = [s for s in broader if (hpo_id, s) in anchor_pairs]
        if anchors:
            return "reject", anchors

        stronger = [
            s for s in broader
            if best_confidence_per_pair.get((hpo_id, s), 0.0) > candidate_confidence
        ]
        if stronger:
            return "reject", stronger

        return "demote", broader

    # ------------------------------------------------------------------
    # Similarity corroboration lookups, shared by steps 2c, 3, 4, 5, and 6.
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

    # All exact lexical pairs have confidence=1.0 by definition.
    exact_lexical_pairs = set(
        zip(
            mappings.loc[mappings[attr_col] == exact_lexical_mappings, id_col],
            mappings.loc[mappings[attr_col] == exact_lexical_mappings, val_col],
        )
    )

    # ------------------------------------------------------------------
    # SCT label lookup, built once and reused by the domain filter and
    # the domain distribution analysis after step 1.
    # ------------------------------------------------------------------
    sct_labels = buildAttributeLookup(sct, sct_ontology, labelClass)

    # ------------------------------------------------------------------
    # Step 1 (domain filter): mark ALL pairs whose SCT target does not
    # belong to an acceptable domain as validity=0. Running this first
    # ensures that domain-invalid pairs are permanently excluded before
    # any acceptance step runs, and since mark_validity only writes to
    # NA rows, no subsequent step can accidentally overwrite these marks.
    # isSCTDomainValid checks whether the SCT concept belongs to the
    # Clinical finding hierarchy (or another domain relevant to HPO
    # phenotype concepts); concepts outside this domain are excluded.
    # ------------------------------------------------------------------
    l.log("Step 1 (domain filter): Marking domain-mismatched pairs as invalid (validity=0)...")

    all_pairs = mappings.drop_duplicates(subset=[id_col, val_col])
    domain_mismatch_count = 0
    domain_rejected_pairs = set()

    for hpo_id, sct_id in zip(all_pairs[id_col], all_pairs[val_col]):
        pair = (hpo_id, sct_id)
        pts = sct_labels.get(sct_id, [])
        if not isSCTDomainValid(pts):
            mark_validity(hpo_id, sct_id, validity_domain_mismatch)
            domain_invalid.add(pair)
            domain_rejected_pairs.add(pair)
            domain_mismatch_count += 1

    l.log(f"{domain_mismatch_count} pairs marked invalid due to domain mismatch.")
    evaluate_rejected(domain_rejected_pairs, "Step 1 (domain filter) — domain mismatch rejections:")

    # ------------------------------------------------------------------
    # Step 2: anchor set = direct HPO xrefs to SNOMED CT concepts that
    # are also confirmed by an exact lexical match (case-insensitive).
    # A pair must appear in BOTH hpo_sct_mappings AND exact_lexical_mappings
    # to qualify as an anchor. This intersection provides evidence from two
    # independent sources: an explicit cross-reference in HPO and a label
    # match, giving high confidence without requiring Jaccard or cosine
    # corroboration. The hierarchy check is still applied to avoid accepting
    # a pair that conflicts with an already-accepted narrower mapping.
    # Anchors that pass are accepted as immutable (validity=10) and cannot
    # be demoted by any subsequent step. Anchors that fail the hierarchy
    # check are marked validity=1 and enter the LLM review queue.
    # ------------------------------------------------------------------
    l.log("Step 2: Building anchor set from existing HPO mappings...")
    l.log("Anchor set criteria: xref to SNOMED CT concept in HPO confirmed "
          "by an exact lexical match (case-insensitive).")

    hpo_sct_pairs = set(
        zip(
            mappings.loc[mappings[attr_col] == hpo_sct_mappings, id_col],
            mappings.loc[mappings[attr_col] == hpo_sct_mappings, val_col],
        )
    )

    anchor_candidates = hpo_sct_pairs & exact_lexical_pairs

    qualifying_count     = 0
    reversal_count       = 0
    non_qualifying_count = 0
    domain_invalid_count = 0

    for hpo_id, sct_id in sorted(anchor_candidates):
        # Domain-invalid pairs (validity=0, step 1) can never become anchors.
        if (hpo_id, sct_id) in domain_invalid:
            domain_invalid_count += 1
            continue

        action, _ = check_hierarchy(
            hpo_id, sct_id,
            best_confidence_per_pair.get((hpo_id, sct_id), 0.0)
        )
        if action in ("ok", "demote"):
            accept(hpo_id, sct_id, is_anchor=True)
            qualifying_count += 1
        else:
            mark_validity(hpo_id, sct_id, validity_anchor_reversal)
            reversal_count += 1

    # HPO xrefs that did not meet the exact lexical criterion are left with
    # validity=NA and will be evaluated in subsequent steps.
    non_qualifying_count = len(hpo_sct_pairs) - len(anchor_candidates)

    l.log(f"Qualifying anchors (hpo_sct AND exact lexical, no hierarchy reversal): {qualifying_count}")
    l.log(f"Qualifying anchors rejected (hierarchy reversal): {reversal_count}")
    l.log(f"Qualifying anchors skipped (domain-invalid, validity=0): {domain_invalid_count}")
    l.log(f"Non-qualifying HPO xrefs (no exact lexical match): {non_qualifying_count}")
    l.log(f"Total accepted after Step 2: {len(accepted_pairs)}")
    evaluation(mappings[mappings[validity_column] == validity_accepted], graham, "After Step 2 — accepted:", hpo_ontology)

    # ------------------------------------------------------------------
    # Step 2 domain analysis: distribution of SCT semantic domains among
    # the accepted anchor pairs. extract_snomed_domain parses the
    # parenthetical tag from the SCT preferred label, e.g.
    # "Physical activity (observable entity)" -> "observable entity".
    # ------------------------------------------------------------------
    anchor_domain_counts = defaultdict(int)
    for hpo_id, sct_id in accepted_pairs:
        labels = sct_labels.get(sct_id, [])
        domain = None
        for label in labels:
            if isinstance(label, str):
                domain = extract_snomed_domain(label)
                if domain:
                    break
        anchor_domain_counts[domain or "unknown"] += 1

    l.log("Step 2 — anchor set domain distribution:")
    for domain, count in sorted(anchor_domain_counts.items(), key=lambda x: -x[1]):
        l.log(f"  {domain}: {count}")

    # ------------------------------------------------------------------
    # Step 2b: exact lexical mappings not already accepted as anchors.
    # All remaining exact lexical pairs (i.e. those not in hpo_sct_pairs)
    # are processed here. Pairs that pass the hierarchy check are accepted
    # (validity=10); pairs that fail are marked validity=2 and enter the
    # LLM review queue. Domain-invalid pairs are already marked validity=0
    # and are skipped by the domain_invalid guard.
    # ------------------------------------------------------------------
    l.log("Step 2b: Processing remaining exact lexical mappings...")

    step1b_candidates = exact_lexical_pairs - anchor_candidates

    step1b_accepted  = 0
    step1b_reversal  = 0

    # sorted() fixes the processing order, which decides which of two
    # related candidates is accepted first; sets have no stable order.
    for hpo_id, sct_id in sorted(step1b_candidates):
        pair = (hpo_id, sct_id)

        if pair in domain_invalid:
            continue

        action, conflicts = check_hierarchy(
            hpo_id, sct_id,
            best_confidence_per_pair.get((hpo_id, sct_id), 0.0)
        )

        if action == "ok":
            accept(hpo_id, sct_id)
            step1b_accepted += 1
        elif action == "demote":
            # Candidate is narrower than accepted non-anchors — demote
            # the broader mappings and accept this one.
            for other_sct in conflicts:
                demote(hpo_id, other_sct, validity_exact_lexical_reversal)
            accept(hpo_id, sct_id)
            step1b_accepted += 1
        else:
            mark_validity(hpo_id, sct_id, validity_exact_lexical_reversal)
            step1b_reversal += 1

    l.log(f"Step 2b: {step1b_accepted} accepted, {step1b_reversal} rejected (hierarchy reversal, validity=2).")
    evaluation(mappings[mappings[validity_column] == validity_accepted], graham, "After Step 2b — accepted:", hpo_ontology)

    # ------------------------------------------------------------------
    # Step 2c: HPO xrefs (hpo_sct_mappings) not accepted as anchors in
    # step 2 that have BOTH Jaccard=1.0 AND cosine=1.0. These have strong
    # similarity corroboration from two independent methods. Pairs passing
    # the hierarchy check are accepted (validity=10). Pairs failing the
    # hierarchy check are marked validity=3 (distinct from steps 2 and 2b,
    # to allow separate analysis of this set). Pairs without both
    # similarity scores are left with validity=NA for downstream steps.
    # ------------------------------------------------------------------
    l.log("Step 2c: Processing HPO xrefs with Jaccard=1 AND cosine=1...")

    step1c_candidates = mappings[
        mappings[attr_col] == hpo_sct_mappings
    ].drop_duplicates(subset=[id_col, val_col])

    step1c_accepted   = 0
    step1c_reversal   = 0
    step1c_skipped    = 0
    step1c_classified = 0
    step1c_demoted    = 0

    step1c_reversal_pairs = set()

    for hpo_id, sct_id in zip(step1c_candidates[id_col], step1c_candidates[val_col]):
        pair = (hpo_id, sct_id)

        if pair in domain_invalid:
            continue

        # Pairs already accepted or rejected in steps 2/2b keep that decision.
        if pair in classified_pairs:
            step1c_classified += 1
            continue

        has_jaccard = pair in jaccard_exact_pairs
        has_cosine  = pair in embedding_exact_pairs

        if not has_jaccard or not has_cosine:
            step1c_skipped += 1
            continue

        action, conflicts = check_hierarchy(
            hpo_id, sct_id,
            best_confidence_per_pair.get((hpo_id, sct_id), 0.0)
        )

        if action == "ok":
            accept(hpo_id, sct_id)
            step1c_accepted += 1
        elif action == "demote":
            # Candidate is narrower than accepted non-anchors — demote
            # the broader mappings and accept this one.
            for other_sct in conflicts:
                demote(hpo_id, other_sct, validity_hpo_xref_reversal)
                step1c_reversal_pairs.add((hpo_id, other_sct))
            step1c_demoted += len(conflicts)
            accept(hpo_id, sct_id)
            step1c_accepted += 1
        else:
            mark_validity(hpo_id, sct_id, validity_hpo_xref_reversal)
            step1c_reversal_pairs.add(pair)
            step1c_reversal += 1

    l.log(f"Step 2c: {step1c_accepted} accepted, {step1c_reversal} rejected "
          f"(hierarchy reversal, validity=3), {step1c_demoted} demoted (broader displaced), "
          f"{step1c_skipped} skipped "
          f"(no Jaccard=1 AND cosine=1, left for downstream steps), "
          f"{step1c_classified} already classified in steps 2/2b.")
    evaluation(mappings[mappings[validity_column] == validity_accepted], graham, "After Step 2c — accepted:", hpo_ontology)
    evaluate_rejected(step1c_reversal_pairs, "Step 2c — hierarchy reversal rejections (validity=3):")



    # ------------------------------------------------------------------
    # Shared implementation for steps 3 and 4.
    # Accepts a pair (validity=10) if:
    #   - not domain-invalid
    #   - not already classified in an earlier step
    #   - Jaccard=1 OR cosine=1
    #   - check_hierarchy returns "ok" or "demote"
    # If similarity passes but hierarchy fails -> reversal_validity
    #   (validity=4 for step 3, validity=5 for step 4)
    # If neither Jaccard=1 nor cosine=1 -> validity remains NA (not
    # written to output); logged as low similarity rejection.
    # ------------------------------------------------------------------
    def process_corroborated_step(
        step_name: str,
        attribute_value: str,
        reversal_validity: int = 1,
    ) -> None:
        """
        Accept pairs for a given mapping attribute if:
          - not domain-invalid
          - Jaccard=1 OR cosine=1
          - check_hierarchy returns "ok" or "demote"
        Pairs already classified in an earlier step (accepted or rejected)
        are skipped: their rows can no longer be changed, so re-evaluating
        them would only inflate the step's counts.
        Pairs passing similarity but failing hierarchy are marked
        reversal_validity (default 1; pass 2 for higher-trust sources).
        Pairs failing similarity are left with validity=NA.
        """
        l.log(step_name)
        candidates = mappings[mappings[attr_col] == attribute_value].drop_duplicates(subset=[id_col, val_col])

        accepted_count       = 0
        reversal_count       = 0
        rejected_count       = 0
        demoted_count        = 0
        classified_count     = 0
        low_similarity_pairs = set()
        reversal_pairs       = set()

        for hpo_id, sct_id in zip(candidates[id_col], candidates[val_col]):
            pair = (hpo_id, sct_id)

            if pair in domain_invalid:
                continue

            if pair in classified_pairs:
                classified_count += 1
                continue

            has_jaccard = pair in jaccard_exact_pairs
            has_cosine  = pair in embedding_exact_pairs

            if not has_jaccard and not has_cosine:
                low_similarity_pairs.add(pair)
                rejected_count += 1
                continue

            cand_conf = best_confidence_per_pair.get((hpo_id, sct_id), 0.0)
            action, conflicts = check_hierarchy(hpo_id, sct_id, cand_conf)

            if action == "ok":
                accept(hpo_id, sct_id)
                accepted_count += 1

            elif action == "demote":
                # Candidate is narrower than the accepted non-anchor conflicts.
                # Demote the broader accepted mappings and accept the candidate.
                for other_sct in conflicts:
                    demote(hpo_id, other_sct, reversal_validity)
                    reversal_pairs.add((hpo_id, other_sct))
                demoted_count += len(conflicts)
                accept(hpo_id, sct_id)
                accepted_count += 1

            elif action == "reject":
                # Candidate is broader than an accepted mapping -> reject it.
                mark_validity(hpo_id, sct_id, reversal_validity)
                reversal_pairs.add(pair)
                reversal_count += 1

        l.log(f"{step_name}: {accepted_count} accepted, {reversal_count} rejected "
              f"(hierarchy reversal, validity={reversal_validity}), "
              f"{demoted_count} demoted (broader displaced), {rejected_count} rejected (low similarity), "
              f"{classified_count} already classified in earlier steps.")
        evaluation(mappings[mappings[validity_column] == validity_accepted], graham, f"After {step_name} — accepted:", hpo_ontology)
        evaluate_rejected(low_similarity_pairs, f"{step_name} — low similarity rejections:", include_printcounts=True)
        evaluate_rejected(reversal_pairs,       f"{step_name} — hierarchy reversal rejections:")

    # ------------------------------------------------------------------
    # Step 3: HPO -> UMLS -> SNOMED CT mappings.
    # Hierarchy reversals are marked validity=4.
    # ------------------------------------------------------------------
    process_corroborated_step(
        "Step 3: HPO -> UMLS -> SNOMED CT mappings",
        hpo_umls_sct_mappings,
        reversal_validity=validity_umls_bridge_reversal,
    )

    # ------------------------------------------------------------------
    # Step 4: mappings connecting HPO and SNOMED CT via a shared UMLS CUI.
    # Hierarchy reversals are marked validity=5.
    # ------------------------------------------------------------------
    process_corroborated_step(
        "Step 4: Shared-UMLS-CUI mappings",
        umls_hpo_sct_mappings,
        reversal_validity=validity_umls_cui_reversal,
    )

    # ------------------------------------------------------------------
    # Step 5: Jaccard=1 AND cosine=1 candidates with no reference-based
    # backing. Pairs that appear in any of hpo_sct_mappings,
    # hpo_umls_sct_mappings, or umls_hpo_sct_mappings are excluded here
    # — they were already handled in steps 1–4. Only pairs whose sole
    # evidence is lexical/embedding similarity are considered.
    # Hierarchy reversals are marked validity=6.
    # ------------------------------------------------------------------
    l.log("Step 5: Adding Jaccard=1 AND cosine=1 mappings with no reference-based backing.")

    # Build the set of pairs that have any reference-based attribute row,
    # regardless of whether they were accepted in earlier steps.
    reference_backed_pairs = set(
        zip(
            mappings.loc[mappings[attr_col].isin([
                hpo_sct_mappings, hpo_umls_sct_mappings, umls_hpo_sct_mappings
            ]), id_col],
            mappings.loc[mappings[attr_col].isin([
                hpo_sct_mappings, hpo_umls_sct_mappings, umls_hpo_sct_mappings
            ]), val_col],
        )
    )

    step5_pool = mappings[mappings[attr_col].isin([jaccard_mappings, embedding_mappings])]
    seen_pairs = set()
    step5_candidate_pairs = []
    for hpo_id, sct_id in zip(step5_pool[id_col], step5_pool[val_col]):
        pair = (hpo_id, sct_id)
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            step5_candidate_pairs.append(pair)

    step5_accepted             = 0
    step5_reversal             = 0
    step5_rejected             = 0
    step5_demoted              = 0
    step5_low_similarity_pairs = set()
    step5_reversal_pairs       = set()

    for hpo_id, sct_id in step5_candidate_pairs:
        pair = (hpo_id, sct_id)

        if pair in accepted_pairs:
            continue

        if pair in domain_invalid:
            continue

        # Skip pairs that have any reference-based mapping row — those
        # were already evaluated in steps 1–4.
        if pair in reference_backed_pairs:
            continue

        # Skip pairs rejected in an earlier step (e.g. exact lexical pairs
        # rejected in step 2b); their decision is final.
        if pair in classified_pairs:
            continue

        if pair not in jaccard_exact_pairs or pair not in embedding_exact_pairs:
            step5_low_similarity_pairs.add(pair)
            step5_rejected += 1
            continue

        cand_conf = best_confidence_per_pair.get((hpo_id, sct_id), 0.0)
        action, conflicts = check_hierarchy(hpo_id, sct_id, cand_conf)

        if action == "ok":
            accept(hpo_id, sct_id)
            step5_accepted += 1

        elif action == "demote":
            for other_sct in conflicts:
                demote(hpo_id, other_sct, validity_jaccard_cosine_reversal)
                step5_reversal_pairs.add((hpo_id, other_sct))
            step5_demoted += len(conflicts)
            accept(hpo_id, sct_id)
            step5_accepted += 1

        elif action == "reject":
            mark_validity(hpo_id, sct_id, validity_jaccard_cosine_reversal)
            step5_reversal_pairs.add(pair)
            step5_reversal += 1

    l.log(f"Step 5: {step5_accepted} accepted, {step5_reversal} rejected "
          f"(hierarchy reversal, validity=6), "
          f"{step5_demoted} demoted (broader displaced), {step5_rejected} rejected (low similarity).")
    evaluation(mappings[mappings[validity_column] == validity_accepted], graham, "After Step 5 — accepted:", hpo_ontology)
    evaluate_rejected(step5_low_similarity_pairs, "Step 5 — low similarity rejections:", include_printcounts=True)
    evaluate_rejected(step5_reversal_pairs,       "Step 5 — hierarchy reversal rejections:")

    # ------------------------------------------------------------------
    # Step 6: Jaccard=1 AND cosine≠1 candidates with no reference-based
    # backing. Pairs that have cosine=1 were already handled in step 5.
    # Pairs that appear in any reference-based attribute are excluded.
    # Hierarchy reversals are marked validity=7.
    # ------------------------------------------------------------------
    l.log("Step 6: Adding Jaccard=1 (cosine≠1) mappings with no reference-based backing.")

    step6_pool = mappings[mappings[attr_col] == jaccard_mappings]
    seen_pairs = set()
    step6_candidate_pairs = []
    for hpo_id, sct_id in zip(step6_pool[id_col], step6_pool[val_col]):
        pair = (hpo_id, sct_id)
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            step6_candidate_pairs.append(pair)

    step6_accepted             = 0
    step6_reversal             = 0
    step6_rejected             = 0
    step6_demoted              = 0
    step6_low_similarity_pairs = set()
    step6_reversal_pairs       = set()

    for hpo_id, sct_id in step6_candidate_pairs:
        pair = (hpo_id, sct_id)

        if pair in accepted_pairs:
            continue

        if pair in domain_invalid:
            continue

        # Skip pairs that have any reference-based mapping row.
        if pair in reference_backed_pairs:
            continue

        # Skip pairs already classified in an earlier step.
        if pair in classified_pairs:
            continue

        # Must have Jaccard=1 but NOT cosine=1 — pairs with both were
        # already handled in step 5.
        has_jaccard = pair in jaccard_exact_pairs
        has_cosine  = pair in embedding_exact_pairs

        if not has_jaccard or has_cosine:
            step6_low_similarity_pairs.add(pair)
            step6_rejected += 1
            continue

        cand_conf = best_confidence_per_pair.get((hpo_id, sct_id), 0.0)
        action, conflicts = check_hierarchy(hpo_id, sct_id, cand_conf)

        if action == "ok":
            accept(hpo_id, sct_id)
            step6_accepted += 1

        elif action == "demote":
            for other_sct in conflicts:
                demote(hpo_id, other_sct, validity_jaccard_only_reversal)
                step6_reversal_pairs.add((hpo_id, other_sct))
            step6_demoted += len(conflicts)
            accept(hpo_id, sct_id)
            step6_accepted += 1

        elif action == "reject":
            mark_validity(hpo_id, sct_id, validity_jaccard_only_reversal)
            step6_reversal_pairs.add(pair)
            step6_reversal += 1

    l.log(f"Step 6: {step6_accepted} accepted, {step6_reversal} rejected "
          f"(hierarchy reversal, validity=7), "
          f"{step6_demoted} demoted (broader displaced), {step6_rejected} skipped "
          f"(no Jaccard=1 without cosine=1, or reference-backed).")
    evaluation(mappings[mappings[validity_column] == validity_accepted], graham, "After Step 6 — accepted:", hpo_ontology)
    evaluate_rejected(step6_low_similarity_pairs, "Step 6 — skipped (cosine=1 or no Jaccard=1):", include_printcounts=True)
    evaluate_rejected(step6_reversal_pairs,       "Step 6 — hierarchy reversal rejections:")

    # ------------------------------------------------------------------
    # Consistency check: after the automatic acceptance steps, no HPO
    # concept should have two accepted SCT targets in an ancestor/
    # descendant relationship (source collapse). Expected to be 0;
    # comparable to WARNING A in MappingErrorAnalysis.
    # ------------------------------------------------------------------
    collapsed_concepts = 0
    collapsed_pairs    = 0
    for hpo_id, sct_ids in accepted_by_hpo.items():
        targets  = sorted(sct_ids)
        conflict = sum(
            1
            for i, s1 in enumerate(targets)
            for s2 in targets[i + 1:]
            if s2 in sct_descendants(s1) or s1 in sct_descendants(s2)
        )
        if conflict:
            collapsed_concepts += 1
            collapsed_pairs    += conflict
    l.log(f"Hierarchy consistency check after step 6: {collapsed_concepts} HPO concepts "
          f"with accepted SCT targets in an ancestor/descendant relationship "
          f"({collapsed_pairs} related target pairs).")

    # ------------------------------------------------------------------
    # Step 7: best-candidate fallback for unmapped HPO concepts.
    # For every HPO concept that has neither an accepted mapping (validity=10)
    # nor any classified candidate already, find the highest-cosine and
    # highest-Jaccard candidates (domain-valid only).
    # If both point to the same SCT concept, run the hierarchy check:
    #   - passes  -> mark validity=8 (LLM review)
    #   - fails   -> skip
    # ------------------------------------------------------------------
    l.log("Step 7: Best-candidate fallback for unmapped HPO concepts.")

    # Determine which HPO concepts already have at least one classified row
    # (validity=1–7 or accepted validity=10) — those are skipped entirely.
    handled_hpo = set(
        mappings.loc[mappings[validity_column].isin(validity_hierarchy_reversals + [validity_accepted]), id_col].unique()
    )

    # All HPO concepts that appear in the mappings dataframe.
    all_hpo = set(mappings[id_col].unique())
    unmapped_hpo = all_hpo - handled_hpo
    l.log(f"HPO concepts with no accepted or classified mapping: {len(unmapped_hpo)}")

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
        return dict(zip(top[id_col], zip(top[val_col], top[confidence_column])))

    best_cosine  = best_per_concept(embedding_pool)
    best_jaccard = best_per_concept(jaccard_pool)

    step7_count            = 0
    step7_skipped_missing  = 0
    step7_skipped_disagree = 0
    step7_skipped_other    = 0
    step7_pairs            = set()

    for hpo_id in unmapped_hpo:
        cosine_result  = best_cosine.get(hpo_id)
        jaccard_result = best_jaccard.get(hpo_id)

        if cosine_result is None or jaccard_result is None:
            step7_skipped_missing += 1
            continue

        cosine_sct,  cosine_conf  = cosine_result
        jaccard_sct, jaccard_conf = jaccard_result

        # Both must agree on the same SCT concept.
        if cosine_sct != jaccard_sct:
            step7_skipped_disagree += 1
            continue

        sct_id = cosine_sct
        pair   = (hpo_id, sct_id)

        # Skip if already accepted or domain-invalid.
        if pair in accepted_pairs or pair in domain_invalid:
            step7_skipped_other += 1
            continue

        action, _ = check_hierarchy(hpo_id, sct_id, best_confidence_per_pair.get((hpo_id, sct_id), 0.0))

        if action in ("ok", "demote"):
            # Hierarchy check passes — mark for LLM review rather than
            # auto-accepting, since similarity is below 1.
            mark_validity(hpo_id, sct_id, validity_best_candidate)
            step7_pairs.add(pair)
            step7_count += 1
        # "reject" -> skip, no validity assigned.

    l.log(f"Step 7: {step7_count} candidates marked for LLM review (validity=8).")
    l.log(f"  Skipped — no cosine or Jaccard candidate:      {step7_skipped_missing}.")
    l.log(f"  Skipped — cosine and Jaccard disagree on SCT:  {step7_skipped_disagree}.")
    l.log(f"  Skipped — already accepted or domain-invalid:  {step7_skipped_other}.")
    evaluate_rejected(step7_pairs, "Step 7 — best-candidate fallback for LLM review (validity 8):")
    # Steps 7 and 8 never accept anything, so the accepted set is the one
    # from step 6. Both views are logged to keep them apart.
    evaluation(
        mappings[mappings[validity_column] == validity_accepted],
        graham,
        "After Step 7 — accepted only (validity 10):",
        hpo_ontology,
    )
    evaluation(
        mappings[mappings[validity_column].isin(validity_hierarchy_reversals + [validity_best_candidate, validity_accepted])],
        graham,
        "After Step 7 — accepted + candidates for LLM review (validity 1–8, 10):",
        hpo_ontology,
    )
    l.log("After Step 7 — candidates for LLM review only (validity 1–8), by confidence:")
    printCounts(mappings[mappings[validity_column].isin(validity_hierarchy_reversals + [validity_best_candidate])], hpo_ontology, similarity_steps, graham, True)

    # ------------------------------------------------------------------
    # Step 8: last-resort fallback for still-unmapped HPO concepts.
    # For every HPO concept that still has no classified mapping after
    # step 7, the single highest-cosine-similarity candidate (domain-valid
    # only) is marked validity=9. No hierarchy check is applied; these are
    # true last-resort candidates and any issues will be caught during LLM
    # review.
    # ------------------------------------------------------------------
    l.log("Step 8: Last-resort fallback for still-unmapped HPO concepts")

    # Recompute handled concepts after step 7.
    handled_hpo_after7 = set(
        mappings.loc[mappings[validity_column].isin(validity_hierarchy_reversals + [validity_best_candidate, validity_accepted]), id_col].unique()
    )
    still_unmapped = all_hpo - handled_hpo_after7
    l.log(f"HPO concepts still without any mapping after step 7: {len(still_unmapped)}")

    # From the domain-valid cosine pool, find the highest-confidence
    # candidate for each still-unmapped concept.
    step8_pool = valid_mappings[
        (valid_mappings[id_col].isin(still_unmapped)) &
        (valid_mappings[attr_col] == embedding_mappings)
    ]

    step8_count   = 0
    step8_skipped = 0

    if not step8_pool.empty:
        idx = step8_pool.groupby(id_col)[confidence_column].idxmax()
        top = step8_pool.loc[idx]

        for _, row in top.iterrows():
            hpo_id = row[id_col]
            sct_id = row[val_col]
            pair   = (hpo_id, sct_id)

            if pair in accepted_pairs or pair in domain_invalid:
                step8_skipped += 1
                continue

            # Use mark_validity to respect the NA-only guard — if this pair
            # was somehow already classified by an earlier step, do not
            # overwrite it. Only count the pair if rows were actually marked.
            if mark_validity(hpo_id, sct_id, validity_last_resort):
                step8_count += 1
            else:
                step8_skipped += 1

    step8_no_candidate = len(still_unmapped) - step8_pool[id_col].nunique()

    l.log(f"Step 8: {step8_count} candidates marked as last-resort (validity=9), "
          f"{step8_skipped} concepts skipped, "
          f"{step8_no_candidate} concepts without any domain-valid embedding candidate.")

    step8_pairs = set(
        zip(
            mappings.loc[mappings[validity_column] == validity_last_resort, id_col],
            mappings.loc[mappings[validity_column] == validity_last_resort, val_col],
        )
    )
    evaluate_rejected(step8_pairs, "Step 8 — last-resort fallback for LLM review (validity 9):")
    evaluation(
        mappings[mappings[validity_column] == validity_accepted],
        graham,
        "After Step 8 — accepted only (validity 10):",
        hpo_ontology,
    )
    evaluation(
        mappings[mappings[validity_column].isin(validity_reviewable)],
        graham,
        "After Step 8 — accepted + candidates for LLM review (validity 1–10):",
        hpo_ontology,
    )

    # ------------------------------------------------------------------
    # Final output.
    # Write the full mappings dataframe (same format as input) with the
    # validity column filled in:
    #   0  = step 1 (domain filter): SCT target outside acceptable domain
    #   1  = step 2 failure: hierarchy reversal for hpo_sct AND
    #        exact_lexical anchor candidates
    #   2  = step 2b failure: hierarchy reversal for remaining exact
    #        lexical candidates
    #   3  = step 2c failure: hierarchy reversal for HPO xrefs with
    #        Jaccard=1 AND cosine=1
    #   4  = step 3 failure: hierarchy reversal for HPO -> UMLS -> SCT
    #        mappings
    #   5  = step 4 failure: hierarchy reversal for shared UMLS CUI
    #        mappings
    #   6  = step 5 failure: hierarchy reversal for Jaccard=1 AND
    #        cosine=1 pairs with no reference-based backing
    #   7  = step 6 failure: hierarchy reversal for Jaccard=1 only pairs
    #        with no reference-based backing
    #   8  = step 7 fallback: best-candidate (cosine+Jaccard agree) for
    #        unmapped HPO concepts, marked for LLM review
    #   9  = step 8 last-resort: highest cosine candidate for HPO
    #        concepts with no other mapping
    #   10 = accepted
    #   NA = generated in script 1 but not reached by any filtering step
    #        (dropped from output)
    # Only rows with validity in {0,1,2,3,4,5,6,7,8,9,10} are written.
    # ------------------------------------------------------------------
    na_validity_rows = mappings[mappings[validity_column].isna()]

    na_unique_pairs  = na_validity_rows.drop_duplicates(subset=[id_col, val_col])
    
    l.log(f"Dropping {len(na_validity_rows.index)} rows "
          f"({len(na_unique_pairs.index)} unique pairs) with validity=NA "
          f"(generated in script 1 but not reached by any filtering step).")

    final = mappings[mappings[validity_column].isin(validity_all)].copy()

    l.log(f"Total rows written: {len(final.index)} "
          f"(accepted={len(final[final[validity_column] == validity_accepted])}, "
          f"domain_mismatch={len(final[final[validity_column] == validity_domain_mismatch])}, "
          f"step2_fail={len(final[final[validity_column] == validity_anchor_reversal])}, "
          f"step2b_fail={len(final[final[validity_column] == validity_exact_lexical_reversal])}, "
          f"step2c_fail={len(final[final[validity_column] == validity_hpo_xref_reversal])}, "
          f"step3_fail={len(final[final[validity_column] == validity_umls_bridge_reversal])}, "
          f"step4_fail={len(final[final[validity_column] == validity_umls_cui_reversal])}, "
          f"step5_fail={len(final[final[validity_column] == validity_jaccard_cosine_reversal])}, "
          f"step6_fail={len(final[final[validity_column] == validity_jaccard_only_reversal])}, "
          f"step7_fallback={len(final[final[validity_column] == validity_best_candidate])}, "
          f"step8_last_resort={len(final[final[validity_column] == validity_last_resort])})")

    evaluation(final[final[validity_column] == validity_accepted], graham, "Final accepted mapping set:", hpo_ontology)
    printCounts(final[final[validity_column] == validity_accepted], hpo_ontology, similarity_steps, graham, True)

    writeHugeCSV(mappings, filtered_path)

    # ------------------------------------------------------------------
    # Mapping constellation analysis.
    # For all non-domain-mismatch mappings (validity=1–10), count how many
    # unique (hpo_id, sct_id) pairs exist and classify them into 1:1, 1:n,
    # n:1, and m:n constellations based on how many SCT concepts each HPO
    # concept maps to and vice versa.
    # ------------------------------------------------------------------
    l.log("Mapping constellation analysis (validity=1–10)")

    llm_pairs = (
        final[final[validity_column].isin(validity_reviewable)]
        .drop_duplicates(subset=[id_col, val_col])[[id_col, val_col]]
    )

    # Count unique SCT targets per HPO concept and unique HPO sources per
    # SCT concept.
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
    l.log(f"Total unique pairs: {total_pairs}")
    for constellation in ["1:1", "1:n", "n:1", "m:n"]:
        count = counts.get(constellation, 0)
        l.log(f"  {constellation}: {count:6} pairs ({count / total_pairs:.1%})")

    l.printHeader("Filtering generated mappings completed.")