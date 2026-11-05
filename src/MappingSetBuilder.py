import sys

sys.dont_write_bytecode = True

from logger                 import Logger
from adapter                import writeHugeCSV
from MappingUtils           import (
    similarity_steps, exact_lexical_mappings, loadInputs, isSCTDomainValid, evaluation,
    confidence_column, validity_column, printCounts,
    hpo_sct_mappings, hpo_umls_sct_mappings, umls_hpo_sct_mappings, embedding_mappings,
    jaccard_mappings, extract_snomed_domain, build_child_map, build_parent_map,
    make_descendant_lookup, buildAttributeLookup, pairSeries, keepHPOConcepts,
    mapping_path, filtered_path, VALIDITY_LABELS,
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

# ==========================================================================
# MappingSetBuilder.py
#
# Filters the candidate mappings written by MappingGeneration and assigns
# each (hpo_id, sct_id) pair a validity code (see MappingUtils):
#
#   Step 1  : domain filter                         -> 0
#   Step 2  : anchors (HPO xref + exact lexical)    -> 10, reversal 1
#   Step 2b : remaining exact lexical matches       -> 10, reversal 2
#   Step 2c : HPO xrefs with Jaccard=1 or cosine=1  -> 10, reversal 3
#   Step 3  : HPO -> UMLS -> SCT, Jaccard/cosine=1  -> 10, reversal 4
#   Step 4  : shared UMLS CUI, Jaccard/cosine=1     -> 10, reversal 5
#   Step 5  : Jaccard=1 AND cosine=1, no reference  -> 10, reversal 6
#   Step 6  : Jaccard=1 only, no reference          -> 10, reversal 7
#   Step 7  : best candidate for unmapped concepts  -> 8 (LLM review)
#   Step 8  : last resort for unmapped concepts     -> 9 (LLM review)
#
# A pair keeps the first code it is given; only demotion (see
# check_hierarchy) overwrites an accepted 10. Pairs not reached by any step
# keep validity=NA and are written to filtered.csv too (MappingErrorAnalysis
# uses all generated pairs); MappingLLM reviews validity 1-10.
# ==========================================================================

EXACT_SIMILARITY = 1.0
# float32 cosine of identical embeddings can be 0.9999997; MappingGeneration
# rounds to 6 decimals.
_EPS             = 1e-6


def is_exact(value: float) -> bool:
    """Floating point safe check for 'similarity == 1.0'."""
    return abs(value - EXACT_SIMILARITY) < _EPS


if __name__ == "__main__":

    l = Logger()

    l.printHeader("Filtering generated mappings")

    l.printHeader("Loading data")

    hpo, sct, umls, hpo_ontology, sct_ontology, umls_ontology, graham = loadInputs(";")

    id_col   = hpo_ontology.config.id_column
    val_col  = hpo_ontology.config.value_column
    attr_col = hpo_ontology.config.attribute_column

    l.log("Loading generated mappings...")
    mappings = pd.read_csv(mapping_path, sep=";")
    mappings = keepHPOConcepts(mappings, id_col)
    n_pairs  = len(mappings.drop_duplicates(subset=[id_col, val_col]).index)
    l.log(f"Generated mappings: {len(mappings.index)} rows, {n_pairs} pairs, "
          f"{mappings[id_col].nunique()} HPO concepts.")
    l.log("Loading generated mappings completed.")

    mappings[validity_column] = pd.NA

    # pair -> row positions (one row per method that found the pair), so
    # mark_validity does not scan the dataframe. Positions equal index labels
    # because keepHPOConcepts resets the index.
    pair_to_indices: dict[tuple, list] = defaultdict(list)
    for idx, (hpo_id, sct_id) in enumerate(zip(mappings[id_col], mappings[val_col])):
        pair_to_indices[(hpo_id, sct_id)].append(idx)

    # The hierarchy check works on SCT only; hpo_child_map is only logged
    # (kept for the TODO at check_hierarchy). len(sct_ancestors(c)) serves as
    # the depth of c, to process anchors narrowest-first in step 2.
    l.log("Building hierarchy lookups...")
    hpo_child_map   = build_child_map(hpo, hpo_ontology)
    sct_child_map   = build_child_map(sct, sct_ontology)
    sct_descendants = make_descendant_lookup(sct_child_map)
    sct_ancestors   = make_descendant_lookup(build_parent_map(sct, sct_ontology))
    l.log(f"SCT hierarchy: {len(sct_child_map)} parent concepts, "
          f"{sum(len(v) for v in sct_child_map.values())} edges.")
    l.log(f"HPO hierarchy: {len(hpo_child_map)} parent concepts, "
          f"{sum(len(v) for v in hpo_child_map.values())} edges.")
    l.log("Building hierarchy lookups completed.")

    accepted_pairs   = set()                # (hpo_id, sct_id) accepted so far
    accepted_by_hpo  = defaultdict(set)     # hpo_id -> accepted sct_ids
    domain_invalid   = set()                # pairs marked validity=0
    classified_pairs = set()                # pairs with any validity code

    def mark_validity(hpo_id: str, sct_id: str, value: int, replace: Optional[int] = None) -> int:
        """
        Set the validity of all rows of this pair that are still NA, or that
        hold the code `replace` (used by demote()). Returns the number of
        rows changed.
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

    def accept(hpo_id: str, sct_id: str) -> None:
        """Register a pair as accepted and mark all its rows validity=10."""
        accepted_pairs.add((hpo_id, sct_id))
        accepted_by_hpo[hpo_id].add(sct_id)
        mark_validity(hpo_id, sct_id, validity_accepted)

    def demote(hpo_id: str, sct_id: str, reversal_validity: int) -> None:
        """
        Un-accept a pair displaced by a narrower candidate (see
        check_hierarchy) and mark it with the current step's reversal code.
        """
        accepted_pairs.discard((hpo_id, sct_id))
        accepted_by_hpo[hpo_id].discard(sct_id)
        mark_validity(hpo_id, sct_id, reversal_validity, replace=validity_accepted)

    def log_step_summary(
        step:        str,
        accepted:    int,
        reversal:    int,
        validity:    int,
        demoted:     int = 0,
        not_accepted: Optional[int] = None,
        not_accepted_reason: str = "",
        skipped:     Optional[dict] = None,
        rejected_by_evidence: int = 0,
    ) -> None:
        """
        Log the pair counts of an acceptance step in a fixed order.
        rejected_by_evidence is the part of the rejections where the
        candidate was narrower but had weaker evidence.
        """
        parts = [
            f"{accepted} accepted",
            f"{reversal} rejected (hierarchy reversal, validity={validity})",
            f"{rejected_by_evidence} of the rejected are narrower but have weaker evidence",
            f"{demoted} demoted (broader target displaced, validity={validity})",
        ]
        if not_accepted is not None:
            parts.append(f"{not_accepted} not accepted here ({not_accepted_reason})")
        for reason, count in (skipped or {}).items():
            parts.append(f"{count} skipped ({reason})")
        for part in parts:
            l.log(f"{step} (pairs): {part}.")

    def evaluate_accepted(step: str) -> None:
        """Evaluate the accepted set (validity=10) after a step."""
        evaluation(
            mappings[mappings[validity_column] == validity_accepted],
            graham,
            f"Evaluation (after {step}, accepted, validity=10):",
            hpo_ontology
        )

    mappings_pairs = pairSeries(mappings, id_col, val_col)

    def evaluate_pairs(pairs: set, label: str, include_printcounts: bool = False) -> None:
        """
        Evaluate the rows whose pair is in `pairs`. include_printcounts also
        logs the confidence distribution of their Jaccard and embedding rows
        (other methods always have confidence 1.0).
        """
        if not pairs:
            return
        selected = mappings[mappings_pairs.isin(pairs)]
        evaluation(selected, graham, f"Evaluation ({label}):", hpo_ontology)
        selected_sim = selected[selected[attr_col].isin([jaccard_mappings, embedding_mappings])]
        if include_printcounts and not selected_sim.empty:
            l.log(f"Confidence distribution ({label}, Jaccard and embedding rows):")
            printCounts(selected_sim, hpo_ontology, similarity_steps, graham, True)

    # TODO: if there is no overlap in hierarchy in three generations above or
    # below, accept both candidates.
    def check_hierarchy(hpo_id: str, sct_id: str):
        """
        Hierarchy check against all accepted targets of the HPO concept.
        Mapping it to an SCT concept and to one of its ancestors (at any
        distance) is a "source collapse", e.g. "Focal seizure" -> "Focal
        seizure" and "Seizure"; normally the narrower stays. Rules in order:
          1. candidate broader than an accepted target        -> "reject"
          2. candidate narrower, a broader target has stronger
             evidence (see evidence())                        -> "reject_evidence"
          3. candidate narrower, no stronger evidence         -> "demote" ALL of them
          4. no relationship                                  -> "ok"
        Rules 1 and 3 make the outcome independent of arrival order for equal
        evidence; rule 2 keeps e.g. a Jaccard-only pair from displacing a
        cross-referenced target. Returns (action, conflicting sct_ids), an
        empty list for "ok".
        """
        candidate_desc_sct = sct_descendants(sct_id)
        broader            = []    # accepted targets the candidate is narrower than
        narrower           = []    # accepted targets the candidate is broader than

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

        candidate_evidence = evidence((hpo_id, sct_id))
        stronger = [s for s in broader if evidence((hpo_id, s)) > candidate_evidence]
        if stronger:
            return "reject_evidence", stronger

        return "demote", broader

    # Similarity lookups for steps 2c-6: a pair is "exact" if at least one
    # of its rows has 1.0.
    def pairs_of(rows: pd.DataFrame) -> set:
        """Unique (hpo_id, sct_id) pairs of the rows."""
        return set(zip(rows[id_col], rows[val_col]))

    jaccard_rows   = mappings[mappings[attr_col] == jaccard_mappings]
    embedding_rows = mappings[mappings[attr_col] == embedding_mappings]

    jaccard_exact_pairs   = pairs_of(jaccard_rows[jaccard_rows[confidence_column].apply(is_exact)])
    embedding_exact_pairs = pairs_of(embedding_rows[embedding_rows[confidence_column].apply(is_exact)])

    exact_lexical_pairs = pairs_of(mappings[mappings[attr_col] == exact_lexical_mappings])

    # Pairs with any reference-based row (HPO xref or UMLS); steps 5 and 6
    # skip them.
    reference_methods = [hpo_sct_mappings, hpo_umls_sct_mappings, umls_hpo_sct_mappings]
    reference_rows    = mappings[mappings[attr_col].isin(reference_methods)]
    reference_backed_pairs = pairs_of(reference_rows)

    # Number of distinct reference sources (0-3) per pair.
    reference_count: dict = (
        reference_rows.groupby([id_col, val_col])[attr_col].nunique().to_dict()
    )

    def evidence(pair: tuple) -> tuple:
        """
        Evidence of a pair for check_hierarchy (not tuned on the gold
        standard), compared as a tuple: (number of reference sources 0-3,
        lexical: 2 exact lexical / 1 Jaccard=1 / 0, embedding: 1 cosine=1 / 0).
        One cross-reference outweighs any similarity; anchors are >= (1, 2, 0).
        """
        lexical = 2 if pair in exact_lexical_pairs else 1 if pair in jaccard_exact_pairs else 0
        return (
            reference_count.get(pair, 0),
            lexical,
            1 if pair in embedding_exact_pairs else 0,
        )

    sct_labels = buildAttributeLookup(sct, sct_ontology, labelClass)

    # Step 1 runs first so that no later step can accept a domain-invalid
    # pair (mark_validity never overwrites a code). The domain is the
    # semantic tag of the SCT label, e.g. "Seizure (finding)" -> "finding";
    # see isSCTDomainValid.
    l.printHeader("Step 1: Domain filter")

    l.log("Step 1: Marking pairs with an SCT target outside the acceptable domains...")

    all_pairs = mappings.drop_duplicates(subset=[id_col, val_col])

    for hpo_id, sct_id in zip(all_pairs[id_col], all_pairs[val_col]):
        if not isSCTDomainValid(sct_labels.get(sct_id, [])):
            mark_validity(hpo_id, sct_id, validity_domain_mismatch)
            domain_invalid.add((hpo_id, sct_id))

    l.log(f"Step 1 (pairs): {len(domain_invalid)} marked as domain mismatch "
          f"(validity={validity_domain_mismatch}).")
    evaluate_pairs(domain_invalid, f"step 1, domain mismatch, validity={validity_domain_mismatch}")
    l.log("Step 1: Marking pairs with an SCT target outside the acceptable domains completed.")

    # Step 2: anchors are HPO xrefs confirmed by an exact lexical match, two
    # independent sources, so no Jaccard/cosine corroboration is needed.
    # Narrowest first per HPO concept, so of two related anchors the broader
    # one fails rule 1 of check_hierarchy instead of being demoted.
    l.printHeader("Step 2: Anchors (HPO xref + exact lexical)")

    l.log("Step 2: Accepting HPO xrefs confirmed by an exact lexical match (case-insensitive)...")

    hpo_sct_pairs     = pairs_of(mappings[mappings[attr_col] == hpo_sct_mappings])
    anchor_candidates = hpo_sct_pairs & exact_lexical_pairs

    def anchor_order(pair: tuple) -> tuple:
        """Sort key: per HPO concept, deepest SCT target first."""
        hpo_id, sct_id = pair
        return (hpo_id, -len(sct_ancestors(sct_id)), sct_id)

    step2_accepted = 0
    step2_reversal = 0
    step2_evidence = 0
    step2_demoted  = 0
    step2_domain   = 0

    for hpo_id, sct_id in sorted(anchor_candidates, key=anchor_order):
        if (hpo_id, sct_id) in domain_invalid:
            step2_domain += 1
            continue

        action, conflicts = check_hierarchy(hpo_id, sct_id)
        if action == "ok":
            accept(hpo_id, sct_id)
            step2_accepted += 1
        elif action == "demote":
            for other_sct in conflicts:
                demote(hpo_id, other_sct, validity_anchor_reversal)
            step2_demoted += len(conflicts)
            accept(hpo_id, sct_id)
            step2_accepted += 1
        else:
            mark_validity(hpo_id, sct_id, validity_anchor_reversal)
            step2_reversal += 1
            step2_evidence += action == "reject_evidence"

    log_step_summary(
        "Step 2", step2_accepted, step2_reversal, validity_anchor_reversal,
        demoted = step2_demoted,
        rejected_by_evidence = step2_evidence,
        not_accepted = len(hpo_sct_pairs - anchor_candidates),
        not_accepted_reason = "HPO xref without exact lexical match, left for step 2c",
        skipped = {"domain mismatch": step2_domain},
    )
    evaluate_accepted("step 2")

    # SCT semantic domains of the accepted anchors.
    anchor_domain_counts = defaultdict(int)
    for hpo_id, sct_id in accepted_pairs:
        domain = None
        for label in sct_labels.get(sct_id, []):
            if isinstance(label, str):
                domain = extract_snomed_domain(label)
                if domain:
                    break
        anchor_domain_counts[domain or "unknown"] += 1

    l.log("Anchor domain distribution (pairs):")
    for domain, count in sorted(anchor_domain_counts.items(), key=lambda x: -x[1]):
        l.log(f"  {domain}: {count}")

    l.log("Step 2: Accepting HPO xrefs confirmed by an exact lexical match "
          "(case-insensitive) completed.")

    # Step 2b: exact lexical pairs without an HPO xref.
    l.printHeader("Step 2b: Remaining exact lexical mappings")

    l.log("Step 2b: Accepting exact lexical matches without an HPO xref...")

    step2b_accepted = 0
    step2b_reversal = 0
    step2b_evidence = 0
    step2b_demoted  = 0
    step2b_domain   = 0

    # sorted() fixes the processing order; sets have no stable order.
    for hpo_id, sct_id in sorted(exact_lexical_pairs - anchor_candidates):
        pair = (hpo_id, sct_id)

        if pair in domain_invalid:
            step2b_domain += 1
            continue

        action, conflicts = check_hierarchy(hpo_id, sct_id)

        if action == "ok":
            accept(hpo_id, sct_id)
            step2b_accepted += 1
        elif action == "demote":
            for other_sct in conflicts:
                demote(hpo_id, other_sct, validity_exact_lexical_reversal)
            step2b_demoted += len(conflicts)
            accept(hpo_id, sct_id)
            step2b_accepted += 1
        else:
            mark_validity(hpo_id, sct_id, validity_exact_lexical_reversal)
            step2b_reversal += 1
            step2b_evidence += action == "reject_evidence"

    log_step_summary(
        "Step 2b", step2b_accepted, step2b_reversal, validity_exact_lexical_reversal,
        demoted = step2b_demoted,
        rejected_by_evidence = step2b_evidence,
        skipped = {"domain mismatch": step2b_domain},
    )
    evaluate_accepted("step 2b")
    l.log("Step 2b: Accepting exact lexical matches without an HPO xref completed.")

    def process_corroborated_step(
        step:              str,
        attribute_value:   str,
        reversal_validity: int,
    ) -> None:
        """
        Steps 2c, 3, 4: accept the pairs of one reference-based method
        (attribute_value) that have Jaccard=1 or cosine=1 and pass
        check_hierarchy; hierarchy failures get reversal_validity. Pairs
        without that similarity stay NA (possible step 7/8 candidates).
        """
        candidates = mappings[mappings[attr_col] == attribute_value].drop_duplicates(subset=[id_col, val_col])

        accepted_count       = 0
        reversal_count       = 0
        evidence_count       = 0
        demoted_count        = 0
        domain_count         = 0
        classified_count     = 0
        low_similarity_pairs = set()
        reversal_pairs       = set()

        for hpo_id, sct_id in zip(candidates[id_col], candidates[val_col]):
            pair = (hpo_id, sct_id)

            if pair in domain_invalid:
                domain_count += 1
                continue

            if pair in classified_pairs:
                classified_count += 1
                continue

            if pair not in jaccard_exact_pairs and pair not in embedding_exact_pairs:
                low_similarity_pairs.add(pair)
                continue

            action, conflicts = check_hierarchy(hpo_id, sct_id)

            if action == "ok":
                accept(hpo_id, sct_id)
                accepted_count += 1

            elif action == "demote":
                for other_sct in conflicts:
                    demote(hpo_id, other_sct, reversal_validity)
                    reversal_pairs.add((hpo_id, other_sct))
                demoted_count += len(conflicts)
                accept(hpo_id, sct_id)
                accepted_count += 1

            else:
                mark_validity(hpo_id, sct_id, reversal_validity)
                reversal_pairs.add(pair)
                reversal_count += 1
                evidence_count += action == "reject_evidence"

        log_step_summary(
            step, accepted_count, reversal_count, reversal_validity,
            demoted = demoted_count,
            rejected_by_evidence = evidence_count,
            not_accepted = len(low_similarity_pairs),
            not_accepted_reason = "neither Jaccard=1 nor cosine=1",
            skipped = {
                "domain mismatch": domain_count,
                "already classified in an earlier step": classified_count,
            },
        )
        evaluate_accepted(step.lower())
        evaluate_pairs(low_similarity_pairs,
            f"{step.lower()}, not accepted here", include_printcounts=True)
        evaluate_pairs(reversal_pairs,
            f"{step.lower()}, hierarchy reversals, validity={reversal_validity}")

    # Step 2c: HPO xrefs without exact lexical match, same criterion as the
    # UMLS-based steps 3 and 4.
    l.printHeader("Step 2c: Remaining HPO xrefs")

    l.log("Step 2c: Accepting HPO xrefs with Jaccard=1 or cosine=1...")

    process_corroborated_step(
        "Step 2c",
        hpo_sct_mappings,
        reversal_validity=validity_hpo_xref_reversal,
    )
    l.log("Step 2c: Accepting HPO xrefs with Jaccard=1 or cosine=1 completed.")

    l.printHeader("Step 3: HPO -> UMLS -> SNOMED CT mappings")

    l.log("Step 3: Accepting HPO -> UMLS -> SNOMED CT mappings with "
          "Jaccard=1 or cosine=1...")

    process_corroborated_step(
        "Step 3",
        hpo_umls_sct_mappings,
        reversal_validity=validity_umls_bridge_reversal,
    )
    l.log("Step 3: Accepting HPO -> UMLS -> SNOMED CT mappings with "
          "Jaccard=1 or cosine=1 completed.")

    l.printHeader("Step 4: Shared UMLS CUI mappings")

    l.log("Step 4: Accepting shared UMLS CUI mappings with Jaccard=1 or "
          "cosine=1...")

    process_corroborated_step(
        "Step 4",
        umls_hpo_sct_mappings,
        reversal_validity=validity_umls_cui_reversal,
    )
    l.log("Step 4: Accepting shared UMLS CUI mappings with Jaccard=1 or "
          "cosine=1 completed.")

    def process_similarity_step(
        step:              str,
        pool:              pd.DataFrame,
        qualifies,
        not_accepted_reason: str,
        reversal_validity: int,
    ) -> None:
        """
        Steps 5, 6: accept the pairs in `pool` without reference-based backing
        for which qualifies(pair) is true and check_hierarchy passes;
        hierarchy failures get reversal_validity, the rest stays NA.
        """
        # Unique pairs in first-seen order.
        candidate_pairs = list(dict.fromkeys(zip(pool[id_col], pool[val_col])))

        accepted_count       = 0
        reversal_count       = 0
        evidence_count       = 0
        demoted_count        = 0
        domain_count         = 0
        reference_count      = 0
        classified_count     = 0
        not_qualifying_pairs = set()
        reversal_pairs       = set()

        for hpo_id, sct_id in candidate_pairs:
            pair = (hpo_id, sct_id)

            if pair in domain_invalid:
                domain_count += 1
                continue

            if pair in reference_backed_pairs:
                reference_count += 1
                continue

            if pair in classified_pairs:
                classified_count += 1
                continue

            if not qualifies(pair):
                not_qualifying_pairs.add(pair)
                continue

            action, conflicts = check_hierarchy(hpo_id, sct_id)

            if action == "ok":
                accept(hpo_id, sct_id)
                accepted_count += 1

            elif action == "demote":
                for other_sct in conflicts:
                    demote(hpo_id, other_sct, reversal_validity)
                    reversal_pairs.add((hpo_id, other_sct))
                demoted_count += len(conflicts)
                accept(hpo_id, sct_id)
                accepted_count += 1

            else:
                mark_validity(hpo_id, sct_id, reversal_validity)
                reversal_pairs.add(pair)
                reversal_count += 1
                evidence_count += action == "reject_evidence"

        log_step_summary(
            step, accepted_count, reversal_count, reversal_validity,
            demoted = demoted_count,
            rejected_by_evidence = evidence_count,
            not_accepted = len(not_qualifying_pairs),
            not_accepted_reason = not_accepted_reason,
            skipped = {
                "domain mismatch": domain_count,
                "reference-backed, handled in steps 2-4": reference_count,
                "already classified in an earlier step": classified_count,
            },
        )
        evaluate_accepted(step.lower())
        evaluate_pairs(not_qualifying_pairs,
            f"{step.lower()}, not accepted here", include_printcounts=True)
        evaluate_pairs(reversal_pairs,
            f"{step.lower()}, hierarchy reversals, validity={reversal_validity}")

    l.printHeader("Step 5: Jaccard=1 and cosine=1 mappings")

    l.log("Step 5: Accepting Jaccard=1 and cosine=1 mappings without "
          "reference-based backing...")

    process_similarity_step(
        "Step 5",
        mappings[mappings[attr_col].isin([jaccard_mappings, embedding_mappings])],
        lambda pair: pair in jaccard_exact_pairs and pair in embedding_exact_pairs,
        "not both Jaccard=1 and cosine=1",
        reversal_validity=validity_jaccard_cosine_reversal,
    )
    l.log("Step 5: Accepting Jaccard=1 and cosine=1 mappings without "
          "reference-based backing completed.")

    # Step 6: Jaccard=1 with cosine below 1 or no embedding candidate.
    l.printHeader("Step 6: Jaccard=1 only mappings")

    l.log("Step 6: Accepting Jaccard=1-only mappings without reference-based backing...")

    process_similarity_step(
        "Step 6",
        mappings[mappings[attr_col] == jaccard_mappings],
        lambda pair: pair in jaccard_exact_pairs and pair not in embedding_exact_pairs,
        "Jaccard<1",
        reversal_validity=validity_jaccard_only_reversal,
    )
    l.log("Step 6: Accepting Jaccard=1-only mappings without reference-based backing completed.")

    # Consistency check: no HPO concept should have two related accepted SCT
    # targets (source collapse); comparable to WARNING A in
    # MappingErrorAnalysis.
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
          f"with related accepted targets.")
    l.log(f"Hierarchy consistency check after step 6: {collapsed_pairs} related target pairs.")
    l.log("Related: SCT targets in an ancestor/descendant relationship; both counts "
          "are expected to be 0.")

    # Step 7: for HPO concepts without a code 1-7 or 10, the best domain-valid
    # cosine and Jaccard candidates are marked 8 if they agree. Few concepts
    # have a Jaccard candidate here (threshold 0.9, mostly handled in steps
    # 5/6), and the hierarchy check always passes for a concept without
    # accepted targets; both kept on purpose for the evaluation.
    l.printHeader("Step 7: Best-candidate fallback")

    l.log("Step 7: Marking agreeing best cosine and Jaccard candidates for LLM review...")

    handled_hpo = set(
        mappings.loc[mappings[validity_column].isin(validity_hierarchy_reversals + [validity_accepted]), id_col].unique()
    )

    all_hpo = set(mappings[id_col].unique())
    unmapped_hpo = all_hpo - handled_hpo
    l.log(f"HPO concepts without an accepted or classified mapping: {len(unmapped_hpo)}.")

    valid_mappings = mappings[~mappings_pairs.isin(domain_invalid)]

    embedding_pool = valid_mappings[valid_mappings[attr_col] == embedding_mappings]
    jaccard_pool   = valid_mappings[valid_mappings[attr_col] == jaccard_mappings]

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
    step7_skipped_reversal = 0
    step7_pairs            = set()

    for hpo_id in sorted(unmapped_hpo):
        cosine_result  = best_cosine.get(hpo_id)
        jaccard_result = best_jaccard.get(hpo_id)

        if cosine_result is None or jaccard_result is None:
            step7_skipped_missing += 1
            continue

        cosine_sct,  cosine_conf  = cosine_result
        jaccard_sct, jaccard_conf = jaccard_result

        if cosine_sct != jaccard_sct:
            step7_skipped_disagree += 1
            continue

        sct_id = cosine_sct
        pair   = (hpo_id, sct_id)

        if pair in accepted_pairs or pair in domain_invalid:
            step7_skipped_other += 1
            continue

        action, _ = check_hierarchy(hpo_id, sct_id)

        if action in ("ok", "demote"):
            # Only for LLM review: no reference backing, no exact similarity.
            mark_validity(hpo_id, sct_id, validity_best_candidate)
            step7_pairs.add(pair)
            step7_count += 1
        else:
            step7_skipped_reversal += 1

    l.log(f"Step 7 (HPO concepts): {step7_count} candidates marked for LLM review "
          f"(validity={validity_best_candidate}).")
    l.log(f"Step 7 (HPO concepts): {step7_skipped_missing} skipped (no cosine or no "
          f"Jaccard candidate).")
    l.log(f"Step 7 (HPO concepts): {step7_skipped_disagree} skipped (cosine and "
          f"Jaccard disagree).")
    l.log(f"Step 7 (HPO concepts): {step7_skipped_other} skipped (already accepted "
          f"or domain mismatch).")
    l.log(f"Step 7 (HPO concepts): {step7_skipped_reversal} skipped (hierarchy reversal).")
    evaluate_pairs(step7_pairs, f"step 7, best candidates, validity={validity_best_candidate}")
    # Steps 7 and 8 accept nothing; accepted and review views are logged apart.
    evaluate_accepted("step 7")
    review_after7 = validity_hierarchy_reversals + [validity_best_candidate]
    evaluation(
        mappings[mappings[validity_column].isin(review_after7 + [validity_accepted])],
        graham,
        f"Evaluation (after step 7, accepted + candidates for LLM review, "
        f"validity=1–{validity_best_candidate} and {validity_accepted}):",
        hpo_ontology,
    )
    l.log(f"Confidence distribution (after step 7, candidates for LLM review, "
          f"validity=1–{validity_best_candidate}):")
    printCounts(mappings[mappings[validity_column].isin(review_after7)], hpo_ontology, similarity_steps, graham, True)
    l.log("Step 7: Marking agreeing best cosine and Jaccard candidates for LLM review completed.")

    # Step 8: the best domain-valid cosine candidate of each concept still
    # unhandled is marked 9, without hierarchy check (left to the LLM).
    l.printHeader("Step 8: Last-resort fallback")

    l.log("Step 8: Marking the best cosine candidate of unmapped HPO concepts "
          "for LLM review...")

    handled_hpo_after7 = set(
        mappings.loc[mappings[validity_column].isin(review_after7 + [validity_accepted]), id_col].unique()
    )
    still_unmapped = all_hpo - handled_hpo_after7
    l.log(f"HPO concepts still without an accepted or classified mapping: {len(still_unmapped)}.")

    step8_pool = valid_mappings[
        (valid_mappings[id_col].isin(still_unmapped)) &
        (valid_mappings[attr_col] == embedding_mappings)
    ]

    step8_count   = 0
    step8_skipped = 0
    step8_pairs   = set()

    if not step8_pool.empty:
        idx = step8_pool.groupby(id_col)[confidence_column].idxmax()
        top = step8_pool.loc[idx]

        for hpo_id, sct_id in zip(top[id_col], top[val_col]):
            pair = (hpo_id, sct_id)

            if pair in accepted_pairs or pair in domain_invalid:
                step8_skipped += 1
                continue

            # Counted only if rows were still NA.
            if mark_validity(hpo_id, sct_id, validity_last_resort):
                step8_pairs.add(pair)
                step8_count += 1
            else:
                step8_skipped += 1

    step8_no_candidate = len(still_unmapped) - step8_pool[id_col].nunique()

    l.log(f"Step 8 (HPO concepts): {step8_count} candidates marked for LLM review "
          f"(validity={validity_last_resort}).")
    l.log(f"Step 8 (HPO concepts): {step8_skipped} skipped (already classified).")
    l.log(f"Step 8 (HPO concepts): {step8_no_candidate} skipped (no domain-valid "
          f"embedding candidate).")
    evaluate_pairs(step8_pairs, f"step 8, last-resort candidates, validity={validity_last_resort}")
    evaluate_accepted("step 8")
    evaluation(
        mappings[mappings[validity_column].isin(validity_reviewable)],
        graham,
        f"Evaluation (after step 8, accepted + candidates for LLM review, "
        f"validity=1–{validity_accepted}):",
        hpo_ontology,
    )
    l.log("Step 8: Marking the best cosine candidate of unmapped HPO concepts "
          "for LLM review completed.")

    l.printHeader("Writing filtered mappings")

    pair_validity = mappings.drop_duplicates(subset=[id_col, val_col, validity_column])
    l.log("Validity codes (pairs / rows):")
    for code in validity_all:
        n_code_pairs = int((pair_validity[validity_column] == code).sum())
        n_code_rows  = int((mappings[validity_column] == code).sum())
        l.log(f"  validity={code:<3} {VALIDITY_LABELS[code]:<52}: "
              f"{n_code_pairs:7} / {n_code_rows:7}")
    l.log(f"  validity=NA  {'not reached by any step':<52}: "
          f"{int(pair_validity[validity_column].isna().sum()):7} / "
          f"{int(mappings[validity_column].isna().sum()):7}")

    writeHugeCSV(mappings, filtered_path)

    l.printHeader("Summary")

    final = mappings[mappings[validity_column].isin(validity_all)]

    evaluation(
        final[final[validity_column] == validity_accepted],
        graham,
        f"Evaluation (final accepted set, validity={validity_accepted}):",
        hpo_ontology
    )

    # Mapping constellations of the validity 1-10 pairs: 1:n = HPO concept
    # with several SCT targets (n_sct > 1), n:1 = SCT concept shared by several
    # HPO concepts (n_hpo > 1). MappingLLM gives HPO concepts with one target
    # (1:1, n:1) a pair-level prompt and the others a group-level prompt.
    llm_pairs = (
        final[final[validity_column].isin(validity_reviewable)]
        .drop_duplicates(subset=[id_col, val_col])[[id_col, val_col]]
    )

    sct_per_hpo = llm_pairs.groupby(id_col)[val_col].nunique()
    hpo_per_sct = llm_pairs.groupby(val_col)[id_col].nunique()

    llm_pairs = llm_pairs.copy()
    llm_pairs["n_sct"] = llm_pairs[id_col].map(sct_per_hpo)
    llm_pairs["n_hpo"] = llm_pairs[val_col].map(hpo_per_sct)

    def classify(row):
        """Constellation of a pair row from its n_sct and n_hpo."""
        if row["n_sct"] == 1 and row["n_hpo"] == 1:
            return "1:1"
        elif row["n_sct"] > 1 and row["n_hpo"] == 1:
            return "1:n"
        elif row["n_sct"] == 1 and row["n_hpo"] > 1:
            return "n:1"
        else:
            return "m:n"

    total_pairs = len(llm_pairs)
    counts = llm_pairs.apply(classify, axis=1).value_counts() if total_pairs else pd.Series(dtype=int)

    l.log(f"Mapping constellations (validity=1–{validity_accepted}, "
          f"{total_pairs} pairs):")
    for constellation in ["1:1", "1:n", "n:1", "m:n"]:
        count = counts.get(constellation, 0)
        share = count / total_pairs if total_pairs else 0
        l.log(f"  {constellation}: {count:6} pairs ({share:.1%})")

    l.printHeader("Filtering generated mappings completed.")
