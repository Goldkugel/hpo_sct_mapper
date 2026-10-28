import sys

# Prevent Python from generating .pyc files (compiled bytecode files)
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
# A pair keeps the first code it is given; only demotion (a narrower
# candidate displacing a broader accepted one) overwrites an accepted 10.
# Pairs not reached by any step keep validity=NA.
#
# Output: all rows of mapping.csv with the validity column filled in,
# written to ./data/output/filtered/filtered.csv. Rows with validity=NA are
# kept: MappingErrorAnalysis uses them as the universe of generated pairs.
# MappingLLM reviews validity 1-10.
# ==========================================================================

EXACT_SIMILARITY = 1.0
# Tolerance for "similarity == 1.0". Cosine similarities are computed in
# float32, where two identical embeddings give values like 0.9999997; 1e-6
# covers that rounding (MappingGeneration also rounds to 6 decimals).
_EPS             = 1e-6


def is_exact(value: float) -> bool:
    """Floating point safe check for 'similarity == 1.0'."""
    return abs(value - EXACT_SIMILARITY) < _EPS


if __name__ == "__main__":

    l = Logger()

    l.printHeader("Filtering generated mappings")

    # ======================================================================
    # Loading data
    # ======================================================================

    l.printHeader("Loading data")

    hpo, sct, umls, hpo_ontology, sct_ontology, umls_ontology, graham = loadInputs(";")

    id_col   = hpo_ontology.config.id_column
    val_col  = hpo_ontology.config.value_column
    attr_col = hpo_ontology.config.attribute_column

    l.log("Loading generated mappings...")
    mappings = pd.read_csv(mapping_path, sep=";")
    mappings = keepHPOConcepts(mappings, id_col)
    n_pairs  = len(mappings.drop_duplicates(subset=[id_col, val_col]).index)
    l.log(f"Loading generated mappings completed. {len(mappings.index)} rows, "
          f"{n_pairs} pairs, {mappings[id_col].nunique()} HPO concepts.")

    # Initialise validity column to NaN; filled in per-row as the pipeline runs.
    mappings[validity_column] = pd.NA

    # Build a pair -> [row positions] lookup once, so mark_validity never has
    # to scan the full dataframe — O(k) per call instead of O(n). A pair has
    # one row per method that found it, and all of them always get the same
    # validity code. Positions equal index labels here, because
    # keepHPOConcepts resets the index.
    pair_to_indices: dict[tuple, list] = defaultdict(list)
    for idx, (hpo_id, sct_id) in enumerate(zip(mappings[id_col], mappings[val_col])):
        pair_to_indices[(hpo_id, sct_id)].append(idx)

    # ------------------------------------------------------------------
    # Hierarchy closures, built once and reused by all hierarchy checks.
    # hpo_child_map is not used by the current hierarchy check (which
    # operates only within the SCT hierarchy) but is retained for the
    # TODO: accept both candidates when there is no overlap within three
    # generations in either hierarchy.
    # sct_descendants(c) returns all concepts below c (children,
    # grandchildren, ...); sct_ancestors(c) all concepts above it. Both are
    # computed on first use per concept and cached. len(sct_ancestors(c))
    # serves as the depth of c, used to process anchors narrowest-first in
    # step 2.
    # ------------------------------------------------------------------
    l.log("Building hierarchy lookups...")
    hpo_child_map   = build_child_map(hpo, hpo_ontology)
    sct_child_map   = build_child_map(sct, sct_ontology)
    sct_descendants = make_descendant_lookup(sct_child_map)
    sct_ancestors   = make_descendant_lookup(build_parent_map(sct, sct_ontology))
    l.log(f"Building hierarchy lookups completed. SCT: {len(sct_child_map)} "
          f"parent concepts, {sum(len(v) for v in sct_child_map.values())} edges. "
          f"HPO: {len(hpo_child_map)} parent concepts, "
          f"{sum(len(v) for v in hpo_child_map.values())} edges.")

    # ------------------------------------------------------------------
    # Incremental accepted-mapping state, built up step by step below.
    # accepted_pairs   : set of (hpo_id, sct_id) pairs accepted so far.
    # accepted_by_hpo  : hpo_id -> {sct_id, ...} for fast per-concept lookup.
    # domain_invalid   : set of (hpo_id, sct_id) pairs marked validity=0.
    # anchor_pairs     : pairs accepted as anchors in step 2; never demoted.
    # classified_pairs : pairs assigned any validity code so far.
    # ------------------------------------------------------------------
    accepted_pairs   = set()
    accepted_by_hpo  = defaultdict(set)
    domain_invalid   = set()
    anchor_pairs     = set()
    classified_pairs = set()

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

    def demote(hpo_id: str, sct_id: str, reversal_validity: int) -> None:
        """
        Remove a non-anchor accepted pair from the accepted set and mark it
        with the given reversal_validity code. Only called when a narrower
        candidate displaces a broader already-accepted non-anchor mapping.
        The reversal_validity is the code of the step that triggered the
        demotion.
        """
        accepted_pairs.discard((hpo_id, sct_id))
        accepted_by_hpo[hpo_id].discard(sct_id)
        # The pair's rows already hold validity=10, so they must be replaced
        # explicitly; the NA-only guard would otherwise leave them accepted.
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
    ) -> None:
        """
        One summary line per acceptance step, always in the same order:
        accepted, rejected, demoted, not accepted here, skipped. All counts
        are pairs.
        """
        parts = [
            f"{accepted} accepted",
            f"{reversal} rejected (hierarchy reversal, validity={validity})",
            f"{demoted} demoted (broader target displaced, validity={validity})",
        ]
        if not_accepted is not None:
            parts.append(f"{not_accepted} not accepted here ({not_accepted_reason})")
        for reason, count in (skipped or {}).items():
            parts.append(f"{count} skipped ({reason})")
        l.log(f"{step} completed (pairs): " + ", ".join(parts) + ".")

    def evaluate_accepted(step: str) -> None:
        """Evaluate the accepted set (validity=10) after a step."""
        evaluation(
            mappings[mappings[validity_column] == validity_accepted],
            graham,
            f"Evaluation (after {step}, accepted, validity=10):",
            hpo_ontology
        )

    # Pre-build a Series of (hpo_id, sct_id) tuples aligned to mappings.index,
    # reused by evaluate_pairs for fast pair membership tests.
    mappings_pairs = pairSeries(mappings, id_col, val_col)

    def evaluate_pairs(pairs: set, label: str, include_printcounts: bool = False) -> None:
        """
        Evaluate mappings rows whose pair appears in `pairs`, logged as
        "Evaluation (<label>):".

        include_printcounts=True: additionally logs the confidence
        distribution of the Jaccard and embedding rows, where confidence
        varies meaningfully. Not useful for UMLS-based or hierarchy
        reversal sets, where confidence is always 1.0.
        """
        if not pairs:
            return
        selected = mappings[mappings_pairs.isin(pairs)]
        evaluation(selected, graham, f"Evaluation ({label}):", hpo_ontology)
        selected_sim = selected[selected[attr_col].isin([jaccard_mappings, embedding_mappings])]
        if include_printcounts and not selected_sim.empty:
            l.log(f"Confidence distribution ({label}, Jaccard and embedding rows):")
            printCounts(selected_sim, hpo_ontology, similarity_steps, graham, True)

    # ------------------------------------------------------------------
    # Pre-compute the highest confidence seen for each (hpo_id, sct_id)
    # pair across all methods. Used by rule 3 of check_hierarchy, where a
    # narrower candidate may only displace broader accepted targets that
    # are not more confident than itself.
    # ------------------------------------------------------------------
    best_confidence_per_pair: dict[tuple, float] = (
        mappings.groupby([id_col, val_col])[confidence_column]
        .max()
        .to_dict()
    )

    # ------------------------------------------------------------------
    # Hierarchy check: transitive within-concept, narrower wins.
    # For the SAME HPO concept, checks whether the candidate's SCT target
    # is in an ancestor/descendant relationship (over any number of
    # levels) with any other already-accepted SCT target for that concept.
    #
    # Mapping one HPO concept to both an SCT concept and one of its
    # ancestors is a "source collapse": e.g. "Focal seizure" mapped to
    # both "Focal seizure" and its ancestor "Seizure". Only the narrower,
    # more specific target should stay accepted.
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
    # Rules 1 and 3 together make the outcome independent of the order in
    # which two related non-anchor candidates arrive (at equal confidence):
    # broader first -> the narrower one demotes it (rule 3); narrower first
    # -> the broader one is rejected (rule 1). Either way the narrower one
    # ends up accepted. Anchors are immutable, so step 2 sorts its
    # candidates narrowest-first to reach the same result.
    #
    # Note on the confidence comparison in rule 3: every candidate of steps
    # 2-6 is a cross-reference or has Jaccard=1 or cosine=1, so its best
    # confidence is 1.0, as is that of every accepted target. The comparison
    # therefore never rejects in steps 2-6; it only matters if lower-
    # confidence candidates are ever checked against accepted targets.
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
    def pairs_of(rows: pd.DataFrame) -> set:
        return set(zip(rows[id_col], rows[val_col]))

    jaccard_rows   = mappings[mappings[attr_col] == jaccard_mappings]
    embedding_rows = mappings[mappings[attr_col] == embedding_mappings]

    jaccard_exact_pairs   = pairs_of(jaccard_rows[jaccard_rows[confidence_column].apply(is_exact)])
    embedding_exact_pairs = pairs_of(embedding_rows[embedding_rows[confidence_column].apply(is_exact)])

    # All exact lexical pairs have confidence=1.0 by definition.
    exact_lexical_pairs = pairs_of(mappings[mappings[attr_col] == exact_lexical_mappings])

    # Pairs with any reference-based row (HPO xref or UMLS), regardless of
    # whether they are accepted later. Steps 5 and 6 skip them.
    reference_backed_pairs = pairs_of(mappings[mappings[attr_col].isin([
        hpo_sct_mappings, hpo_umls_sct_mappings, umls_hpo_sct_mappings
    ])])

    # ------------------------------------------------------------------
    # SCT label lookup (fully specified names), built once and reused by
    # the domain filter and the anchor domain distribution.
    # ------------------------------------------------------------------
    sct_labels = buildAttributeLookup(sct, sct_ontology, labelClass)

    # ======================================================================
    # Step 1: domain filter
    # ======================================================================

    # Mark ALL pairs whose SCT target does not belong to an acceptable domain
    # as validity=0. Running this first ensures that domain-invalid pairs are
    # permanently excluded before any acceptance step runs, and since
    # mark_validity only writes to NA rows, no subsequent step can overwrite
    # these marks.
    #
    # The domain is the semantic tag at the end of the SCT fully specified
    # name, e.g. "Seizure (finding)" -> "finding". isSCTDomainValid accepts a
    # concept if any of its labels has a tag in acceptableDomains
    # (MappingUtils: disorder, finding, morphologic abnormality) or no tag
    # at all. Concepts without any label are rejected.

    l.printHeader("Step 1: Domain filter")

    l.log("Step 1: Marking pairs with an SCT target outside the acceptable domains...")

    all_pairs = mappings.drop_duplicates(subset=[id_col, val_col])

    for hpo_id, sct_id in zip(all_pairs[id_col], all_pairs[val_col]):
        if not isSCTDomainValid(sct_labels.get(sct_id, [])):
            mark_validity(hpo_id, sct_id, validity_domain_mismatch)
            domain_invalid.add((hpo_id, sct_id))

    l.log(f"Step 1 completed (pairs): {len(domain_invalid)} marked as domain "
          f"mismatch (validity={validity_domain_mismatch}).")
    evaluate_pairs(domain_invalid, f"step 1, domain mismatch, validity={validity_domain_mismatch}")

    # ======================================================================
    # Step 2: anchors
    # ======================================================================

    # Anchor set = direct HPO xrefs to SNOMED CT concepts that are also
    # confirmed by an exact lexical match (case-insensitive). A pair must
    # appear in BOTH hpo_sct_mappings AND exact_lexical_mappings to qualify
    # as an anchor. This intersection provides evidence from two independent
    # sources: an explicit cross-reference in HPO and a label match, giving
    # high confidence without requiring Jaccard or cosine corroboration.
    #
    # Anchors that pass the hierarchy check are accepted as immutable
    # (validity=10) and cannot be demoted by any subsequent step. Anchors
    # that fail it are marked validity=1 and enter the LLM review queue.
    #
    # Within each HPO concept the candidates are processed narrowest first
    # (most SCT ancestors first). Because anchors are immutable, the first
    # of two related anchors would always win; processing the narrowest
    # first makes that the narrower one, as in all other steps: the broader
    # candidate then fails rule 1 of check_hierarchy. For the same reason
    # check_hierarchy can never return "demote" here.

    l.printHeader("Step 2: Anchors (HPO xref + exact lexical)")

    l.log("Step 2: Accepting HPO xrefs to SNOMED CT that are confirmed by an "
          "exact lexical match (case-insensitive)...")

    hpo_sct_pairs     = pairs_of(mappings[mappings[attr_col] == hpo_sct_mappings])
    anchor_candidates = hpo_sct_pairs & exact_lexical_pairs

    def anchor_order(pair: tuple) -> tuple:
        """Sort key: per HPO concept, deepest SCT target first."""
        hpo_id, sct_id = pair
        return (hpo_id, -len(sct_ancestors(sct_id)), sct_id)

    step2_accepted = 0
    step2_reversal = 0
    step2_domain   = 0

    for hpo_id, sct_id in sorted(anchor_candidates, key=anchor_order):
        # Domain-invalid pairs (validity=0, step 1) can never become anchors.
        if (hpo_id, sct_id) in domain_invalid:
            step2_domain += 1
            continue

        action, _ = check_hierarchy(
            hpo_id, sct_id,
            best_confidence_per_pair.get((hpo_id, sct_id), 0.0)
        )
        if action == "ok":
            accept(hpo_id, sct_id, is_anchor=True)
            step2_accepted += 1
        else:
            mark_validity(hpo_id, sct_id, validity_anchor_reversal)
            step2_reversal += 1

    # HPO xrefs without an exact lexical match are left with validity=NA and
    # are evaluated in step 2c.
    log_step_summary(
        "Step 2", step2_accepted, step2_reversal, validity_anchor_reversal,
        not_accepted = len(hpo_sct_pairs - anchor_candidates),
        not_accepted_reason = "HPO xref without exact lexical match, left for step 2c",
        skipped = {"domain mismatch": step2_domain},
    )
    evaluate_accepted("step 2")

    # Distribution of SCT semantic domains among the accepted anchors.
    # extract_snomed_domain parses the parenthetical tag from the SCT label,
    # e.g. "Physical activity (observable entity)" -> "observable entity".
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

    # ======================================================================
    # Step 2b: remaining exact lexical mappings
    # ======================================================================

    # All exact lexical pairs that were not anchor candidates (i.e. those
    # without an HPO xref) are processed here. Pairs that pass the hierarchy
    # check are accepted (validity=10); pairs that fail are marked
    # validity=2 and enter the LLM review queue. From this step on, a
    # narrower candidate can demote a broader target accepted earlier in
    # the same or a previous step (but never an anchor); the demoted pair
    # gets the current step's reversal code.

    l.printHeader("Step 2b: Remaining exact lexical mappings")

    l.log("Step 2b: Accepting exact lexical matches without an HPO xref...")

    step2b_accepted = 0
    step2b_reversal = 0
    step2b_demoted  = 0
    step2b_domain   = 0

    # sorted() fixes the processing order; sets have no stable order.
    for hpo_id, sct_id in sorted(exact_lexical_pairs - anchor_candidates):
        pair = (hpo_id, sct_id)

        if pair in domain_invalid:
            step2b_domain += 1
            continue

        action, conflicts = check_hierarchy(
            hpo_id, sct_id,
            best_confidence_per_pair.get(pair, 0.0)
        )

        if action == "ok":
            accept(hpo_id, sct_id)
            step2b_accepted += 1
        elif action == "demote":
            # Candidate is narrower than accepted non-anchors — demote
            # the broader mappings and accept this one.
            for other_sct in conflicts:
                demote(hpo_id, other_sct, validity_exact_lexical_reversal)
            step2b_demoted += len(conflicts)
            accept(hpo_id, sct_id)
            step2b_accepted += 1
        else:
            mark_validity(hpo_id, sct_id, validity_exact_lexical_reversal)
            step2b_reversal += 1

    log_step_summary(
        "Step 2b", step2b_accepted, step2b_reversal, validity_exact_lexical_reversal,
        demoted = step2b_demoted,
        skipped = {"domain mismatch": step2b_domain},
    )
    evaluate_accepted("step 2b")

    # ======================================================================
    # Shared implementation for steps 2c, 3 and 4.
    # ======================================================================

    def process_corroborated_step(
        step:              str,
        attribute_value:   str,
        reversal_validity: int,
    ) -> None:
        """
        Process all pairs of one reference-based method (attribute_value).
        A pair is accepted (validity=10) if it is
          - not domain-invalid and not classified in an earlier step,
          - corroborated by Jaccard=1 OR cosine=1, and
          - check_hierarchy returns "ok" or "demote".
        Pairs passing the similarity criterion but failing the hierarchy
        check are marked reversal_validity. Pairs without Jaccard=1 or
        cosine=1 are left with validity=NA: they are not accepted here, but
        can still become step 7/8 candidates.
        """
        candidates = mappings[mappings[attr_col] == attribute_value].drop_duplicates(subset=[id_col, val_col])

        accepted_count       = 0
        reversal_count       = 0
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

            # Pairs classified in an earlier step keep that decision: their
            # rows can no longer be changed.
            if pair in classified_pairs:
                classified_count += 1
                continue

            if pair not in jaccard_exact_pairs and pair not in embedding_exact_pairs:
                low_similarity_pairs.add(pair)
                continue

            action, conflicts = check_hierarchy(
                hpo_id, sct_id,
                best_confidence_per_pair.get(pair, 0.0)
            )

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

            else:
                # Candidate conflicts with an accepted mapping -> reject it.
                mark_validity(hpo_id, sct_id, reversal_validity)
                reversal_pairs.add(pair)
                reversal_count += 1

        log_step_summary(
            step, accepted_count, reversal_count, reversal_validity,
            demoted = demoted_count,
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

    # ======================================================================
    # Step 2c: remaining HPO xrefs
    # ======================================================================

    # HPO xrefs that were not anchor candidates in step 2 (no exact lexical
    # match). They are held to the same similarity criterion as the UMLS-
    # based steps 3 and 4: Jaccard=1 OR cosine=1. Hierarchy reversals are
    # marked validity=3, distinct from steps 2 and 2b, to allow separate
    # analysis of this set.

    l.printHeader("Step 2c: Remaining HPO xrefs")

    l.log("Step 2c: Accepting HPO xrefs with Jaccard=1 or cosine=1...")

    process_corroborated_step(
        "Step 2c",
        hpo_sct_mappings,
        reversal_validity=validity_hpo_xref_reversal,
    )

    # ======================================================================
    # Step 3: HPO -> UMLS -> SNOMED CT mappings
    # ======================================================================

    l.printHeader("Step 3: HPO -> UMLS -> SNOMED CT mappings")

    l.log("Step 3: Accepting HPO -> UMLS -> SNOMED CT mappings with "
          "Jaccard=1 or cosine=1...")

    process_corroborated_step(
        "Step 3",
        hpo_umls_sct_mappings,
        reversal_validity=validity_umls_bridge_reversal,
    )

    # ======================================================================
    # Step 4: shared UMLS CUI mappings
    # ======================================================================

    l.printHeader("Step 4: Shared UMLS CUI mappings")

    l.log("Step 4: Accepting shared UMLS CUI mappings with Jaccard=1 or "
          "cosine=1...")

    process_corroborated_step(
        "Step 4",
        umls_hpo_sct_mappings,
        reversal_validity=validity_umls_cui_reversal,
    )

    # ======================================================================
    # Shared implementation for steps 5 and 6.
    # ======================================================================

    def process_similarity_step(
        step:              str,
        pool:              pd.DataFrame,
        qualifies,
        not_accepted_reason: str,
        reversal_validity: int,
    ) -> None:
        """
        Process the pairs of the Jaccard/embedding rows in `pool` that have
        no reference-based backing. A pair is accepted (validity=10) if it
        is not domain-invalid, not classified in an earlier step,
        qualifies(pair) is true, and check_hierarchy returns "ok" or
        "demote". Hierarchy failures are marked reversal_validity; pairs
        that do not qualify are left with validity=NA.
        """
        # Unique pairs in first-seen order.
        candidate_pairs = list(dict.fromkeys(zip(pool[id_col], pool[val_col])))

        accepted_count       = 0
        reversal_count       = 0
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

            # Reference-backed pairs belong to steps 2-4.
            if pair in reference_backed_pairs:
                reference_count += 1
                continue

            # Pairs classified in an earlier step (accepted, rejected, or
            # e.g. exact lexical pairs from step 2b) keep that decision.
            if pair in classified_pairs:
                classified_count += 1
                continue

            if not qualifies(pair):
                not_qualifying_pairs.add(pair)
                continue

            action, conflicts = check_hierarchy(
                hpo_id, sct_id,
                best_confidence_per_pair.get(pair, 0.0)
            )

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

        log_step_summary(
            step, accepted_count, reversal_count, reversal_validity,
            demoted = demoted_count,
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

    # ======================================================================
    # Step 5: Jaccard=1 AND cosine=1, no reference-based backing
    # ======================================================================

    # Only pairs whose sole evidence is lexical/embedding similarity are
    # considered. Hierarchy reversals are marked validity=6.

    l.printHeader("Step 5: Jaccard=1 and cosine=1 mappings")

    l.log("Step 5: Accepting mappings with Jaccard=1 and cosine=1 and no "
          "reference-based backing...")

    process_similarity_step(
        "Step 5",
        mappings[mappings[attr_col].isin([jaccard_mappings, embedding_mappings])],
        lambda pair: pair in jaccard_exact_pairs and pair in embedding_exact_pairs,
        "not both Jaccard=1 and cosine=1",
        reversal_validity=validity_jaccard_cosine_reversal,
    )

    # ======================================================================
    # Step 6: Jaccard=1 only, no reference-based backing
    # ======================================================================

    # Pairs with Jaccard=1 and a cosine similarity below 1 (or no embedding
    # candidate at all). Pairs with Jaccard=1 and cosine=1 were classified
    # in step 5 and are skipped as already classified. Hierarchy reversals
    # are marked validity=7.

    l.printHeader("Step 6: Jaccard=1 only mappings")

    l.log("Step 6: Accepting mappings with Jaccard=1 (cosine below 1 or none) "
          "and no reference-based backing...")

    process_similarity_step(
        "Step 6",
        mappings[mappings[attr_col] == jaccard_mappings],
        lambda pair: pair in jaccard_exact_pairs and pair not in embedding_exact_pairs,
        "Jaccard<1",
        reversal_validity=validity_jaccard_only_reversal,
    )

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
    l.log(f"Hierarchy consistency check after step 6: {collapsed_concepts} HPO "
          f"concepts with accepted SCT targets in an ancestor/descendant "
          f"relationship ({collapsed_pairs} related target pairs; expected 0).")

    # ======================================================================
    # Step 7: best-candidate fallback
    # ======================================================================

    # For every HPO concept that has neither an accepted mapping (validity=10)
    # nor any classified candidate already, find the highest-cosine and
    # highest-Jaccard candidates (domain-valid only).
    # If both point to the same SCT concept, run the hierarchy check:
    #   - passes  -> mark validity=8 (LLM review)
    #   - fails   -> skip
    # "Unmapped" includes concepts whose only candidates are domain
    # mismatches (validity=0) or were never reached (validity=NA).
    #
    # Because Jaccard candidates only exist above threshold_jaccard_similarity
    # (in practice Jaccard=1, mostly handled in steps 5/6), few concepts reach
    # this step with a Jaccard candidate. And since an unmapped concept has
    # no accepted targets, its hierarchy check always passes. This is kept
    # on purpose for the evaluation.

    l.printHeader("Step 7: Best-candidate fallback")

    l.log("Step 7: Marking agreeing best cosine and Jaccard candidates of "
          "unmapped HPO concepts for LLM review...")

    # HPO concepts with at least one row of validity 1–7 or 10 are handled.
    handled_hpo = set(
        mappings.loc[mappings[validity_column].isin(validity_hierarchy_reversals + [validity_accepted]), id_col].unique()
    )

    # All HPO concepts that appear in the mappings dataframe.
    all_hpo = set(mappings[id_col].unique())
    unmapped_hpo = all_hpo - handled_hpo
    l.log(f"HPO concepts without an accepted or classified mapping: {len(unmapped_hpo)}")

    # Build per-concept best-cosine and best-Jaccard lookups from the full
    # mappings pool, excluding domain-invalid pairs.
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

        action, _ = check_hierarchy(hpo_id, sct_id, best_confidence_per_pair.get(pair, 0.0))

        if action in ("ok", "demote"):
            # Hierarchy check passes — mark for LLM review rather than
            # auto-accepting, since the pair has neither reference backing
            # nor the exact similarity required in steps 2-6.
            mark_validity(hpo_id, sct_id, validity_best_candidate)
            step7_pairs.add(pair)
            step7_count += 1
        else:
            # "reject" -> skip, no validity assigned.
            step7_skipped_reversal += 1

    l.log(f"Step 7 completed (HPO concepts): {step7_count} candidates marked "
          f"for LLM review (validity={validity_best_candidate}), "
          f"{step7_skipped_missing} skipped (no cosine or no Jaccard candidate), "
          f"{step7_skipped_disagree} skipped (cosine and Jaccard disagree), "
          f"{step7_skipped_other} skipped (already accepted or domain mismatch), "
          f"{step7_skipped_reversal} skipped (hierarchy reversal).")
    evaluate_pairs(step7_pairs, f"step 7, best candidates, validity={validity_best_candidate}")
    # Steps 7 and 8 never accept anything, so the accepted set is the one
    # from step 6. Both views are logged to keep them apart.
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

    # ======================================================================
    # Step 8: last-resort fallback
    # ======================================================================

    # For every HPO concept that still has no classified mapping after step 7,
    # the single highest-cosine-similarity candidate (domain-valid only) is
    # marked validity=9. No hierarchy check is applied; these are true
    # last-resort candidates and any issues will be caught during LLM review.

    l.printHeader("Step 8: Last-resort fallback")

    l.log("Step 8: Marking the best cosine candidate of still unmapped HPO "
          "concepts for LLM review...")

    handled_hpo_after7 = set(
        mappings.loc[mappings[validity_column].isin(review_after7 + [validity_accepted]), id_col].unique()
    )
    still_unmapped = all_hpo - handled_hpo_after7
    l.log(f"HPO concepts still without an accepted or classified mapping: {len(still_unmapped)}")

    # From the domain-valid cosine pool, find the highest-confidence
    # candidate for each still-unmapped concept.
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

            # Use mark_validity to respect the NA-only guard — if this pair
            # was somehow already classified by an earlier step, do not
            # overwrite it. Only count the pair if rows were actually marked.
            if mark_validity(hpo_id, sct_id, validity_last_resort):
                step8_pairs.add(pair)
                step8_count += 1
            else:
                step8_skipped += 1

    step8_no_candidate = len(still_unmapped) - step8_pool[id_col].nunique()

    l.log(f"Step 8 completed (HPO concepts): {step8_count} candidates marked "
          f"for LLM review (validity={validity_last_resort}), "
          f"{step8_skipped} skipped (already classified), "
          f"{step8_no_candidate} skipped (no domain-valid embedding candidate).")
    evaluate_pairs(step8_pairs, f"step 8, last-resort candidates, validity={validity_last_resort}")
    evaluate_accepted("step 8")
    evaluation(
        mappings[mappings[validity_column].isin(validity_reviewable)],
        graham,
        f"Evaluation (after step 8, accepted + candidates for LLM review, "
        f"validity=1–{validity_accepted}):",
        hpo_ontology,
    )

    # ======================================================================
    # Writing filtered mappings
    # ======================================================================

    # All rows of mapping.csv are written with their validity code. Rows with
    # validity=NA (not reached by any step) are kept: MappingErrorAnalysis
    # uses all generated pairs as its universe, and MappingLLM only reviews
    # validity 1-10.

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

    # ======================================================================
    # Summary
    # ======================================================================

    l.printHeader("Summary")

    final = mappings[mappings[validity_column].isin(validity_all)]

    evaluation(
        final[final[validity_column] == validity_accepted],
        graham,
        f"Evaluation (final accepted set, validity={validity_accepted}):",
        hpo_ontology
    )

    # ------------------------------------------------------------------
    # Mapping constellation analysis.
    # For all non-domain-mismatch mappings (validity=1–10), count how many
    # unique (hpo_id, sct_id) pairs exist and classify each pair by how many
    # SCT targets its HPO concept has (n_sct) and how many HPO sources its
    # SCT concept has (n_hpo):
    #   1:1 : n_sct = 1, n_hpo = 1 (the pair stands alone)
    #   1:n : n_sct > 1, n_hpo = 1 (HPO concept with several SCT targets)
    #   n:1 : n_sct = 1, n_hpo > 1 (SCT concept shared by several HPO concepts)
    #   m:n : n_sct > 1, n_hpo > 1
    # MappingLLM groups by HPO concept only: concepts with one SCT target
    # (1:1 and n:1) get a pair-level prompt, concepts with several (1:n and
    # m:n) a group-level prompt.
    # ------------------------------------------------------------------
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

    total_pairs = len(llm_pairs)
    counts = llm_pairs.apply(classify, axis=1).value_counts() if total_pairs else pd.Series(dtype=int)

    l.log(f"Mapping constellations (validity=1–{validity_accepted}, "
          f"{total_pairs} pairs):")
    for constellation in ["1:1", "1:n", "n:1", "m:n"]:
        count = counts.get(constellation, 0)
        share = count / total_pairs if total_pairs else 0
        l.log(f"  {constellation}: {count:6} pairs ({share:.1%})")

    l.printHeader("Filtering generated mappings completed.")
