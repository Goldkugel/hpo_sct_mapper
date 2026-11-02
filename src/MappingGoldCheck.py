import sys

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

from logger         import Logger
from model          import Model
from adapter        import labelClass, writeHugeCSV
from MappingUtils   import (
    loadInputs, confidence_column, hpo_id_column, sct_id_column,
    gold_check_path, build_child_map, make_descendant_lookup,
    REVIEW_REASON, validity_accepted,
)
from MappingLLM     import (
    build_ontology_lookups, pair_prompt_fields, group_prompt_fields, start_pass,
    share, CONFIDENCE_MAX, FALLBACK_RESULT, PAIR_PROMPT_INDEX, GROUP_PROMPT_INDEX,
)
import os
import pandas       as pd

# ==========================================================================
# MappingGoldCheck.py
#
# Checks the Graham gold standard with the LLM. Every unique (hpo_id,
# sct_id) pair of the gold standard is reviewed with the prompts of
# MappingLLM, as if the pipeline had accepted it:
#
#   Pass 1 : HPO concepts with one gold target   -> pair-level prompt
#                                                   (single_mapping.md)
#   Pass 2 : HPO concepts with several targets   -> group-level prompt
#                                                   (multiple_mapping.md)
#
# The review reason is always ACCEPTED, so the LLM re-evaluates each pair
# as a judge and rejects it if it finds a clinically meaningful mismatch.
# A rejected gold pair is a candidate for an error in the gold standard.
#
# Pairs whose HPO or SNOMED CT concept is missing from the loaded data
# (e.g. obsolete HPO concepts, inactive SNOMED CT concepts) cannot be
# described to the LLM. They are not reviewed and keep their status.
#
# Independent of the mapping pipeline: only the ontologies and the gold
# standard are read, so the script can run before or after steps 1-4.
#
# Output: every gold pair with its status and the LLM's decision,
# confidence and rationale, written to
# ./data/output/evaluation/gold_check.csv.
# ==========================================================================

# ------------------------------------------------------------------
# Constants.
# ------------------------------------------------------------------

# Review reason shown to the LLM for every gold pair.
GOLD_REVIEW_REASON = REVIEW_REASON[validity_accepted]

# Prefix of the checkpoint files in data/tmp/: gold_0_<model>.tmp (pass 1)
# and gold_1_<model>.tmp (pass 2). MappingLLM uses "prompt", so neither
# script resumes from the other's checkpoints.
CHECKPOINT_PREFIX = "gold"

# Status of a gold pair: reviewed by the LLM, or why it was not.
status_checked       = "CHECKED"
status_hpo_not_found = "HPO_NOT_FOUND"
status_sct_not_found = "SCT_NOT_FOUND"

# How many rejected gold pairs to print as examples in the log.
REJECTED_EXAMPLE_LIMIT = 20


# ------------------------------------------------------------------
# Helper functions.
# ------------------------------------------------------------------
def select_gold_pairs(gold: pd.DataFrame, hpo_data: dict, sct_data: dict) -> pd.DataFrame:
    """
    Unique (hpo_id, sct_id) pairs of the gold standard with their status:
    CHECKED if both concepts have a label in the loaded data, otherwise
    why the pair cannot be sent to the LLM. Sorted, so that the prompt
    indices (and thereby the checkpoints) are reproducible.
    """
    pairs = (
        gold[[hpo_id_column, sct_id_column]]
        .drop_duplicates()
        .sort_values([hpo_id_column, sct_id_column], kind="stable")
        .reset_index(drop=True)
    )
    hpo_found = pairs[hpo_id_column].isin(hpo_data[labelClass].keys())
    sct_found = pairs[sct_id_column].isin(sct_data[labelClass].keys())
    pairs["status"] = status_checked
    pairs.loc[~sct_found, "status"] = status_sct_not_found
    # A missing HPO concept takes precedence: without it, nothing can be shown.
    pairs.loc[~hpo_found, "status"] = status_hpo_not_found
    return pairs

def clean_result(result) -> dict | None:
    """
    The decision, confidence and rationale of one LLM decision, or None if
    it is missing or has no valid decision or confidence. The model
    sometimes answers "accept" or "4" instead of "ACCEPT" or 4; both are
    accepted. Unlike MappingLLM, any other decision counts as unusable, so
    that every reviewed gold pair ends up accepted or rejected.
    """
    if not isinstance(result, dict):
        return None
    decision = str(result.get("decision", "")).strip().upper()
    try:
        confidence = int(result.get("confidence"))
    except (TypeError, ValueError):
        return None
    if decision not in ("ACCEPT", "REJECT") or not 1 <= confidence <= CONFIDENCE_MAX:
        return None
    return {
        "decision":   decision,
        "confidence": confidence,
        "rationale":  str(result.get("rationale", "")),
    }


if __name__ == "__main__":

    l = Logger()
    l.printHeader("LLM check of the gold standard")

    # ======================================================================
    # Loading data
    # ======================================================================

    l.printHeader("Loading data")

    # umls is loaded by loadInputs but not used in this script.
    hpo, sct, umls, hpo_ontology, sct_ontology, umls_ontology, graham = loadInputs(";")
    del umls, umls_ontology  # not used in this script; free the memory

    l.log("Building ontology lookup tables...")
    hpo_data = build_ontology_lookups(hpo, hpo_ontology)
    sct_data = build_ontology_lookups(sct, sct_ontology)
    # All SCT descendants of a concept, used to tell the LLM which gold
    # targets of an HPO concept are ancestors/descendants of each other.
    sct_descendants = make_descendant_lookup(build_child_map(sct, sct_ontology))
    l.log("Building ontology lookup tables completed.")

    # ======================================================================
    # Selecting gold pairs
    # ======================================================================

    # Routing is per HPO concept, by its number of reviewable gold targets,
    # like in MappingLLM: targets missing from the SCT data are not shown
    # and do not count.

    l.printHeader("Selecting gold pairs")

    pairs   = select_gold_pairs(graham, hpo_data, sct_data)
    checked = pairs[pairs["status"] == status_checked].reset_index(drop=True)
    checked["n_sct"] = checked[hpo_id_column].map(checked[hpo_id_column].value_counts())

    one_to_one = checked[checked["n_sct"] == 1].reset_index(drop=True)
    one_to_n   = checked[checked["n_sct"] >  1].reset_index(drop=True)

    # Groups for the group-level pass: hpo_id -> list of its sct_ids.
    groups_1n     = one_to_n.groupby(hpo_id_column, sort=False)[sct_id_column].apply(list).to_dict()
    group_hpo_ids = list(groups_1n.keys())

    l.log(f"Gold pairs: {len(pairs)} for {pairs[hpo_id_column].nunique()} HPO concepts.")
    for status in [status_checked, status_hpo_not_found, status_sct_not_found]:
        count = int((pairs["status"] == status).sum())
        l.log(f"  {status + ':':<16}{count:7} ({share(count, len(pairs))})")
    l.log(f"HPO concepts with one gold target (pass 1, pair-level prompt): "
          f"{len(one_to_one)} concepts, {len(one_to_one)} pairs.")
    l.log(f"HPO concepts with several gold targets (pass 2, group-level prompt): "
          f"{len(group_hpo_ids)} concepts, {len(one_to_n)} pairs.")
    l.log(f"Review reason of all pairs: {GOLD_REVIEW_REASON}.")

    # ======================================================================
    # Pass 1: pair-level review
    # ======================================================================

    # One conversation per gold pair.

    l.printHeader("Pass 1: Pair-level review")

    l.log(f"Pass 1: Reviewing {len(one_to_one)} gold pairs with the pair-level prompt...")

    pair_results     = []
    pass1_unparsable = 0

    with Model(index=0) as m:
        if start_pass(m, PAIR_PROMPT_INDEX, len(one_to_one), CHECKPOINT_PREFIX):

            l.log("Formatting pair-level prompts...")
            for row in one_to_one.itertuples(index=True):
                m.formatPromptFromFile(row.Index, pair_prompt_fields(
                    getattr(row, hpo_id_column), getattr(row, sct_id_column),
                    GOLD_REVIEW_REASON, hpo_data, sct_data,
                ))
            l.log("Formatting pair-level prompts completed.")

        m.generate()

        for i in range(len(one_to_one)):
            result = clean_result(m.toJSON(i))
            if result is None:
                result = FALLBACK_RESULT.copy()
                pass1_unparsable += 1
            pair_results.append(result)

    one_to_one["llm_pass"] = 1

    l.log(f"Pass 1 completed: {pass1_unparsable} unparsable answers, counted as rejected.")

    # ======================================================================
    # Pass 2: group-level review
    # ======================================================================

    # One conversation per HPO concept with all its gold targets, so the
    # LLM can judge them together (one-to-many mappings, specificity). The
    # answer holds one decision per target, matched back by SNOMED CT ID.

    l.printHeader("Pass 2: Group-level review")

    l.log(f"Pass 2: Reviewing {len(one_to_n)} gold pairs of {len(group_hpo_ids)} HPO "
          f"concepts with the group-level prompt...")

    group_results    = {}   # (hpo_id, sct_id) -> decision
    pass2_unparsable = 0    # groups whose answer could not be used
    pass2_missing    = 0    # targets without a usable decision in a usable answer

    with Model(index=0) as m:
        if start_pass(m, GROUP_PROMPT_INDEX, len(group_hpo_ids), CHECKPOINT_PREFIX):

            l.log("Formatting group-level prompts...")
            for g_idx, hpo_id in enumerate(group_hpo_ids):
                candidates = [(sct_id, GOLD_REVIEW_REASON) for sct_id in groups_1n[hpo_id]]
                m.formatPromptFromFile(g_idx, group_prompt_fields(
                    hpo_id, candidates, hpo_data, sct_data, sct_descendants,
                ))
            l.log("Formatting group-level prompts completed.")

        m.generate()

        for g_idx, hpo_id in enumerate(group_hpo_ids):
            result = m.toJSON(g_idx)

            if not isinstance(result, dict) or not isinstance(result.get("mappings"), list):
                # Fallback: reject all targets of this HPO concept.
                for sct_id in groups_1n[hpo_id]:
                    group_results[hpo_id, sct_id] = FALLBACK_RESULT.copy()
                pass2_unparsable += 1
                continue

            # Index the returned decisions by SCT ID, so a reordered or
            # incomplete answer is still matched to the right target.
            decisions_by_sct_id = {
                item.get("sct_id"): item for item in result["mappings"] if isinstance(item, dict)
            }
            for sct_id in groups_1n[hpo_id]:
                decision = clean_result(decisions_by_sct_id.get(sct_id))
                if decision is None:
                    decision = FALLBACK_RESULT.copy()
                    pass2_missing += 1
                group_results[hpo_id, sct_id] = decision

    one_to_n["llm_pass"] = 2

    l.log(f"Pass 2 completed: {pass2_unparsable} unparsable group answers and "
          f"{pass2_missing} targets without a usable decision, all counted as rejected.")

    # ======================================================================
    # Writing results
    # ======================================================================

    l.printHeader("Writing results")

    # Decisions in the row order of one_to_one and one_to_n.
    results = pair_results + [
        group_results[getattr(row, hpo_id_column), getattr(row, sct_id_column)]
        for row in one_to_n.itertuples(index=False)
    ]
    reviewed = pd.concat([one_to_one, one_to_n], ignore_index=True)
    reviewed["llm_decision"]    = [r.get("decision",   FALLBACK_RESULT["decision"])                    for r in results]
    reviewed[confidence_column] = [r.get("confidence", FALLBACK_RESULT["confidence"]) / CONFIDENCE_MAX for r in results]
    reviewed["llm_rationale"]   = [r.get("rationale", "")                                               for r in results]

    # All gold pairs: the reviewed ones with the LLM's answer, the others
    # with their status and empty LLM columns. Both labels are added so
    # the file can be read without the ontologies.
    out = pd.concat([reviewed, pairs[pairs["status"] != status_checked]], ignore_index=True)
    out["hpo_label"] = out[hpo_id_column].map(lambda i: "; ".join(hpo_data[labelClass].get(i, [])))
    out["sct_label"] = out[sct_id_column].map(lambda i: "; ".join(sct_data[labelClass].get(i, [])))
    # Nullable integers, so the pairs that were not reviewed stay empty.
    out["n_sct"]    = out["n_sct"].astype("Int64")
    out["llm_pass"] = out["llm_pass"].astype("Int64")
    out = out[[hpo_id_column, "hpo_label", sct_id_column, "sct_label", "status", "n_sct",
               "llm_pass", "llm_decision", confidence_column, "llm_rationale"]]

    l.log(f"Rows to write: {len(out.index)} (one per gold pair).")
    # The check can run before MappingErrorAnalysis has created the folder.
    os.makedirs(os.path.dirname(gold_check_path), exist_ok=True)
    writeHugeCSV(out, gold_check_path)

    # ======================================================================
    # Summary
    # ======================================================================

    l.printHeader("Summary")

    l.log("LLM check of the gold pairs:")
    l.log(f"  {'Pass':<34} {'Reviewed':>8} {'Accepted':>9} {'Rejected':>9} {'Rejection Rate':>15}")
    for label, subset in [("Pass 1 (one gold target)",     reviewed[reviewed["llm_pass"] == 1]),
                          ("Pass 2 (several gold targets)", reviewed[reviewed["llm_pass"] == 2]),
                          ("Total",                         reviewed)]:
        accepted = int((subset["llm_decision"] == "ACCEPT").sum())
        rejected = len(subset) - accepted
        l.log(f"  {label:<34} {len(subset):>8} {accepted:>9} {rejected:>9} "
              f"{share(rejected, len(subset)):>15}")
    l.log(f"  Rejections include {pass1_unparsable + pass2_missing} pairs without a usable "
          f"decision and the pairs of {pass2_unparsable} unparsable group answers.")

    # The rejections the LLM is surest about come first; they are the first
    # candidates for errors in the gold standard.
    rejected = (
        out[(out["status"] == status_checked) & (out["llm_decision"] == "REJECT")]
        .sort_values(confidence_column, ascending=False, kind="stable")
    )
    l.log(f"Rejected gold pairs (up to {REJECTED_EXAMPLE_LIMIT}, highest confidence first; "
          f"all in {gold_check_path}):")
    for row in rejected.head(REJECTED_EXAMPLE_LIMIT).itertuples(index=False):
        l.log(f"  {getattr(row, hpo_id_column)} ({row.hpo_label}) -> "
              f"{getattr(row, sct_id_column)} ({row.sct_label}): "
              f"confidence {getattr(row, confidence_column):.1f}. "
              f"{row.llm_rationale or '(no usable answer)'}")

    l.printHeader("LLM check of the gold standard completed.")
