import sys

sys.dont_write_bytecode = True

from logger         import Logger
from model          import Model
from adapter        import (
    definitionClass, synonymClass, commentClass, labelClass,
    childrenClass, writeHugeCSV,
)
from MappingUtils   import (
    loadInputs, evaluation, validity_column, confidence_column,
    REVIEW_REASON, llm_mappings, filtered_path, llm_filtered_path,
    llm_review_path, validity_accepted, validity_reviewable,
    buildAttributeLookup, is_exact_synonym, build_child_map,
    make_descendant_lookup,
)
import pandas       as pd

# ==========================================================================
# MappingLLM.py
#
# Reviews the candidate mappings of MappingSetBuilder with an LLM. All pairs
# with validity 1-10 are reviewed: candidates (1-9) and auto-accepted pairs
# (10), so the LLM can judge the full set of mappings per HPO concept.
#
#   Pass 1 : HPO concepts with one candidate   -> pair-level prompt
#   Pass 2 : HPO concepts with several         -> group-level prompt
#
# Every pair the LLM accepts is added as a new row with attribute="llm" and
# validity=10. Existing rows are not changed, so an auto-accepted pair the
# LLM rejects stays accepted, only without "llm" row; both the LLM-accepted
# set and all validity=10 rows are evaluated.
#
# Output: llm_filtered.csv (filtered.csv plus the "llm" rows) and
# llm_review.csv (every reviewed pair with decision, confidence, rationale).
# ==========================================================================

# LLM confidence is 1-5; divided by this to normalise to [0, 1].
CONFIDENCE_MAX = 5

# Used when the LLM answer is missing or unparsable.
FALLBACK_RESULT = {"decision": "REJECT", "confidence": 1}

# Prompt file indices in config.prompt_files.
PAIR_PROMPT_INDEX  = 0   # One HPO concept vs. one SNOMED CT concept.
GROUP_PROMPT_INDEX = 1   # One HPO concept vs. multiple SNOMED CT candidates.

# Children shown per concept in the prompt. SNOMED CT concepts can have
# 1,500+ children, exceeding the context window; children matter least for
# equivalence decisions.
MAX_CHILDREN = 10


def quote(txt: str, quotation_char: str = '"') -> str:
    """txt in quotation marks for the prompt; "None" for a missing value."""
    if txt is None:
        return "None"
    return quotation_char + str(txt) + quotation_char

def quote_list(txt_list: list, quotation_char: str = '"') -> list:
    """Every value quoted; ["None"] for an empty list, so the field is never blank."""
    if not txt_list:
        return ["None"]
    return [quote(txt, quotation_char) for txt in txt_list]

def build_ontology_lookups(df, ont):
    """
    Per-attribute lookups of an ontology: attribute class (labelClass,
    definitionClass, ..., plus "parents") -> {concept id: [values]}.
    Synonyms are exact synonyms only.
    """
    ont_config  = ont.config
    labels      = buildAttributeLookup(df, ont, labelClass)
    definitions = buildAttributeLookup(df, ont, definitionClass)
    comments    = buildAttributeLookup(df, ont, commentClass)

    syn_df = df[df[ont_config.attribute_column] == synonymClass]
    synonyms = {}
    if ont_config.additional_column in syn_df.columns:
        syn_mask = syn_df[ont_config.additional_column].apply(is_exact_synonym)
        synonyms = syn_df[syn_mask].groupby(ont_config.id_column)[ont_config.value_column].apply(list).to_dict()

    # childrenClass rows are (id=parent, value=child).
    parents   = df[df[ont_config.attribute_column] == childrenClass].groupby(ont_config.value_column)[ont_config.id_column].apply(list).to_dict()
    children  = buildAttributeLookup(df, ont, childrenClass)

    return {
        labelClass:      labels,
        definitionClass: definitions,
        commentClass:    comments,
        synonymClass:    synonyms,
        "parents":       parents,
        childrenClass:   children,
    }

def get_labels_for_ids(id_list, label_dict):
    """Resolve a list of concept IDs to their preferred labels."""
    result = []
    for _id in id_list:
        result.extend(label_dict.get(_id, []))
    return result

def join_quoted(values: list) -> str:
    """Quoted values as one comma-separated prompt field."""
    return ", ".join(quote_list(values))

def children_text(concept_id: str, data: dict) -> str:
    """
    Labels of at most MAX_CHILDREN children for the prompt, with a note
    when truncated so the LLM does not take the list as complete.
    """
    child_ids = data[childrenClass].get(concept_id, [])
    text = join_quoted(get_labels_for_ids(child_ids[:MAX_CHILDREN], data[labelClass]))
    if len(child_ids) > MAX_CHILDREN:
        text += f" ({MAX_CHILDREN} of {len(child_ids)} shown)"
    return text

def share(count: int, total: int) -> str:
    """Percentage for log lines; 0 when there is nothing to divide by."""
    return f"{count / total if total else 0:.1%}"


# ------------------------------------------------------------------
# Prompt fields. Shared with MappingGoldCheck, which sends the gold
# standard pairs through the same two prompts.
# ------------------------------------------------------------------
def hpo_prompt_fields(hpo_id: str, hpo_data: dict) -> dict:
    """Prompt fields describing the HPO concept, shared by both prompts."""
    return {
        "hpo_label":      join_quoted(hpo_data[labelClass].get(hpo_id, [])),
        "hpo_synonyms":   join_quoted(hpo_data[synonymClass].get(hpo_id, [])),
        "hpo_definition": join_quoted(hpo_data[definitionClass].get(hpo_id, [])),
        "hpo_comment":    join_quoted(hpo_data[commentClass].get(hpo_id, [])),
        "hpo_parents":    join_quoted(get_labels_for_ids(hpo_data["parents"].get(hpo_id, []), hpo_data[labelClass])),
        "hpo_children":   children_text(hpo_id, hpo_data),
    }

def pair_prompt_fields(hpo_id: str, sct_id: str, reason: str, hpo_data: dict, sct_data: dict) -> dict:
    """All fields of the pair-level prompt (single_mapping.md)."""
    return {
        **hpo_prompt_fields(hpo_id, hpo_data),
        "sct_term":       join_quoted(sct_data[labelClass].get(sct_id, [])),
        "sct_synonyms":   join_quoted(sct_data[synonymClass].get(sct_id, [])),
        "sct_parents":    join_quoted(get_labels_for_ids(sct_data["parents"].get(sct_id, []), sct_data[labelClass])),
        "sct_children":   children_text(sct_id, sct_data),
        "review_reason":  reason,
    }

def related_candidates_text(sct_id: str, group_sct_ids: list, sct_data: dict, sct_descendants) -> str:
    """
    The other candidates of the group that sct_id is broader or narrower
    than in the SCT hierarchy, or "None".
    """
    parts = []
    for other in group_sct_ids:
        if other == sct_id:
            continue
        # First label only, to keep the line short.
        label = join_quoted(sct_data[labelClass].get(other, [])[:1])
        if other in sct_descendants(sct_id):
            parts.append(f"broader than {other} ({label})")
        elif sct_id in sct_descendants(other):
            parts.append(f"narrower than {other} ({label})")
    return "; ".join(parts) if parts else "None"

def group_prompt_fields(hpo_id: str, candidates: list, hpo_data: dict, sct_data: dict,
                        sct_descendants) -> dict:
    """
    All fields of the group-level prompt (multiple_mapping.md).
    candidates: (sct_id, review reason) per SNOMED CT candidate.
    """
    group_sct_ids = [sct_id for sct_id, _ in candidates]
    sct_blocks = []
    for sct_id, reason in candidates:
        # The ID lets the answers be matched to rows by ID, not position.
        sct_blocks.append(
            f"* SNOMED CT ID: {sct_id}\n"
            f"* Preferred Term / FSN: {join_quoted(sct_data[labelClass].get(sct_id, []))}\n"
            f"* Synonyms: {join_quoted(sct_data[synonymClass].get(sct_id, []))}\n"
            f"* Parents: {join_quoted(get_labels_for_ids(sct_data['parents'].get(sct_id, []), sct_data[labelClass]))}\n"
            f"* Children: {children_text(sct_id, sct_data)}\n"
            f"* Related Candidates: {related_candidates_text(sct_id, group_sct_ids, sct_data, sct_descendants)}\n"
            f"* Review Reason: {reason}"
        )
    return {
        **hpo_prompt_fields(hpo_id, hpo_data),
        "sct_concepts":   "\n\n---\n\n".join(sct_blocks),
    }

def start_pass(m: Model, prompt_index: int, n_prompts: int, prefix: str = "prompt") -> None:
    """
    Give each pass its own answer store "<prefix>_<model>_<prompt_index>.json"
    in data/tmp/ and add the pass's prompt template to n_prompts histories,
    to be formatted by the caller. generate() reuses the stored answers of
    identical prompts (keyed by their SHA-256), so an interrupted run resumes;
    a store written with other model settings is refused.
    """
    l = Logger()
    m.config.prompt_tmp_file = f"{prefix}.json"
    m.setPostfix(str(prompt_index))
    if m.hasCheckpoint():
        l.log("Answer store found; its answers are reused, the others are generated.")
    else:
        l.log("No answer store found; starting a new review.")
    m.addPromptFromFile(prompt_index, n_prompts)


if __name__ == "__main__":

    l = Logger()
    l.printHeader("LLM review of candidate mappings")

    l.printHeader("Loading data")

    hpo, sct, umls, hpo_ontology, sct_ontology, umls_ontology, graham = loadInputs(";")
    del umls, umls_ontology  # not used in this script; free the memory

    # In the mappings the HPO ID is in id_column, the SCT ID in value_column.
    id_col   = hpo_ontology.config.id_column
    val_col  = hpo_ontology.config.value_column
    attr_col = hpo_ontology.config.attribute_column

    l.log("Loading filtered mappings...")
    mappings = pd.read_csv(filtered_path, sep=";")
    l.log(f"Filtered mappings: {len(mappings.index)} rows, "
          f"{len(mappings.drop_duplicates(subset=[id_col, val_col]).index)} pairs.")
    l.log("Loading filtered mappings completed.")

    l.log("Building ontology lookup tables...")
    hpo_data = build_ontology_lookups(hpo, hpo_ontology)
    sct_data = build_ontology_lookups(sct, sct_ontology)
    sct_descendants = make_descendant_lookup(build_child_map(sct, sct_ontology))
    l.log("Building ontology lookup tables completed.")

    # All rows of a pair have the same validity code, so deduplicating keeps
    # one row per pair; the stable sort keeps the prompt indices
    # reproducible. Routing is by the number of SCT candidates per HPO
    # concept only, so pass 1 covers MappingSetBuilder's 1:1 and n:1 pairs,
    # pass 2 its 1:n and m:n pairs.
    l.printHeader("Selecting pairs for review")

    pair_candidates = (
        mappings[mappings[validity_column].isin(validity_reviewable)]
        .sort_values(validity_column, ascending=False, kind="stable")
        .drop_duplicates(subset=[id_col, val_col])
        .reset_index(drop=True)
    )

    n_auto_accepted = int((pair_candidates[validity_column] == validity_accepted).sum())

    l.log(f"Pairs to review (validity=1–{validity_accepted}): {len(pair_candidates)}.")
    l.log(f"Pairs to review: {len(pair_candidates) - n_auto_accepted} candidates with "
          f"validity=1–{validity_accepted - 1}.")
    l.log(f"Pairs to review: {n_auto_accepted} auto-accepted with validity={validity_accepted}.")

    sct_per_hpo = pair_candidates.groupby(id_col)[val_col].nunique()
    pair_candidates["n_sct"] = pair_candidates[id_col].map(sct_per_hpo)

    one_to_one = pair_candidates[pair_candidates["n_sct"] == 1].reset_index(drop=True)
    one_to_n   = pair_candidates[pair_candidates["n_sct"] >  1].reset_index(drop=True)

    groups_1n = {}
    for row in one_to_n.itertuples(index=True):
        groups_1n.setdefault(getattr(row, id_col), []).append(row)
    group_hpo_ids = list(groups_1n.keys())

    l.log(f"HPO concepts with one candidate (pass 1, pair-level prompt): "
          f"{len(one_to_one)} concepts, {len(one_to_one)} pairs.")
    l.log(f"HPO concepts with several candidates (pass 2, group-level prompt): "
          f"{len(group_hpo_ids)} concepts, {len(one_to_n)} pairs.")

    evaluation(
        mappings[mappings[validity_column] == validity_accepted],
        graham,
        f"Evaluation (before LLM review, auto-accepted, validity={validity_accepted}):",
        hpo_ontology,
    )

    l.printHeader("Pass 1: Pair-level review")

    l.log(f"Pass 1: Reviewing {len(one_to_one)} pairs with the pair-level prompt...")

    pair_results     = []
    pass1_unparsable = 0

    with Model(index=0) as m:
        start_pass(m, PAIR_PROMPT_INDEX, len(one_to_one))

        l.log("Formatting pair-level prompts...")
        for row in one_to_one.itertuples(index=True):
            hpo_id = getattr(row, id_col)
            sct_id = getattr(row, val_col)
            reason = REVIEW_REASON.get(int(getattr(row, validity_column)), "UNKNOWN")

            m.formatPromptFromFile(
                row.Index, pair_prompt_fields(hpo_id, sct_id, reason, hpo_data, sct_data)
            )

        l.log("Formatting pair-level prompts completed.")

        m.generate()

        for i in range(len(one_to_one)):
            result = m.toJSON(i)
            if result is None:
                result = FALLBACK_RESULT.copy()
                pass1_unparsable += 1
            pair_results.append(result)

    pass1_out = one_to_one.copy()
    pass1_out["llm_decision"]    = [r.get("decision",   FALLBACK_RESULT["decision"])                    for r in pair_results]
    pass1_out[confidence_column] = [r.get("confidence", FALLBACK_RESULT["confidence"]) / CONFIDENCE_MAX for r in pair_results]
    pass1_out["llm_rationale"]   = [r.get("rationale", "")                                               for r in pair_results]
    pass1_out["llm_pass"]        = 1

    p1_accepted = int((pass1_out["llm_decision"] == "ACCEPT").sum())
    p1_rejected = int((pass1_out["llm_decision"] == "REJECT").sum())
    l.log(f"Pass 1 (pairs): {p1_accepted} accepted, {p1_rejected} rejected, "
          f"{len(pass1_out.index) - p1_accepted - p1_rejected} with another decision.")
    l.log(f"Pass 1: {pass1_unparsable} unparsable answers, counted as rejected.")
    l.log(f"Pass 1: Reviewing {len(one_to_one)} pairs with the pair-level prompt completed.")

    # Pass 2: one prompt per HPO concept with all its SNOMED CT candidates;
    # the answer has one decision per candidate, identified by SNOMED CT ID.
    l.printHeader("Pass 2: Group-level review")

    l.log(f"Pass 2: Reviewing {len(one_to_n)} pairs of {len(group_hpo_ids)} HPO "
          f"concepts with the group-level prompt...")

    group_results    = {}   # hpo_id -> list of per-sct decisions (same order as groups_1n[hpo_id])
    pass2_unparsable = 0    # groups whose answer could not be used
    pass2_missing    = 0    # candidates without a decision in a usable answer

    with Model(index=0) as m:
        start_pass(m, GROUP_PROMPT_INDEX, len(group_hpo_ids))

        l.log("Formatting group-level prompts...")
        for g_idx, hpo_id in enumerate(group_hpo_ids):

            candidates = [
                (getattr(row, val_col),
                 REVIEW_REASON.get(int(getattr(row, validity_column)), "UNKNOWN"))
                for row in groups_1n[hpo_id]
            ]

            m.formatPromptFromFile(g_idx, group_prompt_fields(
                hpo_id, candidates, hpo_data, sct_data, sct_descendants,
            ))

        l.log("Formatting group-level prompts completed.")

        m.generate()

        for g_idx, hpo_id in enumerate(group_hpo_ids):
            result = m.toJSON(g_idx)
            rows   = groups_1n[hpo_id]

            if result is None or "mappings" not in result:
                group_results[hpo_id] = [FALLBACK_RESULT.copy() for _ in rows]
                pass2_unparsable += 1
                continue

            # By SCT ID, in case the LLM reorders or skips candidates.
            decisions_by_sct_id = {item.get("sct_id"): item for item in result["mappings"]}

            per_group = []
            for row in rows:
                sct_id = getattr(row, val_col)
                if sct_id in decisions_by_sct_id:
                    per_group.append(decisions_by_sct_id[sct_id])
                else:
                    per_group.append(FALLBACK_RESULT.copy())
                    pass2_missing += 1

            group_results[hpo_id] = per_group

    group_rows_out = []
    for hpo_id, decisions in group_results.items():
        for row, decision in zip(groups_1n[hpo_id], decisions):
            group_rows_out.append({
                id_col:            getattr(row, id_col),
                val_col:           getattr(row, val_col),
                validity_column:   getattr(row, validity_column),
                "n_sct":           getattr(row, "n_sct"),
                "llm_decision":    decision.get("decision",   FALLBACK_RESULT["decision"]),
                confidence_column: decision.get("confidence", FALLBACK_RESULT["confidence"]) / CONFIDENCE_MAX,
                "llm_rationale":   decision.get("rationale", ""),
                "llm_pass":        2,
            })
    # Without group prompts an empty list would give a frame without columns.
    pass2_out = (
        pd.DataFrame(group_rows_out)
        if group_rows_out
        else pd.DataFrame(columns=pass1_out.columns)
    )

    p2_accepted = int((pass2_out["llm_decision"] == "ACCEPT").sum())
    p2_rejected = int((pass2_out["llm_decision"] == "REJECT").sum())
    l.log(f"Pass 2 (pairs): {p2_accepted} accepted, {p2_rejected} rejected, "
          f"{len(pass2_out.index) - p2_accepted - p2_rejected} with another decision.")
    l.log(f"Pass 2: {pass2_missing} candidates missing in the answers, counted as rejected.")
    l.log(f"Pass 2: The pairs of {pass2_unparsable} unparsable group answers are counted "
          f"as rejected.")
    l.log(f"Pass 2: Reviewing {len(one_to_n)} pairs of {len(group_hpo_ids)} HPO "
          f"concepts with the group-level prompt completed.")

    l.printHeader("Writing LLM-reviewed mappings")

    all_results = pd.concat([pass1_out, pass2_out], ignore_index=True)

    # pandas reads validity as float (filtered.csv has NA rows); the reviewed
    # pairs all have a code.
    all_results[validity_column] = all_results[validity_column].astype(int)

    # Every LLM-accepted pair (candidate or auto-accepted) gets a new "llm"
    # row with validity=10, so downstream it counts like an accepted pair.
    all_accepted_llm = all_results[all_results["llm_decision"] == "ACCEPT"].copy()

    # Counted before the validity of the new rows is set to 10.
    was_accepted         = all_results[validity_column] == validity_accepted
    n_new_accepted       = int((all_accepted_llm[validity_column] != validity_accepted).sum())
    n_confirmed_accepted = int((all_accepted_llm[validity_column] == validity_accepted).sum())
    n_rejected_accepted  = int((was_accepted & (all_results["llm_decision"] == "REJECT")).sum())

    all_accepted_llm[validity_column] = validity_accepted
    all_accepted_llm[attr_col]        = llm_mappings

    # Drop the LLM-specific columns (llm_decision, llm_rationale, ...).
    retained_cols = [col for col in mappings.columns if col in all_accepted_llm.columns]
    assert validity_column in retained_cols, (
        f"validity_column '{validity_column}' was unexpectedly dropped during "
        f"column filtering — check that filtered.csv was written with the "
        f"validity column present."
    )
    all_accepted_llm = all_accepted_llm[retained_cols]

    n_rows_before = len(mappings.index)
    mappings = pd.concat([mappings, all_accepted_llm], ignore_index=True)

    l.log(f"Rows to write: {len(mappings.index)} ({n_rows_before} from filtered.csv, "
          f"{len(all_accepted_llm.index)} new).")
    l.log(f"The new rows have attribute={llm_mappings} and validity={validity_accepted}.")

    writeHugeCSV(mappings, llm_filtered_path)

    # Audit trail; unparsable or missing answers have the fallback decision
    # and an empty rationale.
    review_cols = [id_col, val_col, validity_column, "n_sct", "llm_pass",
                   "llm_decision", confidence_column, "llm_rationale"]
    l.log(f"Rows to write to the review file: {len(all_results.index)} "
          f"(one per reviewed pair).")
    writeHugeCSV(all_results[review_cols], llm_review_path)

    l.printHeader("Summary")

    total_reviewed = len(all_results.index)
    total_accepted = int((all_results["llm_decision"] == "ACCEPT").sum())
    total_rejected = int((all_results["llm_decision"] == "REJECT").sum())
    total_other    = total_reviewed - total_accepted - total_rejected

    l.log("LLM review (pairs):")
    l.log(f"  {'Reviewed:':<34}{total_reviewed:7}")
    l.log(f"  {'Accepted:':<34}{total_accepted:7} ({share(total_accepted, total_reviewed)})")
    l.log(f"  {'Rejected:':<34}{total_rejected:7} ({share(total_rejected, total_reviewed)})")
    l.log(f"  {'Other decision:':<34}{total_other:7} ({share(total_other, total_reviewed)})")
    l.log(f"  {'Candidates accepted (new):':<34}{n_new_accepted:7} "
          f"(validity=1–{validity_accepted - 1} accepted by the LLM)")
    l.log(f"  {'Auto-accepted confirmed:':<34}{n_confirmed_accepted:7} "
          f"(validity={validity_accepted} accepted by the LLM)")
    l.log(f"  {'Auto-accepted rejected:':<34}{n_rejected_accepted:7} "
          f"(validity={validity_accepted} rejected by the LLM; kept, no llm row)")

    l.log("LLM review by validity code (pairs):")
    l.log(f"  {'Code':<6} {'Reason':<40} {'Reviewed':>8} {'Accepted':>9} {'Rejected':>9} {'Other':>6} {'Rejection Rate':>15}")
    for code in sorted(REVIEW_REASON.keys()):
        subset   = all_results[all_results[validity_column] == code]
        reviewed = len(subset)
        if reviewed == 0:
            continue
        accepted = int((subset["llm_decision"] == "ACCEPT").sum())
        rejected = int((subset["llm_decision"] == "REJECT").sum())
        other    = reviewed - accepted - rejected
        l.log(f"  {code:<6} {REVIEW_REASON[code]:<40} {reviewed:>8} {accepted:>9} "
              f"{rejected:>9} {other:>6} {share(rejected, reviewed):>15}")

    # The LLM's own verdict per pass and in total, then the final set
    # (all validity=10 rows, including auto-accepted pairs the LLM rejected).
    evaluation(
        pass1_out[pass1_out["llm_decision"] == "ACCEPT"],
        graham,
        "Evaluation (pass 1, LLM-accepted):",
        hpo_ontology,
    )
    evaluation(
        pass2_out[pass2_out["llm_decision"] == "ACCEPT"],
        graham,
        "Evaluation (pass 2, LLM-accepted):",
        hpo_ontology,
    )
    evaluation(
        all_accepted_llm,
        graham,
        "Evaluation (after LLM review, LLM-accepted only):",
        hpo_ontology,
    )
    evaluation(
        mappings[mappings[validity_column] == validity_accepted],
        graham,
        f"Evaluation (after LLM review, auto-accepted + LLM-accepted, "
        f"validity={validity_accepted}):",
        hpo_ontology,
    )

    l.printHeader("LLM review completed.")
