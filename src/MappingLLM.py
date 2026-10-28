import sys

# Prevent Python from generating .pyc files (compiled bytecode files)
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
    validity_accepted, validity_reviewable,
    buildAttributeLookup, is_exact_synonym,
)
import pandas       as pd

# ------------------------------------------------------------------
# Constants.
# ------------------------------------------------------------------

# Confidence scale is 1–5; used to normalise LLM output to [0, 1].
CONFIDENCE_MAX = 5

# Fallback result when the LLM returns nothing or an unparseable response.
FALLBACK_RESULT = {"decision": "REJECT", "confidence": 1}

# Prompt file indices in config.prompt_files.
PAIR_PROMPT_INDEX  = 0   # One HPO concept vs. one SNOMED CT concept.
GROUP_PROMPT_INDEX = 1   # One HPO concept vs. multiple SNOMED CT candidates.

# Maximum number of children to include per concept in the prompt.
# SNOMED CT concepts can have 1,500+ children, which causes prompts to
# exceed the model's context window. Children are the least diagnostically
# useful field for equivalence decisions compared to parents and definitions,
# so truncating them is the safest way to reduce prompt length.
MAX_CHILDREN = 10


# ------------------------------------------------------------------
# Initialise logger.
# ------------------------------------------------------------------
l = Logger()
l.printHeader("LLM review of candidate mappings")

# ------------------------------------------------------------------
# Load ontologies and mappings.
# ------------------------------------------------------------------
# umls is loaded by loadOntologies but not used in this script.
# If loadOntologies supports a parameter to skip UMLS loading, passing
# it here would save ~30 seconds. Until then it is loaded and discarded.
hpo, sct, umls, hpo_ontology, sct_ontology, umls_ontology, graham = loadInputs(";")
del umls, umls_ontology  # not used in this script; free the memory


l.log("Loading filtered mappings...")
mappings = pd.read_csv(filtered_path, sep=";")
l.log("Loading filtered mappings completed.")

# ------------------------------------------------------------------
# Helper utilities.
# ------------------------------------------------------------------
def quote(txt: str, quotation_char: str = '"') -> str:
    if txt is None:
        return "None"
    return quotation_char + str(txt) + quotation_char

def quote_list(txt_list: list, quotation_char: str = '"') -> list:
    if not txt_list:
        return ["None"]
    return [quote(txt, quotation_char) for txt in txt_list]

def build_ontology_lookups(df, ont):
    """
    Build per-attribute lookup dicts for a loaded ontology dataframe.
    Returns a dict keyed by attribute class (labelClass, definitionClass,
    etc.) where each value is a concept-id -> [values] dict.
    Parents and children are also included under "parents" and
    childrenClass respectively, derived from the children relation rows.
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

    # childrenClass rows encode parent->child edges as (id=parent, value=child).
    # "parents" maps each concept to its parent concept IDs.
    # childrenClass maps each concept to its child concept IDs.
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
    return ", ".join(quote_list(values))

# ------------------------------------------------------------------
# Build lookup tables.
# ------------------------------------------------------------------
l.log("Building ontology lookup tables...")
hpo_data = build_ontology_lookups(hpo, hpo_ontology)
sct_data = build_ontology_lookups(sct, sct_ontology)
l.log("Building ontology lookup tables completed.")

# ------------------------------------------------------------------
# Select candidate pairs for LLM review.
#
# Validity codes 1–9 are hierarchy reversals and fallback candidates
# from the MappingSetBuilder pipeline. Code 10 (already accepted) is
# also included so the LLM acts as a judge over the full set for each
# HPO concept — auto-accepted pairs are presented with review reason
# ACCEPTED and re-evaluated independently. Only code 0 (domain
# mismatch) is excluded.
#
# Deduplicate to one row per (hpo_id, sct_id) pair. For pairs that
# appear under multiple validity codes, the highest validity value is
# kept as representative. Validity=10 (ACCEPTED) is the most
# informative label for a pair the pipeline accepted — if the same
# pair also has a failure code from an earlier step, the LLM should
# see it as ACCEPTED, not as a hierarchy reversal. Sorting descending
# then deduplicating keeps the highest code per pair.
#
# All constellations (1:1, 1:n, n:1, m:n) are handled uniformly:
# for each HPO concept all its SNOMED CT candidates are evaluated
# individually, and the prompt explicitly permits set mappings.
# Including accepted pairs means most HPO concepts now have multiple
# candidates, so the routing naturally favours the group prompt.
# ------------------------------------------------------------------
pair_candidates = (
    mappings[mappings[validity_column].isin(validity_reviewable)]
    .sort_values(validity_column, ascending=False)
    .drop_duplicates(subset=[hpo_ontology.config.id_column, hpo_ontology.config.value_column])
    .reset_index(drop=True)
)

review_pairs      = pair_candidates[pair_candidates[validity_column] != 10]
auto_accepted_df  = pair_candidates[pair_candidates[validity_column] == validity_accepted]

l.log(f"Total unique pairs for LLM review: {len(pair_candidates)} "
      f"({len(review_pairs)} under review, {len(auto_accepted_df)} already accepted).")

# ------------------------------------------------------------------
# Classify each HPO concept by how many unique SNOMED CT candidates
# it has in the review set.
#
#   1:1  — one HPO concept, one SCT candidate  -> pair-level prompt
#   1:n  — one HPO concept, many SCT candidates -> group-level prompt
#
# n:1 and m:n constellations are intentionally not routed to a
# separate group prompt: from the HPO perspective they reduce to
# 1:1 or 1:n once deduplicated, so the same two prompts cover them.
# ------------------------------------------------------------------
# Note: in the mappings EAV frame, the HPO concept ID is in id_column
# and the SCT concept ID is stored in value_column (the "value" of the
# cross-reference relation). This is why hpo_ontology.config.value_column
# is used as the SCT ID column throughout this script.
sct_per_hpo = pair_candidates.groupby(hpo_ontology.config.id_column)[hpo_ontology.config.value_column].nunique()
pair_candidates["n_sct"] = pair_candidates[hpo_ontology.config.id_column].map(sct_per_hpo)

one_to_one = pair_candidates[pair_candidates["n_sct"] == 1].reset_index(drop=True)
one_to_n   = pair_candidates[pair_candidates["n_sct"] >  1].reset_index(drop=True)

# Groups for the 1:n pass: hpo_id -> list of rows.
groups_1n = {}
for row in one_to_n.itertuples(index=True):
    hpo_id = getattr(row, hpo_ontology.config.id_column)
    groups_1n.setdefault(hpo_id, []).append(row)
group_hpo_ids = list(groups_1n.keys())

l.log(f"1:1 pairs  (pair-level prompt): {len(one_to_one)}.")
l.log(f"1:n groups (group-level prompt): {len(group_hpo_ids)} HPO concepts, "
      f"{len(one_to_n)} pairs total.")

# ------------------------------------------------------------------
# PASS 1: Pair-level prompt (1:1).
# One history per (hpo_id, sct_id) pair.
# ------------------------------------------------------------------
l.log("Pass 1: Pair-level LLM review (1:1)...")

pair_results = []

with Model(index=0) as m:
    # Each pass needs its own checkpoint file so that Pass 2 does not
    # import Pass 1's histories on a resumed run.
    model_name = str(m.config.model_id[0].split("/")[-1])
    m.config.prompt_tmp_file = f"prompt_{PAIR_PROMPT_INDEX}_{model_name}.tmp"

    if m.importHistoriesFromFile() <= 0:

        m.addPromptFromFile(PAIR_PROMPT_INDEX, len(one_to_one))

        l.log("Formatting pair-level prompts...")
        for row in one_to_one.itertuples(index=True):
            idx    = row.Index
            hpo_id = getattr(row, hpo_ontology.config.id_column)
            sct_id = getattr(row, hpo_ontology.config.value_column)
            reason = REVIEW_REASON.get(int(getattr(row, validity_column)), "UNKNOWN")

            obj = {
                "hpo_label":      join_quoted(hpo_data[labelClass].get(hpo_id, [])),
                "hpo_synonyms":   join_quoted(hpo_data[synonymClass].get(hpo_id, [])),
                "hpo_definition": join_quoted(hpo_data[definitionClass].get(hpo_id, [])),
                "hpo_comment":    join_quoted(hpo_data[commentClass].get(hpo_id, [])),
                "hpo_parents":    join_quoted(get_labels_for_ids(hpo_data["parents"].get(hpo_id, []),     hpo_data[labelClass])),
                "hpo_children":   join_quoted(get_labels_for_ids(hpo_data[childrenClass].get(hpo_id, []), hpo_data[labelClass])),
                "sct_term":       join_quoted(sct_data[labelClass].get(sct_id, [])),
                "sct_synonyms":   join_quoted(sct_data[synonymClass].get(sct_id, [])),
                "sct_parents":    join_quoted(get_labels_for_ids(sct_data["parents"].get(sct_id, []), sct_data[labelClass])),
                "sct_children":   join_quoted(get_labels_for_ids(sct_data[childrenClass].get(sct_id, [])[:MAX_CHILDREN], sct_data[labelClass])),
                "review_reason":  reason,
            }

            m.formatPromptFromFile(idx, obj)

        l.log("Formatting pair-level prompts completed.")

    m.generate()

    for i in range(len(one_to_one)):
        result = m.toJSON(i)
        if result is None:
            result = FALLBACK_RESULT.copy()
        pair_results.append(result)

l.log("Pass 1: Pair-level LLM review (1:1) completed.")

# ------------------------------------------------------------------
# PASS 2: Group-level prompt (1:n).
# One history per HPO concept; the prompt receives the full list of
# SNOMED CT candidates for that HPO concept and returns one decision
# per candidate, identified by index.
# ------------------------------------------------------------------
l.log("Pass 2: Group-level LLM review (1:n)...")

group_results = {}   # hpo_id -> list of per-sct decisions (same order as groups_1n[hpo_id])

with Model(index=0) as m:
    # Separate checkpoint from Pass 1.
    model_name = str(m.config.model_id[0].split("/")[-1])
    m.config.prompt_tmp_file = f"prompt_{GROUP_PROMPT_INDEX}_{model_name}.tmp"

    if m.importHistoriesFromFile() <= 0:

        m.addPromptFromFile(GROUP_PROMPT_INDEX, len(group_hpo_ids))

        l.log("Formatting group-level prompts...")
        for g_idx, hpo_id in enumerate(group_hpo_ids):
            rows = groups_1n[hpo_id]

            # Build the SNOMED CT candidates block for this HPO concept.
            sct_blocks = []
            for local_idx, row in enumerate(rows):
                sct_id = getattr(row, hpo_ontology.config.value_column)
                reason = REVIEW_REASON.get(int(getattr(row, validity_column)), "UNKNOWN")

                # The SNOMED CT ID is included explicitly so the LLM can
                # populate the sct_id field in its output schema, enabling
                # the code to match decisions back to rows by ID rather
                # than by position.
                block  = (
                    f"* SNOMED CT ID: {sct_id}\n"
                    f"* Preferred Term / FSN: {join_quoted(sct_data[labelClass].get(sct_id, []))}\n"
                    f"* Synonyms: {join_quoted(sct_data[synonymClass].get(sct_id, []))}\n"
                    f"* Parents: {join_quoted(get_labels_for_ids(sct_data['parents'].get(sct_id, []), sct_data[labelClass]))}\n"
                    f"* Children: {join_quoted(get_labels_for_ids(sct_data[childrenClass].get(sct_id, [])[:MAX_CHILDREN], sct_data[labelClass]))}\n"
                    f"* Review Reason: {reason}"
                )

                sct_blocks.append(block)

            obj = {
                "hpo_label":      join_quoted(hpo_data[labelClass].get(hpo_id, [])),
                "hpo_synonyms":   join_quoted(hpo_data[synonymClass].get(hpo_id, [])),
                "hpo_definition": join_quoted(hpo_data[definitionClass].get(hpo_id, [])),
                "hpo_comment":    join_quoted(hpo_data[commentClass].get(hpo_id, [])),
                "hpo_parents":    join_quoted(get_labels_for_ids(hpo_data["parents"].get(hpo_id, []),     hpo_data[labelClass])),
                "hpo_children":   join_quoted(get_labels_for_ids(hpo_data[childrenClass].get(hpo_id, []), hpo_data[labelClass])),
                "sct_concepts":   "\n\n---\n\n".join(sct_blocks),
            }

            m.formatPromptFromFile(g_idx, obj)

        l.log("Formatting group-level prompts completed.")

    m.generate()

    for g_idx, hpo_id in enumerate(group_hpo_ids):
        result = m.toJSON(g_idx)
        rows   = groups_1n[hpo_id]

        if result is None or "mappings" not in result:
            # Fallback: reject all candidates in this group.
            group_results[hpo_id] = [FALLBACK_RESULT.copy() for _ in rows]
            continue

        # Index the returned decisions by SCT ID so that if the LLM
        # reorders its output or skips a candidate, each decision is
        # still matched to the correct row rather than silently
        # misaligned.
        decisions_by_sct_id = {item.get("sct_id"): item for item in result["mappings"]}

        per_group = []
        for local_idx, row in enumerate(rows):
            sct_id   = getattr(row, hpo_ontology.config.value_column)
            decision = decisions_by_sct_id.get(sct_id, FALLBACK_RESULT.copy())
            per_group.append(decision)

        group_results[hpo_id] = per_group

l.log("Pass 2: Group-level LLM review (1:n) completed.")

# ------------------------------------------------------------------
# Assemble results from both passes into a unified dataframe.
# ------------------------------------------------------------------
l.log("Assembling results...")

# Pass 1 results (1:1).
pass1_out = one_to_one.copy()
pass1_out["llm_decision"]    = [r.get("decision",   FALLBACK_RESULT["decision"])                    for r in pair_results]
pass1_out[confidence_column] = [r.get("confidence", FALLBACK_RESULT["confidence"]) / CONFIDENCE_MAX for r in pair_results]

# Pass 2 results (1:n).
# Note: hpo_ontology.config.value_column holds the SCT ID in the EAV frame.
group_rows_out = []
for hpo_id, decisions in group_results.items():
    for row, decision in zip(groups_1n[hpo_id], decisions):
        group_rows_out.append({
            hpo_ontology.config.id_column:    getattr(row, hpo_ontology.config.id_column),
            hpo_ontology.config.value_column: getattr(row, hpo_ontology.config.value_column),
            validity_column:                  getattr(row, validity_column),
            "n_sct":                          getattr(row, "n_sct"),
            "llm_decision":                   decision.get("decision",   FALLBACK_RESULT["decision"]),
            confidence_column:                decision.get("confidence", FALLBACK_RESULT["confidence"]) / CONFIDENCE_MAX,
        })
# Guard against the edge case where all HPO concepts had only one
# candidate and no group prompts were run — an empty list would
# produce a dataframe with no columns, breaking the concat.
pass2_out = (
    pd.DataFrame(group_rows_out)
    if group_rows_out
    else pd.DataFrame(columns=pass1_out.columns)
)

all_results = pd.concat([pass1_out, pass2_out], ignore_index=True)

# Ensure validity_column is an integer type after concat — pandas may
# promote it to float64 if any NaN values were introduced during the
# group result assembly, which would silently break equality comparisons.
all_results[validity_column] = all_results[validity_column].astype(int)

# ------------------------------------------------------------------
# Add LLM-accepted mappings as new rows (validity=10).
#
# Two groups are handled separately:
#
#   1. Previously under-review pairs (validity=1–9) that the LLM
#      accepted: added as new rows with attribute="llm", validity=10.
#
#   2. Previously auto-accepted pairs (validity=10) that the LLM
#      accepted: confirmed by the LLM, added as new rows with
#      attribute="llm", validity=10, alongside the existing row.
#      Auto-accepted pairs that the LLM rejected are simply not
#      added — their original validity=10 row remains in the output
#      but no llm row is written, making the rejection visible by
#      absence.
#
# In both cases validity=10 is used to ensure LLM-accepted pairs are
# treated identically to auto-accepted pairs by downstream consumers.
# ------------------------------------------------------------------
all_accepted_llm = all_results[all_results["llm_decision"] == "ACCEPT"].copy()

# Compute logging counts before overwriting validity to 10 — these
# counts depend on the original validity values and must be captured
# here. Using lengths rather than retaining sub-dataframes avoids any
# fragility around the subsequent in-place assignment below.
n_new_accepted       = len(all_accepted_llm[all_accepted_llm[validity_column] != 10])
n_confirmed_accepted = len(all_accepted_llm[all_accepted_llm[validity_column] == validity_accepted])
n_rejected_accepted  = len(all_results[
    (all_results[validity_column] == validity_accepted) &
    (all_results["llm_decision"] == "REJECT")
])

# Mark all LLM-accepted rows as accepted and record the source.
all_accepted_llm[validity_column]                      = 10
all_accepted_llm[hpo_ontology.config.attribute_column] = llm_mappings

# Keep only the columns present in the original mappings dataframe,
# dropping LLM-specific columns (llm_decision, n_sct).
retained_cols = [col for col in mappings.columns if col in all_accepted_llm.columns]
assert validity_column in retained_cols, (
    f"validity_column '{validity_column}' was unexpectedly dropped during "
    f"column filtering — check that filtered.csv was written with the "
    f"validity column present."
)
all_accepted_llm = all_accepted_llm[retained_cols]

# Append the new rows to the existing mappings.
mappings = pd.concat([mappings, all_accepted_llm], ignore_index=True)

writeHugeCSV(mappings, llm_filtered_path)

# ------------------------------------------------------------------
# LLM review statistics.
# Log overall acceptance/rejection counts, the overall rejection rate,
# and a per-validity-code breakdown so it is easy to see which pipeline
# steps benefited most from LLM review and which produced mostly noise.
# ------------------------------------------------------------------
total_reviewed  = len(all_results)
total_accepted  = len(all_results[all_results["llm_decision"] == "ACCEPT"])
total_rejected  = len(all_results[all_results["llm_decision"] == "REJECT"])
rejection_rate  = total_rejected / total_reviewed if total_reviewed > 0 else 0.0

l.log(f"LLM review summary:")
l.log(f"  Total reviewed:        {total_reviewed}")
l.log(f"  Accepted:              {total_accepted} ({total_accepted / total_reviewed:.1%})")
l.log(f"  Rejected:              {total_rejected} ({rejection_rate:.1%})")
l.log(f"  New mappings added:    {n_new_accepted} (validity=1–9 pairs accepted by LLM)")
l.log(f"  Confirmed accepted:    {n_confirmed_accepted} (validity=10 pairs confirmed by LLM)")
l.log(f"  Rejected accepted:     {n_rejected_accepted} (validity=10 pairs rejected by LLM — no llm row written)")

l.log("LLM review breakdown by validity code:")
l.log(f"  {'Code':<6} {'Reason':<40} {'Reviewed':>8} {'Accepted':>9} {'Rejected':>9} {'Rejection Rate':>15}")
for code in sorted(REVIEW_REASON.keys()):
    subset   = all_results[all_results[validity_column] == code]
    reviewed = len(subset)
    if reviewed == 0:
        continue
    accepted = len(subset[subset["llm_decision"] == "ACCEPT"])
    rejected = reviewed - accepted
    rate     = rejected / reviewed
    l.log(f"  {code:<6} {REVIEW_REASON[code]:<40} {reviewed:>8} {accepted:>9} {rejected:>9} {rate:>14.1%}")

l.log(f"LLM accepted: {len(all_accepted_llm)} new rows added with validity=10.")
l.log(f"Total mappings written: {len(mappings)}.")

# ------------------------------------------------------------------
# Evaluate the final accepted mapping set (validity=10) against the
# Graham gold standard, including both auto-accepted and LLM-accepted
# pairs.
# ------------------------------------------------------------------
l.log("Evaluating final accepted mapping set...")
evaluation(
    mappings[mappings[validity_column] == validity_accepted],
    graham,
    "Final accepted mapping set (auto + LLM):",
    hpo_ontology,
)

l.printHeader("LLM review completed.")