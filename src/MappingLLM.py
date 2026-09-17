import sys

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

from logger         import Logger
from model          import Model
from adapter        import (
    definitionClass, synonymClass, commentClass, labelClass,
    childrenClass, exactSynonymClass, semanticClass, writeHugeCSV,
)
from MappingUtils   import (
    loadOntologies, loadGold, validity_column, confidence_column,
    hpo_id_column, sct_id_column,
)
import pandas       as pd
import os

# ------------------------------------------------------------------
# Validity -> review reason mapping.
# ------------------------------------------------------------------
REVIEW_REASON = {
    1: "HIERARCHY_REVERSAL",
    2: "ANCHOR_HIERARCHY_REVERSAL",
    3: "LOW_SIMILARITY",
    4: "LAST_RESORT",
}

# Prompt file indices in config.prompt_files.
PAIR_PROMPT_INDEX  = 0
GROUP_PROMPT_INDEX = 1

l = Logger()

l.printHeader("LLM review of candidate mappings")

# ------------------------------------------------------------------
# Load ontologies and mappings.
# ------------------------------------------------------------------
hpo, sct, umls, hpo_ontology, sct_ontology, umls_ontology = loadOntologies(";")
hpo = hpo[hpo[hpo_ontology.config.id_column].str.startswith("HP:")].reset_index(drop=True)

mask        = sct[sct_ontology.config.attribute_column] == childrenClass
val_col_sct = sct_ontology.config.value_column
sct_vals    = sct.loc[mask, val_col_sct].astype(str)
already_prefixed = sct_vals.str.startswith("SNOMEDCT_US:")
sct.loc[mask, val_col_sct] = "SNOMEDCT_US:" + sct_vals.where(already_prefixed, sct_vals)

graham = loadGold("./data/input/graham/mapping_graham.csv", ";")

l.log("Loading filtered mappings...")
mappings = pd.read_csv("./data/output/filtered/filtered.csv", sep=";")
l.log("Loading filtered mappings completed.")

id_col   = hpo_ontology.config.id_column
val_col  = hpo_ontology.config.value_column
attr_col = hpo_ontology.config.attribute_column

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

def build_ontology_lookups(df, ont_config):
    id_c   = ont_config.id_column
    attr_c = ont_config.attribute_column
    val_c  = ont_config.value_column
    add_c  = ont_config.additional_column

    labels      = df[df[attr_c] == labelClass].groupby(id_c)[val_c].apply(list).to_dict()
    definitions = df[df[attr_c] == definitionClass].groupby(id_c)[val_c].apply(list).to_dict()
    comments    = df[df[attr_c] == commentClass].groupby(id_c)[val_c].apply(list).to_dict()

    syn_df = df[df[attr_c] == synonymClass]
    if add_c in syn_df.columns:
        syn_mask = syn_df[add_c].apply(
            lambda x: isinstance(x, dict) and x.get(semanticClass) == exactSynonymClass
        )
        synonyms = syn_df[syn_mask].groupby(id_c)[val_c].apply(list).to_dict()
    else:
        synonyms = {}

    children = df[df[attr_c] == childrenClass].groupby(val_c)[id_c].apply(list).to_dict()
    parents  = df[df[attr_c] == childrenClass].groupby(id_c)[val_c].apply(list).to_dict()

    return {
        labelClass:      labels,
        definitionClass: definitions,
        commentClass:    comments,
        synonymClass:    synonyms,
        "parents":       parents,
        childrenClass:   children,
    }

def get_labels_for_ids(id_list, label_dict):
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
hpo_data = build_ontology_lookups(hpo, hpo_ontology.config)
sct_data = build_ontology_lookups(sct, sct_ontology.config)
l.log("Building ontology lookup tables completed.")

# ------------------------------------------------------------------
# Select candidate pairs for LLM review (validity 1-4).
# Deduplicate to one row per (hpo_id, sct_id) pair, keeping the
# highest validity value (most informative origin) as representative.
# ------------------------------------------------------------------
to_check = (
    mappings[mappings[validity_column].isin([1, 2, 3, 4])]
    .sort_values(validity_column, ascending=False)
    .drop_duplicates(subset=[id_col, val_col])
    .reset_index(drop=True)
)

l.log(f"Total unique pairs for LLM review: {len(to_check)}")

# ------------------------------------------------------------------
# Classify pairs into constellations based on how many SCT targets
# each HPO concept has and vice versa (within the LLM review set).
# ------------------------------------------------------------------
sct_per_hpo  = to_check.groupby(id_col)[val_col].nunique()
hpo_per_sct  = to_check.groupby(val_col)[id_col].nunique()
to_check["n_sct"] = to_check[id_col].map(sct_per_hpo)
to_check["n_hpo"] = to_check[val_col].map(hpo_per_sct)

def classify_constellation(row):
    if row["n_sct"] == 1 and row["n_hpo"] == 1:
        return "1:1"
    elif row["n_sct"] > 1 and row["n_hpo"] == 1:
        return "1:n"
    elif row["n_sct"] == 1 and row["n_hpo"] > 1:
        return "n:1"
    else:
        return "m:n"

to_check["constellation"] = to_check.apply(classify_constellation, axis=1)

pair_candidates  = to_check[to_check["constellation"].isin(["1:1", "1:n"])].reset_index(drop=True)
group_candidates = to_check[to_check["constellation"].isin(["n:1", "m:n"])].reset_index(drop=True)

l.log(f"Pair-level  (1:1, 1:n): {len(pair_candidates)} pairs")
l.log(f"Group-level (n:1, m:n): {len(group_candidates)} pairs across "
      f"{group_candidates[val_col].nunique()} SCT groups")

# ------------------------------------------------------------------
# PASS 1: Pair-level prompt (1:1 and 1:n).
# One history per (hpo_id, sct_id) pair.
# ------------------------------------------------------------------
l.printHeader("Pass 1: Pair-level LLM review")

pair_results = []

with Model(index=0) as m:
    # Pass 1 and Pass 2 each need their own checkpoint file -- both use
    # Model(index=0) with the same config, so without this override they'd
    # share the default prompt_tmp_file and Pass 2 would import Pass 1's
    # leftover pair-level histories instead of building its own.
    m.config.prompt_tmp_file = "pair_review_raw.tmp"

    if m.importHistoriesFromFile() <= 0:

        m.addPromptFromFile(PAIR_PROMPT_INDEX, len(pair_candidates))

        l.log("Formatting pair-level prompts...")
        for row in pair_candidates.itertuples(index=True):
            idx     = row.Index
            hpo_id  = getattr(row, id_col)
            sct_id  = getattr(row, val_col)
            reason  = REVIEW_REASON.get(int(getattr(row, validity_column)), "UNKNOWN")

            obj = {
                "hpo_label":      join_quoted(hpo_data[labelClass].get(hpo_id, [])),
                "hpo_synonyms":   join_quoted(hpo_data[synonymClass].get(hpo_id, [])),
                "hpo_definition": join_quoted(hpo_data[definitionClass].get(hpo_id, [])),
                "hpo_comment":    join_quoted(hpo_data[commentClass].get(hpo_id, [])),
                "hpo_parents":    join_quoted(get_labels_for_ids(hpo_data["parents"].get(hpo_id, []), hpo_data[labelClass])),
                "hpo_children":   join_quoted(get_labels_for_ids(hpo_data[childrenClass].get(hpo_id, []), hpo_data[labelClass])),
                "sct_term":       join_quoted(sct_data[labelClass].get(sct_id, [])),
                "sct_synonyms":   join_quoted(sct_data[synonymClass].get(sct_id, [])),
                "sct_parents":    join_quoted(get_labels_for_ids(sct_data["parents"].get(sct_id, []), sct_data[labelClass])),
                "sct_children":   join_quoted(get_labels_for_ids(sct_data[childrenClass].get(sct_id, []), sct_data[labelClass])),
                "review_reason":  reason,
            }
            m.formatPromptFromFile(idx, obj)

        l.log("Formatting pair-level prompts completed.")

    imported_count = len(m.getMessageHistories())
    if imported_count != len(pair_candidates):
        raise RuntimeError(
            f"Pair-level checkpoint has {imported_count} histories but "
            f"{len(pair_candidates)} candidate pairs were computed this run "
            f"-- the checkpoint is stale (upstream mappings/filtering changed "
            f"since it was written). Delete "
            f"{os.path.join(m.config.prompt_tmp_folder, m.config.prompt_tmp_file)} "
            f"and rerun."
        )

    m.generate()

    for i in range(len(pair_candidates)):
        result = m.toJSON(i)
        if result is None:
            result = {"decision": "REJECT", "confidence": 0}
        pair_results.append(result)

# ------------------------------------------------------------------
# PASS 2: Group-level prompt (n:1 and m:n).
# One history per unique SCT concept (group).
# ------------------------------------------------------------------
l.printHeader("Pass 2: Group-level LLM review")

# Build groups: sct_id -> list of rows in group_candidates.
groups = {}
for row in group_candidates.itertuples(index=True):
    sct_id = getattr(row, val_col)
    groups.setdefault(sct_id, []).append(row)

group_sct_ids = list(groups.keys())
group_results = {}   # sct_id -> list of per-hpo decisions

with Model(index=0) as m:
    # See the matching comment in Pass 1: each pass needs its own checkpoint.
    m.config.prompt_tmp_file = "group_review_raw.tmp"

    if m.importHistoriesFromFile() <= 0:

        m.addPromptFromFile(GROUP_PROMPT_INDEX, len(group_sct_ids))

        l.log("Formatting group-level prompts...")
        for g_idx, sct_id in enumerate(group_sct_ids):
            rows = groups[sct_id]

            # Build the HPO concepts block for this group.
            hpo_blocks = []
            for local_idx, row in enumerate(rows):
                hpo_id = getattr(row, id_col)
                reason = REVIEW_REASON.get(int(getattr(row, validity_column)), "UNKNOWN")
                block  = (
                    f"* Index: {local_idx}\n"
                    f"* Label: {join_quoted(hpo_data[labelClass].get(hpo_id, []))}\n"
                    f"* Synonyms: {join_quoted(hpo_data[synonymClass].get(hpo_id, []))}\n"
                    f"* Definition: {join_quoted(hpo_data[definitionClass].get(hpo_id, []))}\n"
                    f"* Comment: {join_quoted(hpo_data[commentClass].get(hpo_id, []))}\n"
                    f"* Parents: {join_quoted(get_labels_for_ids(hpo_data['parents'].get(hpo_id, []), hpo_data[labelClass]))}\n"
                    f"* Children: {join_quoted(get_labels_for_ids(hpo_data[childrenClass].get(hpo_id, []), hpo_data[labelClass]))}\n"
                    f"* HPO ID: {hpo_id}\n"
                    f"* Review Reason: {reason}"
                )
                hpo_blocks.append(block)

            obj = {
                "sct_term":     join_quoted(sct_data[labelClass].get(sct_id, [])),
                "sct_synonyms": join_quoted(sct_data[synonymClass].get(sct_id, [])),
                "sct_parents":  join_quoted(get_labels_for_ids(sct_data["parents"].get(sct_id, []), sct_data[labelClass])),
                "sct_children": join_quoted(get_labels_for_ids(sct_data[childrenClass].get(sct_id, []), sct_data[labelClass])),
                "hpo_concepts": "\n\n---\n\n".join(hpo_blocks),
            }
            m.formatPromptFromFile(g_idx, obj)

        l.log("Formatting group-level prompts completed.")

    imported_count = len(m.getMessageHistories())
    if imported_count != len(group_sct_ids):
        raise RuntimeError(
            f"Group-level checkpoint has {imported_count} histories but "
            f"{len(group_sct_ids)} candidate groups were computed this run "
            f"-- the checkpoint is stale (upstream mappings/filtering changed "
            f"since it was written). Delete "
            f"{os.path.join(m.config.prompt_tmp_folder, m.config.prompt_tmp_file)} "
            f"and rerun."
        )

    m.generate()

    for g_idx, sct_id in enumerate(group_sct_ids):
        result = m.toJSON(g_idx)
        rows   = groups[sct_id]

        if result is None or "mappings" not in result:
            # Fallback: reject all in group.
            group_results[sct_id] = [
                {"decision": "REJECT", "confidence": 0} for _ in rows
            ]
            continue

        # Index the returned decisions by hpo_id for safe lookup.
        decisions_by_hpo = {}
        for item in result["mappings"]:
            hpo_id = item.get("hpo_id")
            if hpo_id:
                decisions_by_hpo[hpo_id] = item

        per_group = []
        for row in rows:
            hpo_id   = getattr(row, id_col)
            decision = decisions_by_hpo.get(hpo_id, {"decision": "REJECT", "confidence": 0})
            per_group.append(decision)

        group_results[sct_id] = per_group

# ------------------------------------------------------------------
# Assemble results into a unified dataframe.
# ------------------------------------------------------------------
l.log("Assembling results...")

# Pair results.
pair_decisions   = [r.get("decision",   "REJECT") for r in pair_results]
pair_confidences = [r.get("confidence", 0)        for r in pair_results]

pair_out = pair_candidates.copy()
pair_out["llm_decision"]  = pair_decisions
pair_out[confidence_column] = [c / 10.0 for c in pair_confidences]

# Group results.
group_rows_out = []
for sct_id, decisions in group_results.items():
    rows = groups[sct_id]
    for row, decision in zip(rows, decisions):
        group_rows_out.append({
            id_col:            getattr(row, id_col),
            val_col:           getattr(row, val_col),
            validity_column:   getattr(row, validity_column),
            "llm_decision":    decision.get("decision",   "REJECT"),
            confidence_column: decision.get("confidence", 0) / 10.0,
            "constellation":   getattr(row, "constellation"),
        })

group_out = pd.DataFrame(group_rows_out)

# Combine both passes.
all_results = pd.concat([pair_out, group_out], ignore_index=True)

# Write full LLM output before filtering.
writeHugeCSV(all_results, "./data/output/mapped/all_llm_mappings.csv")

# ------------------------------------------------------------------
# Update validity: ACCEPT -> validity=5, REJECT stays as-is.
# Merge accepted LLM mappings back into the full mappings dataframe.
# ------------------------------------------------------------------
accepted_llm = all_results[all_results["llm_decision"] == "ACCEPT"][[id_col, val_col]].copy()
accepted_llm["_llm_accepted"] = True

mappings = mappings.merge(accepted_llm, on=[id_col, val_col], how="left")
llm_accept_mask = (
    mappings[validity_column].isin([1, 2, 3, 4]) &
    (mappings["_llm_accepted"] == True)
)
mappings.loc[llm_accept_mask, validity_column] = 5
mappings = mappings.drop(columns=["_llm_accepted"])

writeHugeCSV(mappings, "./data/output/filtered/llm_filtered.csv")

l.log(f"LLM accepted: {llm_accept_mask.sum()} rows updated to validity=5.")
l.log(f"Total mappings written: {len(mappings)}")

l.printHeader("LLM review completed.")