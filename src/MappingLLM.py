import sys

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

from logger                 import Logger
from model                  import Model
from adapter                import definitionClass, synonymClass, commentClass, labelClass, childrenClass, exactSynonymClass, semanticClass, writeHugeCSV
from MappingUtils           import loadOntologies, loadGold, validity_column, confidence_column, attribute_column
import pandas               as pd

l = Logger()

l.printHeader("Check valid and not auto accepted mappings")

hpo, sct, umls, hpo_ontology, sct_ontology, umls_ontology = loadOntologies(";")
hpo = hpo[hpo[hpo_ontology.config.id_column].str.startswith("HP:")].reset_index(drop = True)

mask = sct[sct_ontology.config.attribute_column] == childrenClass

# 2. Extract the target column name
val_col = sct_ontology.config.value_column

# 3. Vectorize string concatenation on only the masked rows
#    (We conditionally prepend 'SNOMEDCT_US:' only if it's not already prefixed)
sct_vals = sct.loc[mask, val_col].astype(str)
already_prefixed = sct_vals.str.startswith("SNOMEDCT_US:")

sct.loc[mask, val_col] = (
    "SNOMEDCT_US:" + sct_vals.where(already_prefixed, sct_vals)
)

graham = loadGold("./data/input/graham/mapping_graham.csv", ";")

l.log("Loading generated mappings...")
mappings = pd.read_csv("./data/output/filtered/filtered.csv", sep=";")
l.log("Loading generated mappings completed.")

def quote(txt: str, quotationChar: chr = '"') -> str:
    if txt is None:
        return "None"
    return quotationChar + str(txt) + quotationChar

def quoteList(txt_list: list[str], quotationChar: chr = '"') -> list[str]:
    if len(txt_list) == 0:
        return ["None"]
    return [quote(txt, quotationChar) for txt in txt_list]

to_check = mappings[(mappings[validity_column] == 1) & (mappings["auto_accept"] == 0)].reset_index(drop=True)

m = Model(index = 0)

# ---------------------------------------------------------
# 1. Updated Helper Function to Include Children Lookups
# ---------------------------------------------------------
def build_ontology_lookups(df, ont_config):
    id_col = ont_config.id_column
    attr_col = ont_config.attribute_column
    val_col = ont_config.value_column
    add_col = ont_config.additional_column

    # DataFrames for attributes
    labels = df[df[attr_col] == labelClass].groupby(id_col)[val_col].apply(list).to_dict()
    definitions = df[df[attr_col] == definitionClass].groupby(id_col)[val_col].apply(list).to_dict()
    comments = df[df[attr_col] == commentClass].groupby(id_col)[val_col].apply(list).to_dict()

    # Extract synonyms
    syn_df = df[df[attr_col] == synonymClass]
    if add_col in syn_df.columns:
        syn_mask = syn_df[add_col].apply(lambda x: isinstance(x, dict) and x.get(semanticClass) == exactSynonymClass)
        synonyms = syn_df[syn_mask].groupby(id_col)[val_col].apply(list).to_dict()
    else:
        synonyms = {}

    # Hierarchy Lookups
    # Parent Lookup: val_col (child ID) -> id_col (parent ID)
    children = df[df[attr_col] == childrenClass].groupby(val_col)[id_col].apply(list).to_dict()
    
    # Child Lookup: id_col (parent ID) -> val_col (child ID)
    parents = df[df[attr_col] == childrenClass].groupby(id_col)[val_col].apply(list).to_dict()

    return {
        labelClass: labels,
        definitionClass: definitions,
        commentClass: comments,
        synonymClass: synonyms,
        "parents": parents,
        childrenClass: children
    }

def get_labels_for_ids(id_list, label_dict):
    result = []
    for _id in id_list:
        result.extend(label_dict.get(_id, []))
    return result

if m.importHistoriesFromFile() <= 0:

    m.addPromptFromFile(0, len(to_check.index))

    # ---------------------------------------------------------
    # 2. Build Lookups
    # ---------------------------------------------------------
    l.log("Creating Lookup-Tables...")
    hpo_data = build_ontology_lookups(hpo, hpo_ontology.config)
    sct_data = build_ontology_lookups(sct, sct_ontology.config)
    l.log("Creating Lookup-Tables completed")

    # ---------------------------------------------------------
    # 3. Main Loop
    # ---------------------------------------------------------
    l.log("Create Prompts...")

    for row in to_check.itertuples(index = True):
        index = row.Index

        obj = {}

        hpo_id = getattr(row, hpo_ontology.config.id_column)
        sct_id = getattr(row, sct_ontology.config.value_column)
        
        # --- HPO Lookups ---
        obj["hpo_label"]        = ", ".join(quoteList(hpo_data[labelClass].get(hpo_id, [])))
        obj["hpo_synonyms"]     = ", ".join(quoteList(hpo_data[synonymClass].get(hpo_id, [])))
        obj["hpo_definition"]   = ", ".join(quoteList(hpo_data[definitionClass].get(hpo_id, [])))
        obj["hpo_comment"]      = ", ".join(quoteList(hpo_data[commentClass].get(hpo_id, [])))
        obj["hpo_children"]     = ", ".join(quoteList(get_labels_for_ids(hpo_data[childrenClass].get(hpo_id, []), hpo_data[labelClass])))
        obj["hpo_parents"]      = ", ".join(quoteList(get_labels_for_ids(hpo_data["parents"].get(hpo_id, []), hpo_data[labelClass])))

        # --- SNOMED CT Lookups ---
        obj["sct_term"]         = ", ".join(quoteList(sct_data[labelClass].get(sct_id, [])))
        obj["sct_synonyms"]     = ", ".join(quoteList(sct_data[synonymClass].get(sct_id, [])))
        obj["sct_children"]     = ", ".join(quoteList(get_labels_for_ids(sct_data[childrenClass].get(sct_id, []), sct_data[labelClass])))
        obj["sct_parents"]      = ", ".join(quoteList(get_labels_for_ids(sct_data["parents"].get(sct_id, []), sct_data[labelClass])))

        # --- Prompt Creation ---
        m.formatPromptFromFile(index, obj)

    l.log("Create Prompts completed")
    
m.generate()

results = []

for i in range(0, len(to_check.index)):
    results.append(m.toJSON(i))

df = pd.DataFrame(results)

to_check[confidence_column] = df[confidence_column]
to_check[confidence_column] = to_check[confidence_column].astype(float) / 10.0
to_check[attribute_column] = ["llm"] * len(to_check.index)
to_check["llm_classification"] = df["mapping_type"]
writeHugeCSV(to_check, "./data/output/mapped/all_llm_mappings.csv")

to_check[validity_column] = to_check["llm_classification"] == "EXACT_MATCH"
to_check = to_check[to_check[validity_column] == 1].reset_index(drop = True)
to_check = to_check.drop(["llm_classification"], axis = 1)

mappings = pd.concat([mappings, to_check], ignore_index = True).reset_index(drop = True)

writeHugeCSV(mappings, "./data/output/mapped/llm_mapping.csv")

l.printHeader("Check valid and not auto accepted mappings compelted.")