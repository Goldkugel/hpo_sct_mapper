import sys

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

from logger         import Logger
from adapter        import labelClass
from MappingUtils   import loadOntologies, loadGold, hpo_id_column, sct_id_column
import pandas        as pd

if __name__ == "__main__":

    l = Logger()

    l.printHeader("Error analysis: gold-standard pairs missed by every generation strategy")

    hpo, sct, umls, hpo_ontology, sct_ontology, umls_ontology = loadOntologies(";")
    graham = loadGold("./data/input/graham/mapping_graham.csv", ";")

    l.log("Loading generated mappings...")
    mappings = pd.read_csv("./data/output/mapped/mapping.csv", sep=";")
    l.log("Loading generated mappings completed.")

    id_col  = hpo_ontology.config.id_column
    val_col = hpo_ontology.config.value_column

    # Gold-standard (hpo_id, sct_id) pairs that no generation strategy in
    # MappingGeneration.py produced at all, i.e. the source of the recall
    # gap reported by evaluation() at each stage.
    generated_pairs = set(zip(mappings[id_col], mappings[val_col]))
    gold_pairs = list(zip(graham[hpo_id_column], graham[sct_id_column]))

    missed = sorted({pair for pair in gold_pairs if pair not in generated_pairs})

    l.log(f"Gold standard pairs: {len(set(gold_pairs))}")
    l.log(
        f"Gold standard pairs missed by every generation strategy: {len(missed)} "
        f"({len(missed) / len(set(gold_pairs)):.1%})"
    )

    hpo_labels = (
        hpo[hpo[hpo_ontology.config.attribute_column] == labelClass]
        [[id_col, val_col]]
        .drop_duplicates(subset=[id_col])
        .set_index(id_col)[val_col]
        .to_dict()
    )

    sample_size = min(20, len(missed))
    l.log(f"Sample of {sample_size} missed pairs:")
    for missed_hpo_id, missed_sct_id in missed[:sample_size]:
        label = hpo_labels.get(missed_hpo_id, "?")
        l.log(f"  \"{missed_hpo_id}\" ({label}) -> \"{missed_sct_id}\"")

    l.printHeader("Error analysis completed")
