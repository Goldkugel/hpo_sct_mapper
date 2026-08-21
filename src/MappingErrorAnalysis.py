import sys

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

from logger                 import Logger
from adapter                import HPOAdapter, SCTAdapter, UMLSAdapter
from MappingUtils           import *
import pandas               as pd
import numpy                as np
import os

l = Logger()

l.printHeader("Gather all mappings from HPO to SNOMED CT")

l.log("Loading necessary data...")

hpo     = HPOAdapter()
if hpo.load() > 0:
    hpo.to_csv()
umls    = UMLSAdapter()
if umls.load() > 0:
    umls.to_csv()
sct     = SCTAdapter()
if sct.load() > 0:
    sct.to_csv()

l.log("Loading necessary data completed.")

# ToDo: filter only exact synonyms.
l.log("Reading HPO data...")
hpo = pd.read_csv(
        os.path.join(
            hpo.config.output_folder, 
            hpo.config.output_file
        ),
        sep=";",
        low_memory = False
    )
hpo = hpo[hpo["additional"].apply(
    lambda x: x.get("semantic_class") != "related" if isinstance(x, dict) else True
)]
l.log("Reading HPO data completed")

l.log("Reading SNOMED CT data...")
sct = pd.read_csv(
        os.path.join(
            sct.config.output_folder, 
            sct.config.output_file
        ),
        sep=";",
        low_memory = False
    )
sct["value"] = sct["value"].astype(str)
l.log("Reading SNOMED CT data completed.")

l.log("Reading UMLS data...")
umls = pd.read_csv(
        os.path.join(
            umls.config.output_folder, 
            umls.config.output_file
        ),
        sep=";",
        low_memory = False
    )
l.log("Reading UMLS data completed.")

l.log("Reading gold standard data...")
graham = pd.read_csv(
        "./data/input/graham/mapping_graham.csv",
        sep=";",
        low_memory = False
    )

graham = graham[graham["Match group"].isin([
    "One to one match", 
    "One to many"
])]
graham = graham.drop([
    "Synonyms", 
    "Preferred Label", 
    "Definition", 
    "HP Parent No", 
    "HP Parent Term", 
    "Match group", 
    "Alternative expression", 
    "Mapping Comment", 
    "SNOMED term/exp"
], axis=1)
graham["canonical view"] = graham["canonical view"].astype(str)
graham["sct_id"] = -1

for index, row in graham.iterrows():
    if ("e+" in row["canonical view"] or 
        (":" not in row["canonical view"] and 
        "+" not in row["canonical view"])):
        try:
            graham.loc[index, "sct_id"] = int(float(row["canonical view"]))
        except ValueError:
            ""
    elif "+" in row["canonical view"]:
        graham.loc[index, "sct_id"] = -2

graham = graham.drop(["canonical view"], axis=1)
graham = graham.rename(columns={"HP_ID": "hpo_id"})
graham = graham[graham["sct_id"] > 0]
graham = graham.reset_index(drop = True)
graham["sct_id"] = graham["sct_id"].astype(str)

for index, row in graham.iterrows():
    graham.loc[index, "sct_id"] = "SNOMEDCT_US:" + row["sct_id"]

l.log("Reading gold standard data completed.")
l.log(f"Mappings for {len(set(graham['hpo_id'].tolist()))} Concepts found.")

l.log("Loading generated mappings...")

mappigns = pd.read_csv("./data/output/mapped/mapping.csv")

l.log("Loading generated mappings completed.")