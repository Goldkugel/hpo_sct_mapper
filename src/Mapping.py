import sys

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

from logger                 import Logger
from embedder               import Embedder
from model                  import Model
from normalizer             import Normalizer
from adapter                import HPOAdapter, SCTAdapter, UMLSAdapter
from MappingUtils           import *
from sentence_transformers  import util
from tqdm                   import tqdm
from scipy                  import sparse
import pandas               as pd
import numpy                as np
import os
import json
import faiss


os.chdir("..")

l = Logger()

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
    )
l.log("Reading UMLS data completed.")

l.log("Reading gold standard data...")
graham = pd.read_csv(
        "./data/input/graham/mapping_graham.csv",
        sep=";",
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

# UMLS -> HPO (to check with HPO -> UMLS) + UMLS -> SCT

l.log("Loading UMLS references...")
umls_references = umls[
    umls["attribute"] == "reference"
].copy().reset_index(drop = True)
l.log(f"Loading UMLS references completed. " \
      f"{len(umls_references.index)} references found.")

umls_hpo_maps = {}
umls_sct_maps = {}

l.log("Collecting HPO and SNOMED CT references in UMLS...")

with newProgress() as progress:
    
    task = newTask(
        progress, 
        len(umls_references.index), 
        "Collect HPO and SNOMED CT references in UMLS"
    )

    for index, row in umls_references.iterrows():
        j = json.loads(str(row["additional"]).replace("'", '"'))

        id = "UMLS:" + str(row["id"])
        value = "SNOMEDCT_US:" + str(row["value"])

        if j["source_abbreviation"] == "HPO":
            if row["id"] not in umls_hpo_maps.keys():
                umls_hpo_maps[row["id"]] = []
            umls_hpo_maps[row["id"]].append(row["value"])
        elif j["source_abbreviation"] == "SNOMEDCT_US":
            if id not in umls_sct_maps.keys():
                umls_sct_maps[id] = []
            umls_sct_maps[id].append(value)

        progress.update(task, advance = 1)
                
    progress.refresh()

l.log("Collecting HPO and SNOMED CT references in UMLS completed.")

# HPO -> UMLS

l.log("Gathering all UMLS references in HPO...")

hpo_umls_references = hpo[(
    hpo["attribute"] == "reference") & (
    hpo["value"].str.startswith("UMLS")
)].copy().reset_index(drop = True)

hpo_umls_maps = {}

for index, row in hpo_umls_references.iterrows():
    if row["id"] not in hpo_umls_maps.keys():
        hpo_umls_maps[row["id"]] = []
    hpo_umls_maps[row["id"]].append(row["value"])
    
l.log("Gathering all UMLS references in HPO completed.")

# HPO -> SCT

l.log("Gathering all SNOMED CT references in HPO...")

hpo_sct_references = hpo[(
    hpo["attribute"] == "reference") & (
    hpo["value"].str.startswith("SNOMEDCT_US")
)].copy().reset_index(drop = True)

hpo_sct_maps = {}

for index, row in hpo_sct_references.iterrows():
    if row["id"] not in hpo_sct_maps.keys():
        hpo_sct_maps[row["id"]] = []
    hpo_sct_maps[row["id"]].append(row["value"])

l.log("Gathering all SNOMED CT references in HPO completed.")

def dictToEAVDataFrame(data: dict, attribute: str) -> pd.DataFrame:
    ids = []
    values = []

    for key in data.keys():
        for element in data[key]:
            ids.append(key)
            values.append(element)

    return pd.DataFrame({
        "id" : ids,
        "attribute" : [attribute] * len(ids),
        "value" : values
    })

hpo_sct_maps_df = dictToEAVDataFrame(hpo_sct_maps, "hpo_sct")

count = 0
for index, row in hpo_sct_maps_df.iterrows():
    if (len(graham[(
        graham["hpo_id"] == row["id"]) & (
        graham["sct_id"] == row["value"]
    )].index) > 0):
        count += 1

l.log(f"Recall: {round(float(count) / len(graham.index), 2)}")
l.log(f"Precision: {round(float(count) / len(hpo_sct_maps_df.index), 2)}")
l.log(f"Concept Mapping Count: {len(set(hpo_sct_maps_df['id'].tolist()))}")
l.log(f"Mapping Count: {len(hpo_sct_maps_df.index)}")

# HPO -> UMLS -> SCT (starting point: hpo_umls_maps and umls_sct_maps)

l.log("Gathering all SNOMED CT references in UMLS concepts " \
    "referred from HPO...")

hpo_umls_ids = {}
for hpo_id in hpo_umls_maps.keys():
    tmp = hpo_umls_maps[hpo_id].copy()
    for i in range(0, len(tmp)):
        tmp[i] = str(tmp[i]).replace("UMLS:", "")
        if tmp[i] not in hpo_umls_ids.keys():
            hpo_umls_ids[tmp[i]] = []

for umls_id in hpo_umls_ids.keys():
    full_umls_id = "UMLS:" + umls_id
    if full_umls_id in umls_sct_maps:
        hpo_umls_ids[umls_id] = umls_sct_maps[full_umls_id]

hpo_umls_sct_maps = {}

for hpo_id in hpo_umls_maps.keys():
    hpo_umls_sct_maps[hpo_id] = []
    for umls_id in hpo_umls_maps[hpo_id]:
        if umls_id in umls_sct_maps.keys():
            hpo_umls_sct_maps[hpo_id] = (hpo_umls_sct_maps[hpo_id] + 
                umls_sct_maps[umls_id])

d = []
for key in hpo_umls_sct_maps.keys():
    if len(hpo_umls_sct_maps[key]) == 0:
        d.append(key)

for key in d:
    del hpo_umls_sct_maps[key]

l.log("Gathering all SNOMED CT references in UMLS concepts " \
    "referred from HPO completed.")

hpo_umls_sct_maps_df = dictToEAVDataFrame(hpo_umls_sct_maps, "hpo_umls_sct")

count = 0
for index, row in hpo_umls_sct_maps_df.iterrows():
    if (len(graham[(
        graham["hpo_id"] == row["id"]) & (
        graham["sct_id"] == row["value"]
    )].index) > 0):
        count += 1

l.log(f"Recall: {round(float(count) / len(graham.index), 2)}")
l.log(f"Precision: {round(float(count) / len(hpo_umls_sct_maps_df.index), 2)}")
l.log(f"Concept Mapping Count: {len(set(hpo_umls_sct_maps_df['id'].tolist()))}")
l.log(f"Mapping Count: {len(hpo_umls_sct_maps_df.index)}")

# HPO -> UMLS -> SCT + HPO -> SCT

l.log("Gathering all direct mappings (HPO -> UMLS -> SCT and HPO -> SCT)...")

directed_mappings = pd.concat([
    hpo_umls_sct_maps_df, 
    dictToEAVDataFrame(hpo_sct_maps, "hpo_sct")
])

directed_mappings = directed_mappings.drop(["attribute"], axis=1)
directed_mappings = directed_mappings.drop_duplicates(ignore_index = True)
directed_mappings = directed_mappings.reset_index(drop = True)

l.log("Gathering all direct mappings " \
    "(HPO -> UMLS -> SCT and HPO -> SCT) completed.")

count = 0
for index, row in directed_mappings.iterrows():
    if (len(graham[(
        graham["hpo_id"] == row["id"]) & (
        graham["sct_id"] == row["value"]
        )].index) > 0):
        count += 1

l.log(f"Recall: {round(float(count) / len(graham.index), 2)}")
l.log(f"Precision: {round(float(count) / len(directed_mappings.index), 2)}")
l.log(f"Concept Mapping Count: {len(set(directed_mappings['id'].tolist()))}")
l.log(f"Mapping Count: {len(directed_mappings.index)}")

# HPO -> SCT + HPO <- UMLS -> SCT + HPO -> UMLS -> SCT

l.log("Create mappings based on shared UMLS concepts...")

umls_hpo_sct_maps = {}
for key in umls_hpo_maps.keys():
    full_umls_id = "UMLS:" + key
    if full_umls_id in umls_sct_maps.keys():
        for hpo_id in umls_hpo_maps[key]:
            if hpo_id not in umls_hpo_sct_maps.keys():
                umls_hpo_sct_maps[hpo_id] = []
            umls_hpo_sct_maps[hpo_id] = umls_hpo_sct_maps[hpo_id] + \
                umls_sct_maps[full_umls_id]

umls_hpo_sct_maps_df = dictToEAVDataFrame(umls_hpo_sct_maps, "umls_hpo_sct")

l.log("Create mappings based on shared UMLS concepts completed.")

l.log("Gathering all mappings " \
    "('HPO -> SCT', 'HPO <- UMLS -> SCT', and 'HPO -> " \
    "UMLS -> SCT')...")

mappings = pd.concat([
    directed_mappings, 
    umls_hpo_sct_maps_df
], ignore_index = True)

mappings = mappings.drop(["attribute"], axis=1)
mappings = mappings.drop_duplicates(ignore_index = True)
mappings = mappings.reset_index(drop = True)

l.log("Gathering all mappings " \
    "('HPO -> SCT', 'HPO <- UMLS -> SCT', and 'HPO -> " \
    "UMLS -> SCT') completed.")

count = 0
for index, row in mappings.iterrows():
    if (len(graham[(
        graham["hpo_id"] == row["id"]) & (
        graham["sct_id"] == row["value"])].index) > 0):
        count += 1

l.log(f"Recall: {float(count) / len(graham.index)}")
l.log(f"Precision: {float(count) / len(mappings.index)}")
l.log(f"Concept Mapping Count: {len(set(mappings['id'].tolist()))}")
l.log(f"Mapping Count: {len(mappings.index)}")

# HPO -> UMLS vs. UMLS -> HPO check

l.log("Comparing 'HPO -> UMLS' vs. 'UMLS -> HPO' mappings...")

contradiction = {
    "umls_not_in_hpo" : {},
    "hpo_not_in_umls" : {}
}

for umls_id in umls_hpo_maps.keys():
    # umls_id has a map to hpo concepts.
    full_umls_id = "UMLS:" + umls_id
    for hpo_id in umls_hpo_maps[umls_id]:
        if hpo_id in hpo_umls_maps:
            if full_umls_id not in hpo_umls_maps[hpo_id]:
                # UMLS -> HPO not in HPO -> UMLS
                obj = contradiction["umls_not_in_hpo"]
                obj[full_umls_id] = hpo_id
        else:
            # HPO -> UMLS not in UMLS -> HPO
            obj = contradiction["hpo_not_in_umls"]
            obj[hpo_id] = full_umls_id
            
l.log("Comparing 'HPO -> UMLS' vs. 'UMLS -> HPO' mappings completed.")
l.log(f"UMLS mapping not found in HPO mapping: " \
    f"{len(contradiction['umls_not_in_hpo'].keys())}")
l.log(f"HPO mapping not found in UMLS mapping: " \
    f"{len(contradiction['hpo_not_in_umls'].keys())}")

hpo_terms_subset = hpo[((
        hpo["attribute"] == "label") | (
        hpo["attribute"] == "synonym")) & (
    hpo["id"].str.startswith("HP:"))]
hpo_terms_subset = hpo_terms_subset.reset_index(drop = True)
l.log(f"HPO Terms: {len(hpo_terms_subset.index)}")

sct_terms_subset = sct[(
    sct["attribute"] == "label") | (
    sct["attribute"] == "synonym"
)]

sct_terms_subset = sct_terms_subset.reset_index(drop = True)

l.log(f"SCT Terms: {len(sct_terms_subset.index)}")

sct_terms = sct_terms_subset["value"].tolist()
hpo_terms = hpo_terms_subset["value"].tolist()

n = Normalizer()

sct_normalized_terms = []
hpo_normalized_terms = []

l.log("Normalizing SNOMED CT terms...")

sct_normalized_terms = n.normalize_batch(
    sct_terms, 
    batch_size = 15000, 
    n_process = 100
)

l.log("Normalizing SNOMED CT terms completed.")

l.log("Normalizing HPO terms...")

hpo_normalized_terms = n.normalize_batch(
    hpo_terms, 
    batch_size = 15000, 
    n_process = 4
)

l.log("Normalizing HPO terms completed.")

ids_jaccard_similarity = []
values_jaccard_similarity = []
confidence_jaccard_similarity = []

threshold_jaccard_similarity = 0.95

l.log("Calculating Jaccard similarities...")
# Build a shared vocabulary
vocab = {}
def encode(term_lists):
    rows, cols = [], []
    for i, tokens in enumerate(term_lists):
        for tok in set(tokens):
            j = vocab.setdefault(tok, len(vocab))
            rows.append(i)
            cols.append(j)
    return rows, cols

rows_a, cols_a = encode(sct_normalized_terms)
rows_b, cols_b = encode(hpo_normalized_terms)

n_vocab = len(vocab)
A = sparse.csr_matrix((np.ones(len(rows_a), dtype=np.int32), (rows_a, cols_a)),
                       shape=(len(sct_normalized_terms), n_vocab))
B = sparse.csr_matrix((np.ones(len(rows_b), dtype=np.int32), (rows_b, cols_b)),
                       shape=(len(hpo_normalized_terms), n_vocab))

intersection_sparse = A.dot(B.T).tocoo()   # STAYS SPARSE - do not call .toarray()

rows = intersection_sparse.row
cols = intersection_sparse.col
inter_vals = intersection_sparse.data.astype(np.float64)

sizes_a = np.asarray(A.sum(axis=1)).ravel()
sizes_b = np.asarray(B.sum(axis=1)).ravel()

# Only compute jaccard for the (sparse) pairs that actually share tokens
a_sizes = sizes_a[rows]
b_sizes = sizes_b[cols]
union_vals = a_sizes + b_sizes - inter_vals

jaccard_vals = inter_vals / union_vals
min_vals = np.minimum(a_sizes, b_sizes)

hit_jac_mask = jaccard_vals > threshold_jaccard_similarity

ids_jaccard_similarity = [hpo_terms_subset.loc[j, "id"] for j in cols[hit_jac_mask]]
values_jaccard_similarity = [sct_terms_subset.loc[i, "id"] for i in rows[hit_jac_mask]]

# Handle the both-empty edge case (jaccard = 1.0) separately — these pairs
# have intersection 0 so they're absent from the sparse matrix above.
empty_a = np.where(sizes_a == 0)[0]
empty_b = np.where(sizes_b == 0)[0]
if len(empty_a) and len(empty_b) and threshold_jaccard_similarity < 1.0:
    for i in empty_a:
        for j in empty_b:
            ids_jaccard_similarity.append(hpo_terms_subset.loc[j, "id"])
            values_jaccard_similarity.append(sct_terms_subset.loc[i, "id"])

for index in range(0, len(values_jaccard_similarity)):
    values_jaccard_similarity[index] = (
        "SNOMEDCT_US:" + str(values_jaccard_similarity[index]))

jaccard_maps = {}
for index in range(0, len(ids_jaccard_similarity)):
    hpo_id = ids_jaccard_similarity[index]
    if hpo_id not in jaccard_maps.keys():
        jaccard_maps[hpo_id] = []
    jaccard_maps[hpo_id].append(values_jaccard_similarity[index])

jaccard_maps_df = dictToEAVDataFrame(jaccard_maps, "jaccard")
jaccard_maps_df = jaccard_maps_df.drop_duplicates().reset_index(drop = True)

l.log("Calculating Jaccard similarities completed.")

count = 0
for index, row in jaccard_maps_df.iterrows():
    if (len(graham[(
        graham["hpo_id"] == row["id"]) & (
        graham["sct_id"] == row["value"]
    )].index) > 0):
        count += 1

l.log(f"Recall: {float(count) / len(graham.index)}")
l.log(f"Precision: {float(count) / len(jaccard_maps_df.index)}")
l.log(f"Concept Mapping Count: {len(set(jaccard_maps_df['id'].tolist()))}")
l.log(f"Mapping Count: {len(jaccard_maps_df.index)}")

l.log("Gathering all mappings...")

mappings = pd.concat([mappings, jaccard_maps_df], ignore_index = True)
mappings = mappings.drop(["attribute"], axis=1)
mappings = mappings.drop_duplicates(ignore_index = True)
mappings = mappings.reset_index(drop = True)

l.log("Gathering all mappings completed.")

count = 0
for index, row in mappings.iterrows():
    if (len(graham[(
        graham["hpo_id"] == row["id"]) & (
        graham["sct_id"] == row["value"]
    )].index) > 0):
        count += 1

l.log(f"Recall: {float(count) / len(graham.index)}")
l.log(f"Precision: {float(count) / len(mappings.index)}")
l.log(f"Concept Mapping Count: {len(set(mappings['id'].tolist()))}")
l.log(f"Mapping Count: {len(mappings.index)}")

e = Embedder()

l.log("Embed HPO terms...")

embedded_hpo = e.embed(hpo_terms)
embedded_hpo = embedded_hpo["cambridgeltl/SapBERT-from-PubMedBERT-fulltext"]

l.log("Embed HPO terms completed.")

l.log("Embed SNOMED CT terms...")

embedded_sct = e.embed(sct_terms)
embedded_sct = embedded_sct["cambridgeltl/SapBERT-from-PubMedBERT-fulltext"]

l.log("Embed SNOMED CT terms completed.")

l.log("Calculating cosine similarity...")

ids_cosine_similarity = []
values_cosine_similarity = []
confidence_cosine_similarity = []
threshold_cosine_similarity = 0.85

# Use all available CPU cores
faiss.omp_set_num_threads(os.cpu_count())

# Normalize once (skip if embedder already outputs unit vectors)
def normalize(mat):
    return mat / np.linalg.norm(mat, axis=1, keepdims=True)

embedded_sct_n = normalize(embedded_sct)
embedded_hpo_n = normalize(embedded_hpo)

dim = embedded_hpo.shape[1]

# Approximate index: cluster HPO embeddings, only search nearby clusters per query
nlist = 1000  # number of clusters; rule of thumb: ~sqrt(n_vectors) to 4*sqrt(n_vectors)
quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

index.train(embedded_hpo_n.astype(np.float32))  # needed once, learns the clusters
index.add(embedded_hpo_n.astype(np.float32))

index.nprobe = 20  # speed/recall knob; increase if recall is too low

# range_search returns all neighbors above a similarity threshold
lims, distances, indices = index.range_search(
    embedded_sct_n.astype(np.float32), threshold_cosine_similarity
)

# Vectorized result collection (no Python-level per-match loop)
n_matches_per_query = np.diff(lims).astype(np.int64)
sct_row_indices = np.repeat(np.arange(len(sct_terms)), n_matches_per_query)

hpo_ids_arr = hpo_terms_subset["id"].to_numpy()
sct_ids_arr = sct_terms_subset["id"].to_numpy()

ids_cosine_similarity = hpo_ids_arr[indices].tolist()
values_cosine_similarity = sct_ids_arr[sct_row_indices].tolist()
confidence_cosine_similarity = distances.tolist()

l.log("Calculating cosine similarity completed.")

l.log("Create mappings from cosine similarity...")

embedding_maps = {}

for index in range(0, len(ids_cosine_similarity)):
    hpo_id = ids_cosine_similarity[index]
    if hpo_id not in embedding_maps.keys():
        embedding_maps[hpo_id] = []
    embedding_maps[hpo_id].append(
        "SNOMEDCT_US:" + str(values_cosine_similarity[index]))

embedding_maps_df = dictToEAVDataFrame(embedding_maps, "embedding")
embedding_maps_df = embedding_maps_df.drop_duplicates(ingore_index = True)
embedding_maps_df = embedding_maps_df.reset_index(drop = True)

l.log("Create mappings from cosine similarity completed.")

count = 0
for index, row in embedding_maps_df.iterrows():
    if (len(graham[(
        graham["hpo_id"] == row["id"]) & (
        graham["sct_id"] == row["value"]
    )].index) > 0):
        count += 1

l.log(f"Recall: {float(count) / len(graham.index)}")
l.log(f"Precision: {float(count) / len(embedding_maps_df.index)}")
l.log(f"Concept Mapping Count: {len(set(embedding_maps_df['id'].tolist()))}")
l.log(f"Mapping Count: {len(embedding_maps_df.index)}")

l.log("Gathering all mappings...")

mappings = pd.concat([
    mappings, embedding_maps_df.drop(["attribute"], axis = 1)
], ignore_index = True)
mappings = mappings.drop_duplicates(ignore_index = True)
mappings = mappings.reset_index(drop = True)

count = 0
for index, row in mappings.iterrows():
    if (len(graham[(
        graham["hpo_id"] == row["id"]) & (
        graham["sct_id"] == row["value"]
    )].index) > 0):
        count += 1

l.log("Gathering all mappings completed.")

l.log(f"Recall: {float(count) / len(graham.index)}")
l.log(f"Precision: {float(count) / len(mappings.index)}")
l.log(f"Concept Mapping Count: {len(set(mappings['id'].tolist()))}")
l.log(f"Mapping Count: {len(mappings.index)}")

all_mappings_df = pd.concat([
    hpo_sct_maps_df,
    hpo_umls_sct_maps_df,
    umls_hpo_sct_maps_df,
    jaccard_maps_df,
    embedding_maps_df
], ignore_index = True).reset_index(drop = True)
writeHugeCSV(all_mappings_df, "./data/output/mapped/mapping.csv")