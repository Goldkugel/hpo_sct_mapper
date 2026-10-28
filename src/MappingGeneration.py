import sys

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

from logger                 import Logger
from embedder               import Embedder
from normalizer             import Normalizer
from adapter                import writeHugeCSV, labelClass, synonymClass
from MappingUtils           import exact_lexical_mappings, loadOntologies, loadGold, evaluation, dictToEAVDataFrame, newProgress, newTask, confidence_column, hpo_sct_mappings, hpo_umls_sct_mappings, umls_hpo_sct_mappings, embedding_mappings, jaccard_mappings, threshold_cosine_similarity, threshold_jaccard_similarity
from adapter                import referenceClass
from scipy                  import sparse
import pandas               as pd
import numpy                as np
import os
import json
import faiss
import logging

# ==========================================================================
# MappingGeneration.py
#
# Generates all candidate HPO -> SNOMED CT mappings using six strategies,
# listed in descending order of expected precision:
#
#   1. hpo_sct       : Direct cross-references embedded in the HPO source file.
#   2. hpo_umls_sct  : HPO -> UMLS -> SCT, bridging via HPO's own UMLS xrefs.
#   3. umls_hpo_sct  : HPO <- UMLS -> SCT, bridging via shared UMLS CUI
#                      (UMLS lists both the HPO concept and an SCT concept
#                      under the same CUI).
#   4. jaccard       : Token-level Jaccard similarity on normalised labels and
#                      synonyms, above a configurable threshold.
#   5. embedding     : Cosine similarity on SapBERT embeddings of labels and
#                      synonyms, above a configurable threshold, using an
#                      approximate FAISS index for scalability.
#   6. exact_lexical : Case-insensitive exact string match on labels and
#                      synonyms, assigned confidence 1.0.
#
# Output:
#   all_mappings_df  : One row per (method, hpo_id, sct_id) triple, with the
#                      originating method preserved in the attribute column.
#                      Written to ./data/output/mapped/mapping.csv and used
#                      as input to the MappingSetBuilder script.
#
#   mappings         : Deduplicated (hpo_id, sct_id) pairs without the
#                      attribute column, used only for incremental evaluation
#                      during this script and discarded afterwards.
#
# Intermediate evaluation against the Graham gold standard is logged after
# each strategy and after each cumulative union, to track the recall and
# precision contribution of each method.
# ==========================================================================

if __name__ == "__main__":

    logging.getLogger("faiss").setLevel(logging.WARNING)

    l = Logger()

    l.printHeader("Gather all mappings from HPO to SNOMED CT")

    hpo, sct, umls, hpo_ontology, sct_ontology, umls_ontology = \
        loadOntologies(separator = ";")
    graham = \
        loadGold(path = "./data/input/graham/mapping_graham.csv", 
            separator = ";")

    # Build ID sets for fast membership tests. Cross-references found in one
    # ontology that point to a concept not present in the loaded version of the
    # target ontology are treated as invalid and counted but not mapped.
    valid_umls_ids = set(umls[umls_ontology.config.id_column].tolist())
    valid_sct_ids = set(sct[sct_ontology.config.id_column].tolist())
    valid_hpo_ids = set(hpo[hpo_ontology.config.id_column].tolist())

    # UMLS -> HPO (to check with HPO -> UMLS) + UMLS -> SCT

    l.log("Loading UMLS references...")
    umls_references = umls[
        umls[umls_ontology.config.attribute_column] == referenceClass
    ].copy().reset_index(drop = True)
    l.log(f"Loading UMLS references completed. " \
        f"{len(umls_references.index)} references found.")

    umls_hpo_maps = {}
    umls_sct_maps = {}

    invalid_sct_count = 0
    invalid_hpo_count = 0

    l.log("Collecting HPO and SNOMED CT references in UMLS...")

    with newProgress() as progress:
        
        task = newTask(
            progress, 
            len(umls_references.index), 
            "Collect HPO and SNOMED CT references in UMLS"
        )

        for row in umls_references.itertuples(index=False):
            # The additional column stores a Python-dict-like string with
            # metadata about the reference, including "source_abbreviation"
            # which identifies the target vocabulary (e.g. "HPO", "SNOMEDCT_US").
            # Single quotes are replaced with double quotes before JSON parsing
            # because the column is stored with Python repr formatting.
            j = json.loads(
                str(getattr(row, umls_ontology.config.additional_column))
                .replace("'", '"')
            )

            raw_id    = getattr(row, umls_ontology.config.id_column)
            raw_value = getattr(row, umls_ontology.config.value_column)

            # Prefixed forms used as keys in the mapping dicts.
            umls_id   = "UMLS:" + str(raw_id)
            sct_value = "SNOMEDCT_US:" + str(raw_value)

            if j["source_abbreviation"] == "HPO":
                # UMLS -> HPO: used later to cross-check against HPO -> UMLS.
                if raw_value in valid_hpo_ids:
                    umls_hpo_maps.setdefault(row.id, []).append(raw_value)
                else:
                    invalid_hpo_count += 1

            elif j["source_abbreviation"] == "SNOMEDCT_US":
                # UMLS -> SCT: used as the bridge target in both UMLS-based
                # mapping strategies.
                if sct_value in valid_sct_ids:
                    umls_sct_maps.setdefault(umls_id, []).append(sct_value)
                else:
                    invalid_sct_count += 1

            progress.update(task, advance = 1)
                    
        progress.refresh()

    l.log("Collecting HPO and SNOMED CT references in UMLS completed. " \
          f"Invalid SNOMED CT references: {invalid_sct_count}. Invalid " \
          f"HPO references: {invalid_hpo_count}.")

    # HPO -> UMLS

    l.log("Gathering all UMLS references in HPO...")

    hpo_umls_references = hpo[(
        hpo[hpo_ontology.config.attribute_column] == referenceClass) & (
        hpo[hpo_ontology.config.value_column].str.startswith("UMLS")
    )].copy().reset_index(drop = True)

    hpo_umls_maps = {}
    invalid_count = 0

    for index, row in hpo_umls_references.iterrows():
        if (row[hpo_ontology.config.value_column].replace("UMLS:", "") in 
            valid_umls_ids):
            if row[hpo_ontology.config.id_column] not in hpo_umls_maps.keys():
                hpo_umls_maps[row[hpo_ontology.config.id_column]] = []
            hpo_umls_maps[row[hpo_ontology.config.id_column]].append(
                row[hpo_ontology.config.value_column]
            )
        else:
            invalid_count += 1
        
    l.log("Gathering all UMLS references in HPO completed. Invalid " \
          f"UMLS references: {invalid_count}.")

    # HPO -> SCT

    l.log("Gathering all SNOMED CT references in HPO...")

    hpo_sct_references = hpo[(
        hpo[hpo_ontology.config.attribute_column] == referenceClass) & (
        hpo[hpo_ontology.config.value_column].str.startswith("SNOMEDCT_US")
    )].copy().reset_index(drop = True)

    hpo_sct_maps = {}
    invalid_count = 0

    for index, row in hpo_sct_references.iterrows():
        if row[hpo_ontology.config.value_column] in valid_sct_ids:
            if row[hpo_ontology.config.id_column] not in hpo_sct_maps.keys():
                hpo_sct_maps[row[hpo_ontology.config.id_column]] = []
            hpo_sct_maps[row[hpo_ontology.config.id_column]].append(
                row[hpo_ontology.config.value_column]
            )
        else:
            invalid_count += 1

    l.log("Gathering all SNOMED CT references in HPO completed. Invalid " \
          f"SNOMED CT references: {invalid_count}.")

    hpo_sct_maps_df = dictToEAVDataFrame(
        hpo_sct_maps, 
        hpo_sct_mappings, 
        hpo_ontology
    )
    hpo_sct_maps_df[confidence_column] = 1

    evaluation(
        hpo_sct_maps_df, 
        graham, 
        "HPO to SNOMED CT references only:",
        hpo_ontology
    )

    # HPO -> UMLS -> SCT (starting point: hpo_umls_maps and umls_sct_maps)

    l.log("Gathering all SNOMED CT references in UMLS concepts " \
        "referred from HPO...")

    # hpo_umls_ids: bare UMLS ID (without "UMLS:" prefix) -> [SCT IDs].
    # Built as an intermediate lookup to resolve which SCT concepts each
    # UMLS CUI referenced by HPO maps to. The prefix is stripped here
    # because hpo_umls_maps stores values with the "UMLS:" prefix while
    # the bare ID is needed to reconstruct the full key for umls_sct_maps.
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

    # Walk HPO -> UMLS -> SCT: for each HPO concept, collect the SCT concepts
    # reachable via every UMLS CUI the HPO concept references.
    hpo_umls_sct_maps = {}

    for hpo_id in hpo_umls_maps.keys():
        hpo_umls_sct_maps[hpo_id] = []
        for umls_id in hpo_umls_maps[hpo_id]:
            if umls_id in umls_sct_maps.keys():
                hpo_umls_sct_maps[hpo_id] = (hpo_umls_sct_maps[hpo_id] + 
                    umls_sct_maps[umls_id])

    # Remove HPO concepts for which no SCT concept was reachable via UMLS.
    # These are HPO concepts that reference UMLS CUIs that have no SCT mapping
    # in the loaded UMLS release.
    d = []
    for key in hpo_umls_sct_maps.keys():
        if len(hpo_umls_sct_maps[key]) == 0:
            d.append(key)

    for key in d:
        del hpo_umls_sct_maps[key]

    l.log("Gathering all SNOMED CT references in UMLS concepts " \
        "referred from HPO completed.")

    hpo_umls_sct_maps_df = dictToEAVDataFrame(
        hpo_umls_sct_maps, 
        hpo_umls_sct_mappings,
        hpo_ontology
    )
    hpo_umls_sct_maps_df[confidence_column] = 1

    evaluation(
        hpo_umls_sct_maps_df, 
        graham, 
        "HPO to SNOMED CT over UMLS references only:",
        hpo_ontology
    )

    # HPO -> UMLS -> SCT + HPO -> SCT

    l.log("Gathering all direct mappings (HPO -> UMLS -> SCT and " \
        "HPO -> SCT)...")

    directed_mappings = pd.concat([
        hpo_umls_sct_maps_df, 
        hpo_sct_maps_df
    ])

    directed_mappings = directed_mappings.drop(
        [hpo_ontology.config.attribute_column], axis = 1
    )
    directed_mappings = directed_mappings.drop_duplicates(ignore_index = True)
    directed_mappings = directed_mappings.reset_index(drop = True)

    l.log("Gathering all direct mappings " \
        "(HPO -> UMLS -> SCT and HPO -> SCT) completed.")

    evaluation(
        directed_mappings, 
        graham, 
        "HPO to SNOMED CT references and HPO to SNOMED CT over UMLS references:",
        hpo_ontology
    )

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

    umls_hpo_sct_maps_df = dictToEAVDataFrame(
        umls_hpo_sct_maps, 
        umls_hpo_sct_mappings, 
        umls_ontology
    )
    umls_hpo_sct_maps_df[confidence_column] = 1

    evaluation(
        umls_hpo_sct_maps_df, 
        graham, 
        "HPO to SNOMED CT references over UMLS references:",
        hpo_ontology
    )

    l.log("Create mappings based on shared UMLS concepts completed.")

    l.log("Gathering all mappings " \
        "('HPO -> SCT', 'HPO <- UMLS -> SCT', and 'HPO -> " \
        "UMLS -> SCT')...")

    mappings = pd.concat([
        directed_mappings, 
        umls_hpo_sct_maps_df
    ], ignore_index = True)

    mappings = mappings.drop([hpo_ontology.config.attribute_column], axis=1)
    mappings = mappings.drop_duplicates(ignore_index = True)
    mappings = mappings.reset_index(drop = True)

    l.log("Gathering all mappings " \
        "('HPO -> SCT', 'HPO <- UMLS -> SCT', and 'HPO -> " \
        "UMLS -> SCT') completed.")

    evaluation(
        mappings, 
        graham, 
        "HPO to SNOMED CT references, HPO to SNOMED CT over UMLS " \
            "references, and shared UMLS CUI references:",
        hpo_ontology
    )

    # HPO -> UMLS vs. UMLS -> HPO cross-check (diagnostic only).
    # HPO records outgoing UMLS xrefs (HPO -> UMLS) and UMLS records incoming
    # xrefs from HPO (UMLS -> HPO). In a fully consistent state these two sets
    # would be mirrors of each other. The asymmetry logged here is informational
    # only — it does not affect any mappings — but a large count of HPO concepts
    # not reflected in UMLS may indicate that some HPO -> UMLS -> SCT paths are
    # silently failing because the referenced UMLS CUI has no back-reference.

    l.log("Comparing 'HPO -> UMLS' vs. 'UMLS -> HPO' mappings...")

    umls_not_in_hpo = "umls_not_in_hpo"
    hpo_not_in_umls = "hpo_not_in_umls"

    contradiction = {
        umls_not_in_hpo : {},   # UMLS -> HPO edges not reflected in HPO -> UMLS
        hpo_not_in_umls : {}    # HPO -> UMLS edges not reflected in UMLS -> HPO
    }

    for umls_id in umls_hpo_maps.keys():
        # umls_id has a map to hpo concepts.
        full_umls_id = "UMLS:" + umls_id
        for hpo_id in umls_hpo_maps[umls_id]:
            if hpo_id in hpo_umls_maps:
                if full_umls_id not in hpo_umls_maps[hpo_id]:
                    # UMLS -> HPO not in HPO -> UMLS
                    obj = contradiction[umls_not_in_hpo]
                    obj[full_umls_id] = hpo_id
            else:
                # HPO -> UMLS not in UMLS -> HPO
                obj = contradiction[hpo_not_in_umls]
                obj[hpo_id] = full_umls_id
                
    l.log("Comparing 'HPO -> UMLS' vs. 'UMLS -> HPO' mappings completed.")
    l.log(f"UMLS mapping not found in HPO mapping: " \
        f"{len(contradiction[umls_not_in_hpo].keys())}")
    l.log(f"HPO mapping not found in UMLS mapping: " \
        f"{len(contradiction[hpo_not_in_umls].keys())}")

    # Build term subsets for lexical and embedding methods.
    # Non-string and empty-string terms are filtered out here once,
    # so all downstream methods (Jaccard, embedding, exact lexical)
    # are guaranteed to receive clean string input.
    hpo_terms_subset = hpo[((
            hpo[hpo_ontology.config.attribute_column] == labelClass) | (
            hpo[hpo_ontology.config.attribute_column] == synonymClass)) & (
        hpo[hpo_ontology.config.id_column].str.startswith("HP:"))]

    hpo_terms_subset = hpo_terms_subset[
        hpo_terms_subset[hpo_ontology.config.value_column].apply(
            lambda x: isinstance(x, str) and bool(x.strip())
        )
    ]

    hpo_terms_subset = hpo_terms_subset.reset_index(drop=True)
    l.log(f"HPO Terms: {len(hpo_terms_subset.index)}")

    sct_terms_subset = sct[(
        sct[sct_ontology.config.attribute_column] == labelClass) | (
        sct[sct_ontology.config.attribute_column] == synonymClass
    )]

    sct_terms_subset = sct_terms_subset[
        sct_terms_subset[sct_ontology.config.value_column].apply(
            lambda x: isinstance(x, str) and bool(x.strip())
        )
    ]

    sct_terms_subset = sct_terms_subset.reset_index(drop = True)
    l.log(f"SCT Terms: {len(sct_terms_subset.index)}")

    sct_terms = sct_terms_subset[sct_ontology.config.value_column].tolist()
    hpo_terms = hpo_terms_subset[hpo_ontology.config.value_column].tolist()

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

    l.log("Calculating Jaccard similarities...")

    # Jaccard similarity is computed as |A ∩ B| / |A ∪ B| over token sets.
    # A shared vocabulary is built across both term lists so that both can be
    # represented as binary sparse matrices over the same token space, enabling
    # the intersection to be computed as a single sparse matrix multiplication.
    vocab = {}

    def encode(term_lists):
        """
        Encode a list of token sets into parallel (row, col) index lists
        suitable for constructing a sparse binary matrix. Each row corresponds
        to one term and each column to one vocabulary token. The vocab dict is
        shared across calls so that SCT and HPO terms are encoded into the same
        column space.
        """
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

    # A: shape (n_sct_terms, n_vocab) — one row per SCT term, binary token presence.
    # B: shape (n_hpo_terms, n_vocab) — one row per HPO term, binary token presence.
    # A.dot(B.T): shape (n_sct, n_hpo) — entry [i, j] = number of shared tokens
    # between SCT term i and HPO term j. Kept sparse throughout.
    A = sparse.csr_matrix((np.ones(len(rows_a), dtype=np.int32), 
        (rows_a, cols_a)), shape=(len(sct_normalized_terms), n_vocab))
    B = sparse.csr_matrix((np.ones(len(rows_b), dtype=np.int32), 
        (rows_b, cols_b)), shape=(len(hpo_normalized_terms), n_vocab))

    intersection_sparse = A.dot(B.T).tocoo()   # STAYS SPARSE - do not call .toarray()

    rows = intersection_sparse.row
    cols = intersection_sparse.col
    inter_vals = intersection_sparse.data.astype(np.float64)

    sizes_a = np.asarray(A.sum(axis = 1)).ravel()
    sizes_b = np.asarray(B.sum(axis = 1)).ravel()

    # Only compute jaccard for the (sparse) pairs that actually share tokens
    a_sizes = sizes_a[rows]
    b_sizes = sizes_b[cols]
    union_vals = a_sizes + b_sizes - inter_vals

    jaccard_vals = inter_vals / union_vals

    hit_jac_mask = jaccard_vals > threshold_jaccard_similarity

    hit_jaccard_vals = jaccard_vals[hit_jac_mask]

    ids_jaccard_similarity = [hpo_terms_subset.loc[j, 
        hpo_ontology.config.id_column] for j in cols[hit_jac_mask]]
    values_jaccard_similarity = [sct_terms_subset.loc[i, 
        sct_ontology.config.id_column] for i in rows[hit_jac_mask]]

    confidence_jaccard_similarity = hit_jaccard_vals.tolist()

    # Handle the both-empty edge case (jaccard = 1.0) separately — these pairs
    # have intersection 0 so they're absent from the sparse matrix above.
    empty_a = np.where(sizes_a == 0)[0]
    empty_b = np.where(sizes_b == 0)[0]
    if len(empty_a) and len(empty_b) and threshold_jaccard_similarity < 1.0:
        for i in empty_a:
            for j in empty_b:
                ids_jaccard_similarity.append(hpo_terms_subset.loc[j, 
                    hpo_ontology.config.id_column])
                values_jaccard_similarity.append(sct_terms_subset.loc[i, 
                    sct_ontology.config.id_column])
                confidence_jaccard_similarity.append(1.0)

    jaccard_maps_df = pd.DataFrame({
        hpo_ontology.config.id_column           : ids_jaccard_similarity,
        hpo_ontology.config.value_column        : values_jaccard_similarity,
        hpo_ontology.config.attribute_column    : [jaccard_mappings] * 
            len(ids_jaccard_similarity),
        confidence_column                       : confidence_jaccard_similarity,
    })

    l.log("Calculating Jaccard similarities completed.")

    evaluation(
        jaccard_maps_df, 
        graham, 
        "Jaccard similarity mapping only:",
        hpo_ontology
    )

    l.log("Gathering all mappings...")

    mappings = pd.concat([mappings, jaccard_maps_df], ignore_index = True)
    mappings = mappings.drop([hpo_ontology.config.attribute_column], axis=1)
    mappings = mappings.drop_duplicates(ignore_index = True)
    mappings = mappings.reset_index(drop = True)

    l.log("Gathering all mappings completed.")

    evaluation(
        mappings, 
        graham, 
        "HPO to SNOMED CT references, HPO to SNOMED CT over UMLS " \
            "references, shared UMLS CUI references, and Jaccard " \
            "similarity mappings:",
        hpo_ontology
    )

    e = Embedder()

    l.log("Embed HPO terms...")

    embedded_hpo = e.embed(hpo_terms)
    embedded_hpo = embedded_hpo[e.config.model_id[0]]

    l.log("Embed HPO terms completed.")

    l.log("Embed SNOMED CT terms...")

    embedded_sct = e.embed(sct_terms)
    embedded_sct = embedded_sct[e.config.model_id[0]]

    l.log("Embed SNOMED CT terms completed.")

    l.log("Calculating cosine similarity...")

    ids_cosine_similarity           = []
    values_cosine_similarity        = []
    confidence_cosine_similarity    = []

    # Use all available CPU cores
    faiss.omp_set_num_threads(os.cpu_count())

    # Normalize once (skip if embedder already outputs unit vectors)
    def normalize(mat):
        return mat / np.linalg.norm(mat, axis=1, keepdims=True)

    embedded_sct_n = normalize(embedded_sct)
    embedded_hpo_n = normalize(embedded_hpo)

    dim = embedded_hpo.shape[1]

    # Approximate index: cluster HPO embeddings, only search nearby clusters
    # per query.
    # Number of clusters; rule of thumb: ~sqrt(n_vectors) to 4*sqrt(n_vectors)
    nlist = 1000
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    # Needed once, learns the clusters
    index.train(embedded_hpo_n.astype(np.float32))
    index.add(embedded_hpo_n.astype(np.float32))

    # Speed/recall knob; increase if recall is too low
    index.nprobe = 20

    # range_search returns all neighbors above a similarity threshold
    lims, distances, indices = index.range_search(
        embedded_sct_n.astype(np.float32), threshold_cosine_similarity
    )

    # Vectorized result collection (no Python-level per-match loop)
    n_matches_per_query = np.diff(lims).astype(np.int64)
    sct_row_indices = np.repeat(np.arange(len(sct_terms)), n_matches_per_query)

    hpo_ids_arr = hpo_terms_subset[hpo_ontology.config.id_column].to_numpy()
    sct_ids_arr = sct_terms_subset[sct_ontology.config.id_column].to_numpy()

    ids_cosine_similarity = hpo_ids_arr[indices].tolist()
    values_cosine_similarity = sct_ids_arr[sct_row_indices].tolist()
    confidence_cosine_similarity = distances.tolist()

    l.log("Calculating cosine similarity completed.")

    l.log("Create mappings from cosine similarity...")

    embedding_maps_df = pd.DataFrame({
        hpo_ontology.config.id_column           : ids_cosine_similarity,
        hpo_ontology.config.value_column        : values_cosine_similarity,
        hpo_ontology.config.attribute_column    : [embedding_mappings] * len(ids_cosine_similarity),
        confidence_column                       : confidence_cosine_similarity,
    })
    embedding_maps_df[hpo_ontology.config.value_column] = \
        embedding_maps_df[hpo_ontology.config.value_column].astype(str)

    l.log("Create mappings from cosine similarity completed.")

    evaluation(
        embedding_maps_df, 
        graham, 
        "Cosine similarity mappings only:",
        hpo_ontology
    )

    l.log("Gathering all mappings...")

    mappings = pd.concat([
        mappings,
        embedding_maps_df.drop([hpo_ontology.config.attribute_column], axis = 1)
    ], ignore_index = True)
    mappings = mappings.drop_duplicates(ignore_index = True)
    mappings = mappings.reset_index(drop = True)

    l.log("Gathering all mappings completed.")

    evaluation(
        mappings, 
        graham, 
        "HPO to SNOMED CT references, HPO to SNOMED CT over UMLS " \
            "references, shared UMLS CUI references, Jaccard similarity, and " \
            "cosine similarity mappings:",
        hpo_ontology
    )

    # Exact lexical matching (case-insensitive).
    # Non-string and empty terms are already excluded from sct_terms_subset
    # and hpo_terms_subset, so no additional guards are needed here.
    l.log("Calculating exact lexical mappings (case-insensitive)...")

    # Build a lowercase term -> [SCT ID, ...] lookup from all SCT labels
    # and synonyms.
    sct_lower_lookup = {}
    for sct_id, term in zip(
        sct_terms_subset[sct_ontology.config.id_column],
        sct_terms_subset[sct_ontology.config.value_column]
    ):
        key = term.lower().strip()
        sct_lower_lookup.setdefault(key, []).append(sct_id)

    # Match each HPO label/synonym against the SCT lookup.
    ids_exact_lexical    = []
    values_exact_lexical = []

    for hpo_id, term in zip(
        hpo_terms_subset[hpo_ontology.config.id_column],
        hpo_terms_subset[hpo_ontology.config.value_column]
    ):
        key = term.lower().strip()
        for sct_id in sct_lower_lookup.get(key, []):
            ids_exact_lexical.append(hpo_id)
            values_exact_lexical.append(sct_id)

    exact_lexical_maps_df = pd.DataFrame({
        hpo_ontology.config.id_column        : ids_exact_lexical,
        hpo_ontology.config.value_column     : values_exact_lexical,
        hpo_ontology.config.attribute_column : [exact_lexical_mappings] * len(ids_exact_lexical),
        confidence_column                    : 1.0,
    })

    l.log("Calculating exact lexical mappings (case-insensitive) completed.")

    evaluation(
        exact_lexical_maps_df,
        graham,
        "Exact lexical mappings only:",
        hpo_ontology
    )

    l.log("Gathering all mappings...")

    mappings = pd.concat([
        mappings,
        exact_lexical_maps_df.drop([hpo_ontology.config.attribute_column], axis = 1)
    ], ignore_index = True)
    mappings = mappings.drop_duplicates(ignore_index = True)
    mappings = mappings.reset_index(drop = True)

    l.log("Gathering all mappings completed.")

    evaluation(
        mappings,
        graham,
        "HPO to SNOMED CT references, HPO to SNOMED CT over UMLS " \
            "references, shared UMLS CUI references, Jaccard similarity, " \
            "cosine similarity, and exact lexical mappings:",
        hpo_ontology
    )

    # all_mappings_df preserves full provenance: one row per (method, hpo_id,
    # sct_id) triple, with the originating method recorded in the attribute
    # column. This means the same (hpo_id, sct_id) pair can appear multiple
    # times if it was found by more than one method — that is intentional.
    # The MappingSetBuilder script uses the attribute column to apply
    # method-specific acceptance rules and to track which methods corroborate
    # each pair.
    #
    # By contrast, the cumulative `mappings` variable used for intermediate
    # evaluation above is deduplicated and has the attribute column dropped —
    # it is used only for logging and is not written to disk.
    all_mappings_df = pd.concat([
        hpo_sct_maps_df,
        hpo_umls_sct_maps_df,
        umls_hpo_sct_maps_df,
        jaccard_maps_df,
        embedding_maps_df,
        exact_lexical_maps_df
    ], ignore_index = True).reset_index(drop = True)

    writeHugeCSV(all_mappings_df, "./data/output/mapped/mapping.csv")

    l.printHeader("Gathered all mappings from HPO to SNOMED CT finalized.")
    l.printHeader("Results")

    evaluation(
        mappings,
        graham,
        "All generated mappings:",
        hpo_ontology
    )

    l.printHeader("Finished")