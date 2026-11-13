import sys

sys.dont_write_bytecode = True

from logger                 import Logger
from embedder               import Embedder
from normalizer             import Normalizer
from adapter                import writeHugeCSV, labelClass, synonymClass, referenceClass
from adapter.BaseAdapterUtils import additionalSeparator
from MappingUtils           import exact_lexical_mappings, loadInputs, evaluation, dictToEAVDataFrame, confidence_column, hpo_sct_mappings, hpo_umls_sct_mappings, umls_hpo_sct_mappings, embedding_mappings, jaccard_mappings, threshold_cosine_similarity, threshold_jaccard_similarity, mapping_path, sct_prefix, umls_prefix
from scipy                  import sparse
import pandas               as pd
import numpy                as np
import os
import ast
import faiss
import logging

# ==========================================================================
# MappingGeneration.py
#
# Generates all candidate HPO -> SNOMED CT mappings using six strategies,
# listed in the order they are run:
#
#   1. hpo_sct       : Direct cross-references embedded in the HPO source file.
#   2. hpo_umls_sct  : HPO -> UMLS -> SCT, bridging via HPO's own UMLS xrefs.
#   3. umls_hpo_sct  : HPO <- UMLS -> SCT, bridging via shared UMLS CUI
#                      (UMLS lists both the HPO concept and an SCT concept
#                      under the same CUI).
#   4. jaccard       : Token-level Jaccard similarity on normalised labels and
#                      synonyms, above threshold_jaccard_similarity.
#   5. embedding     : Cosine similarity on embeddings of labels and synonyms
#                      (model set in config.yaml, currently SapBERT), above
#                      threshold_cosine_similarity, using an exact FAISS
#                      inner-product search.
#   6. exact_lexical : Case-insensitive exact string match on labels and
#                      synonyms.
#
# Strategies 1-3 and exact_lexical get confidence 1.0, Jaccard and embedding
# keep their similarity. Each strategy is evaluated on its own and
# cumulatively with the previous ones.
#
# Output: ./data/output/mapped/mapping.csv, one row per (method, hpo_id,
# sct_id), the method in the attribute column.
# ==========================================================================

source_abbreviation = "source_abbreviation"

def sourceAbbreviation(entry: str):
    """
    Source vocabularies (e.g. ["HPO"] or ["SNOMEDCT_US"]) of a UMLS reference
    row's additional column (a dict-like string), or [] if unparsable.
    Joined sources (adapter 1.5.0, e.g. "SNOMEDCT_US, SNOMEDCT_VET") are split.
    """
    try:
        parsed = ast.literal_eval(entry)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, dict) or parsed.get(source_abbreviation) is None:
        return []
    return str(parsed[source_abbreviation]).split(additionalSeparator)

def dedupePairs(data: pd.DataFrame, id_col: str, val_col: str) -> pd.DataFrame:
    """
    Keep one row per (hpo_id, sct_id) pair within a single method, with the
    highest confidence. A pair can be found several times, e.g. through
    several UMLS CUIs or several labels/synonyms of the same concepts.
    """
    return (
        data
        .sort_values(confidence_column, ascending = False, kind = "stable")
        .drop_duplicates(subset = [id_col, val_col])
        .reset_index(drop = True)
    )

def addMethod(
    cumulative: pd.DataFrame,
    included:   list,
    data:       pd.DataFrame,
    method:     str,
    gold:       pd.DataFrame,
    adapter
) -> pd.DataFrame:
    """
    Evaluate one method on its own, add its pairs to the cumulative set and,
    from the second method on, evaluate the cumulative set. Returns the new
    cumulative set; `included` is extended with `method` in place.
    """
    l = Logger()
    id_col  = adapter.config.id_column
    val_col = adapter.config.value_column

    evaluation(data, gold, f"Evaluation ({method} only):", adapter)

    pairs = data[[id_col, val_col]]
    before = 0 if cumulative is None else len(cumulative.index)
    cumulative = pairs if cumulative is None else pd.concat(
        [cumulative, pairs], ignore_index = True
    )
    cumulative = cumulative.drop_duplicates(ignore_index = True)
    included.append(method)

    if len(included) > 1:
        l.log(f"New pairs added by {method}: {len(cumulative.index) - before}.")
        evaluation(
            cumulative,
            gold,
            f"Evaluation (cumulative: {' + '.join(included)}):",
            adapter
        )

    return cumulative

def logPairs(data: pd.DataFrame, id_col: str) -> str:
    """Short summary of a mapping DataFrame for result log rows."""
    return f"{len(data.index)} pairs for {data[id_col].nunique()} HPO concepts."

if __name__ == "__main__":

    logging.getLogger("faiss").setLevel(logging.WARNING)

    l = Logger()

    # CPUs available to this process (respects Slurm/cgroup CPU affinity).
    n_cpus = len(os.sched_getaffinity(0))

    l.printHeader("Gather all mappings from HPO to SNOMED CT")

    l.printHeader("1. Loading data")

    # The gold standard is used for the evaluations only.
    hpo, sct, umls, hpo_ontology, sct_ontology, umls_ontology, graham = \
        loadInputs(separator = ";")

    id_col   = hpo_ontology.config.id_column
    val_col  = hpo_ontology.config.value_column
    attr_col = hpo_ontology.config.attribute_column

    # Cross-references to concepts missing from the loaded target ontology
    # are counted as invalid and not mapped.
    valid_umls_ids = set(umls[umls_ontology.config.id_column].tolist())
    valid_sct_ids  = set(sct[sct_ontology.config.id_column].tolist())
    valid_hpo_ids  = set(hpo[id_col].tolist())

    l.log(f"Thresholds: Jaccard > {threshold_jaccard_similarity}, " \
        f"cosine > {threshold_cosine_similarity}.")
    l.log(f"CPUs: {n_cpus}.")

    l.printHeader("2. Collecting cross-references")

    # Lookups umls_hpo_maps, umls_sct_maps (UMLS CUI -> IDs, from UMLS) and
    # hpo_umls_maps, hpo_sct_maps (HPO ID -> IDs, from HPO), all IDs prefixed.
    # UMLS reference rows are CUI -> code, e.g. "UMLS:C0036572" -> "HP:0001250"
    # or "91175000"; the adapter has already prefixed the CUIs, bare SCT codes
    # get the "SNOMEDCT_US:" prefix here.
    l.log("Collecting HPO and SNOMED CT references in UMLS...")

    umls_references = umls[
        umls[umls_ontology.config.attribute_column] == referenceClass
    ].reset_index(drop = True)

    # Few distinct values, so each is parsed once.
    additional = umls_references[umls_ontology.config.additional_column].astype(str)
    parsed = {entry: sourceAbbreviation(entry) for entry in additional.unique()}
    sources = additional.map(parsed)

    umls_ids    = umls_references[umls_ontology.config.id_column].astype(str)
    umls_values = umls_references[umls_ontology.config.value_column].astype(str)

    umls_hpo_maps = {}
    invalid_hpo_count = 0
    is_hpo = sources.map(lambda s: "HPO" in s).to_numpy()
    for umls_id, hpo_id in zip(umls_ids[is_hpo], umls_values[is_hpo]):
        if hpo_id in valid_hpo_ids:
            umls_hpo_maps.setdefault(umls_id, []).append(hpo_id)
        else:
            invalid_hpo_count += 1

    umls_sct_maps = {}
    invalid_sct_count = 0
    is_sct = sources.map(lambda s: "SNOMEDCT_US" in s).to_numpy()
    for umls_id, sct_id in zip(umls_ids[is_sct], sct_prefix + umls_values[is_sct]):
        if sct_id in valid_sct_ids:
            umls_sct_maps.setdefault(umls_id, []).append(sct_id)
        else:
            invalid_sct_count += 1

    l.log(f"References in UMLS: {len(umls_references.index)}.")
    l.log(f"UMLS concepts with HPO references: {len(umls_hpo_maps)}, " \
        f"with SNOMED CT references: {len(umls_sct_maps)}.")
    l.log(f"Invalid HPO references in UMLS: {invalid_hpo_count}.")
    l.log(f"Invalid SNOMED CT references in UMLS: {invalid_sct_count}.")
    l.log("Collecting HPO and SNOMED CT references in UMLS completed.")

    # HPO xrefs and the UMLS IDs of the adapter carry the same prefix
    # ("UMLS:C0036572"), so they are compared as they are.
    l.log("Collecting UMLS references in HPO...")

    hpo_umls_references = hpo[
        (hpo[attr_col] == referenceClass) &
        (hpo[val_col].str.startswith(umls_prefix, na = False))
    ]

    hpo_umls_maps = {}
    invalid_count = 0
    for hpo_id, umls_id in zip(hpo_umls_references[id_col], hpo_umls_references[val_col]):
        if umls_id in valid_umls_ids:
            hpo_umls_maps.setdefault(hpo_id, []).append(umls_id)
        else:
            invalid_count += 1

    l.log(f"HPO concepts with UMLS references: {len(hpo_umls_maps)}.")
    l.log(f"Invalid UMLS references in HPO: {invalid_count}.")
    l.log("Collecting UMLS references in HPO completed.")

    l.log("Collecting SNOMED CT references in HPO...")

    hpo_sct_references = hpo[
        (hpo[attr_col] == referenceClass) &
        (hpo[val_col].str.startswith(sct_prefix, na = False))
    ]

    hpo_sct_maps = {}
    invalid_count = 0
    for hpo_id, sct_id in zip(hpo_sct_references[id_col], hpo_sct_references[val_col]):
        if sct_id in valid_sct_ids:
            hpo_sct_maps.setdefault(hpo_id, []).append(sct_id)
        else:
            invalid_count += 1

    l.log(f"HPO concepts with SNOMED CT references: {len(hpo_sct_maps)}.")
    l.log(f"Invalid SNOMED CT references in HPO: {invalid_count}.")
    l.log("Collecting SNOMED CT references in HPO completed.")

    # cumulative: unique pairs of all methods so far, for the evaluations
    # only (not written). included: the methods in cumulative.
    cumulative = None
    included   = []

    l.printHeader(f"3. {hpo_sct_mappings}: HPO -> SNOMED CT")

    l.log(f"Creating {hpo_sct_mappings} mappings...")

    hpo_sct_maps_df = dictToEAVDataFrame(
        hpo_sct_maps,
        hpo_sct_mappings,
        hpo_ontology
    )
    hpo_sct_maps_df[confidence_column] = 1.0
    hpo_sct_maps_df = dedupePairs(hpo_sct_maps_df, id_col, val_col)

    l.log(f"{hpo_sct_mappings} mappings: {logPairs(hpo_sct_maps_df, id_col)}")
    l.log(f"Creating {hpo_sct_mappings} mappings completed.")

    cumulative = addMethod(cumulative, included, hpo_sct_maps_df,
        hpo_sct_mappings, graham, hpo_ontology)

    l.printHeader(f"4. {hpo_umls_sct_mappings}: HPO -> UMLS -> SNOMED CT")

    l.log(f"Creating {hpo_umls_sct_mappings} mappings...")

    hpo_umls_sct_maps = {}
    for hpo_id, hpo_umls_ids in hpo_umls_maps.items():
        for umls_id in hpo_umls_ids:
            if umls_id in umls_sct_maps:
                hpo_umls_sct_maps.setdefault(hpo_id, []).extend(umls_sct_maps[umls_id])

    hpo_umls_sct_maps_df = dictToEAVDataFrame(
        hpo_umls_sct_maps,
        hpo_umls_sct_mappings,
        hpo_ontology
    )
    hpo_umls_sct_maps_df[confidence_column] = 1.0
    hpo_umls_sct_maps_df = dedupePairs(hpo_umls_sct_maps_df, id_col, val_col)

    l.log(f"{hpo_umls_sct_mappings} mappings: {logPairs(hpo_umls_sct_maps_df, id_col)}")
    l.log(f"Creating {hpo_umls_sct_mappings} mappings completed.")

    cumulative = addMethod(cumulative, included, hpo_umls_sct_maps_df,
        hpo_umls_sct_mappings, graham, hpo_ontology)

    l.printHeader(f"5. {umls_hpo_sct_mappings}: HPO <- UMLS -> SNOMED CT")

    l.log(f"Creating {umls_hpo_sct_mappings} mappings...")

    # Unlike section 4 this uses UMLS's records only, so it also finds pairs
    # whose HPO concept has no xref to the CUI (see section 6).
    umls_hpo_sct_maps = {}
    for umls_id, hpo_ids in umls_hpo_maps.items():
        if umls_id in umls_sct_maps:
            for hpo_id in hpo_ids:
                umls_hpo_sct_maps.setdefault(hpo_id, []).extend(umls_sct_maps[umls_id])

    umls_hpo_sct_maps_df = dictToEAVDataFrame(
        umls_hpo_sct_maps,
        umls_hpo_sct_mappings,
        hpo_ontology
    )
    umls_hpo_sct_maps_df[confidence_column] = 1.0
    umls_hpo_sct_maps_df = dedupePairs(umls_hpo_sct_maps_df, id_col, val_col)

    l.log(f"{umls_hpo_sct_mappings} mappings: {logPairs(umls_hpo_sct_maps_df, id_col)}")
    l.log(f"Creating {umls_hpo_sct_mappings} mappings completed.")

    cumulative = addMethod(cumulative, included, umls_hpo_sct_maps_df,
        umls_hpo_sct_mappings, graham, hpo_ontology)

    # Section 6 is diagnostic only: in a consistent state the HPO -> UMLS and
    # UMLS -> HPO edges would mirror each other; the asymmetry shows how much
    # sections 4 and 5 can differ.
    l.printHeader("6. Cross-check of HPO -> UMLS and UMLS -> HPO")

    l.log("Comparing HPO -> UMLS and UMLS -> HPO references...")

    hpo_side = {
        (hpo_id, umls_id)
        for hpo_id, hpo_umls_ids in hpo_umls_maps.items()
        for umls_id in hpo_umls_ids
    }
    umls_side = {
        (hpo_id, umls_id)
        for umls_id, hpo_ids in umls_hpo_maps.items()
        for hpo_id in hpo_ids
    }

    only_in_hpo  = hpo_side - umls_side
    only_in_umls = umls_side - hpo_side

    l.log(f"References in both directions:   {len(hpo_side & umls_side):7} " \
        f"edges")
    l.log(f"HPO -> UMLS missing in UMLS:      {len(only_in_hpo):7} edges, " \
        f"{len({hpo_id for hpo_id, _ in only_in_hpo})} HPO concepts")
    l.log(f"UMLS -> HPO missing in HPO:       {len(only_in_umls):7} edges, " \
        f"{len({hpo_id for hpo_id, _ in only_in_umls})} HPO concepts")
    l.log("Comparing HPO -> UMLS and UMLS -> HPO references completed.")

    l.printHeader("7. Preparing terms")

    l.log("Collecting labels and synonyms...")

    def termSubset(data: pd.DataFrame, adapter) -> pd.DataFrame:
        """Label and synonym rows whose value is a non-empty string."""
        subset = data[data[adapter.config.attribute_column].isin([labelClass, synonymClass])]
        subset = subset[subset[adapter.config.value_column].apply(
            lambda x: isinstance(x, str) and bool(x.strip())
        )]
        return subset.reset_index(drop = True)

    hpo_terms_subset = termSubset(hpo, hpo_ontology)
    sct_terms_subset = termSubset(sct, sct_ontology)

    hpo_terms = hpo_terms_subset[val_col].tolist()
    sct_terms = sct_terms_subset[sct_ontology.config.value_column].tolist()

    # Aligned with the term lists: the similarity methods work on term
    # positions and map them back to concept IDs through these arrays.
    hpo_ids_arr = hpo_terms_subset[id_col].to_numpy()
    sct_ids_arr = sct_terms_subset[sct_ontology.config.id_column].to_numpy()

    l.log(f"Terms: {len(hpo_terms)} HPO, {len(sct_terms)} SNOMED CT.")
    l.log("Collecting labels and synonyms completed.")

    # Token sets for the Jaccard similarity (normalisation set in
    # config.yaml). The HPO list is small: more worker processes would cost
    # more to start than they save.
    n = Normalizer()

    l.log("Normalizing SNOMED CT terms...")

    sct_normalized_terms = n.normalize_batch(
        sct_terms,
        batch_size = 15000,
        n_process = n_cpus
    )

    l.log("Normalizing SNOMED CT terms completed.")

    l.log("Normalizing HPO terms...")

    hpo_normalized_terms = n.normalize_batch(
        hpo_terms,
        batch_size = 15000,
        n_process = min(4, n_cpus)
    )

    l.log("Normalizing HPO terms completed.")

    l.printHeader(f"8. {jaccard_mappings}: Jaccard similarity")

    l.log(f"Creating {jaccard_mappings} mappings...")

    # Jaccard |S ∩ H| / |S ∪ H| for all SCT x HPO terms via sparse 0/1
    # token matrices A (SCT) and B (HPO): A · Bᵀ holds |S ∩ H| for pairs
    # sharing a token (others are not stored), |S ∪ H| = |S| + |H| - |S ∩ H|.
    # With the 0.90 threshold only identical token sets pass, unless the
    # union has more than 10 tokens (10 of 11 gives 0.909).
    vocab = {}

    def encode(term_lists):
        """
        (row, col) indices of a sparse binary term x token matrix. vocab is
        shared, so SCT and HPO terms use the same columns.
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

    A = sparse.csr_matrix((np.ones(len(rows_a), dtype=np.int32),
        (rows_a, cols_a)), shape=(len(sct_normalized_terms), n_vocab))
    B = sparse.csr_matrix((np.ones(len(rows_b), dtype=np.int32),
        (rows_b, cols_b)), shape=(len(hpo_normalized_terms), n_vocab))

    intersection_sparse = A.dot(B.T).tocoo()   # STAYS SPARSE - do not call .toarray()

    # rows: SCT term positions, cols: HPO term positions.
    rows = intersection_sparse.row
    cols = intersection_sparse.col
    inter_vals = intersection_sparse.data.astype(np.float64)

    sizes_a = np.asarray(A.sum(axis = 1)).ravel()
    sizes_b = np.asarray(B.sum(axis = 1)).ravel()

    # Terms with an empty token set (e.g. only stop words) are never matched.
    union_vals = sizes_a[rows] + sizes_b[cols] - inter_vals
    jaccard_vals = inter_vals / union_vals

    hit_jac_mask = jaccard_vals > threshold_jaccard_similarity

    jaccard_maps_df = pd.DataFrame({
        id_col              : hpo_ids_arr[cols[hit_jac_mask]],
        val_col             : sct_ids_arr[rows[hit_jac_mask]],
        attr_col            : jaccard_mappings,
        confidence_column   : jaccard_vals[hit_jac_mask],
    })
    jaccard_maps_df = dedupePairs(jaccard_maps_df, id_col, val_col)

    l.log(f"{jaccard_mappings} mappings: {logPairs(jaccard_maps_df, id_col)}")
    l.log(f"Terms without tokens after normalisation (never matched): " \
        f"{int((sizes_b == 0).sum())} HPO, {int((sizes_a == 0).sum())} SNOMED CT.")
    l.log(f"Creating {jaccard_mappings} mappings completed.")

    cumulative = addMethod(cumulative, included, jaccard_maps_df,
        jaccard_mappings, graham, hpo_ontology)

    l.printHeader(f"9. {embedding_mappings}: cosine similarity")

    e = Embedder()

    l.log("Embedding HPO terms...")

    embedded_hpo = e.embed(hpo_terms)
    embedded_hpo = embedded_hpo[e.config.model_id[0]]

    l.log("Embedding HPO terms completed.")

    l.log("Embedding SNOMED CT terms...")

    embedded_sct = e.embed(sct_terms)
    embedded_sct = embedded_sct[e.config.model_id[0]]

    l.log("Embedding SNOMED CT terms completed.")

    l.log(f"Creating {embedding_mappings} mappings...")

    faiss.omp_set_num_threads(n_cpus)

    # Normalised here so the inner product is a cosine whatever config.yaml
    # sets; FAISS requires float32.
    def normalize(mat):
        """Scale rows to unit length (zero rows stay zero), as float32."""
        norms = np.linalg.norm(mat, axis = 1, keepdims = True)
        norms[norms == 0] = 1
        return (mat / norms).astype(np.float32)

    embedded_sct_n = normalize(embedded_sct)
    embedded_hpo_n = normalize(embedded_hpo)

    # Exact search; the smaller HPO side is indexed, SCT vectors are queries.
    index = faiss.IndexFlatIP(embedded_hpo_n.shape[1])
    index.add(embedded_hpo_n)

    # Flat results over all queries; those of SCT term i are at
    # lims[i]:lims[i+1].
    lims, similarities, indices = index.range_search(
        embedded_sct_n, threshold_cosine_similarity
    )

    # Identical embeddings give e.g. 0.9999997 or 1.0000001 in float32;
    # clipping and rounding make them exactly 1.0, so cosine=1 is counted
    # and compared exactly (printCounts, MappingSetBuilder).
    similarities = np.round(np.minimum(similarities.astype(np.float64), 1.0), 6)

    # SCT term position of every result.
    sct_row_indices = np.repeat(
        np.arange(len(sct_terms)),
        np.diff(lims).astype(np.int64)
    )

    embedding_maps_df = pd.DataFrame({
        id_col              : hpo_ids_arr[indices],
        val_col             : sct_ids_arr[sct_row_indices].astype(str),
        attr_col            : embedding_mappings,
        confidence_column   : similarities,
    })
    embedding_maps_df = dedupePairs(embedding_maps_df, id_col, val_col)

    l.log(f"{embedding_mappings} mappings: {logPairs(embedding_maps_df, id_col)}")
    l.log(f"Creating {embedding_mappings} mappings completed.")

    cumulative = addMethod(cumulative, included, embedding_maps_df,
        embedding_mappings, graham, hpo_ontology)

    l.printHeader(f"10. {exact_lexical_mappings}: exact lexical match")

    l.log(f"Creating {exact_lexical_mappings} mappings...")

    # Stricter than Jaccard (raw strings, only lower-cased and trimmed);
    # MappingSetBuilder uses identical strings as evidence of its own.
    sct_lower_lookup = {}
    for sct_id, term in zip(sct_ids_arr, sct_terms):
        sct_lower_lookup.setdefault(term.lower().strip(), []).append(sct_id)

    ids_exact_lexical    = []
    values_exact_lexical = []

    for hpo_id, term in zip(hpo_ids_arr, hpo_terms):
        for sct_id in sct_lower_lookup.get(term.lower().strip(), []):
            ids_exact_lexical.append(hpo_id)
            values_exact_lexical.append(sct_id)

    exact_lexical_maps_df = pd.DataFrame({
        id_col              : ids_exact_lexical,
        val_col             : values_exact_lexical,
        attr_col            : exact_lexical_mappings,
        confidence_column   : 1.0,
    })
    exact_lexical_maps_df = dedupePairs(exact_lexical_maps_df, id_col, val_col)

    l.log(f"{exact_lexical_mappings} mappings: {logPairs(exact_lexical_maps_df, id_col)}")
    l.log(f"Creating {exact_lexical_mappings} mappings completed.")

    cumulative = addMethod(cumulative, included, exact_lexical_maps_df,
        exact_lexical_mappings, graham, hpo_ontology)

    l.printHeader("11. Writing mappings")

    # A pair appears once per method that found it, on purpose:
    # MappingSetBuilder's rules depend on the methods.
    all_mappings_df = pd.concat([
        hpo_sct_maps_df,
        hpo_umls_sct_maps_df,
        umls_hpo_sct_maps_df,
        jaccard_maps_df,
        embedding_maps_df,
        exact_lexical_maps_df
    ], ignore_index = True)

    l.log("Rows per method:")
    for method, count in all_mappings_df[attr_col].value_counts().items():
        l.log(f"  {method}: {count}")
    l.log(f"Total: {len(all_mappings_df.index)} rows, {logPairs(cumulative, id_col)}")

    writeHugeCSV(all_mappings_df, mapping_path)

    l.printHeader("Gather all mappings from HPO to SNOMED CT completed.")
