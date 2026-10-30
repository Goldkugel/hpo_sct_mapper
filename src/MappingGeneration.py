import sys

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

from logger                 import Logger
from embedder               import Embedder
from normalizer             import Normalizer
from adapter                import writeHugeCSV, labelClass, synonymClass, referenceClass
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
# The three cross-reference strategies (1-3) carry no score of their own and
# get confidence 1.0, as does exact_lexical. Jaccard and embedding mappings
# keep their similarity as confidence. Both thresholds are defined in
# MappingUtils.
#
# Output:
#   all_mappings_df  : One row per (method, hpo_id, sct_id) triple, with the
#                      originating method preserved in the attribute column.
#                      Written to ./data/output/mapped/mapping.csv and used
#                      as input to the MappingSetBuilder script.
#
#   cumulative       : Deduplicated (hpo_id, sct_id) pairs of all methods
#                      without the attribute column, used only for the
#                      evaluations and log summaries; not written to disk.
#
# Each strategy is evaluated against the Graham gold standard on its own and,
# from the second strategy on, cumulatively with all previous strategies, to
# track the recall and precision contribution of each method.
# ==========================================================================

source_abbreviation = "source_abbreviation"

def sourceAbbreviation(entry: str):
    """
    Return the "source_abbreviation" (target vocabulary, e.g. "HPO" or
    "SNOMEDCT_US") stored in a UMLS reference row's additional column,
    which holds a Python-dict-like string, or None if it can't be parsed.
    """
    try:
        parsed = ast.literal_eval(entry)
    except (ValueError, SyntaxError):
        return None
    return parsed.get(source_abbreviation) if isinstance(parsed, dict) else None

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
        l.log(f"New pairs added by {method}: {len(cumulative.index) - before}")
        evaluation(
            cumulative,
            gold,
            f"Evaluation (cumulative: {' + '.join(included)}):",
            adapter
        )

    return cumulative

def logPairs(data: pd.DataFrame, id_col: str) -> str:
    """Short summary of a mapping DataFrame for 'completed' log lines."""
    return f"{len(data.index)} pairs for {data[id_col].nunique()} HPO concepts."

if __name__ == "__main__":

    logging.getLogger("faiss").setLevel(logging.WARNING)

    l = Logger()

    # CPUs available to this process (respects Slurm/cgroup CPU affinity).
    n_cpus = len(os.sched_getaffinity(0))

    l.printHeader("Gather all mappings from HPO to SNOMED CT")

    # ======================================================================
    # 1. Loading data
    # ======================================================================

    l.printHeader("1. Loading data")

    # All three ontologies come in EAV form (id, attribute, value, additional).
    # loadInputs keeps only HPO concepts ("HP:") in hpo and loads the Graham
    # gold standard, which is used for the evaluations only.
    hpo, sct, umls, hpo_ontology, sct_ontology, umls_ontology, graham = \
        loadInputs(separator = ";")

    id_col   = hpo_ontology.config.id_column
    val_col  = hpo_ontology.config.value_column
    attr_col = hpo_ontology.config.attribute_column

    # Build ID sets for fast membership tests. Cross-references found in one
    # ontology that point to a concept not present in the loaded version of the
    # target ontology are treated as invalid and counted but not mapped.
    valid_umls_ids = set(umls[umls_ontology.config.id_column].tolist())
    valid_sct_ids  = set(sct[sct_ontology.config.id_column].tolist())
    valid_hpo_ids  = set(hpo[id_col].tolist())

    l.log(f"Thresholds: Jaccard > {threshold_jaccard_similarity}, " \
        f"cosine > {threshold_cosine_similarity}. CPUs: {n_cpus}.")

    # ======================================================================
    # 2. Collecting cross-references
    # ======================================================================

    l.printHeader("2. Collecting cross-references")

    # Five lookups are built here and used by the strategies below:
    #   umls_hpo_maps : UMLS CUI -> [HPO IDs]   (from UMLS)
    #   umls_sct_maps : UMLS CUI -> [SCT IDs]   (from UMLS)
    #   hpo_umls_maps : HPO ID   -> [UMLS CUIs] (from HPO)
    #   hpo_sct_maps  : HPO ID   -> [SCT IDs]   (from HPO)
    # All IDs carry their prefix ("UMLS:", "HP:", "SNOMEDCT_US:").

    # In UMLS, a reference row links a CUI (id column) to a code of a source
    # vocabulary (value column), e.g. C0036572 -> "HP:0001250" or
    # C0036572 -> "91175000". HPO codes already carry the "HP:" prefix, SCT
    # codes are bare and get the "SNOMEDCT_US:" prefix here.

    l.log("Collecting HPO and SNOMED CT references in UMLS...")

    umls_references = umls[
        umls[umls_ontology.config.attribute_column] == referenceClass
    ].reset_index(drop = True)

    # The additional column holds a Python-dict-like string such as
    # "{'source_abbreviation': 'HPO'}". There are only few distinct values,
    # so each one is parsed once and mapped onto all rows.
    additional = umls_references[umls_ontology.config.additional_column].astype(str)
    parsed = {entry: sourceAbbreviation(entry) for entry in additional.unique()}
    sources = additional.map(parsed)

    umls_ids    = umls_prefix + umls_references[umls_ontology.config.id_column].astype(str)
    umls_values = umls_references[umls_ontology.config.value_column].astype(str)

    # UMLS -> HPO, keyed by prefixed UMLS ID.
    umls_hpo_maps = {}
    invalid_hpo_count = 0
    is_hpo = (sources == "HPO").to_numpy()
    for umls_id, hpo_id in zip(umls_ids[is_hpo], umls_values[is_hpo]):
        if hpo_id in valid_hpo_ids:
            umls_hpo_maps.setdefault(umls_id, []).append(hpo_id)
        else:
            invalid_hpo_count += 1

    # UMLS -> SCT, keyed by prefixed UMLS ID.
    umls_sct_maps = {}
    invalid_sct_count = 0
    is_sct = (sources == "SNOMEDCT_US").to_numpy()
    for umls_id, sct_id in zip(umls_ids[is_sct], sct_prefix + umls_values[is_sct]):
        if sct_id in valid_sct_ids:
            umls_sct_maps.setdefault(umls_id, []).append(sct_id)
        else:
            invalid_sct_count += 1

    l.log("Collecting HPO and SNOMED CT references in UMLS completed. " \
        f"{len(umls_references.index)} references, " \
        f"{len(umls_hpo_maps)} UMLS concepts with HPO references, " \
        f"{len(umls_sct_maps)} with SNOMED CT references. Invalid HPO " \
        f"references: {invalid_hpo_count}. Invalid SNOMED CT references: " \
        f"{invalid_sct_count}.")

    # HPO -> UMLS. HPO stores its xrefs as prefixed values ("UMLS:C0036572"),
    # while the UMLS data uses bare CUIs as IDs, so the prefix is stripped
    # only for the validity check.

    l.log("Collecting UMLS references in HPO...")

    hpo_umls_references = hpo[
        (hpo[attr_col] == referenceClass) &
        (hpo[val_col].str.startswith(umls_prefix, na = False))
    ]

    hpo_umls_maps = {}
    invalid_count = 0
    for hpo_id, umls_id in zip(hpo_umls_references[id_col], hpo_umls_references[val_col]):
        if umls_id.replace(umls_prefix, "", 1) in valid_umls_ids:
            hpo_umls_maps.setdefault(hpo_id, []).append(umls_id)
        else:
            invalid_count += 1

    l.log("Collecting UMLS references in HPO completed. " \
        f"{len(hpo_umls_maps)} HPO concepts with UMLS references. " \
        f"Invalid UMLS references: {invalid_count}.")

    # HPO -> SCT

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

    l.log("Collecting SNOMED CT references in HPO completed. " \
        f"{len(hpo_sct_maps)} HPO concepts with SNOMED CT references. " \
        f"Invalid SNOMED CT references: {invalid_count}.")

    # cumulative: deduplicated (hpo_id, sct_id) pairs of all methods so far,
    # used only for evaluation. included: the methods in cumulative.
    cumulative = None
    included   = []

    # ======================================================================
    # 3. hpo_sct: HPO -> SCT
    # ======================================================================

    l.printHeader(f"3. {hpo_sct_mappings}: HPO -> SNOMED CT")

    l.log(f"Creating {hpo_sct_mappings} mappings...")

    hpo_sct_maps_df = dictToEAVDataFrame(
        hpo_sct_maps,
        hpo_sct_mappings,
        hpo_ontology
    )
    hpo_sct_maps_df[confidence_column] = 1.0
    hpo_sct_maps_df = dedupePairs(hpo_sct_maps_df, id_col, val_col)

    l.log(f"Creating {hpo_sct_mappings} mappings completed. " \
        f"{logPairs(hpo_sct_maps_df, id_col)}")

    cumulative = addMethod(cumulative, included, hpo_sct_maps_df,
        hpo_sct_mappings, graham, hpo_ontology)

    # ======================================================================
    # 4. hpo_umls_sct: HPO -> UMLS -> SCT
    # ======================================================================

    l.printHeader(f"4. {hpo_umls_sct_mappings}: HPO -> UMLS -> SNOMED CT")

    l.log(f"Creating {hpo_umls_sct_mappings} mappings...")

    # For each HPO concept, collect the SCT concepts reachable via every UMLS
    # CUI the HPO concept references. HPO concepts without any reachable SCT
    # concept are left out.
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

    l.log(f"Creating {hpo_umls_sct_mappings} mappings completed. " \
        f"{logPairs(hpo_umls_sct_maps_df, id_col)}")

    cumulative = addMethod(cumulative, included, hpo_umls_sct_maps_df,
        hpo_umls_sct_mappings, graham, hpo_ontology)

    # ======================================================================
    # 5. umls_hpo_sct: HPO <- UMLS -> SCT (shared UMLS CUI)
    # ======================================================================

    l.printHeader(f"5. {umls_hpo_sct_mappings}: HPO <- UMLS -> SNOMED CT")

    l.log(f"Creating {umls_hpo_sct_mappings} mappings...")

    # For each UMLS CUI that lists both HPO and SCT codes, pair every HPO
    # concept with every SCT concept of that CUI. Unlike section 4, this relies
    # on UMLS's records only, so it also finds pairs whose HPO concept has no
    # xref to that CUI (see the cross-check in section 6).
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

    l.log(f"Creating {umls_hpo_sct_mappings} mappings completed. " \
        f"{logPairs(umls_hpo_sct_maps_df, id_col)}")

    cumulative = addMethod(cumulative, included, umls_hpo_sct_maps_df,
        umls_hpo_sct_mappings, graham, hpo_ontology)

    # ======================================================================
    # 6. HPO -> UMLS vs. UMLS -> HPO cross-check (diagnostic only)
    # ======================================================================

    # HPO records outgoing UMLS xrefs (HPO -> UMLS) and UMLS records incoming
    # xrefs from HPO (UMLS -> HPO). In a fully consistent state these two edge
    # sets would be mirrors of each other. The asymmetry logged here does not
    # affect any mappings, but it shows how much the two UMLS-based strategies
    # (sections 4 and 5) can differ.

    l.printHeader("6. Cross-check of HPO -> UMLS and UMLS -> HPO")

    l.log("Comparing HPO -> UMLS and UMLS -> HPO references...")

    # Both edge sets as (hpo_id, prefixed umls_id) pairs.
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

    l.log("Comparing HPO -> UMLS and UMLS -> HPO references completed.")
    l.log(f"References in both directions:   {len(hpo_side & umls_side):7} " \
        f"edges")
    l.log(f"HPO -> UMLS missing in UMLS:      {len(only_in_hpo):7} edges, " \
        f"{len({hpo_id for hpo_id, _ in only_in_hpo})} HPO concepts")
    l.log(f"UMLS -> HPO missing in HPO:       {len(only_in_umls):7} edges, " \
        f"{len({hpo_id for hpo_id, _ in only_in_umls})} HPO concepts")

    # ======================================================================
    # 7. Preparing terms
    # ======================================================================

    l.printHeader("7. Preparing terms")

    # Build term subsets for lexical and embedding methods.
    # Non-string and empty-string terms are filtered out here once,
    # so all downstream methods (Jaccard, embedding, exact lexical)
    # are guaranteed to receive clean string input.
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

    # ID arrays aligned with the term lists: position i of hpo_terms belongs
    # to concept hpo_ids_arr[i]. Every similarity method below works on term
    # positions and maps them back to concept IDs through these arrays, so a
    # concept with several labels/synonyms can match through any of them.
    hpo_ids_arr = hpo_terms_subset[id_col].to_numpy()
    sct_ids_arr = sct_terms_subset[sct_ontology.config.id_column].to_numpy()

    l.log(f"Collecting labels and synonyms completed. {len(hpo_terms)} HPO " \
        f"terms, {len(sct_terms)} SNOMED CT terms.")

    # The normaliser turns each term into a set of tokens (lower-casing,
    # lemmatising, stemming and stop-word removal as set in config.yaml).
    # The token sets are the input of the Jaccard similarity in section 8.
    # n_process is the number of spaCy worker processes; the HPO term list
    # is small enough that more processes would cost more to start than
    # they save.
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

    # ======================================================================
    # 8. jaccard: Jaccard similarity
    # ======================================================================

    l.printHeader(f"8. {jaccard_mappings}: Jaccard similarity")

    l.log(f"Creating {jaccard_mappings} mappings...")

    # Jaccard similarity of two token sets S and H is |S ∩ H| / |S ∪ H|.
    # Comparing 1.5M SCT terms with 54k HPO terms pair by pair in Python
    # would be far too slow, so it is computed with sparse matrices:
    #
    #   1. Every distinct token gets a column number (vocab), shared by SCT
    #      and HPO terms.
    #   2. Each term becomes a row of 0/1 entries over these columns
    #      (A for SCT, B for HPO).
    #   3. A · Bᵀ then holds |S ∩ H| for every pair: the dot product of two
    #      0/1 rows counts the tokens they share. Pairs without a shared
    #      token are 0 and not stored at all, so the result stays small.
    #   4. |S ∪ H| = |S| + |H| - |S ∩ H|, where |S| and |H| are the row sums.
    #
    # With threshold_jaccard_similarity = 0.90, a pair only passes when the
    # token sets are identical, unless both terms together have more than
    # 10 distinct tokens (e.g. 10 of 11 shared tokens gives 0.909).
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

    # One entry per pair sharing at least one token: rows[k] is the SCT term
    # position, cols[k] the HPO term position, inter_vals[k] = |S ∩ H|.
    rows = intersection_sparse.row
    cols = intersection_sparse.col
    inter_vals = intersection_sparse.data.astype(np.float64)

    # Number of distinct tokens per term (|S| per SCT term, |H| per HPO term).
    sizes_a = np.asarray(A.sum(axis = 1)).ravel()
    sizes_b = np.asarray(B.sum(axis = 1)).ravel()

    # Only pairs that share at least one token are in the sparse matrix.
    # Terms that normalise to an empty token set (e.g. only stop words) are
    # therefore never matched: Jaccard is undefined for two empty sets.
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

    l.log(f"Creating {jaccard_mappings} mappings completed. " \
        f"{logPairs(jaccard_maps_df, id_col)} Terms without tokens after " \
        f"normalisation (never matched): {int((sizes_b == 0).sum())} HPO, " \
        f"{int((sizes_a == 0).sum())} SNOMED CT.")

    cumulative = addMethod(cumulative, included, jaccard_maps_df,
        jaccard_mappings, graham, hpo_ontology)

    # ======================================================================
    # 9. embedding: cosine similarity
    # ======================================================================

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

    # The embedder may already return unit vectors (normalization in
    # config.yaml), but normalising here makes the inner product below a
    # cosine similarity either way. FAISS requires float32.
    def normalize(mat):
        """Scale rows to unit length (zero rows stay zero), as float32."""
        norms = np.linalg.norm(mat, axis = 1, keepdims = True)
        norms[norms == 0] = 1
        return (mat / norms).astype(np.float32)

    embedded_sct_n = normalize(embedded_sct)
    embedded_hpo_n = normalize(embedded_hpo)

    # The HPO vectors form the index and every SCT vector is a query. The
    # flat index compares each query with every HPO vector (exact search, no
    # approximation), using batched matrix products; indexing the smaller
    # HPO side keeps the index small. Both directions give the same pairs.
    index = faiss.IndexFlatIP(embedded_hpo_n.shape[1])
    index.add(embedded_hpo_n)

    # range_search returns, for each SCT term, all HPO terms with similarity
    # above the threshold, as flat arrays over all queries:
    #   similarities[k], indices[k] : similarity and HPO term position of
    #                                 result k
    #   lims                        : length n_sct_terms + 1; the results of
    #                                 SCT term i are at positions
    #                                 lims[i] to lims[i+1] - 1
    lims, similarities, indices = index.range_search(
        embedded_sct_n, threshold_cosine_similarity
    )

    # float32 rounding makes the similarity of two identical embeddings
    # come out as e.g. 0.9999997 or 1.0000001 instead of 1.0. Clipping to
    # 1.0 and rounding to 6 decimals gives identical embeddings exactly 1.0,
    # which MappingSetBuilder relies on for its "cosine=1" criterion.
    similarities = np.round(np.minimum(similarities.astype(np.float64), 1.0), 6)

    # Expand lims into the SCT term position of every result: SCT term i is
    # repeated once per result it has (lims[i+1] - lims[i] times).
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

    l.log(f"Creating {embedding_mappings} mappings completed. " \
        f"{logPairs(embedding_maps_df, id_col)}")

    cumulative = addMethod(cumulative, included, embedding_maps_df,
        embedding_mappings, graham, hpo_ontology)

    # ======================================================================
    # 10. exact_lexical: case-insensitive exact match
    # ======================================================================

    l.printHeader(f"10. {exact_lexical_mappings}: exact lexical match")

    l.log(f"Creating {exact_lexical_mappings} mappings...")

    # Unlike Jaccard, this compares the raw strings (only lower-cased and
    # trimmed), not normalised token sets, so it is stricter: word order,
    # stop words and inflections must match too. Most of these pairs are
    # also found by Jaccard or embedding; the separate method records that
    # the strings are identical, which MappingSetBuilder uses as evidence.

    # Build a lowercase term -> [SCT ID, ...] lookup from all SCT labels
    # and synonyms.
    sct_lower_lookup = {}
    for sct_id, term in zip(sct_ids_arr, sct_terms):
        sct_lower_lookup.setdefault(term.lower().strip(), []).append(sct_id)

    # Match each HPO label/synonym against the SCT lookup.
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

    l.log(f"Creating {exact_lexical_mappings} mappings completed. " \
        f"{logPairs(exact_lexical_maps_df, id_col)}")

    cumulative = addMethod(cumulative, included, exact_lexical_maps_df,
        exact_lexical_mappings, graham, hpo_ontology)

    # ======================================================================
    # 11. Writing mappings
    # ======================================================================

    l.printHeader("11. Writing mappings")

    # all_mappings_df preserves full provenance: one row per (method, hpo_id,
    # sct_id) triple, with the originating method recorded in the attribute
    # column. The same (hpo_id, sct_id) pair appears once per method that
    # found it — that is intentional. The MappingSetBuilder script uses the
    # attribute column to apply method-specific acceptance rules and to track
    # which methods corroborate each pair.
    all_mappings_df = pd.concat([
        hpo_sct_maps_df,
        hpo_umls_sct_maps_df,
        umls_hpo_sct_maps_df,
        jaccard_maps_df,
        embedding_maps_df,
        exact_lexical_maps_df
    ], ignore_index = True)

    l.log(f"Rows per method: " + ", ".join(
        f"{method} {count}"
        for method, count in all_mappings_df[attr_col].value_counts().items()
    ) + f". Total: {len(all_mappings_df.index)} rows, " \
        f"{logPairs(cumulative, id_col)}")

    writeHugeCSV(all_mappings_df, mapping_path)

    l.printHeader("Gather all mappings from HPO to SNOMED CT completed.")
