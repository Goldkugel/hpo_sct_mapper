from adapter                import BaseAdapter
from adapter                import HPOAdapter, SCTAdapter, UMLSAdapter
from adapter                import exactSynonymClass, semanticClass, childrenClass
from logger                 import Logger
from collections            import defaultdict
import pandas               as pd
import numpy                as np
import re
import ast
import os

# Column names of the gold standard (loadGold) and of the validity code
# added by MappingSetBuilder. The mapping files themselves use the
# adapters' EAV column names (adapter.config.id_column etc.).
sct_id_column           = "sct_id"
hpo_id_column           = "hpo_id"
validity_column         = "validity"
confidence_column       = "confidence"

# Mapping methods, stored in the attribute column of the mapping files.
hpo_sct_mappings        = "hpo_sct"
hpo_umls_sct_mappings   = "hpo_umls_sct"
umls_hpo_sct_mappings   = "umls_hpo_sct"
embedding_mappings      = "embedding"
jaccard_mappings        = "jaccard"
exact_lexical_mappings  = "exact_lexical"
llm_mappings            = "llm"

# All mapping methods, and the similarity-based subset.
ALL_METHODS = [
    hpo_sct_mappings,
    hpo_umls_sct_mappings,
    umls_hpo_sct_mappings,
    jaccard_mappings,
    embedding_mappings,
    exact_lexical_mappings,
    llm_mappings,
]
SIM_METHODS = [jaccard_mappings, embedding_mappings]

# Concept ID prefixes.
hpo_prefix = "HP:"
sct_prefix = "SNOMEDCT_US:"
umls_prefix = "UMLS:"

# Input and output files shared between the pipeline scripts.
gold_path            = "./data/input/graham/mapping_graham.csv"
mapping_path         = "./data/output/mapped/mapping.csv"
filtered_path        = "./data/output/filtered/filtered.csv"
llm_filtered_path    = "./data/output/filtered/llm_filtered.csv"
llm_review_path      = "./data/output/filtered/llm_review.csv"
evaluation_path      = "./data/output/evaluation/evaluation_final.csv"
no_gold_path         = "./data/output/evaluation/accepted_without_gold.csv"
llm_disagreements_path = "./data/output/evaluation/llm_disagreements.csv"

# ------------------------------------------------------------------
# Validity codes written by MappingSetBuilder (and read by MappingLLM
# and MappingErrorAnalysis). Rows not reached by any step keep
# validity=NA and are written too: MappingErrorAnalysis uses them as the
# universe of generated pairs.
# ------------------------------------------------------------------
validity_domain_mismatch          = 0   # step 1: SCT target outside acceptable domain
validity_anchor_reversal          = 1   # step 2 failure: hierarchy reversal, anchor candidate
validity_exact_lexical_reversal   = 2   # step 2b failure: hierarchy reversal, exact lexical
validity_hpo_xref_reversal        = 3   # step 2c failure: hierarchy reversal, HPO xref
validity_umls_bridge_reversal     = 4   # step 3 failure: hierarchy reversal, HPO -> UMLS -> SCT
validity_umls_cui_reversal        = 5   # step 4 failure: hierarchy reversal, shared UMLS CUI
validity_jaccard_cosine_reversal  = 6   # step 5 failure: hierarchy reversal, Jaccard=1 AND cosine=1
validity_jaccard_only_reversal    = 7   # step 6 failure: hierarchy reversal, Jaccard=1 only
validity_best_candidate           = 8   # step 7 fallback: best candidate, for LLM review
validity_last_resort              = 9   # step 8 last resort: highest cosine candidate
validity_accepted                 = 10  # accepted

# Hierarchy-reversal failures of steps 2–6.
validity_hierarchy_reversals = [
    validity_anchor_reversal,
    validity_exact_lexical_reversal,
    validity_hpo_xref_reversal,
    validity_umls_bridge_reversal,
    validity_umls_cui_reversal,
    validity_jaccard_cosine_reversal,
    validity_jaccard_only_reversal,
]
# Everything except domain mismatches, i.e. what the LLM reviews.
validity_reviewable = validity_hierarchy_reversals + [
    validity_best_candidate,
    validity_last_resort,
    validity_accepted,
]
# Every code that is written to the output.
validity_all = [validity_domain_mismatch] + validity_reviewable

# Human-readable labels for the validity codes.
VALIDITY_LABELS = {
    validity_domain_mismatch:         "domain mismatch",
    validity_anchor_reversal:         "step 2 failure — hierarchy reversal anchor",
    validity_exact_lexical_reversal:  "step 2b failure — hierarchy reversal exact lexical",
    validity_hpo_xref_reversal:       "step 2c failure — hierarchy reversal HPO xref",
    validity_umls_bridge_reversal:    "step 3 failure — hierarchy reversal UMLS bridge",
    validity_umls_cui_reversal:       "step 4 failure — hierarchy reversal UMLS CUI",
    validity_jaccard_cosine_reversal: "step 5 failure — hierarchy reversal Jaccard+cosine",
    validity_jaccard_only_reversal:   "step 6 failure — hierarchy reversal Jaccard only",
    validity_best_candidate:          "step 7 fallback — best candidate",
    validity_last_resort:             "step 8 last resort",
    validity_accepted:                "accepted",
}

# Validity -> review reason label passed to the LLM prompt.
# Code 0 (domain mismatch) is excluded from review entirely.
# Code 10 (already accepted) is included so the LLM can act as a
# judge over the full set of mappings for each HPO concept.
REVIEW_REASON = {
    validity_anchor_reversal:         "HIERARCHY_REVERSAL_ANCHOR",
    validity_exact_lexical_reversal:  "HIERARCHY_REVERSAL_EXACT_LEXICAL",
    validity_hpo_xref_reversal:       "HIERARCHY_REVERSAL_HPO_XREF",
    validity_umls_bridge_reversal:    "HIERARCHY_REVERSAL_UMLS_BRIDGE",
    validity_umls_cui_reversal:       "HIERARCHY_REVERSAL_UMLS_CUI",
    validity_jaccard_cosine_reversal: "HIERARCHY_REVERSAL_JACCARD_COSINE",
    validity_jaccard_only_reversal:   "HIERARCHY_REVERSAL_JACCARD_ONLY",
    validity_best_candidate:          "BEST_CANDIDATE_FALLBACK",
    validity_last_resort:             "LAST_RESORT",
    validity_accepted:                "ACCEPTED",
}

# Minimum similarity for MappingGeneration to keep a candidate pair
# (strictly greater than the threshold).
threshold_cosine_similarity     = 0.80
threshold_jaccard_similarity    = 0.95

# Confidence thresholds for printCounts: num_points evenly spaced steps
# from 1.0 down to the cosine threshold, plus 0.99 to separate exact from
# near-exact matches. Result: [1.0, 0.99, 0.95, 0.9, 0.85, 0.8].
num_points = 5
similarity_steps = np.linspace(1.0, threshold_cosine_similarity,
    num = num_points).tolist()
similarity_steps.append(0.99)
# Keep the steps in descending order: printCounts(diff=True) builds its
# buckets from consecutive thresholds and needs them sorted.
similarity_steps = sorted(set(similarity_steps), reverse=True)

# SCT semantic tags (the last parenthesised part of a fully specified
# name) that are acceptable targets for HPO phenotype concepts. ""
# accepts a label without any tag.
acceptableDomains = [
    "disorder",
    "finding",
    "morphologic abnormality",
    ""
]

def isSCTDomainValid(terms: list[str]) -> bool:
    """
    True if any of the SCT concept's labels has an acceptable semantic tag
    (see acceptableDomains). Concepts without labels are not valid.
    """
    return bool(terms) and any(
        extract_snomed_domain(term) in acceptableDomains
        for term in terms if isinstance(term, str)
    )

def getRepresentativeLabel(terms: list) -> str:
    """
    Readable label of an SCT concept for the audit trail: the first label
    with an acceptable semantic tag, else the first label, else None.
    """
    if not terms:
        return None
    for term in terms:
        if isinstance(term, str) and extract_snomed_domain(term) in acceptableDomains:
            return term
    return terms[0]

def dictToEAVDataFrame(data: dict, attribute: str, adapter: BaseAdapter) -> pd.DataFrame:
    """
    Turn a {concept id: [values]} dict into EAV rows (id, attribute, value),
    one row per value, all with the given attribute.
    """
    ids = []
    values = []

    for key in data.keys():
        for element in data[key]:
            ids.append(key)
            values.append(element)

    return pd.DataFrame({
        adapter.config.id_column        : ids,
        adapter.config.attribute_column : [attribute] * len(ids),
        adapter.config.value_column     : values
    })

def is_exact_synonym(entry) -> bool:
    """
    True if an EAV row's additional-column entry marks an exact synonym,
    or carries no synonym type at all (including an empty or missing
    entry). Entries may arrive as dicts or as stringified dicts (as pandas
    reads them from CSV).

    HPO synonyms are typed (exact, broad, narrow, related); only exact
    synonyms name the same phenotype and are used for matching. Applied to
    all HPO rows: labels, references, hierarchy rows etc. carry no
    synonym type ("{}") and are kept.
    """
    if entry is None or (isinstance(entry, float) and np.isnan(entry)):
        return True
    if isinstance(entry, str):
        try:
            entry = ast.literal_eval(entry)
        except (ValueError, SyntaxError):
            return False
    if isinstance(entry, dict):
        if semanticClass in entry.keys():
            return entry.get(semanticClass) == exactSynonymClass
        else:
            return True
    return False

def keepHPOConcepts(data: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Keep only rows whose id_col is an HPO concept (prefix "HP:")."""
    return data[data[id_col].str.startswith(hpo_prefix, na=False)].reset_index(drop=True)

def buildAttributeLookup(data: pd.DataFrame, adapter: BaseAdapter, attribute: str) -> dict:
    """
    concept id -> [values] for all EAV rows of the given attribute,
    e.g. labelClass for labels or childrenClass for direct children.
    """
    rows = data[data[adapter.config.attribute_column] == attribute]
    return (
        rows.groupby(adapter.config.id_column)[adapter.config.value_column]
        .apply(list)
        .to_dict()
    )

# ------------------------------------------------------------------
# Hierarchy helpers. The adapters write hierarchy rows with
# attribute=childrenClass as (id = parent, value = child).
# ------------------------------------------------------------------
def build_child_map(data: pd.DataFrame, adapter: BaseAdapter) -> dict:
    """
    Direct hierarchy lookup: parent -> set of direct children.
    make_descendant_lookup walks this map downwards.
    """
    rows = data[data[adapter.config.attribute_column] == childrenClass]
    child_map = defaultdict(set)
    for parent, child in zip(rows[adapter.config.id_column], rows[adapter.config.value_column]):
        child_map[parent].add(child)
    return child_map

def build_parent_map(data: pd.DataFrame, adapter: BaseAdapter) -> dict:
    """
    Direct hierarchy lookup: child -> set of direct parents.
    """
    rows = data[data[adapter.config.attribute_column] == childrenClass]
    parent_map = defaultdict(set)
    for parent, child in zip(rows[adapter.config.id_column], rows[adapter.config.value_column]):
        parent_map[child].add(parent)
    return parent_map

def make_descendant_lookup(child_map: dict):
    """
    Returns a memoized function descendants(node) -> frozenset of ALL nodes
    transitively reachable from `node` through child_map. Cycle-safe.
    Computed lazily per node and cached.

    Passed a parent -> children map (build_child_map) it returns
    descendants; passed a child -> parents map (build_parent_map) it
    returns ancestors.

    The walk is an iterative depth-first search: `stack` holds nodes still
    to expand, `seen` every node reached so far. A node is pushed only the
    first time it is reached, so concepts with several parents (common in
    SNOMED CT and HPO) are expanded once, and a cycle cannot loop forever.
    The start node itself is not in the result (unless a cycle leads back
    to it). Each node's result is cached, but a walk does not reuse the
    cached results of the nodes it passes.
    """
    cache: dict = {}

    def descendants(node):
        if node in cache:
            return cache[node]
        seen = set()
        stack = [node]
        while stack:
            current = stack.pop()
            for child in child_map.get(current, ()):
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        cache[node] = frozenset(seen)
        return cache[node]

    return descendants

# ------------------------------------------------------------------
# (hpo_id, sct_id) pair helpers.
# ------------------------------------------------------------------
def pairSeries(data: pd.DataFrame, id_col: str, val_col: str) -> pd.Series:
    """Series of (id, value) tuples aligned to data.index."""
    return pd.Series(list(zip(data[id_col], data[val_col])), index=data.index)

def inGoldSeries(data: pd.DataFrame, gold_pairs: set, id_col: str, val_col: str) -> pd.Series:
    """
    Boolean Series: True when the (id_col, val_col) pair is in gold_pairs.
    The pairs are built once and looked up by hash, instead of filtering
    the data per gold pair.
    """
    return pairSeries(data, id_col, val_col).isin(gold_pairs)

def loadOntologies(
    separator: str = ";"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, HPOAdapter, SCTAdapter, UMLSAdapter]:
    """
    Loads, processes, and exports HPO, UMLS, and SNOMED CT ontology data.

    Runs the adapter of each ontology to (re)generate its EAV CSV file, then
    reads the three CSV files into DataFrames. The adapters already format
    the IDs (e.g. the "SNOMEDCT_US:" prefix); here only non-exact HPO
    synonyms are removed and SNOMED CT values are read as strings.

    Parameters:
        separator (str): Delimiter used for reading output CSV files. Defaults to ';'.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, HPOAdapter, SCTAdapter, UMLSAdapter]:
            The loaded (hpo, sct, umls) DataFrames alongside the adapter instances used to
            load them (needed downstream for their .config, e.g. column names). Raises
            rather than returning a sentinel value if a source file can't be read.

    Note:
        If an adapter's load() returns 0 (usually because skip_if_present is set and
        the output CSV exists; also if the source file is missing or empty), its
        to_csv() is skipped and the corresponding DataFrame below is read from whatever
        output CSV already exists on disk -- which may be stale from a previous run.
    """
    l = Logger()

    def runAdapter(adapter: BaseAdapter, name: str) -> None:
        """Load an ontology from its source files and write its EAV CSV."""
        l.log(f"Loading {name} from source files...")
        if adapter.load() > 0:
            adapter.to_csv()
            l.log(f"Loading {name} from source files completed.")
        else:
            l.log(f"Loading {name} from source files skipped (load() returned "
                  f"no rows, e.g. skip_if_present); using the existing output "
                  f"CSV, which may be from an earlier run.")

    def readCSV(adapter: BaseAdapter, name: str) -> pd.DataFrame:
        """Read an ontology's EAV CSV and log its size."""
        l.log(f"Reading {name} data...")
        path = os.path.join(adapter.config.output_folder, adapter.config.output_file)
        data = pd.read_csv(path, sep = separator, engine = "python")
        l.log(f"Reading {name} data completed. {len(data.index)} rows, "
              f"{data[adapter.config.id_column].nunique()} concepts.")
        return data

    hpo_ontology  = HPOAdapter()
    umls_ontology = UMLSAdapter()
    sct_ontology  = SCTAdapter()

    runAdapter(hpo_ontology,  "HPO")
    runAdapter(umls_ontology, "UMLS")
    runAdapter(sct_ontology,  "SNOMED CT")

    hpo = readCSV(hpo_ontology, "HPO")

    l.log("Removing non-exact HPO synonyms...")
    before = len(hpo.index)
    hpo = hpo[hpo[hpo_ontology.config.additional_column].apply(is_exact_synonym)]
    l.log(f"Removing non-exact HPO synonyms completed. {before - len(hpo.index)} "
          f"rows removed.")

    umls = readCSV(umls_ontology, "UMLS")

    sct = readCSV(sct_ontology, "SNOMED CT")

    # Make all SCT values strings, whatever type pandas inferred while
    # reading; missing values stay missing instead of becoming "nan".
    value_col = sct_ontology.config.value_column
    sct[value_col] = sct[value_col].map(lambda v: v if pd.isna(v) else str(v))

    return hpo, sct, umls, hpo_ontology, sct_ontology, umls_ontology

def loadGold(path: str, separator: str = ";") -> pd.DataFrame:
    """
    Loads, cleans, and standardizes gold standard mapping data between HPO
    and SNOMED CT.

    Each row of the Graham file has an HPO concept (HP_ID), a "Match group"
    and the SNOMED CT target in "SNOMED term/exp", written as
    "<SCTID> |<term>|" for a single concept or as a post-coordinated
    expression (e.g. "<SCTID>:<attribute>=<value>") for composite mappings.
    If the file has a "canonical view" column, it is used as the bare
    expression instead. Only single-concept targets are kept.

    Parameters:
        path (str): File path to the CSV dataset.
        separator (str): Delimiter used in the CSV file. Defaults to ';'.

    Returns:
        pd.DataFrame: A cleaned DataFrame containing mapped 'hpo_id' and 
            formatted 'sct_id'.
    """
    # Initialize the custom logging utility
    l = Logger()

    l.log("Reading gold standard data...")
    
    # 1. Load the raw dataset using the specified separator
    ret = pd.read_csv(
        path,
        sep=separator,
    )
    rows_read = len(ret.index)

    # 2. Keep only the match groups used as gold mappings ("One to one
    # match", "One to many"); rows of all other groups are removed.
    ret = ret[ret["Match group"].isin([
        "One to one match",
        "One to many"
    ])]
    rows_matched = len(ret.index)

    # 3. Drop unneeded metadata columns to streamline the dataset
    ret = ret.drop([
        "Synonyms", 
        "Preferred Label", 
        "Definition", 
        "HP Parent No", 
        "HP Parent Term", 
        "Match group", 
        "Alternative expression", 
        "Mapping Comment"
    ], axis=1)

    # 4. Get the bare SNOMED CT expression into 'canonical view'. Without
    # that column, it is the part of "SNOMED term/exp" before the first
    # " |", i.e. "12345 |Seizure|" -> "12345".

    if "canonical view" not in ret.columns:
        ret["SNOMED term/exp"] = ret["SNOMED term/exp"].astype(str)
        ret["canonical view"] = ret["SNOMED term/exp"].str.split(" |", regex=False).str[0]
    else:
        ret = ret.drop(["SNOMED term/exp"], axis=1)
    
    ret["canonical view"] = ret["canonical view"].astype(str)

    # 5. Extract numerical SNOMED CT IDs from the 'canonical view' string.
    # Plain integers are parsed as Python ints, so long (extension) SCTIDs
    # keep every digit. Scientific notation (e.g. '1.2e+08' or '1.2E+08')
    # goes through float, as it has already lost precision in the file.
    # Composite expressions (containing ':' or '+') and anything else
    # unparsable yield None and are dropped below. Scientific notation also
    # contains a '+', so it is exempted from the composite check.
    def parseSCTID(text: str):
        text = text.strip()
        if "e+" not in text.lower() and (":" in text or "+" in text):
            return None
        try:
            value = int(text)
        except ValueError:
            try:
                number = float(text)
            except ValueError:
                return None
            if not number.is_integer():
                return None
            value = int(number)
        return str(value) if value > 0 else None

    ret[sct_id_column] = ret["canonical view"].map(parseSCTID)

    # 6. Post-processing and column renaming
    ret = ret.drop(["canonical view"], axis=1)
    ret = ret.rename(columns={"HP_ID": hpo_id_column})

    # Keep only rows with valid, successfully extracted positive numeric IDs
    ret = ret[ret[sct_id_column].notna()]
    ret = ret.reset_index(drop = True)

    # 7. Prefix SNOMED CT IDs with standard ontology namespace format (e.g., 'SNOMEDCT_US:12345')
    ret[sct_id_column] = sct_prefix + ret[sct_id_column]

    # 8. Log completion with the counts of each filtering stage
    n_pairs = len(ret[[hpo_id_column, sct_id_column]].drop_duplicates().index)
    l.log(f"Reading gold standard data completed. {rows_read} rows read, "
          f"{rows_read - rows_matched} removed (match group not one-to-one "
          f"or one-to-many), {rows_matched - len(ret.index)} removed "
          f"(composite or unparsable SNOMED CT expression). {n_pairs} unique "
          f"pairs for {ret[hpo_id_column].nunique()} HPO concepts.")

    return ret

def loadInputs(
    separator: str = ";"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, HPOAdapter, SCTAdapter, UMLSAdapter, pd.DataFrame]:
    """
    Shared setup of the filtering, LLM and evaluation scripts: loads the
    ontologies (see loadOntologies), keeps only HPO concepts ("HP:") in the
    HPO data, and loads the gold standard from gold_path.

    Returns:
        (hpo, sct, umls, hpo_ontology, sct_ontology, umls_ontology, gold)
    """
    l = Logger()

    hpo, sct, umls, hpo_ontology, sct_ontology, umls_ontology = loadOntologies(separator)

    l.log("Removing non-HPO concepts from the HPO data...")
    before = len(hpo.index)
    hpo = keepHPOConcepts(hpo, hpo_ontology.config.id_column)
    l.log(f"Removing non-HPO concepts from the HPO data completed. "
          f"{before - len(hpo.index)} rows removed, "
          f"{hpo[hpo_ontology.config.id_column].nunique()} HPO concepts left.")

    gold = loadGold(gold_path, separator)
    return hpo, sct, umls, hpo_ontology, sct_ontology, umls_ontology, gold

def evaluation(data: pd.DataFrame, gold: pd.DataFrame, text: str, adapter: BaseAdapter) -> None:
    """
    Log recall and precision of the (hpo_id, sct_id) pairs in data against
    the gold standard. Both sides are reduced to unique pairs first, so
    pairs found by several methods, and any other columns (confidence,
    attribute, validity, LLM output, ...), do not affect the counts.

    recall    = correct pairs / unique gold pairs
    precision = correct pairs / unique predicted pairs

    Every predicted pair counts in the precision, including pairs of HPO
    concepts that the gold standard does not cover.
    """
    l = Logger()

    id_col  = adapter.config.id_column
    val_col = adapter.config.value_column

    predicted  = data[[id_col, val_col]].drop_duplicates(ignore_index=True)
    gold_pairs = gold[[hpo_id_column, sct_id_column]].drop_duplicates()

    # Predicted pairs that are present in the gold standard.
    matches = predicted.merge(
        gold_pairs,
        left_on     = [id_col,        val_col],
        right_on    = [hpo_id_column, sct_id_column],
        how         = "inner"
    )

    count = len(matches)

    # Avoid division by zero
    recall    = count / len(gold_pairs) if len(gold_pairs) else 0
    precision = count / len(predicted)  if len(predicted)  else 0

    l.log(f"{text}")
    l.log(f"{'Recall:':<26}{recall:7.2f}")
    l.log(f"{'Precision:':<26}{precision:7.2f}")
    l.log(f"{'Correct pairs (in gold):':<26}{count:7}")
    l.log(f"{'Pairs:':<26}{len(predicted):7}")
    l.log(f"{'HPO concepts:':<26}{predicted[id_col].nunique():7}")

def extract_snomed_domain(term: str) -> str:
    """
    Extracts the trailing SNOMED CT semantic tag (domain) from a fully
    specified name: the content of the last parentheses at the end of the
    string. Non-string input and names without a tag give "".

    Examples:
        "Ependymoma (disorder)" -> "disorder"
        "Structure of bone of lower leg (body structure)" -> "body structure"
        "History of (contextual qualifier) (qualifier value)" -> "qualifier value"
        "Plain term with no tag" -> ""
    """
    if not isinstance(term, str):
        return ""
    match = re.search(r"\(([^)]+)\)$", term.strip())
    return match.group(1) if match else ""

def printCounts(
    data:       pd.DataFrame,
    adapter:    BaseAdapter,
    thresholds: list,
    gold:       pd.DataFrame = None,
    diff:       bool         = False,
) -> None:
    """
    For each confidence threshold, report how many unique (hpo_id, sct_id)
    pairs reach it (confidence >= threshold) and how many of those appear
    in the gold standard; the percentage is the precision of that line.

    Deduplication is by (id_col, val_col), keeping the row with the highest
    confidence per pair, so mappings produced by multiple strategies are
    counted once. When diff=True each bucket counts only pairs whose
    confidence falls in [threshold, previous_threshold), so the lines add
    up to the total; the first bucket has no upper bound. thresholds must
    be in descending order.
    """
    l = Logger()
 
    if data is None or data.empty:
        return
 
    id_col  = adapter.config.id_column
    val_col = adapter.config.value_column

    # Unique gold pairs, so a pair listed twice in the gold standard is
    # not counted twice.
    gold_pairs = (
        gold[[hpo_id_column, sct_id_column]].drop_duplicates()
        if gold is not None else None
    )

    # One row per unique pair, highest confidence wins.
    deduped = (
        data
        .sort_values(confidence_column, ascending=False)
        .drop_duplicates(subset=[id_col, val_col])
    )
 
    # Upper bound of the current bucket; 2.0 is above any confidence, so
    # the first bucket is open-ended.
    old_threshold = 2.0
    for threshold in thresholds:
        subset = deduped[deduped[confidence_column] >= threshold]
        if diff:
            subset = subset[subset[confidence_column] < old_threshold]
 
        count = len(subset)

        # In diff mode each line is a bin [threshold, previous threshold);
        # the first bin has no upper bound.
        if diff and old_threshold < 2.0:
            label = f"in [{threshold:.2f}, {old_threshold:.2f})"
        else:
            label = f">= {threshold:.2f}"

        if gold_pairs is not None:
            in_gold = subset.merge(
                gold_pairs,
                left_on=[id_col, val_col],
                right_on=[hpo_id_column, sct_id_column],
                how="inner",
            )
            gold_count    = len(in_gold)
            gold_fraction = gold_count / count if count else 0
            l.log(
                f"Mappings with confidence {label:<16}: {count:7} "
                f"(in gold standard: {gold_count:7}, {gold_fraction:.2%})"
            )
        else:
            l.log(f"Mappings with confidence {label:<16}: {count:7}")
 
        old_threshold = threshold

