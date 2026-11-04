from adapter                import BaseAdapter
from adapter                import HPOAdapter, SCTAdapter, UMLSAdapter
from adapter                import exactSynonymClass, semanticClass, childrenClass
from adapter.BaseAdapterUtils import additionalSeparator
from logger                 import Logger
from collections            import defaultdict
import pandas               as pd
import numpy                as np
import re
import ast
import os

# Column names of the gold standard (loadGold), of the confidence added by
# MappingGeneration and of the validity code added by MappingSetBuilder. The
# mapping files otherwise use the adapters' EAV column names.
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
gold_check_path      = "./data/output/evaluation/gold_check.csv"

# Validity codes written by MappingSetBuilder. Rows not reached by any step
# keep validity=NA.
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
validity_all = [validity_domain_mismatch] + validity_reviewable

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

# Review reason passed to the LLM prompt. Code 0 is not reviewed; code 10 is,
# so the LLM judges all mappings of an HPO concept.
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
threshold_cosine_similarity     = 0.75
threshold_jaccard_similarity    = 0.90

# Confidence thresholds for printCounts: evenly spaced from 1.0 down to the
# cosine threshold, plus 0.99 to separate exact from near-exact matches,
# i.e. [1.0, 0.99, 0.9375, 0.875, 0.8125, 0.75]. Descending order is
# required by printCounts(diff=True).
num_points = 5
similarity_steps = np.linspace(1.0, threshold_cosine_similarity,
    num = num_points).tolist()
similarity_steps.append(0.99)
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
    True if an EAV row's additional-column entry (dict or stringified dict)
    marks an exact synonym or has no synonym type at all (labels, hierarchy
    rows etc., also empty or missing entries). Only exact HPO synonyms name
    the same phenotype. Joined types (adapter 1.4.0, e.g. "exact, related")
    count as exact if one of them is.
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
            return exactSynonymClass in str(entry.get(semanticClass)).split(additionalSeparator)
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

# Hierarchy rows are written by the adapters with attribute=childrenClass as
# (id = parent, value = child).
def build_child_map(data: pd.DataFrame, adapter: BaseAdapter) -> dict:
    """Direct hierarchy lookup: parent -> set of direct children."""
    rows = data[data[adapter.config.attribute_column] == childrenClass]
    child_map = defaultdict(set)
    for parent, child in zip(rows[adapter.config.id_column], rows[adapter.config.value_column]):
        child_map[parent].add(child)
    return child_map

def build_parent_map(data: pd.DataFrame, adapter: BaseAdapter) -> dict:
    """Direct hierarchy lookup: child -> set of direct parents."""
    rows = data[data[adapter.config.attribute_column] == childrenClass]
    parent_map = defaultdict(set)
    for parent, child in zip(rows[adapter.config.id_column], rows[adapter.config.value_column]):
        parent_map[child].add(parent)
    return parent_map

def make_descendant_lookup(child_map: dict):
    """
    Returns a memoized function descendants(node) -> frozenset of all nodes
    transitively reachable from `node` through child_map (cycle-safe). Given
    a child -> parents map (build_parent_map) it returns ancestors instead.
    The start node is not included unless a cycle leads back to it.
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

def make_ancestor_distance_lookup(parent_map: dict):
    """
    Returns a memoized function ancestor_distances(node) -> dict mapping the
    node itself (0) and every transitive ancestor to the smallest number of
    is_a steps up from `node` (breadth-first, cycle-safe). parent_map comes
    from build_parent_map.
    """
    cache: dict = {}

    def ancestor_distances(node):
        if node in cache:
            return cache[node]
        distances = {node: 0}
        frontier  = [node]
        steps     = 0
        while frontier:
            steps += 1
            next_frontier = []
            for current in frontier:
                for parent in parent_map.get(current, ()):
                    if parent not in distances:
                        distances[parent] = steps
                        next_frontier.append(parent)
            frontier = next_frontier
        cache[node] = distances
        return distances

    return ancestor_distances

def make_hierarchy_distance(ancestor_distances):
    """
    Returns a function hierarchy_distance(a, b) -> int or None: the number
    of is_a edges on the shortest path from a up to a common ancestor and
    down to b. Equal concepts have distance 0, a parent and its child 1,
    siblings and grandparent/grandchild 2, uncle/nephew 3, cousins 4.
    Paths over a common descendant are not counted. None means no common
    ancestor (e.g. a concept not in the hierarchy). ancestor_distances comes
    from make_ancestor_distance_lookup, so its cache can be shared.
    """
    def hierarchy_distance(a, b):
        if a == b:
            return 0
        up_a = ancestor_distances(a)
        up_b = ancestor_distances(b)
        if len(up_a) > len(up_b):
            up_a, up_b = up_b, up_a
        best = None
        for ancestor, steps_a in up_a.items():
            steps_b = up_b.get(ancestor)
            if steps_b is not None and (best is None or steps_a + steps_b < best):
                best = steps_a + steps_b
        return best

    return hierarchy_distance

def pairSeries(data: pd.DataFrame, id_col: str, val_col: str) -> pd.Series:
    """Series of (id, value) tuples aligned to data.index."""
    return pd.Series(list(zip(data[id_col], data[val_col])), index=data.index)

def inGoldSeries(data: pd.DataFrame, gold_pairs: set, id_col: str, val_col: str) -> pd.Series:
    """Boolean Series: True when the (id_col, val_col) pair is in gold_pairs."""
    return pairSeries(data, id_col, val_col).isin(gold_pairs)

def loadOntologies(
    separator: str = ";"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, HPOAdapter, SCTAdapter, UMLSAdapter]:
    """
    Loads, processes, and exports HPO, UMLS, and SNOMED CT ontology data.

    Runs each adapter to (re)generate its EAV CSV, then reads the CSVs. Only
    non-exact HPO synonyms are removed and SNOMED CT values made strings;
    the adapters already format the IDs.

    Parameters:
        separator (str): Delimiter used for reading output CSV files. Defaults to ';'.

    Returns:
        tuple: (hpo, sct, umls) DataFrames and the adapters used to load them
            (needed downstream for their .config). Raises if a source file
            can't be read.

    Note:
        If an adapter's load() returns 0 (e.g. skip_if_present with an existing
        output CSV), the existing CSV is read, which may be from an earlier run.
    """
    l = Logger()

    def runAdapter(adapter: BaseAdapter, name: str) -> None:
        """Load an ontology from its source files and write its EAV CSV."""
        l.log(f"Loading {name} from source files...")
        if adapter.load() > 0:
            adapter.to_csv()
        else:
            l.log(f"No {name} rows loaded from source files (e.g. skip_if_present).")
            l.log(f"The existing {name} output CSV is used; it may be from an earlier run.")
        l.log(f"Loading {name} from source files completed.")

    def readCSV(adapter: BaseAdapter, name: str) -> pd.DataFrame:
        """Read an ontology's EAV CSV and log its size."""
        l.log(f"Reading {name} data...")
        path = os.path.join(adapter.config.output_folder, adapter.config.output_file)
        data = pd.read_csv(path, sep = separator, engine = "python")
        l.log(f"{name} data: {len(data.index)} rows, "
              f"{data[adapter.config.id_column].nunique()} concepts.")
        l.log(f"Reading {name} data completed.")
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
    l.log(f"Non-exact HPO synonyms: {before - len(hpo.index)} rows removed.")
    l.log("Removing non-exact HPO synonyms completed.")

    umls = readCSV(umls_ontology, "UMLS")

    sct = readCSV(sct_ontology, "SNOMED CT")

    # Missing values stay missing instead of becoming "nan".
    value_col = sct_ontology.config.value_column
    sct[value_col] = sct[value_col].map(lambda v: v if pd.isna(v) else str(v))

    return hpo, sct, umls, hpo_ontology, sct_ontology, umls_ontology

def loadGold(path: str, separator: str = ";") -> pd.DataFrame:
    """
    Loads the HPO -> SNOMED CT gold standard (Graham file).

    "SNOMED term/exp" holds "<SCTID> |<term>|" or a post-coordinated
    expression (e.g. "<SCTID>:<attribute>=<value>"); a "canonical view"
    column, if present, is used as the bare expression instead. Only
    single-concept targets of the one-to-one / one-to-many groups are kept.

    Parameters:
        path (str): File path to the CSV dataset.
        separator (str): Delimiter used in the CSV file. Defaults to ';'.

    Returns:
        pd.DataFrame: A cleaned DataFrame containing mapped 'hpo_id' and 
            formatted 'sct_id'.
    """
    l = Logger()

    l.log("Reading gold standard data...")
    
    ret = pd.read_csv(
        path,
        sep=separator,
    )
    rows_read = len(ret.index)

    ret = ret[ret["Match group"].isin([
        "One to one match",
        "One to many"
    ])]
    rows_matched = len(ret.index)

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

    # Without "canonical view": "12345 |Seizure|" -> "12345".
    if "canonical view" not in ret.columns:
        ret["SNOMED term/exp"] = ret["SNOMED term/exp"].astype(str)
        ret["canonical view"] = ret["SNOMED term/exp"].str.split(" |", regex=False).str[0]
    else:
        ret = ret.drop(["SNOMED term/exp"], axis=1)
    
    ret["canonical view"] = ret["canonical view"].astype(str)

    # Integers are parsed as int so long (extension) SCTIDs keep every digit;
    # scientific notation ('1.2E+08', precision already lost in the file)
    # goes through float and is exempt from the '+' composite check.
    # Composite or unparsable expressions give None and are dropped.
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

    ret = ret.drop(["canonical view"], axis=1)
    ret = ret.rename(columns={"HP_ID": hpo_id_column})

    ret = ret[ret[sct_id_column].notna()]
    ret = ret.reset_index(drop = True)

    ret[sct_id_column] = sct_prefix + ret[sct_id_column]

    n_pairs = len(ret[[hpo_id_column, sct_id_column]].drop_duplicates().index)
    l.log(f"Gold standard: {rows_read} rows read.")
    l.log(f"Gold standard: {rows_read - rows_matched} rows removed (match group "
          f"not one-to-one or one-to-many).")
    l.log(f"Gold standard: {rows_matched - len(ret.index)} rows removed (composite "
          f"or unparsable SNOMED CT expression).")
    l.log(f"Gold standard: {n_pairs} unique pairs for "
          f"{ret[hpo_id_column].nunique()} HPO concepts.")
    l.log("Reading gold standard data completed.")

    return ret

def loadInputs(
    separator: str = ";"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, HPOAdapter, SCTAdapter, UMLSAdapter, pd.DataFrame]:
    """
    Shared setup of all pipeline scripts and the gold standard check: loads the
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
    l.log(f"Non-HPO concepts: {before - len(hpo.index)} rows removed, "
          f"{hpo[hpo_ontology.config.id_column].nunique()} HPO concepts left.")
    l.log("Removing non-HPO concepts from the HPO data completed.")

    gold = loadGold(gold_path, separator)
    return hpo, sct, umls, hpo_ontology, sct_ontology, umls_ontology, gold

def evaluation(data: pd.DataFrame, gold: pd.DataFrame, text: str, adapter: BaseAdapter) -> None:
    """
    Log recall and precision of the unique (hpo_id, sct_id) pairs in data
    against the unique gold pairs. Every predicted pair counts in the
    precision, also pairs of HPO concepts the gold standard does not cover.
    """
    l = Logger()

    id_col  = adapter.config.id_column
    val_col = adapter.config.value_column

    predicted  = data[[id_col, val_col]].drop_duplicates(ignore_index=True)
    gold_pairs = gold[[hpo_id_column, sct_id_column]].drop_duplicates()

    matches = predicted.merge(
        gold_pairs,
        left_on     = [id_col,        val_col],
        right_on    = [hpo_id_column, sct_id_column],
        how         = "inner"
    )

    count = len(matches)

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

    A pair found by several methods counts once, with its highest
    confidence. With diff=True each line counts [threshold, previous
    threshold), so the lines add up; the first has no upper bound.
    thresholds must be in descending order.
    """
    l = Logger()
 
    if data is None or data.empty:
        return
 
    id_col  = adapter.config.id_column
    val_col = adapter.config.value_column

    gold_pairs = (
        gold[[hpo_id_column, sct_id_column]].drop_duplicates()
        if gold is not None else None
    )

    deduped = (
        data
        .sort_values(confidence_column, ascending=False)
        .drop_duplicates(subset=[id_col, val_col])
    )
 
    # 2.0 is above any confidence: the first bucket is open-ended.
    old_threshold = 2.0
    for threshold in thresholds:
        subset = deduped[deduped[confidence_column] >= threshold]
        if diff:
            subset = subset[subset[confidence_column] < old_threshold]
 
        count = len(subset)

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

