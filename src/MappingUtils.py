from rich.progress          import Progress, BarColumn, TextColumn, TaskID
from rich.progress          import TaskProgressColumn, TimeElapsedColumn
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

sct_id_column           = "sct_id"
hpo_id_column           = "hpo_id"
validity_column         = "validity"
hpo_sct_mappings        = "hpo_sct"
hpo_umls_sct_mappings   = "hpo_umls_sct"
umls_hpo_sct_mappings   = "umls_hpo_sct"
embedding_mappings      = "embedding"
jaccard_mappings        = "jaccard"
exact_lexical_mappings  = "exact_lexical"
llm_mappings            = "llm"
confidence_column       = "confidence"
attribute_column        = "attribute"

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

# Input and output files shared between the pipeline scripts.
gold_path            = "./data/input/graham/mapping_graham.csv"
mapping_path         = "./data/output/mapped/mapping.csv"
filtered_path        = "./data/output/filtered/filtered.csv"
llm_filtered_path    = "./data/output/filtered/llm_filtered.csv"
evaluation_path      = "./data/output/evaluation/evaluation_final.csv"

# ------------------------------------------------------------------
# Validity codes written by MappingSetBuilder (and read by MappingLLM
# and MappingErrorAnalysis). Rows with validity=NA are not written.
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

threshold_cosine_similarity     = 0.80
threshold_jaccard_similarity    = 0.95

num_points = 5
similarity_steps = np.linspace(1.0, threshold_cosine_similarity,
    num = num_points).tolist()
similarity_steps.append(0.99)
# Keep the steps in descending order: printCounts(diff=True) builds its
# buckets from consecutive thresholds and needs them sorted.
similarity_steps = sorted(set(similarity_steps), reverse=True)

acceptableDomains = [
    "disorder",
    "finding",
    "morphologic abnormality",
    "contextual qualifier",
    ""
]

progressBarColor        = "cyan"
progressBarTextLength   = 40

def isSCTDomainValid(terms: list[str]) -> bool:
    ret = False
    if terms is not None and len(terms) > 0 and any(extract_snomed_domain(pt) in acceptableDomains for pt in terms):
        ret = True
    return ret

# Readable label for the audit trail: prefer a label with an acceptable domain, else the first available
def getRepresentativeLabel(terms: list) -> str:
    if not terms:
        return None
    for term in terms:
        if term is not None and extract_snomed_domain(term) in acceptableDomains:
            return term
    return terms[0]

def newProgress() -> Progress:
    """
    Create a Rich progress bar with consistent formatting.
    """
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style = progressBarColor),
        TaskProgressColumn(),
        TextColumn("ET:"),
        TimeElapsedColumn(),
        TextColumn("Elem.s: {task.completed}/{task.total}"),
    )

def newTask(
    progress: Progress,
    iterations: int,
    text: str = "Task"
) -> TaskID:
    """
    Add a new task to a Rich progress bar.
    """
    return progress.add_task(
        text.ljust(progressBarTextLength),
        total=iterations
    )

def dictToEAVDataFrame(data: dict, attribute: str, adapter: BaseAdapter) -> pd.DataFrame:
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
    or carries no synonym type at all. Entries may arrive as dicts or as
    stringified dicts (as pandas reads them from CSV).
    """
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
    return data[data[id_col].str.startswith(hpo_prefix)].reset_index(drop=True)

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
    Vectorised — no Python-level row loop.
    """
    return pairSeries(data, id_col, val_col).isin(gold_pairs)

def loadOntologies(
    separator: str = ";"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, HPOAdapter, SCTAdapter, UMLSAdapter]:
    """
    Loads, processes, and exports HPO, SNOMED CT, and UMLS ontology data.

    Processes adapters for each ontology to generate CSV files, then loads them
    into DataFrames. Applies standard filtering (e.g., exact HPO synonyms) and
    ID formatting (e.g., prefixing SNOMED CT IDs).

    Parameters:
        separator (str): Delimiter used for reading output CSV files. Defaults to ';'.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, HPOAdapter, SCTAdapter, UMLSAdapter]:
            The loaded (hpo, sct, umls) DataFrames alongside the adapter instances used to
            load them (needed downstream for their .config, e.g. column names). Raises
            rather than returning a sentinel value if a source file can't be read.

    Note:
        If an adapter's load() returns 0 (e.g. the source file is missing or empty), its
        to_csv() is skipped and the corresponding DataFrame below is read from whatever
        output CSV already exists on disk -- which may be stale from a previous run.
    """
    # Initialize custom logger
    l = Logger()

    l.log("Loading necessary data from source files...")

    # 1. Process and extract HPO (Human Phenotype Ontology) data
    hpo_ontology = HPOAdapter()
    if hpo_ontology.load() > 0:
        hpo_ontology.to_csv()
    else:
        l.log("HPO load() returned no rows (skip_if_present or empty source); using existing output CSV.")

    # 2. Process and extract UMLS (Unified Medical Language System) data
    umls_ontology = UMLSAdapter()
    if umls_ontology.load() > 0:
        umls_ontology.to_csv()
    else:
        l.log("UMLS load() returned no rows (skip_if_present or empty source); using existing output CSV.")

    # 3. Process and extract SNOMED CT data
    sct_ontology = SCTAdapter()
    if sct_ontology.load() > 0:
        sct_ontology.to_csv()
    else:
        l.log("SNOMED CT load() returned no rows (skip_if_present or empty source); using existing output CSV.")

    l.log("Loading necessary data from source files completed.")
    l.log("Reading HPO, SNOMED CT, and UMLS data...")

    # --- HPO PROCESSING ---
    l.log("Reading HPO data...")
    hpo_file_path = os.path.join(
        hpo_ontology.config.output_folder, 
        hpo_ontology.config.output_file
    )
    hpo = pd.read_csv(hpo_file_path, sep = separator, engine = "python")
    
    l.log("Keep only exact synonyms from HPO...")

    # Filter DataFrame to retain only exact synonyms
    hpo = hpo[hpo[hpo_ontology.config.additional_column].apply(is_exact_synonym)]
    l.log("Removing non exact synonyms completed.")
    l.log("Reading HPO data completed.")

    # --- SNOMED CT PROCESSING ---
    l.log("Reading SNOMED CT data...")
    sct_file_path = os.path.join(
        sct_ontology.config.output_folder, 
        sct_ontology.config.output_file
    )
    sct = pd.read_csv(sct_file_path, sep = separator, engine = "python")
    
    # Ensure value column is string type
    sct[sct_ontology.config.value_column] = \
        sct[sct_ontology.config.value_column].astype(str)
    
    l.log("Reading SNOMED CT data completed.")

    # --- UMLS PROCESSING ---
    l.log("Reading UMLS data...")
    umls_file_path = os.path.join(
        umls_ontology.config.output_folder, 
        umls_ontology.config.output_file
    )
    umls = pd.read_csv(umls_file_path, sep = separator, engine = "python")
    l.log("Reading UMLS data completed.")

    l.log("Reading HPO, SNOMED CT, and UMLS data completed.")
    
    return hpo, sct, umls, hpo_ontology, sct_ontology, umls_ontology

def loadGold(path: str, separator: str = ";") -> pd.DataFrame:
    """
    Loads, cleans, and standardizes gold standard mapping data between HPO 
    and SNOMED CT.

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

    # 2. Filter for specific matching criteria (keep only valid mappings)
    ret = ret[ret["Match group"].isin([
        "One to one match", 
        "One to many"
    ])]

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

    # 4. Prepare 'canonical view' for extraction and initialize 'sct_id' column

    if "canonical view" not in ret.columns:
        ret["SNOMED term/exp"] = ret["SNOMED term/exp"].astype(str)
        ret["canonical view"] = ret["SNOMED term/exp"].str.split(" |", regex=False).str[0]
    else:
        ret = ret.drop(["SNOMED term/exp"], axis=1)
    
    ret["canonical view"] = ret["canonical view"].astype(str)
    ret[sct_id_column] = -1

    # 5. Extract numerical SNOMED CT IDs from the 'canonical view' string.
    # Scientific notation (e.g. '1.2e+08') or a plain numeric string (no
    # ':' or '+') parses to a positive int; anything else containing '+'
    # is a composite multi-mapping, flagged -2; everything else keeps the
    # -1 default set above.
    val = ret["canonical view"]
    is_scientific_or_plain = val.str.contains("e+", regex=False) | (
        ~val.str.contains(":", regex=False) & ~val.str.contains("+", regex=False)
    )
    is_composite = ~is_scientific_or_plain & val.str.contains("+", regex=False)

    parsed = pd.to_numeric(val.where(is_scientific_or_plain), errors="coerce")
    ret.loc[parsed.notna(), sct_id_column] = parsed.dropna().astype(int)
    ret.loc[is_composite, sct_id_column] = -2

    # 6. Post-processing and column renaming
    ret = ret.drop(["canonical view"], axis=1)
    ret = ret.rename(columns={"HP_ID": hpo_id_column})
    
    # Keep only rows with valid, successfully extracted positive numeric IDs
    ret = ret[ret[sct_id_column] > 0]
    ret = ret.reset_index(drop = True)
    
    # 7. Prefix SNOMED CT IDs with standard ontology namespace format (e.g., 'SNOMEDCT_US:12345')
    ret[sct_id_column] = sct_prefix + ret[sct_id_column].astype(str)

    # 8. Log completion and count unique HPO terms successfully mapped
    l.log("Reading gold standard data completed.")
    l.log(f"Mappings for {len(set(ret[hpo_id_column].tolist()))} Concepts found.")

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
    hpo, sct, umls, hpo_ontology, sct_ontology, umls_ontology = loadOntologies(separator)
    hpo  = keepHPOConcepts(hpo, hpo_ontology.config.id_column)
    gold = loadGold(gold_path, separator)
    return hpo, sct, umls, hpo_ontology, sct_ontology, umls_ontology, gold

def evaluation(data: pd.DataFrame, gold: pd.DataFrame, text: str, adapter: BaseAdapter) -> None:
    l = Logger()

    # Remove columns that are not relevant for evaluation
    drop_cols = [
        col for col in (confidence_column, attribute_column)
        if col in data.columns
    ]

    eval = (
        data
        .drop(columns=drop_cols)
        .drop_duplicates(ignore_index=True)
    )

    # Only the columns needed for matching
    gold_pairs = gold[[hpo_id_column, sct_id_column]].drop_duplicates()

    # Find which predictions are present in gold
    matches = (
        eval[[adapter.config.id_column, adapter.config.value_column]]
        .merge(
            gold_pairs,
            left_on     = [adapter.config.id_column,    adapter.config.value_column],
            right_on    = [hpo_id_column,               sct_id_column],
            how         = "inner"
        )
    )

    count = len(matches)

    # Avoid division by zero
    recall = count / len(gold) if len(gold) else 0
    precision = count / len(eval) if len(eval) else 0

    l.log(f"{text}")
    l.log(f"Recall:                   {recall:.2f}")
    l.log(f"Precision:                {precision:.2f}")
    l.log(f"Concept Mapping Count: {eval[adapter.config.id_column].nunique():7}")
    l.log(f"Mapping Count:         {len(eval):7}")

def extract_snomed_domain(term: str) -> str:
    """
    Extracts the trailing SNOMED CT semantic tag (domain) from a preferred term string.
    
    Examples:
        "Ependymoma (disorder)" -> "disorder"
        "Structure of bone of lower leg (body structure)" -> "body structure"
        "Plain term with no tag" -> ""
    """
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
    pairs exceed it and how many of those appear in the gold standard.
 
    Deduplication is by (id_col, val_col), keeping the row with the highest
    confidence per pair, so mappings produced by multiple strategies are
    counted once. When diff=True each bucket counts only pairs whose
    confidence falls in [threshold, previous_threshold).
    """
    l = Logger()
 
    if data is None or data.empty:
        return
 
    id_col  = adapter.config.id_column
    val_col = adapter.config.value_column
 
    # One row per unique pair, highest confidence wins.
    deduped = (
        data
        .sort_values(confidence_column, ascending=False)
        .drop_duplicates(subset=[id_col, val_col])
    )
 
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

        if gold is not None:
            in_gold = subset.merge(
                gold[[hpo_id_column, sct_id_column]],
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

