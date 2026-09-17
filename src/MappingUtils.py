from rich.progress          import Progress, BarColumn, TextColumn, TaskID
from rich.progress          import TaskProgressColumn, TimeElapsedColumn
from adapter                import BaseAdapter
from adapter                import HPOAdapter, SCTAdapter, UMLSAdapter
from adapter                import exactSynonymClass, semanticClass
from logger                 import Logger
import pandas               as pd
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
confidence_column       = "confidence"
attribute_column        = "attribute"

threshold_cosine_similarity     = 0.75
threshold_jaccard_similarity    = 0.95

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
        l.log("HPO load() returned no rows; reusing existing (possibly stale) output CSV.")

    # 2. Process and extract UMLS (Unified Medical Language System) data
    umls_ontology = UMLSAdapter()
    if umls_ontology.load() > 0:
        umls_ontology.to_csv()
    else:
        l.log("UMLS load() returned no rows; reusing existing (possibly stale) output CSV.")

    # 3. Process and extract SNOMED CT data
    sct_ontology = SCTAdapter()
    if sct_ontology.load() > 0:
        sct_ontology.to_csv()
    else:
        l.log("SNOMED CT load() returned no rows; reusing existing (possibly stale) output CSV.")

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
    
    # Helper to parse dictionary entries if pandas imported them as 
    # stringified dicts
    def is_exact_synonym(entry):
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
    
    l.log("Extending the ID of SNOMED CT concepts...")
    # Standardize SNOMED CT IDs with standard ontology namespace 
    # prefix (vectorized)
    sct[sct_ontology.config.id_column] = "SNOMEDCT_US:" + \
        sct[sct_ontology.config.id_column].astype(str)
    l.log("Extending the ID of SNOMED CT concepts completed.")
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
        "Mapping Comment", 
        "SNOMED term/exp"
    ], axis=1)

    # 4. Prepare 'canonical view' for extraction and initialize 'sct_id' column
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
    ret[sct_id_column] = "SNOMEDCT_US:" + ret[sct_id_column].astype(str)

    # 8. Log completion and count unique HPO terms successfully mapped
    l.log("Reading gold standard data completed.")
    l.log(f"Mappings for {len(set(ret[hpo_id_column].tolist()))} Concepts found.")

    return ret

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
                f"Mappings with confidence above {threshold:.2f}: {count:7} "
                f"(in gold standard: {gold_count:7}, {gold_fraction:.2%})"
            )
        else:
            l.log(f"Mappings with confidence above {threshold:.2f}: {count:7}")
 
        old_threshold = threshold

