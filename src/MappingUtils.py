from rich.progress          import Progress, BarColumn, TextColumn, TaskID
from rich.progress          import TaskProgressColumn, TimeElapsedColumn
from adapter                import BaseAdapter
from adapter                import HPOAdapter, SCTAdapter, UMLSAdapter
from adapter                import exactSynonymClass, semanticClass, childrenClass
from scipy.optimize         import linprog
from logger                 import Logger
import pandas               as pd
import numpy                as np
import torch
import re
import ast
import os

sct_id_column           = "sct_id"
hpo_id_column           = "hpo_id"
validity_column         = "validity"
narrow_threshold        = 0.75 
borad_threshold         = 0.85
min_sources             = 2
min_confidence          = 0.95
hpo_sct_mappings        = "hpo_sct"
hpo_umls_sct_mappings   = "hpo_umls_sct"
umls_hpo_sct_mappings   = "umls_hpo_sct"
embedding_mappings      = "embedding"
jaccard_mappings        = "jaccard"
confidence_column       = "confidence"
attribute_column        = "attribute"
accepted_column         = "accepted"

threshold_cosine_similarity     = 0.75
threshold_jaccard_similarity    = 0.95

reference_methods = [
    hpo_sct_mappings, 
    hpo_umls_sct_mappings, 
    #umls_hpo_sct_mappings
]

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

def isChild(data: pd.DataFrame, id: str, child_id: str, recursive: bool, adapter: BaseAdapter) -> bool:
    ret = False

    if data is not None and len(data.index) > 0:
        children = data[(data[adapter.config.attribute_column] == childrenClass) & (data[adapter.config.id_column] == id)]
        if children is not None and len(children.index) > 0:
            if child_id in children[adapter.config.value_column].tolist():
                ret = True
            elif recursive:
                for c in children[adapter.config.value_column].tolist():
                    if not ret:
                        ret = isChild(data, c, child_id, True, adapter)
                
    return ret

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
    text: str = "Taks"
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

def jaccard_similarity(tokens_a: list, tokens_b: list):
    """
    Compute the Jaccard similarity between two lists of normalized tokens.

    Jaccard = |A ∩ B| / |A ∪ B|

    Returns
    -------
    float in [0, 1]. Returns 1.0 if both inputs are empty.
    """
    ret = 0
    set_a, set_b = set(tokens_a), set(tokens_b)

    if not set_a and not set_b:
        ret = 1.0
    else:    
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        ret = intersection / union

    return ret

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
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing (hpo, sct, umls)
            DataFrames upon success, or (None, None, None) if loading fails for any adapter.
    """
    # Initialize custom logger
    l = Logger()

    l.log("Loading necessary data from source files...")

    # 1. Process and extract HPO (Human Phenotype Ontology) data
    hpo_ontology = HPOAdapter()
    if hpo_ontology.load() > 0:
        hpo_ontology.to_csv()

    # 2. Process and extract UMLS (Unified Medical Language System) data
    umls_ontology = UMLSAdapter()
    if umls_ontology.load() > 0:
        umls_ontology.to_csv()

    # 3. Process and extract SNOMED CT data
    sct_ontology = SCTAdapter()
    if sct_ontology.load() > 0:
        sct_ontology.to_csv()

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

    # 5. Extract numerical SNOMED CT IDs from the 'canonical view' string
    for index, row in ret.iterrows():
        # Handle cases where the ID is formatted in scientific notation (e.g., '1.2e+08') 
        # or simple string numbers without complex mapping syntax (no ':' or '+')
        val = str(row["canonical view"])
        if ("e+" in val or 
            (":" not in val and 
            "+" not in val)):
            try:
                # Convert string -> float -> int to cleanly parse scientific notation
                ret.loc[index, sct_id_column] = int(float(val))
            except ValueError:
                # Ignore values that cannot be parsed into numbers
                ""
        # Flag complex multi-mappings or composite terms with -2
        elif "+" in val:
            ret.loc[index, sct_id_column] = -2

    # 6. Post-processing and column renaming
    ret = ret.drop(["canonical view"], axis=1)
    ret = ret.rename(columns={"HP_ID": hpo_id_column})
    
    # Keep only rows with valid, successfully extracted positive numeric IDs
    ret = ret[ret[sct_id_column] > 0]
    ret = ret.reset_index(drop = True)
    
    # 7. Prefix SNOMED CT IDs with standard ontology namespace format (e.g., 'SNOMEDCT_US:12345')
    ret[sct_id_column] = ret[sct_id_column].astype(str)
    for index, row in ret.iterrows():
        ret.loc[index, sct_id_column] = "SNOMEDCT_US:" + row[sct_id_column]

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

def extract_snomed_label(term: str) -> str:
    """
    Strips the trailing SNOMED CT semantic tag from a preferred term string.
    
    Example:
        "Ependymoma (disorder)" -> "Ependymoma"
        "Structure of bone of lower leg (body structure)" -> "Structure of bone of lower leg"
    """
    return re.sub(r"\s*\([^)]*\)$", "", term.strip())

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

def batch_cosine_similarity_threshold(
    hpo_embeddings: np.ndarray,
    snomed_embeddings: np.ndarray,
    threshold: float = 0.8,
    batch_size: int = 5000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes exact cosine similarity between HPO and SNOMED CT embeddings using

    batched matrix multiplication on PyTorch (GPU/CPU) and filters all pair
    combinations above a given similarity threshold.

    Parameters
    ----------
    hpo_embeddings : np.ndarray
        HPO vectors array of shape (N_hpo, d).
    snomed_embeddings : np.ndarray
        SNOMED CT vectors array of shape (M_snomed, d).
    threshold : float, default 0.8
        Minimum cosine similarity score (between -1.0 and 1.0) required to keep a
        match.
    batch_size : int, default 5000
        Number of HPO queries processed in a single GPU batch.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - hpo_indices: 1D array of HPO array indices for matched pairs.
        - snomed_indices: 1D array of SNOMED array indices for matched pairs.
        - scores: 1D array of similarity scores for matched pairs.
    """
    l = Logger()
    l.log(
        f"Filtering pairs with Cosine Similarity >= {threshold} on 'cuda'..."
    )

    # 1. Convert SNOMED array to tensor, normalize L2, and transpose
    snomed_tensor = torch.tensor(
        snomed_embeddings, dtype=torch.float32, device="cuda"
    )
    snomed_tensor = torch.nn.functional.normalize(snomed_tensor, p=2, dim=1)
    snomed_tensor_T = snomed_tensor.T

    all_hpo_indices = []
    all_snomed_indices = []
    all_scores = []

    n_hpo = len(hpo_embeddings)

    # 2. Iterate through HPO terms in batches
    for i in range(0, n_hpo, batch_size):
        batch_arr = hpo_embeddings[i : i + batch_size]

        hpo_batch = torch.tensor(
            batch_arr, dtype=torch.float32, device="cuda"
        )
        hpo_batch = torch.nn.functional.normalize(hpo_batch, p=2, dim=1)

        # Batch MatMul: (batch_size, d) @ (d, M_snomed) -> (batch_size, M_snomed)
        sim_matrix = torch.matmul(hpo_batch, snomed_tensor_T)

        # 3. Create boolean mask and extract matching pairs directly in GPU memory
        mask = sim_matrix >= threshold
        rel_hpo_idx, snomed_idx = torch.where(mask)
        scores = sim_matrix[rel_hpo_idx, snomed_idx]

        # Convert relative batch row indices to absolute global HPO array indices
        abs_hpo_idx = rel_hpo_idx + i

        # Move extracted batch pairs to CPU
        all_hpo_indices.append(abs_hpo_idx.cpu().numpy())
        all_snomed_indices.append(snomed_idx.cpu().numpy())
        all_scores.append(scores.cpu().numpy())

    # 4. Concatenate batch results into single 1D arrays
    if all_hpo_indices:
        hpo_idx_out = np.concatenate(all_hpo_indices)
        snomed_idx_out = np.concatenate(all_snomed_indices)
        scores_out = np.concatenate(all_scores)
    else:
        hpo_idx_out = np.array([], dtype=np.int64)
        snomed_idx_out = np.array([], dtype=np.int64)
        scores_out = np.array([], dtype=np.float32)

    return hpo_idx_out, snomed_idx_out, scores_out

def printCounts(data: pd.DataFrame, adapter: BaseAdapter, thresholds: list = [], gold: pd.DataFrame = None, diff: bool = False) -> None:
    l = Logger()
    if data is not None:
        old_threshold = 2
        for threshold in thresholds:
            subset = data[
                (data[confidence_column] >= threshold)
                & ((validity_column not in data.columns) or (data[validity_column] == 1))
                & ((not diff) or (data[confidence_column] < old_threshold))
            ]
            count = len(subset.index)

            if gold is not None:
                in_gold = subset.merge(
                    gold[[hpo_id_column, sct_id_column]],
                    left_on=[adapter.config.id_column, adapter.config.value_column],
                    right_on=[hpo_id_column, sct_id_column],
                    how="inner"
                )
                gold_count = len(in_gold.index)
                gold_fraction = gold_count / count if count else 0
                l.log(f"Mappings with confidence above {threshold:.2f}: {count:7} "
                      f"(in gold standard: {gold_count:7}, {gold_fraction:.2%})")
            else:
                l.log(f"Mappings with confidence above {threshold:.2f}: {count:7} ")

            old_threshold = threshold