from rich.progress  import Progress, BarColumn, TextColumn, TaskID
from rich.progress  import TaskProgressColumn, TimeElapsedColumn
from scipy.optimize import linprog
from logger         import Logger
import pandas       as pd
import numpy        as np
import torch
import re
import ast
import os

progressBarColor        = "cyan"
progressBarTextLength   = 40

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

def cosine_similarity(embedding1: list, embedding2: list) -> float:
    """
    Compute the cosine similarity between two embedding vectors.

    Parameters
    ----------
    embedding1 : array-like
        First embedding vector.
    embedding2 : array-like
        Second embedding vector.

    Returns
    -------
    float
        Cosine similarity in the range [-1, 1].
    """
    ret = float("inf")

    norm1 = np.linalg.norm(np.asarray(embedding1, dtype=np.float32))
    norm2 = np.linalg.norm(np.asarray(embedding2, dtype=np.float32))

    if norm1 != 0 and norm2 != 0:
        ret = np.dot(embedding1, embedding2) / (norm1 * norm2)

    return ret

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


def overlap_coefficient(tokens_a: list, tokens_b: list) -> int:
    """
    Compute the overlap coefficient between two lists of normalized tokens.

    Overlap = |A ∩ B| / min(|A|, |B|)

    Lenient toward subset relationships, e.g. "hypertension" vs.
    "essential hypertension" scores high despite differing lengths.

    Returns
    -------
    float in [0, 1]. Returns 1.0 if both inputs are empty, 0.0 if only
    one is empty.
    """
    ret = 0.0
    set_a, set_b = set(tokens_a), set(tokens_b)

    if not set_a and not set_b:
        ret = 1.0
    elif not set_a or not set_b:
        ret = 0.0
    else:
        ret = len(set_a & set_b) / min(len(set_a), len(set_b))

    return ret

def extract_snomed_label(term: str) -> str:
    """
    Strips the trailing SNOMED CT semantic tag from a preferred term string.
    
    Example:
        "Ependymoma (disorder)" -> "Ependymoma"
        "Structure of bone of lower leg (body structure)" -> "Structure of bone of lower leg"
    """
    return re.sub(r"\s*\([^)]*\)$", "", term.strip())

def write_embeddings_to_csv(
    filepath: str,
    terms: list[str],
    embeddings: list[np.ndarray | list[float]],
) -> int:
    """Formats terms and embeddings into a DataFrame and writes them via

    writeHugeCSV.

    Parameters
    ----------
    filepath : str
        Target CSV file path.
    terms : list[str]
        List of term labels.
    embeddings : list[np.ndarray | list[float]]
        List of 1D vector arrays corresponding to the terms.

    Returns
    -------
    int
        The number of rows written to the CSV file.
    """
    ret = 0

    if len(terms) == len(embeddings):
        # Convert NumPy arrays or lists to string representation for CSV storage
        string_embeddings = [
            str(emb.tolist() if isinstance(emb, np.ndarray) else emb)
            for emb in embeddings
        ]

        # Construct DataFrame expected by writeHugeCSV
        df = pd.DataFrame({"term": terms, "embedding": string_embeddings})

        # Delegate CSV creation to the existing writeHugeCSV function
        ret = writeHugeCSV(df, filepath)

    return ret

def read_embeddings_from_csv(
    filepath: str,
) -> tuple[list[str], list[np.ndarray]]:
    """Reads a CSV file using pandas and returns parallel lists of terms and

    embedding arrays.

    Parameters
    ----------
    filepath : str
        Source CSV file path.

    Returns
    -------
    tuple[list[str], list[np.ndarray]]
        A tuple containing (list_of_terms, list_of_numpy_embedding_arrays).
    """
    # Read CSV into a pandas DataFrame
    df = pd.read_csv(filepath)

    # Extract terms as a standard list
    terms = df["term"].tolist()

    # Reconstruct lists of floats from string representations and cast to NumPy arrays
    embeddings = [
        np.array(ast.literal_eval(emb_str), dtype=np.float32)
        for emb_str in df["embedding"]
    ]

    return terms, embeddings

def writeCSV(
    data: pd.DataFrame = None,
    file: str = "",
    separator: str = ";",
    encoding: str = "utf-8"
) -> int:
    """
    Write a DataFrame to disk as a CSV file with logging.

    Parameters
    ----------
    data : pd.DataFrame, optional
        The DataFrame to write. If None, nothing is written and a
        message is logged instead.
    file : str, optional
        Path of the CSV file to write to. If empty, nothing is written
        and a message is logged instead.
    separator : str, optional
        Field delimiter used in the output CSV (default ";").
    encoding : str, optional
        Character encoding used when writing the file (default "utf-8").

    Returns
    -------
    int
        amount of lines written in CSV file.
    """
    ret: int = 0
    l: Logger = Logger()

    # Only proceed if a DataFrame was actually provided.
    if data is not None:
        # Only proceed if a target file path was actually provided.
        if len(file) > 0:
            # Log before starting the write, in case it's a large file
            # and takes noticeable time.
            l.printWriteFileStart(file)

            # Write the DataFrame to disk without the pandas row index,
            # using the given separator and encoding.
            data.to_csv(
                file,
                sep=separator,
                encoding=encoding,
                index=False
            )

            # Log that the write completed.
            l.printWriteFileEnd(file)
            ret = len(data.index)
        else:
            # No file path given — log and skip writing.
            l.log("File has not been specified and is empty.")
    else:
        # No DataFrame given — log and skip writing.
        l.log("No data provided.")

    return ret


def writeHugeCSV(
    data: pd.DataFrame = None,
    file: str = "",
    separator: str = ";",
    encoding: str = "utf-8"
) -> int:
    """
    Write a large DataFrame to disk safely by first writing to a
    temporary file, then atomically replacing the target file.

    This avoids leaving a corrupted or partially-written file at
    `file` if the write is interrupted, since the original file is
    only replaced once the temporary file has been fully written.

    Parameters
    ----------
    data : pd.DataFrame, optional
        The DataFrame to write.
    file : str, optional
        Path of the final CSV file to write to.
    separator : str, optional
        Field delimiter used in the output CSV (default ";").
    encoding : str, optional
        Character encoding used when writing the file (default "utf-8").

    Returns
    -------
    int
        amount of lines written in CSV file.
    """
    ret: int = 0
    l: Logger = Logger()

    l.log("Writing in temporary file first...")

    # Build the temporary file path by appending ".tmp" to the target path.
    tmpfile = file + ".tmp"

    # Write to the temporary file first, reusing writeCSV's logic.
    ret = writeCSV(data, tmpfile, separator, encoding)

    # Only replace the original file if the temporary write succeeded.
    if ret > 0:
        l.log("Replacing original data with temporary data...")

        # Atomically replace the target file with the temporary file
        # (os.replace is atomic on both POSIX and Windows).
        os.replace(tmpfile, file)

        l.log("Replacing original data with temporary data completed.")

    return ret

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

    print(
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