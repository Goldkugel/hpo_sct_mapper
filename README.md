# hpo_sct_mapper

Maps concepts of the [Human Phenotype Ontology (HPO)](https://hpo.jax.org/) to [SNOMED CT](https://www.snomed.org/) concepts. Candidate mappings are collected from cross-references and from lexical and embedding similarity. A rule-based filter then accepts, rejects or flags them, and a local LLM reviews the flagged and accepted candidates. Every stage is evaluated against a gold standard mapping, which can itself be checked with the LLM.

## Pipeline

The pipeline consists of four scripts in `src/`. They are run one after the other and pass their results on as CSV files. A fifth script, `MappingGoldCheck.py`, checks the gold standard with the LLM and runs on its own (see below).

| Step | Script | Input | Output |
|---|---|---|---|
| 1 | `MappingGeneration.py` | HPO, SNOMED CT, UMLS, gold standard | `data/output/mapped/mapping.csv` |
| 2 | `MappingSetBuilder.py` | `mapping.csv` | `data/output/filtered/filtered.csv` |
| 3 | `MappingLLM.py` | `filtered.csv` | `data/output/filtered/llm_filtered.csv`, `llm_review.csv` |
| 4 | `MappingErrorAnalysis.py` | all of the above | `evaluation_final.csv`, `accepted_without_gold.csv`, `llm_disagreements.csv` in `data/output/evaluation/`, and the log |

Shared constants (file paths, method names, validity codes, thresholds) and helper functions live in `MappingUtils.py`.

### 1. Candidate generation (`MappingGeneration.py`)

Collects candidate pairs (HPO concept, SNOMED CT concept) with six methods. Each pair is stored once per method that found it, with a confidence:

| Method (`attribute`) | Source | Confidence |
|---|---|---|
| `hpo_sct` | SNOMED CT cross-references in HPO | 1.0 |
| `hpo_umls_sct` | HPO → UMLS → SNOMED CT (UMLS cross-reference in HPO) | 1.0 |
| `umls_hpo_sct` | HPO ← UMLS → SNOMED CT (a UMLS concept lists both) | 1.0 |
| `jaccard` | Jaccard similarity of the normalised label and synonym tokens, > 0.90 | similarity |
| `embedding` | Cosine similarity of SapBERT embeddings of labels and synonyms, > 0.75 (exact FAISS search) | similarity |
| `exact_lexical` | Identical lower-cased label or synonym | 1.0 |

Only cross-references that point to a concept present in the loaded version of the target terminology are kept. The SNOMED CT data contains active concepts only.

### 2. Rule-based filter (`MappingSetBuilder.py`)

Assigns every pair a **validity code** in a fixed order of steps, from the strongest evidence to the weakest:

| Code | Step | Meaning |
|---|---|---|
| 0 | 1 | Domain mismatch: no label of the SNOMED CT concept has the semantic tag disorder, finding or morphologic abnormality (or no tag) |
| 1 | 2 | Hierarchy reversal of an anchor (HPO cross-reference confirmed by an exact lexical match) |
| 2 | 2b | Hierarchy reversal of an exact lexical match |
| 3 | 2c | Hierarchy reversal of an HPO cross-reference |
| 4 | 3 | Hierarchy reversal of an HPO → UMLS → SNOMED CT mapping |
| 5 | 4 | Hierarchy reversal of a shared UMLS CUI mapping |
| 6 | 5 | Hierarchy reversal of a Jaccard = 1 and cosine = 1 mapping |
| 7 | 6 | Hierarchy reversal of a Jaccard = 1 only mapping |
| 8 | 7 | Best candidate: the HPO concept has no accepted mapping, and the best cosine and best Jaccard candidate agree |
| 9 | 8 | Last resort: the HPO concept has no accepted mapping, highest cosine candidate |
| 10 | 2–6 | Accepted |
| NA | – | Not reached by any step |

Steps 2–6 accept candidates. A **hierarchy reversal** means that a candidate and an SNOMED CT concept already accepted for the same HPO concept are ancestor and descendant of each other. Only one of the two is kept: the narrower one, unless the broader one has stronger evidence. The evidence of a pair is compared as the tuple (number of cross-reference sources, lexical match strength, cosine = 1). The pair that is not kept gets the reversal code of its step.

### 3. LLM review (`MappingLLM.py`)

Sends all pairs with a reviewable code (1–10) to a local LLM through the [`model`](https://github.com/Goldkugel/model) package (vLLM):

- **Pass 1** reviews HPO concepts with exactly one candidate, using `prompts/single_mapping.md`.
- **Pass 2** reviews HPO concepts with several candidates together, using `prompts/multiple_mapping.md`, so the model can compare them and apply the specificity rule.

The model answers with a rationale, a decision (ACCEPT/REJECT) and a confidence from 1 to 5. Accepted pairs are added as `llm` rows. The final set is the union of the automatically accepted pairs (code 10) and the LLM-accepted pairs; both are also evaluated separately.

### 4. Evaluation (`MappingErrorAnalysis.py`)

Compares every stage with the gold standard and logs precision, recall and F1 per stage, per validity code and per method. It also logs method agreement, confidence thresholds and precision–recall curves, the LLM's decisions per review reason, the reasons gold pairs were missed, hierarchy-depth and structural checks, and error examples. For HPO concepts in the gold standard it also logs how far each mapping is from the nearest gold target in the SNOMED CT is_a hierarchy (0 = correct, 1 = parent or child, 2 = sibling, grandparent or grandchild, ...), whether it is broader, narrower or lateral, and how close the missed gold pairs came. It writes three CSV files to `data/output/evaluation/`:

| File | Content |
|---|---|
| `evaluation_final.csv` | All rows of the final mapping set, with gold status and pair-level columns (`auto_accepted`, `llm_accepted`, `original_validity`, `methods`, `best_conf`, `gold_distance`, `nearest_gold_sct`, `gold_relation`) and the LLM's rationale from `llm_review.csv` (`llm_rationale`) |
| `accepted_without_gold.csv` | Final pairs of HPO concepts that have no mapping in the gold standard |
| `llm_disagreements.csv` | Pairs where the LLM and the rule-based filter disagree, with the LLM's decision, confidence and rationale |

### Gold standard check (`MappingGoldCheck.py`)

Checks the gold standard itself with the LLM. It is independent of the pipeline and can run before or after steps 1–4. Every unique pair of the gold standard is reviewed with the prompts of step 3, with the review reason always set to `ACCEPTED`, so the model judges each gold pair like a mapping the pipeline accepted:

- **Pass 1** reviews HPO concepts with exactly one gold target, using `prompts/single_mapping.md`.
- **Pass 2** reviews HPO concepts with several gold targets together, using `prompts/multiple_mapping.md`, so the model sees how the targets relate in the SNOMED CT hierarchy.

Pairs whose HPO or SNOMED CT concept is missing from the loaded data (obsolete or inactive concepts) cannot be shown to the model. They are not reviewed and get the status `HPO_NOT_FOUND` or `SCT_NOT_FOUND`. Answers that cannot be parsed count as rejected with confidence 1, as in step 3.

The results are written to `data/output/evaluation/gold_check.csv`:

| Column | Content |
|---|---|
| `hpo_id`, `hpo_label` | HPO concept |
| `sct_id`, `sct_label` | SNOMED CT target in the gold standard |
| `status` | `CHECKED`, `HPO_NOT_FOUND` or `SCT_NOT_FOUND` |
| `n_sct` | Number of reviewed gold targets of the HPO concept |
| `llm_pass` | 1 (pair-level prompt) or 2 (group-level prompt) |
| `llm_decision` | `ACCEPT` or `REJECT` |
| `confidence` | The model's confidence (1–5) divided by 5 |
| `llm_rationale` | The model's rationale |

The columns from `llm_pass` on are empty for pairs that were not reviewed. The log shows the counts per pass and the rejected pairs with the highest confidence, which are the first candidates for errors in the gold standard.

## Repository Structure

```
.
├── config/
│   └── config.yaml            # Configuration of all libraries (see config/README.md)
├── data/                      # Input data, outputs, logs and checkpoints (see data/README.md)
├── prompts/
│   ├── single_mapping.md      # LLM prompt for one candidate (pass 1 of step 3 and of the gold check)
│   └── multiple_mapping.md    # LLM prompt for several candidates (pass 2 of step 3 and of the gold check)
├── src/
│   ├── MappingGeneration.py   # Step 1: candidate generation
│   ├── MappingSetBuilder.py   # Step 2: rule-based filter
│   ├── MappingLLM.py          # Step 3: LLM review
│   ├── MappingErrorAnalysis.py# Step 4: evaluation
│   ├── MappingGoldCheck.py    # LLM check of the gold standard
│   └── MappingUtils.py        # Shared constants and helpers
├── start.sh                   # Runs steps 1 and 2
├── startLLM.sh                # Slurm job for step 3
├── startGoldCheck.sh          # Slurm job for the gold standard check
├── LICENSE
└── pyproject.toml
```

## Requirements & Installation

- Python 3.10+
- A CUDA GPU for the LLM review (step 3) and the gold standard check; steps 1, 2 and 4 run on the CPU
- The libraries [`logger`](https://github.com/Goldkugel/logger), [`adapter`](https://github.com/Goldkugel/adapter), [`embedder`](https://github.com/Goldkugel/embedder), [`model`](https://github.com/Goldkugel/model) and [`normalizer`](https://github.com/Goldkugel/normalizer), pinned to fixed versions in `pyproject.toml`
- `faiss` (e.g. `faiss-cpu`) and `scipy`, which are imported by `MappingGeneration.py` but not declared in `pyproject.toml`

```bash
pip install .
pip install faiss-cpu scipy
```

## Input Data

The input files are not part of the repository (licences). Place them in `data/input/`:

| Folder | Files |
|---|---|
| `data/input/hpo/` | `hp.owl` |
| `data/input/sct/` | SNOMED CT International RF2 **Full** release files (Concept, Description, Relationship, …), as listed in `config.yaml` |
| `data/input/umls/` | UMLS Metathesaurus RRF files (`MRCONSO.RRF`, `MRDEF.RRF`, `MRREL.RRF`, …) |
| `data/input/graham/` | Gold standard mapping `mapping_graham.csv` |

The adapters convert the three terminologies into CSV files in `data/output/transformed/` on the first run and reuse them afterwards (`skip_if_present`).

## Usage

All scripts use relative paths and must be started from the repository root.

```bash
./start.sh
```

```bash
sbatch ./startLLM.sh
```

```bash
python3 ./src/MappingErrorAnalysis.py
```

`start.sh` runs steps 1 and 2. `startLLM.sh` submits step 3 as a Slurm job on two A40 GPUs. Step 4 is run once the LLM review has finished.

```bash
sbatch ./startGoldCheck.sh
```

`startGoldCheck.sh` submits the gold standard check as a Slurm job on two A40 GPUs, like `startLLM.sh`. It only needs the input data, not the outputs of steps 1–4.

The LLM review and the gold standard check save a checkpoint after every chunk to `data/tmp/` (`prompt_<pass>_<model>.tmp` and `gold_<pass>_<model>.tmp`) and resume from it if the job is interrupted. **Delete these files before starting a new review or check**, otherwise answers from an earlier run are reused.

All scripts log to `data/logs/output.log`. New runs append to it. The file is tracked by git, so commit it with the results of a run to keep the log of the latest run in the repository (see `data/logs/README.md`).

## License

GNU Affero General Public License v3.0, see `LICENSE`.
