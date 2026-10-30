# data

Input data, outputs, logs and checkpoints of the pipeline. Except for the READMEs and the Slurm logs, the contents are not tracked by git.

```
data/
├── input/                 # Input data (not included, see the main README)
│   ├── hpo/               # hp.owl
│   ├── sct/               # SNOMED CT RF2 Full release files
│   ├── umls/              # UMLS RRF files
│   └── graham/            # Gold standard mapping_graham.csv
├── output/
│   ├── transformed/       # Terminologies as CSV files, written by the adapters
│   │   ├── hpo/hpo.csv
│   │   ├── sct/sct.csv
│   │   └── umls/umls.csv
│   ├── mapped/
│   │   └── mapping.csv        # Step 1: candidate pairs, one row per pair and method
│   ├── filtered/
│   │   ├── filtered.csv       # Step 2: candidate pairs with validity codes
│   │   ├── llm_filtered.csv   # Step 3: filtered.csv plus the LLM-accepted pairs ("llm" rows)
│   │   └── llm_review.csv     # Step 3: every LLM decision with confidence and rationale
│   └── evaluation/
│       ├── evaluation_final.csv
│       ├── accepted_without_gold.csv
│       └── llm_disagreements.csv
├── logs/                  # Logs and run metadata (see logs/README.md)
├── slurmlogs/             # stdout and stderr of the Slurm jobs (job_<id>.out / .err)
└── tmp/                   # Checkpoints of the LLM review (see tmp/README.md)
```

All CSV files use `;` as separator. The output files are described in the main README.
