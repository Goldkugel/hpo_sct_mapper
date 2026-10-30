# logs

Log output of the pipeline. Log files are not tracked by git.

`output.log` is written by all four scripts (via [`logger`](https://github.com/Goldkugel/logger)). It holds the progress, counts and evaluations of every step, and the evaluation report of `MappingErrorAnalysis.py`. The location and file name are set in the `logger` section of `config/config.yaml`.

New runs append to `output.log`. Rename or move it before a run to keep the logs of different runs apart (e.g. `output_old.log`).

The full prompts and answers of the LLM review are not logged here: they are kept in the checkpoints in `data/tmp/`, and every decision with its rationale is written to `data/output/filtered/llm_review.csv`.
