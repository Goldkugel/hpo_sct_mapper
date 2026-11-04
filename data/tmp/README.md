# tmp

Answer stores of the LLM review (`MappingLLM.py`) and of the gold standard check (`MappingGoldCheck.py`), written by the `model` package. They are not tracked by git.

Each review pass has its own store, a JSON file written after every chunk of `chunk_size` prompts (see `config.yaml`):

| File | Script and pass |
|---|---|
| `prompt_<model>_0.json` | LLM review, pass 1: HPO concepts with one candidate (`single_mapping.md`) |
| `prompt_<model>_1.json` | LLM review, pass 2: HPO concepts with several candidates (`multiple_mapping.md`) |
| `gold_<model>_0.json` | Gold standard check, pass 1: HPO concepts with one gold target (`single_mapping.md`) |
| `gold_<model>_1.json` | Gold standard check, pass 2: HPO concepts with several gold targets (`multiple_mapping.md`) |

`<model>` is the last part of the configured model ID, e.g. `Qwen2.5-72B-Instruct-GPTQ-Int4`. A `.lock` file next to a store is the lock runs take while writing it.

Every answer is stored with the SHA-256 of its prompts and the settings it depends on. If a job is interrupted, the next run builds the same prompts again, takes their answers from the store and only generates the missing ones. Answers of other prompts or data are never reused, and a store written with other model settings is refused, so the files do not have to be deleted before a new review; delete them only to generate all answers again.
