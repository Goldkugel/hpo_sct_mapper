# tmp

Checkpoints of the LLM review (`MappingLLM.py`) and of the gold standard check (`MappingGoldCheck.py`). They are not tracked by git.

Each review pass writes its conversations to its own file after every chunk of `chunk_size` prompts (see `config.yaml`):

| File | Script and pass |
|---|---|
| `prompt_0_<model>.tmp` | LLM review, pass 1: HPO concepts with one candidate (`single_mapping.md`) |
| `prompt_1_<model>.tmp` | LLM review, pass 2: HPO concepts with several candidates (`multiple_mapping.md`) |
| `gold_0_<model>.tmp` | Gold standard check, pass 1: HPO concepts with one gold target (`single_mapping.md`) |
| `gold_1_<model>.tmp` | Gold standard check, pass 2: HPO concepts with several gold targets (`multiple_mapping.md`) |

`<model>` is the last part of the configured model ID, e.g. `Qwen2.5-72B-Instruct-GPTQ-Int4`.

If a job is interrupted, the next run imports the checkpoint and only generates the missing answers. The checkpoint is not compared with the current prompts, so **delete these files before every new review or check**. Otherwise answers from an earlier run (with other prompts or data) are reused.
