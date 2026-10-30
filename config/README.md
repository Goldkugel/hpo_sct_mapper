# config

`config.yaml` configures the libraries the pipeline uses. Each library reads its own section and validates it against its configuration class; unknown keys raise an error.

| Section | Read by | Configures |
|---|---|---|
| `adapter.hpo`, `adapter.sct`, `adapter.umls` | [`adapter`](https://github.com/Goldkugel/adapter) | Input files of each terminology, the transformed CSV file, and whether an existing CSV file is reused (`skip_if_present`) |
| `logger` | [`logger`](https://github.com/Goldkugel/logger) | Log folder and file (`data/logs/output.log`), header style, time format |
| `normalizer` | [`normalizer`](https://github.com/Goldkugel/normalizer) | Token normalisation for the Jaccard similarity: lower-casing, lemmatising, stemming, stop-word removal |
| `llm` | [`model`](https://github.com/Goldkugel/model) | LLM (`model_id`), sampling (`temperature`, `top_p`, `seed`), context and answer length (`max_model_len`, `max_tokens`), chunk size, prompt files |
| `embedder` | [`embedder`](https://github.com/Goldkugel/embedder) | Embedding model (SapBERT), device, batch size, normalisation; the UMAP settings are unused (`dimensionality_reduction: False`) |

Settings that belong to the pipeline itself, such as file paths, similarity thresholds and acceptable SNOMED CT domains, are constants in `src/MappingUtils.py`.

When changing the SNOMED CT release, update the file names in `adapter.sct.input_files` and delete the transformed CSV files in `data/output/transformed/`, so they are created again from the new release.
