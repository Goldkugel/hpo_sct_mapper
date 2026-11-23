#!/bin/bash
# ==========================================================================
# cleanUp.sh
#
# Deletes every file the pipeline writes, so the next run starts from the
# input data only and reuses nothing of an earlier run:
#   - data/output/transformed/   ontology CSVs of the adapters
#   - data/output/mapped/        mapping.csv
#   - data/output/filtered/      filtered.csv, llm_filtered.csv, llm_review.csv
#   - data/output/evaluation/    evaluation CSVs, gold_check.csv and plots/
#   - data/tmp/                  LLM answer stores (*.json) and their locks
#   - data/logs/                 output.log, prompts.log, run_metadata.json
#                                and older logs
#   - data/slurmlogs/            job_<id>.out / .err
# Never touched: data/input/, the README.md files, code and configuration.
#
# Usage (from any directory):
#   ./cleanUp.sh       list the files and ask before deleting
#   ./cleanUp.sh -n    only list the files (dry run)
#   ./cleanUp.sh -y    delete without asking (e.g. in a job script)
# ==========================================================================

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

mode="ask"
case "${1:-}" in
    -n) mode="dry" ;;
    -y) mode="yes" ;;
    "") ;;
    *)  echo "Usage: $0 [-n | -y]" >&2; exit 1 ;;
esac

folders=(
    data/output/transformed
    data/output/mapped
    data/output/filtered
    data/output/evaluation
    data/tmp
    data/logs
    data/slurmlogs
)

# All files below the folders except the README.md files.
files=()
for folder in "${folders[@]}"; do
    [ -d "$folder" ] || continue
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(find "$folder" -type f ! -name README.md -print0 | sort -z)
done

if [ ${#files[@]} -eq 0 ]; then
    echo "Nothing to delete."
    exit 0
fi

printf '%s\n' "${files[@]}"
echo "${#files[@]} files."

if [ "$mode" = "dry" ]; then
    exit 0
fi
if [ "$mode" = "ask" ]; then
    read -r -p "Delete these files? [y/N] " answer
    case "$answer" in
        y|Y|yes|Yes) ;;
        *) echo "Nothing deleted."; exit 0 ;;
    esac
fi

rm -f -- "${files[@]}"
# Empty subfolders (e.g. evaluation/plots, transformed/hpo); the adapters
# and scripts create them again.
for folder in "${folders[@]}"; do
    [ -d "$folder" ] && find "$folder" -mindepth 1 -type d -empty -delete
done
echo "Deleted ${#files[@]} files."
