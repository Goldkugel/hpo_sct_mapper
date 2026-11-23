#!/bin/bash

clear

python3 ./src/MappingGeneration.py
python3 ./src/MappingSetBuilder.py

sbatch ./startLLM.sh
sbatch ./startGoldCheck.sh

#python3 ./src/MappingErrorAnalysis.py
