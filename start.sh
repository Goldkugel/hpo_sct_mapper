#!/bin/bash

clear

python3 ./src/MappingGeneration.py
python3 ./src/MappingSetBuilder.py

#sbatch ./startLLM.sh

#python3 ./src/MappingErrorAnalysis.py
