#!/bin/bash
#SBATCH --job-name=mapping
#SBATCH --partition=batch_long
#SBATCH --gres=gpu:a40:2                        # Fordert 1x NVIDIA A40 an
#SBATCH --time=7-00:00:00                       # Maximale Laufzeit (Format: HH:MM:SS)
#SBATCH --output=./data/slurmlogs/job_%j.out        # Speicherort für Ausgaben (%j = Job-ID)
#SBATCH --error=./data/slurmlogs/job_%j.err         # Speicherort für Fehlermeldungen

# Prioritize virtual environment libraries over system CUDA/NCCL modules
export LD_LIBRARY_PATH=/home/pallaoro/user/install/envs/gpu/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH

# Fully disable FlashInfer JIT compilation and force vLLM to use standard PyTorch/FlashAttention
export FLASHINFER_ENABLE_JIT=0
export VLLM_USE_FLASHINFER=0
export VLLM_USE_FLASHINFER_SAMPLER=0

python3 ./src/MappingLLM.py