#!/bin/bash
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --exclude=dgx-station,cn01,cn02,cn03,cn04,cn05,cn06,cn07,cn08,cn09,cn11,cn12
#SBATCH -o /home/reddi/bluffjax-dev/runs/logs/%x-%j.out
# Final BluffJAX runs (code: ~/bluffjax-dev/runs, commit in COMMIT).
# Usage: sbatch -J <name> -t <time> run_final.sbatch <game>/<script>.py [hydra overrides]
set -euo pipefail
cd ~/bluffjax-dev/runs
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline WANDB_SILENT=true
echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) commit $(cat COMMIT) start $(date)"
SCRIPT=$1; shift
.venv/bin/python bluffjax/examples/$SCRIPT wandb=True project=bluffjax-final "$@"
echo "end $(date)"
