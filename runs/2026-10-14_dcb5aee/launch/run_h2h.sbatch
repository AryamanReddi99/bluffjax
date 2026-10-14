#!/bin/bash
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --exclude=dgx-station,cn01,cn02,cn03,cn04,cn05,cn06,cn07,cn08,cn09,cn11,cn12
#SBATCH -o /home/reddi/bluffjax-dev/runs2/logs/%x-%j.out
# Hold'em head-to-head of the NFSP reruns (runs2) and the existing ReBeL runs (runs), via ~/bluffjax-dev/h2h_v2_root.
# Usage: sbatch -J h2h_<game> -t <time> run_h2h.sbatch <hul|hunl>
set -euo pipefail
cd ~/bluffjax-dev/runs2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) commit $(cat COMMIT) start $(date)"
.venv/bin/python -m bluffjax.examples.head_to_head.holdem ~/bluffjax-dev/h2h_v2_root $1 ~/bluffjax-dev/runs2/head_to_head \
    --commit "dcb5aee (PPO-NFSP, PQN-NFSP); 1156562 (ReBeL)"
echo "end $(date)"
