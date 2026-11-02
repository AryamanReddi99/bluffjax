#!/bin/bash
# usage: run_nash.sh <phase> [hydra overrides...]  (runs under the nash GPU lock)
PHASE=$1; shift
cd ~/bluffjax-dev/nfsp_pqn
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline
echo "host $(hostname) start $(date)"
flock ~/bluffjax-dev/gpu.lock .venv/bin/python bluffjax/examples/leduc/leduc_pqn_nfsp.py \
  wandb=true project=bluffjax-leduc-push job_type=leduc_pqn_nfsp_${PHASE} \
  save_dir=$HOME/bluffjax-dev/nfsp_pqn/runs/${PHASE} log_br_gap=true \
  hydra.run.dir=$HOME/bluffjax-dev/nfsp_pqn/outputs/${PHASE} "$@"
echo "end $(date)"
