#!/bin/bash
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --exclude=dgx-station,cn01,cn02,cn03,cn04,cn05,cn06,cn07,cn08,cn09,cn11,cn12
#SBATCH -o /home/reddi/bluffjax-dev/nfsp_pqn/logs/%x_%j.out
# usage: sbatch -J <phase> --time=HH:MM:SS run_phase.sbatch <phase> [hydra overrides...]
PHASE=$1; shift
cd /home/reddi/bluffjax-dev/nfsp_pqn
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline
echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) start $(date)"
.venv/bin/python bluffjax/examples/leduc/leduc_pqn_nfsp.py \
  wandb=true project=bluffjax-leduc-push job_type=leduc_pqn_nfsp_${PHASE} \
  save_dir=/home/reddi/bluffjax-dev/nfsp_pqn/runs/${PHASE} log_br_gap=true \
  hydra.run.dir=/home/reddi/bluffjax-dev/nfsp_pqn/outputs/${PHASE}_${SLURM_JOB_ID} "$@"
echo "end $(date)"
