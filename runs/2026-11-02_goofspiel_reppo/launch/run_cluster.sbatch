#!/bin/bash
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=8G  # MaxRSS of a job is about 1.5 GB
#SBATCH --exclude=dgx-station,cn01,cn02,cn03,cn04,cn05,cn06,cn07,cn08,cn09,cn11,cn12
#SBATCH -o /home/reddi/bluffjax-dev/reppo/runs/2026-11-02_goofspiel_reppo/logs/%x-%j.out
# One ReMAPPO job on the cluster (code: ~/bluffjax-dev/reppo, the
# reppo-goofspiel branch). Usage (launch/submit_cluster.sh does this for a
# jobs file): sbatch -J <name> -t <time> run_cluster.sbatch <script> [hydra overrides]
set -euo pipefail
CODE=~/bluffjax-dev/reppo
RUN=$CODE/runs/2026-11-02_goofspiel_reppo
PY=~/bluffjax-dev/ppo/.venv/bin/python  # jax 0.7.2 + CUDA 12
cd "$CODE"
export PYTHONPATH=$CODE XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
NAME=$SLURM_JOB_NAME
echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) commit $(cat $RUN/launch/COMMIT 2>/dev/null) start $(date)"
echo "$NAME start $(date +%F_%T) job $SLURM_JOB_ID $(hostname)" >> "$RUN/logs/status_cluster.txt"
SCRIPT=$1; shift
if $PY bluffjax/examples/goofspiel/$SCRIPT job_type=$NAME save_dir=$RUN/jobs/$NAME \
    hydra.run.dir=$RUN/hydra/$NAME "$@"; then
  echo "$NAME done $(date +%F_%T)" >> "$RUN/logs/status_cluster.txt"
else
  echo "$NAME FAILED $(date +%F_%T)" >> "$RUN/logs/status_cluster.txt"; exit 1
fi
echo "end $(date)"
