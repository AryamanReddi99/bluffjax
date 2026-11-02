#!/bin/bash
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=dgx-station,cn01,cn02,cn03,cn04,cn05,cn06,cn07,cn08,cn09,cn11,cn12
# Usage: sbatch -J NAME deep_cfr_job.sh GAME TAG [extra flags...]
GAME=$1; TAG=$2; shift 2
cd ~/bluffjax-dev/cfr
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
OUT=~/bluffjax-dev/cfr/runs/leduc_push/$TAG
mkdir -p $OUT
nvidia-smi --query-gpu=name,memory.total --format=csv > $OUT/gpu.txt; hostname >> $OUT/gpu.txt
for s in ${SEEDS:-0 1 2 3 4}; do
  mkdir -p $OUT/seed$s
  .venv/bin/python bluffjax/examples/$GAME/${GAME}_deep_cfr.py --seed $s --num_seeds 1 \
    --save_dir $OUT/seed$s --job_type ${GAME}_deep_cfr_$TAG --wandb --project bluffjax-leduc-push \
    --output $OUT/seed$s/results.json "$@" > $OUT/seed$s/log.txt 2>&1 &
done
wait
