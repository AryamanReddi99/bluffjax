#!/bin/bash
#SBATCH -J BluffJAX
#SBATCH -a 0
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem-per-cpu 32G
#SBATCH -t 12:00:00
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH -o ./logs_sbatch/%A_%a.out
#SBATCH -e ./logs_sbatch/%A_%a.err

# seed 0 already exists in checkpoints/10e7 (ppo_nfsp_2026-07-25_18-37-00_s0) --
# only seeds 1,2 are needed to reach 3 total.
echo "Running script."
for seed in 1 2; do
    python bluff_ppo_nfsp.py --config-name=config_ppo_nfsp_10e7 seed=$seed
done
