#!/bin/bash
#SBATCH -J BluffJAX
#SBATCH -a 0
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem-per-cpu 32G
#SBATCH -t 04:00:00
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH -o ./logs_sbatch/%A_%a.out
#SBATCH -e ./logs_sbatch/%A_%a.err

echo "Running script."
for seed in 0 1 2; do
    python goofspiel_ppo_nfsp.py seed=$seed
done
