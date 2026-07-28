#!/bin/bash
#SBATCH -J BluffJAX
#SBATCH -a 0
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem-per-cpu 32G
#SBATCH -t 18:00:00
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH -o ./logs_sbatch/%A_%a.out
#SBATCH -e ./logs_sbatch/%A_%a.err

# No existing werewolf 10e7 run yet -- need all 3 seeds.
echo "Running script."
for seed in 0 1 2; do
    python werewolf_pqn_nfsp.py --config-name=config_pqn_nfsp_10e7 seed=$seed
done
