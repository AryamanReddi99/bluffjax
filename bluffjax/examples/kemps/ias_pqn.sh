#!/bin/bash
#SBATCH -J BluffJAX
#SBATCH -a 0 # Controls the number of replication
#SBATCH -n 1  ## ALWAYS leave this value to 1. This is only used for MPI, which is not supported now.
#SBATCH -c 1
#SBATCH --mem-per-cpu 32G
#SBATCH -t 04:00:00
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH -o ./logs_sbatch/%A_%a.out
#SBATCH -e ./logs_sbatch/%A_%a.err ## Make sure to create the logs directory


echo "Running script."
for seed in 0 1 2; do
    python kemps_pqn_nfsp.py seed=$seed
done
