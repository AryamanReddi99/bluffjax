#!/bin/bash
#SBATCH -J BluffJAX_PGX
#SBATCH -a 0 # Controls the number of replication
#SBATCH -n 1 # Always leave this value at 1. This is only used for MPI, which is not supported now.
#SBATCH -c 1
#SBATCH --mem-per-cpu 2G
#SBATCH -t 00:30:00
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx6000Ada:1
#SBATCH -o ./logs_sbatch/%A_%a.out
#SBATCH -e ./logs_sbatch/%A_%a.err ## Make sure to create the logs directory

echo "Running PGX memory benchmark."
XLA_PYTHON_CLIENT_PREALLOCATE=false python benchmark_pgx.py "$@"
