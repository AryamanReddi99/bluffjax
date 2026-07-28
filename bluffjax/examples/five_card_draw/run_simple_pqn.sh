#!/bin/bash
# Plain (non-SLURM) equivalent of ias_pqn.sh, for machines without a SLURM
# scheduler (e.g. nash). Run from this directory with the `bluff` conda env active.
echo "Running script."
for seed in 0 1 2; do
    python five_card_draw_pqn_nfsp.py --config-name=config_pqn_nfsp_simple seed=$seed
done
