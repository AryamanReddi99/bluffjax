#!/bin/bash
# Syncs every not-yet-synced offline wandb run of the NFSP reruns and the sweep.
cd ~/bluffjax-dev || exit 1
LOG=~/bluffjax-dev/wandb_sync_v2.log; : > $LOG
for d in runs2/wandb runs2/wandb_cluster runs/wandb runs/wandb_cluster_sweep; do
  for r in $d/offline-run-*; do
    ls $r/*.synced >/dev/null 2>&1 && continue
    if runs2/.venv/bin/wandb sync "$r" >> $LOG 2>&1; then echo "OK $r" >> $LOG; else echo "FAIL $r" >> $LOG; fi
  done
done
echo "SYNC_FINISHED $(grep -c '^OK ' $LOG) ok $(grep -c '^FAIL ' $LOG) failed" >> $LOG
