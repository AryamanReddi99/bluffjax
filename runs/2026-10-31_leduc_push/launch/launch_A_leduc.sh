#!/bin/bash
cd ~/bluffjax-dev/cfr-push
rm -rf runs/leduc_push/A_leduc_*
for v in cfr cfr_plus linear dcfr; do
  d=runs/leduc_push/A_leduc_$v; mkdir -p $d
  WANDB_MODE=offline JAX_PLATFORMS=cpu nohup .venv/bin/python bluffjax/examples/leduc/leduc_cfr.py --variant $v --iterations 2000 --eval_interval 100 --save_dir $d --job_type leduc_cfr_$v --wandb --project bluffjax-leduc-push > $d/log.txt 2>&1 < /dev/null &
done
