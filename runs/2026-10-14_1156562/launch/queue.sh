#!/bin/bash
# Runs the jobs of jobs.txt one after another under the GPU lock, skipping
# those already done. Detached with setsid/nohup, so it survives the ssh
# session. Status: logs/status.txt (name start/end rc), logs/<name>.log.
cd ~/bluffjax-dev/runs || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline WANDB_SILENT=true
while IFS='|' read -r name cmd; do
  name=$(echo "$name" | xargs); cmd=$(echo "$cmd" | sed 's/^ *//')
  [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
  grep -q "^$name done" logs/status.txt 2>/dev/null && continue
  echo "$name start $(date +%F_%T)" >> logs/status.txt
  flock ~/bluffjax-dev/gpu.lock bash -c "$cmd" > "logs/$name.log" 2>&1 < /dev/null
  rc=$?
  if [ $rc = 0 ]; then echo "$name done $(date +%F_%T)" >> logs/status.txt
  else echo "$name FAILED rc=$rc $(date +%F_%T)" >> logs/status.txt; fi
done < jobs.txt
echo "QUEUE_FINISHED $(date +%F_%T)" >> logs/status.txt
