#!/bin/bash
# Resubmits a failed cluster job (state FAILED/NODE_FAIL/OUT_OF_MEMORY, not
# completed or still queued) up to 2 times. Runs on nash, detached; checks
# every 3 minutes until every job has completed or used its retries.
# jobs: name|time|script args (the run_final.sbatch arguments)
LOG=~/bluffjax-dev/cluster_watchdog_v2.log
JOBS=$(cat <<'EOF'
hul_ppo_nfsp|3:00:00|HUL/hul_ppo_nfsp.py
hul_pqn_nfsp|3:00:00|HUL/hul_pqn_nfsp.py
hunl_ppo_nfsp|3:00:00|HUNL/hunl_ppo_nfsp.py
hunl_pqn_nfsp|3:00:00|HUNL/hunl_pqn_nfsp.py
seven_card_stud_ppo_nfsp|3:00:00|seven_card_stud/seven_card_stud_ppo_nfsp.py
seven_card_stud_pqn_nfsp|3:00:00|seven_card_stud/seven_card_stud_pqn_nfsp.py
kemps_ppo_nfsp|3:00:00|kemps/kemps_ppo_nfsp.py
kemps_pqn_nfsp|3:00:00|kemps/kemps_pqn_nfsp.py
bluff_ppo_nfsp|3:00:00|bluff/bluff_ppo_nfsp.py
bluff_pqn_nfsp|3:00:00|bluff/bluff_pqn_nfsp.py
EOF
)
declare -A RETRIES
while true; do
  pending=0
  # latest state of every job name (sacct since the launch)
  STATES=$(ssh -o ConnectTimeout=30 cluster "sacct -u reddi -S 2026-10-14T21:00 -X -n -P -o jobid,jobname,state" 2>/dev/null)
  if [ -z "$STATES" ]; then echo "$(date +%F_%T) cluster unreachable" >> $LOG; sleep 180; continue; fi
  while IFS='|' read -r name time args; do
    state=$(echo "$STATES" | awk -F'|' -v n="$name" '$2==n {s=$3} END {print s}')
    case "$state" in
      COMPLETED) ;;
      FAILED|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL)
        r=${RETRIES[$name]:-0}
        if [ "$r" -lt 2 ]; then
          out=$(ssh -n -o ConnectTimeout=30 cluster "cd ~/bluffjax-dev/runs2 && sbatch -J $name -t $time run_final.sbatch $args" 2>&1)
          RETRIES[$name]=$((r + 1)); pending=1
          echo "$(date +%F_%T) $name was $state, resubmitted ($((r + 1))/2): $out" >> $LOG
        fi ;;
      PENDING|RUNNING|REQUEUED|CONFIGURING|COMPLETING|"") pending=1 ;;
      *) ;;  # TIMEOUT / CANCELLED: left for me to look at
    esac
  done <<< "$JOBS"
  [ $pending = 0 ] && { echo "$(date +%F_%T) all jobs settled" >> $LOG; break; }
  sleep 180
done
