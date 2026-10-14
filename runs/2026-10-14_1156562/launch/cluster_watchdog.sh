#!/bin/bash
# Resubmits a failed cluster job (state FAILED/NODE_FAIL/OUT_OF_MEMORY, not
# completed or still queued) up to 2 times. Runs on nash, detached; checks
# every 10 minutes until every job has completed or used its retries.
# jobs: name|time|script args (the run_final.sbatch arguments)
LOG=~/bluffjax-dev/cluster_watchdog.log
JOBS=$(cat <<'EOF'
hul_ppo_nfsp|8:00:00|HUL/hul_ppo_nfsp.py
hul_pqn_nfsp|8:00:00|HUL/hul_pqn_nfsp.py
hunl_ppo_nfsp|8:00:00|HUNL/hunl_ppo_nfsp.py
hunl_pqn_nfsp|8:00:00|HUNL/hunl_pqn_nfsp.py
seven_card_stud_ppo_nfsp|8:00:00|seven_card_stud/seven_card_stud_ppo_nfsp.py
seven_card_stud_pqn_nfsp|8:00:00|seven_card_stud/seven_card_stud_pqn_nfsp.py
kemps_ppo_nfsp|8:00:00|kemps/kemps_ppo_nfsp.py
kemps_pqn_nfsp|8:00:00|kemps/kemps_pqn_nfsp.py
bluff_ppo_nfsp|8:00:00|bluff/bluff_ppo_nfsp.py
bluff_pqn_nfsp|8:00:00|bluff/bluff_pqn_nfsp.py
hul_rebel_s0|1-00:00:00|HUL/hul_rebel.py seed=0
hul_rebel_s1|1-00:00:00|HUL/hul_rebel.py seed=1
hul_rebel_s2|1-00:00:00|HUL/hul_rebel.py seed=2
hul_rebel_s3|1-00:00:00|HUL/hul_rebel.py seed=3
hul_rebel_s4|1-00:00:00|HUL/hul_rebel.py seed=4
hunl_rebel_s0|2-00:00:00|HUNL/hunl_rebel.py seed=0
hunl_rebel_s1|2-00:00:00|HUNL/hunl_rebel.py seed=1
hunl_rebel_s2|2-00:00:00|HUNL/hunl_rebel.py seed=2
hunl_rebel_s3|2-00:00:00|HUNL/hunl_rebel.py seed=3
hunl_rebel_s4|2-00:00:00|HUNL/hunl_rebel.py seed=4
EOF
)
declare -A RETRIES
while true; do
  pending=0
  # latest state of every job name (sacct since the launch)
  STATES=$(ssh -o ConnectTimeout=30 cluster "sacct -u reddi -S 2026-10-14T00:30 -X -n -P -o jobid,jobname,state" 2>/dev/null)
  if [ -z "$STATES" ]; then echo "$(date +%F_%T) cluster unreachable" >> $LOG; sleep 600; continue; fi
  while IFS='|' read -r name time args; do
    state=$(echo "$STATES" | awk -F'|' -v n="$name" '$2==n {s=$3} END {print s}')
    case "$state" in
      COMPLETED) ;;
      FAILED|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL)
        r=${RETRIES[$name]:-0}
        if [ "$r" -lt 2 ]; then
          out=$(ssh -n -o ConnectTimeout=30 cluster "cd ~/bluffjax-dev/runs && sbatch -J $name -t $time run_final.sbatch $args" 2>&1)
          RETRIES[$name]=$((r + 1)); pending=1
          echo "$(date +%F_%T) $name was $state, resubmitted ($((r + 1))/2): $out" >> $LOG
        fi ;;
      PENDING|RUNNING|REQUEUED|CONFIGURING|COMPLETING|"") pending=1 ;;
      *) ;;  # TIMEOUT / CANCELLED: left for me to look at
    esac
  done <<< "$JOBS"
  [ $pending = 0 ] && { echo "$(date +%F_%T) all jobs settled" >> $LOG; break; }
  sleep 600
done
