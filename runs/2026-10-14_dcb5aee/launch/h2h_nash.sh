#!/bin/bash
# Head-to-head of the 8 non-Hold'em games on nash, after the NFSP training queue: NFSP from runs2, the rest from runs.
cd ~/bluffjax-dev/runs2 || exit 1
until grep -q QUEUE_FINISHED logs/status.txt; do sleep 60; done
export XLA_PYTHON_CLIENT_PREALLOCATE=false
C="dcb5aee (PPO-NFSP, PQN-NFSP); 1156562 (other baselines)"
R=~/bluffjax-dev/h2h_v2_root
mkdir -p head_to_head logs/h2h
.venv/bin/python make_h2h_root.py ~/bluffjax-dev/runs/checkpoints checkpoints $R kuhn leduc five_card_draw seven_card_stud goofspiel bluff kemps werewolf > logs/h2h/root.log 2>&1 || { echo "ROOT_FAILED" >> logs/h2h/status.txt; exit 1; }
run() { name=$1; shift; echo "$name start $(date +%F_%T)" >> logs/h2h/status.txt
  flock ~/bluffjax-dev/gpu.lock "$@" > logs/h2h/$name.log 2>&1 < /dev/null
  echo "$name rc=$? $(date +%F_%T)" >> logs/h2h/status.txt; }
run kuhn_leduc .venv/bin/python -m bluffjax.examples.head_to_head.kuhn_leduc $R head_to_head --commit "$C" --runs_root_label "~/bluffjax-dev/h2h_v2_root"
for g in five_card_draw seven_card_stud goofspiel bluff kemps werewolf; do
  run $g .venv/bin/python -m bluffjax.examples.head_to_head.games $R $g head_to_head --commit "$C"
done
echo "H2H_FINISHED $(date +%F_%T)" >> logs/h2h/status.txt
