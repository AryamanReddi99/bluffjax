# BluffJAX baseline runs, 14 October 2026

Every baseline in every game, trained fresh with the code at commit `1156562`: 5 seeds per baseline (CFR: one deterministic run), each run saving a **half** and a **full** checkpoint from the same run. Settings: `bluffjax-experiments.md` in the project notes, and each run's own `config.yaml`.

## Layout

```
checkpoints/<game>/<job_type>_<timestamp>/
    config.yaml           the resolved config of the run (job_type, env_name, env_kwargs, seeds, ...)
    results.jsonl.gz      everything the run logged, one JSON object per line (gzipped)
    half/ , full/         the checkpoints after half and all of the budget
head_to_head/<game>.json  head-to-head matrices (every checkpoint against every other)
launch/                   the scripts that ran everything (nash queue, cluster job script, watchdog, collector)
logs/                     stdout of every run (gzipped); logs/cluster/ for the cluster jobs
```

Read a run's log with `bluffjax.examples.run_utils.read_results(run_dir)`, which reads `results.jsonl.gz`. Load a checkpoint with `run_utils.load_params(path, template)`; each training script's docstring (`save_checkpoints`) names the network in every file:

| Baseline | Files (per seed index i) | Budget (half at 1/2) |
|---|---|---|
| PPO-NFSP, PQN-NFSP | `avg_policy_{i}` (NFSP's output), `br_{i}` | 1e8 env steps |
| PPO, PQN (Kuhn, Leduc) | `policy_{i}`, `q_network_{i}` | 1e8 env steps |
| CFR (Kuhn, Leduc) | `average_policy` (one run) | 100 iterations |
| Deep CFR (Kuhn, Leduc) | `avg_policy_{i}` | 50 iterations |
| ReBeL, Kuhn | `value_net_{i}` | 4,000 updates |
| ReBeL, Hold'em | `value_net_0` (+ `.json` metadata), one run directory per seed | 1e7 samples |

## Where they ran

- nash (RTX 3080): Kuhn and Leduc (all baselines), Goofspiel, Werewolf and 5-Card Draw, one after another (`launch/queue.sh`, `launch/jobs.txt`).
- IAS cluster (RTX 3090 / A5000): both Hold'em games (NFSP and ReBeL), 7-Card Stud, Kemps and 3-player Bluff (`launch/run_final.sbatch`). Five jobs failed at start-up on a transient missing-file error in the shared Python environment and were resubmitted unchanged.
- wandb logged offline to project `bluffjax-final`.
