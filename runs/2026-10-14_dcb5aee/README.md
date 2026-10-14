# BluffJAX NFSP reruns, 14 October 2026

PPO-NFSP and PQN-NFSP in all 10 games, trained fresh with the code at commit `dcb5aee`, which differs from `1156562` (the other baselines, in `runs/2026-10-14_1156562/`) only in the NFSP configs. The new settings are the best of the Kuhn/Leduc sweep (`runs/2026-10-14_nfsp_sweep/`), applied to every game:

- PPO-NFSP: `ent_coef: 0.1` (was 0.0), `anneal_lr: false` (was true)
- PQN-NFSP: `exploration_fraction: 0.01` (ε 1 → 0.005 over the first 1% of the run; was 0.1), `anneal_lr: false` (was true)

Everything else is unchanged: 5 seeds per baseline, 1e8 env steps, a **half** and a **full** checkpoint from the same run. The head-to-head matrices in `head_to_head/` are new for all 10 games. They pair these NFSP checkpoints with the other baselines' unchanged checkpoints from `runs/2026-10-14_1156562/checkpoints/`.

## Layout

```
checkpoints/<game>/<job_type>_<timestamp>/
    config.yaml           the resolved config of the run
    results.jsonl.gz      everything the run logged, one JSON object per line (gzipped)
    half/ , full/         avg_policy_{i} (NFSP's output) and br_{i} after half and all of the budget
head_to_head/<game>.json  head-to-head matrices (every checkpoint against every other, 32 episodes per pair)
head_to_head/checks/      the Hold'em evaluator's sanity checks and wall times
launch/                   the scripts that ran everything
logs/                     stdout of every run and evaluation (gzipped); logs/cluster/ for the cluster jobs
```

Read a run's log with `bluffjax.examples.run_utils.read_results(run_dir)`; load a checkpoint with `run_utils.load_params(path, template)`.

## Where they ran

- nash (RTX 3080): Kuhn, Leduc, Goofspiel, Werewolf, 5-Card Draw (`launch/queue.sh`, `launch/jobs.txt`).
- IAS cluster (RTX 3090 / A5000): both Hold'em games, 7-Card Stud, Kemps, Bluff (`launch/run_final.sbatch`). One job (No-Limit Hold'em PPO-NFSP) failed at start-up on the shared filesystem's transient missing-file error and was resubmitted unchanged.
- Head-to-head: Hold'em on the cluster (`launch/run_h2h.sbatch`), the other 8 games on nash (`launch/h2h_nash.sh`). Each evaluator reads one runs root, built by `launch/make_h2h_root.py` from symlinks: the NFSP runs here and every other baseline's run in `runs/2026-10-14_1156562/checkpoints/`. That root is `~/bluffjax-dev/h2h_v2_root`, the `runs_root` recorded in the JSONs.
- wandb: logged offline to project `bluffjax-final-v2`, then synced.
