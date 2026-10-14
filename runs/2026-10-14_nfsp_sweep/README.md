# NFSP sweep on Kuhn and Leduc, 14 October 2026

The sweep that chose the NFSP settings of `runs/2026-10-14_dcb5aee/`. Code `1156562`, with each setting given as a Hydra override. 28 configurations × 3 seeds, 1e8 env steps each, every other setting as in the main runs. Kuhn and Leduc are the games with exact exploitability, which is the selection criterion.

- PPO-NFSP: `ent_coef` ∈ {0.01, 0.02, 0.05, 0.1, 0.5} × `anneal_lr` ∈ {true, false}
- PQN-NFSP: ε 1 → 0.005 with `exploration_fraction: 0.01`, or `start_e: 0.1` with `exploration_fraction: 1.0`, × `anneal_lr` ∈ {true, false}

Final exploitability (mean of the last 1% of training, 3 seeds; the main runs on seeds 0–2 in brackets):

| | Kuhn | Leduc |
|---|---|---|
| PPO-NFSP, entropy 0.1, constant lr | **0.0176** (0.205) | **0.299** (1.52) |
| PPO-NFSP, entropy 0.1, lr decay | 0.0250 | 0.455 |
| PQN-NFSP, ε over 1%, constant lr | **0.0072** (0.0210) | **0.156** (0.203) |
| PQN-NFSP, 0.1 → 0.005 over 100%, constant lr | 0.0138 | 0.160 |

Job names are `sweep_<game>_ppo_nfsp_ent<x>_{decay|constlr}` and `sweep_<game>_pqn_nfsp_eps{1to0.005over1pct|0.1to0.005over100pct}_{decay|constlr}`. Kuhn ran on nash (`launch/sweep_queue.sh`, `launch/sweep_jobs.txt`), Leduc on the IAS cluster (`launch/run_sweep.sbatch`, `launch/sweep_cluster.txt`). `launch/collect_sweep.py` extracts the per-seed curves. wandb: project `bluffjax-sweep`.

```
checkpoints/{kuhn,leduc}/<job_type>_<timestamp>/   config.yaml, results.jsonl.gz, half/, full/
logs/{kuhn,leduc}/                                 stdout of every run (gzipped)
```
