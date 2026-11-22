# Goofspiel: CoPPO and extragradient PPO at the matched budget, 21 November 2026

Two more baselines for the comparison in `runs/2026-11-03_goofspiel_matched`, with the same rules: lr 1e-3, entropy 0.1, 10 seeds, N = 4, 5, 6, exact exploitability every 50 updates, half/full checkpoints, PPO's 244,128 policy backward passes.
- **CoPPO** (`goofspiel_reppo.py`: `reppo_k=1 level1_weight=current clip_mode=product nonself_eps2=<eps2>`): one backward pass per minibatch step, 1e9 samples.
- **Extragradient PPO** (`goofspiel_egppo.py`): two backward passes per minibatch step, 5e8 samples.

Each method's own knob was screened first at N = 4, 3 seeds, at the screens' budget (the backward passes of 1e8 PPO samples).

## Screens

Settings are ranked by the mean of each run's last 10% of exact evaluations (`collect_screen.py`; all points are in `../screens.json`).

| Screen | Where | Values → last-10% exploitability | Picked |
|---|---|---|---|
| CoPPO `nonself_eps2`, 1e8 samples (`jobs/scr_coppo_n4`) | nash RTX 3080 | 0.05 / 0.1 / 0.2 / 0.4 → 2.342 / 2.070 / 2.505 / 2.459 | 0.1 |
| EG `eg_probe_scale`, round 1, 5e7 samples | Determined A100 | 0.1 / 0.25 / 0.5 / 1 / 2 → 3.019 / 2.768 / 2.774 / 2.694 / 2.494 | (edge) |
| EG round 2 | Determined A100 | 2 / 4 / 8 / 16 → 1.839 / 2.169 / 1.746 / 1.333 | (edge) |
| EG round 3 | Determined A100 | 16 / 32 / 64 / 128 → 0.936 / 2.202 / 1.356 / 1.482 | 16 |

Every CoPPO setting collapsed (entropy < 0.005). The inner clip kept the opponent weight within about 1% of 1, and the joint clip fraction was about 0.001. The EG screens ran on Determined from a context dir on nash (`~/det_ctx/egscreen`: a `git archive` of this branch, `run.sh`, `job.yaml`). Results came back as a base64 tar in the command logs. GPUs differ between rounds, so the same setting can score differently: scale 2 gave 2.494 in round 1 and 1.839 in round 2.

## Full runs (`jobs_coppo.txt`, `jobs_eg.txt`; SLURM A5000, jobs 224524–224530)

Final exploitability, mean ± std over 10 seeds:

| N | CoPPO (ε₂ 0.1, 1e9) | Extragradient PPO (×16, 5e8) | PPO (1e9, earlier) |
|---|---|---|---|
| 4 | 2.598 ± 0.339 | 2.090 ± 0.720 | 2.409 |
| 5 | 1.109 ± 0.173 | 1.950 ± 0.364 | 4.517 |
| 6 | 1.408 ± 0.159 | 2.011 ± 1.007 | 2.574 |

Both are far behind MMD (0.045 / 0.162 / 0.472), tuned ReMAPPO (0.105 / 0.379 / 0.910) and POLA (0.083 / 0.378 / 0.924). The seed mean of extragradient PPO reaches 0.944 at N = 4 (82M samples) and 1.159 at N = 5 (207M), then drifts back up. At N = 4, 8 of 10 seeds end near uniform play. The 5e7-sample screen could not see this. The data is in `recursive-reasoning-marl/.../evaluation_new/goofspiel/n{4,5,6}/data/{COPPO,EG_PPO}*.npy` (fetched with `download_results.py COPPO EG_PPO`).
