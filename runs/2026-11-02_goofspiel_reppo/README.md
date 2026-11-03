# Goofspiel: recursive self-play PPO (ReMAPPO), 2 November 2026

ReMAPPO (`bluffjax/examples/goofspiel/goofspiel_reppo.py`, `config_reppo.yaml`) against self-play PPO on two-player Goofspiel with N = 4, 5 and 6 cards. Exploitability is exact (NashConv / 2, points per game; `bluffjax/utils/game_utils/goofspiel_exploitability.py`). The configs are those of `runs/2026-11-02_goofspiel_exploitability`; only the recursion knobs and the swept hyperparameters change. PPO runs through the same script with `reppo_k=1`, which reproduces `goofspiel_ppo.py` exactly (checked on N = 3: every logged loss, gradient norm and exploitability identical).

**Hypothesis.** In the earlier tuning, self-play PPO was stable only with a strong constant entropy bonus (0.3). With 0.1 or less, or with the bonus annealed, it collapsed to deterministic bids. Gradient play on a regularised zero-sum game is stable only for a learning rate of about 2d / (d² + ω²), with d the regularisation and ω the coupling, so it needs strong regularisation, whose fixed point is far from Nash. The recursive forecast stabilises the rotational coupling (a window of about 1/ω at K = 2), so ReMAPPO should stay stable at lower entropy and reach lower exploitability.

**Cost.** At N = 6 with 24 vmapped instances, 1e7 env steps took 91 s for K = 1 and 167 s for K = 2 on nash (RTX 3080), so K = 2 costs about 1.8× PPO's wall-clock.

## Jobs

- `launch/stage1.txt`: lr {1e-4, 3e-4, 1e-3} × ent_coef {0.01, 0.03, 0.1, 0.3}, 3 seeds, 1e8 steps, PPO (K = 1) and ReMAPPO (K = 2, forecast, self clip).
- `launch/controls.txt`: at lr {3e-4, 1e-3} × ent_coef {0.03, 0.1, 0.3}, 3 seeds:
  - PPO with 8 epochs (ReMAPPO's gradient computations);
  - K = 2 with the other players at the current parameters (no forecast);
  - K = 2 with the product ratio clipped;
  - K = 3 and K = 4.

Results go to `jobs/<name>/<name>_<timestamp>/`, stdout to `logs/` (`status.txt` tracks the queue).

Where each job ran:
- **nash:** `s1_*_n3`, `s1_*_n4` and `c_*_n4`.
- **cluster:** every other job (`launch/run_cluster.sbatch`, `launch/submit_cluster.sh`), each on one RTX 3090, A5000 or A6000.

Exploitability does not depend on the GPU; wall-clock times do. The raw `results.jsonl` files stay on the machine that ran the job, under `~/bluffjax-dev/reppo/runs/2026-11-02_goofspiel_reppo/jobs/`. `summary_nash.json` and `summary_cluster.json` hold every grid point's numbers and mean curve (`launch/collect.py`). `collect_nash.txt` and `collect_cluster.txt` are the printed tables.

## Results (3 seeds, 1e8 env steps)

Each method's best setting, chosen by the mean over the last 10% of exact evaluations (robust to last-iterate oscillation). The choice is over the grid all methods share: lr {3e-4, 1e-3} × ent_coef {0.03, 0.1, 0.3}. Each cell is last 10% (final). Every best setting has ent_coef 0.3. ReMAPPO is K = 2, forecast, self clip.

| N (uniform) | PPO | PPO, 8 epochs | ReMAPPO K = 2 | current (no forecast) | product clip | K = 3 |
|---|---|---|---|---|---|---|
| 4 (2.50) | 0.350 (0.312) | 0.359 (0.367) | **0.213 (0.215)** | 0.218 (0.234) | 0.225 (0.222) | 0.214 (0.209) |
| 5 (4.00) | 0.833 (0.903) | 0.822 (0.823) | **0.664 (0.660)** | 0.668 (0.665) | 0.674 (0.687) | 0.659 (0.668) |
| 6 (5.83) | 1.526 (1.630) | 1.465 (1.451) | **1.328 (1.330)** | 1.333 (1.321) | 1.335 (1.328) | 1.337 (1.350) |

Best final exploitability over each method's full grid:

| N | PPO | ReMAPPO K = 2 | Reduction |
|---|---|---|---|
| 4 | 0.312 (lr 3e-4, ent 0.3) | 0.215 (lr 3e-4, ent 0.3) | 31% |
| 5 | 0.886 (lr 1e-4, ent 0.3) | 0.660 (lr 3e-4, ent 0.3) | 26% |
| 6 | 1.442 (lr 1e-4, ent 0.3) | 1.311 (lr 1e-3, ent 0.1) | 9% |

For comparison, the earlier 5-seed finals (runs/2026-11-02_goofspiel_exploitability) gave:

| N | PPO | PQN-NFSP |
|---|---|---|
| 4 | 0.356 | 0.262 |
| 5 | 0.822 | 1.340 |
| 6 | 1.484 | 2.631 |

At N = 3, PPO and ReMAPPO both reach 0.000 at their best settings. ReMAPPO is worse at lr 3e-4, ent 0.3 (0.33 against 0.00).

At lower entropy (final, lr 1e-3):

| N | ent_coef | PPO | PPO, 8 epochs | ReMAPPO K = 2 | current | product clip |
|---|---|---|---|---|---|---|
| 4 | 0.1 | 2.54 | 1.33 | **0.31** | 0.31 | 2.93 |
| 5 | 0.1 | 2.45 | 1.96 | **0.78** | 0.81 | 0.84 |
| 5 | 0.03 | 5.02 | 5.94 | **2.66** | 3.16 | 5.63 |
| 6 | 0.1 | 3.73 | 2.99 | **1.31** | 1.40 | 1.47 |
| 6 | 0.03 | 8.32 | 7.69 | **3.80** | 4.27 | 4.27 |

Findings:
- **ReMAPPO beats PPO at every N ≥ 4, and the extra gradient computation does not explain it.** Its last-10% exploitability is 39 / 20 / 13% lower for N = 4 / 5 / 6, and its best final is 31 / 26 / 9% lower. PPO with 8 epochs, which matches ReMAPPO's gradient computations, gains little and remains well behind.
- **ReMAPPO keeps PPO stable at lower entropy.** At ent 0.1 it reaches 0.31 / 0.78 / 1.31 for N = 4 / 5 / 6, which matches or beats PPO's best at 0.3 for N = 5 and 6. At 0.01 every method collapses.
- **Most of the gain comes without the forecast.** `current` weights the other player's bids by the shared network's parameters at the start of the minibatch step, divided by the rollout policy. That is an importance correction that keeps the opponent in the surrogate consistent with the current policy over PPO's epochs, instead of frozen at the rollout. It matches the forecast at ent 0.3 and nearly matches it at 0.1. The forecast adds a clear gain only at the lowest stable entropy, 0.03 (2.66 against 3.16, and 3.80 against 4.27).
- **Clipping the product ratio hurts at low entropy**, as bridge.tex rem:clip predicts: 2.93 against 0.31 (N = 4, ent 0.1) and 5.63 against 2.66 (N = 5, ent 0.03). It is harmless at 0.3.
- **Depth saturates at K = 2.** K = 3 and K = 4 match K = 2 within noise.
- **Cost.** ReMAPPO K = 2 took 2,139 s against 1,115 s for PPO at N = 4 on nash, 1.9×. `current` as implemented also runs the level-1 step it does not use. A version without that step would cost about PPO plus one forward pass.
