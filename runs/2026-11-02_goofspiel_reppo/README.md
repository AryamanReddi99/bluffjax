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

## Results

Pending.
