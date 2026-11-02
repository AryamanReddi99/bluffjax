# Leduc exploitability push, 31 October – 2 November 2026

The goal was to get Leduc Hold'em exploitability to ≤ 0.05 with as many of our implementations as possible. First came an independent audit of every algorithm (its reports are in the project notes, `bluffjax-audit/`). Then came fixes and experiments, one phase per algorithm.

Exploitability is NashConv/2 in chips per hand, computed exactly over the full game tree. This is OpenSpiel's `exploitability`: the audit checked our evaluator against OpenSpiel to 1e-6 on every first-round checkpoint, and the new runs use the same evaluator. It is on the same scale as the NFSP paper's ≈ 0.06. Stochastic methods ran 5 seeds; the CFR variants are deterministic, so each has one run.

| Method | First round | Best now | Folder |
|---|---|---|---|
| CFR / CFR+ / Linear CFR / Discounted CFR | 0.0957 (CFR, 100 it.) | 0.0068 / 0.000085 / 0.0032 / 0.000032 (2,000 it.) | `A/` |
| ReBeL (new for Leduc) | – | 0.0174 ± 0.0021 | `E/` |
| PQN-NFSP | 0.183 ± 0.043 | 0.0443 ± 0.0041 (3e9 env steps) | `C5b/` |
| SD-CFR (new) | – | 0.0482 ± 0.0024 | `D/` |
| PPO-NFSP | 0.316 ± 0.023 | 0.0535 ± 0.0022 (1e9 env steps) | `G/` |
| Deep CFR | 0.293 ± 0.045 | 0.0847 ± 0.0036 | `D/` |
| Self-play PPO | 1.669 ± 0.255 | 1.199 ± 0.397 (ent_coef 0.01) | `PPO_ent001/` |

Every run directory keeps:
- `config.yaml`;
- `results.jsonl.gz`, read with `bluffjax.examples.run_utils.read_results`;
- the `half/` and `full/` checkpoints.

Every run's results file is complete. Three extra run directories carry a `README.txt` saying what they are and are not used in any result: the cancelled 3e9 PPO-NFSP run (`F7/`), the C0 reproducibility rerun (`C0/`) and a ReBeL timing test (`E/`). New behaviour is behind config options whose defaults reproduce the earlier runs.

## Folders

**A, D: CFR variants, Deep CFR and SD-CFR (Kuhn and Leduc).**
- `A/<game>/<variant>/`: `{kuhn,leduc}_cfr.py --variant {cfr,cfr_plus,linear,dcfr} --iterations 2000` on nash CPU. Leduc regrets are now weighted by the chance probability, as in OpenSpiel; all four variants match OpenSpiel to 3e-16.
- `D/<game>/adv<steps>/seed<s>/`: `{kuhn,leduc}_deep_cfr.py --sd_cfr` with:
  - 1,000 iterations × 1,500 traversals;
  - 750 or 3,000 advantage steps × batch 2,048;
  - 5,000 policy steps;
  - a 2e6 advantage memory and a 1e7 strategy memory.

  Each run holds Deep CFR's policy network (`avg_policy_0`) and SD-CFR's exact average strategy (`sd_cfr_policy_0`). Launch scripts are in `launch/`.

**B1–B3, F1–F7, G: PPO-NFSP (Leduc).**
- **B1, B2:** greedy or sharpened (T 0.1) reservoir targets with γ 1, and B2 adds an entropy decay. 1e8 steps.
- **B3:** the sharpened target at 1e9.
- **F1–F7 (1e9; the `_screen` folders are 1e8):**
  - F1: the sampled-target control;
  - F2: colder targets;
  - F3: a 2M reservoir;
  - F4: lr 1e-3, or 8 epochs with a target-KL stop;
  - F5: a learner seat playing the pure average policy;
  - F6, F7: combinations.
- **G:** supervised-learning settings (a decaying SL lr, lr 1e-4, batches of 1,024 or 4,096) on top of the best configuration. Best: `config_ppo_nfsp_best.yaml` plus `sl_anneal_lr`.

**PPO_ent001: self-play PPO with ent_coef 0.01, 1e8 steps.**

**C0–C5c: PQN-NFSP (Leduc).**
- **C0:** the published config, rerun with best-response-gap logging.
- **C1:** ε 0.06 → 0 and γ 1.
- **C2:** C1 plus RAdam, gradient clip 10 and a decaying lr.
- **C3a–C3d:** 32 or 16 minibatches, and/or λ 0.65.
- **C4, C4b:** the C3c and C3a settings at 1e9 steps with a 2e6 reservoir.
- **C5a1:** 64 minibatches.
- **C5a2:** 32 minibatches × 2 epochs.
- **C5b:** the C4 settings at 3e9 steps with a 3e6 reservoir. Its env-step counts were recomputed from the update index, because the logged int32 counter wrapped.
- **C5c:** an lr floor at 10%.

Scripts are in `launch_nfsp_pqn/`.

**E: ReBeL.**
- `E/leduc/`: the new `leduc_rebel.py`, solving one subgame per betting round with Linear CFR-D.
  - First run: 512 CFR iterations in self-play, 1,024 at evaluation, 4,000 updates.
  - Long run: 1,024 iterations in self-play and at evaluation, 16,000 updates.
- `E/kuhn/`: the existing Kuhn checkpoints re-evaluated at full matmul precision with 512 and 1,024 test-time iterations, giving 0.0049 and 0.0036.

## Bug fixed along the way

The logged `env_steps` was an int32 device counter that wrapped negative past 2³¹ env steps. Only logging was affected; schedules use update indices and optimiser step counts. Every model-free script now computes it on the host with `run_utils.env_steps_after`. Values below 2³¹ are unchanged.

## wandb

Project `bluffjax-leduc-push`. The 5 seeds of a cancelled 3e9-step PPO-NFSP run were not synced.
