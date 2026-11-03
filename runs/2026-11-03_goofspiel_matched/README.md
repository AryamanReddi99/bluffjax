# Goofspiel: six methods at a matched backward-pass budget, 3 November 2026

Every method gets PPO's 244,128 backward passes at 1e9 env samples (15,258 updates × 4 epochs × 4 minibatches), lr 1e-3, entropy 0.1 (MMD: temperature 0.1), 10 seeds, N = 4, 5, 6, exact exploitability every 50 updates, checkpoints at half and full budget.
- ReMAPPO K = 2 (`goofspiel_reppo.py`): 2 backward passes per minibatch step, so 5e8 samples.
- POLA (`goofspiel_pola.py`, fresh rollouts): 2 passes per step and 2 rollouts per update, so 7,629 updates and 1e9 samples.
- PPO, optimistic-Adam PPO (`goofspiel_optppo.py`), ReppoMomentum (`goofspiel_reppo_mom.py`, the SMAX version as is, product clip) and MMD (`goofspiel_mmd.py`): 1 pass per step, 1e9 samples.

POLA's and MMD's own knobs come from screens at N = 4 (1e8 samples, 3 seeds; `launch/jobs_*screen*.txt`, `jobs/scr*`):
- POLA: inner_scale 16, beta_out 10 (beta_in has no effect with one inner step), dice_lambda 0.9.
- MMD: moving-average magnet, KL coefficient 3, rate 0.03.

## Final exploitability (mean ± std over 10 seeds)

| N (uniform) | PPO | Optimistic PPO | ReppoMomentum | ReMAPPO K = 2 | POLA | MMD | MMD magnet |
|---|---|---|---|---|---|---|---|
| 4 (2.50) | 2.409 ± 0.299 | 2.727 ± 0.462 | 2.365 ± 0.527 | 0.394 ± 0.068 | 0.086 ± 0.013 | **0.045 ± 0.017** | 0.040 |
| 5 (4.00) | 4.517 ± 1.524 | 3.203 ± 0.987 | 0.708 ± 0.064 | 0.627 ± 0.071 | 0.378 ± 0.019 | **0.162 ± 0.005** | 0.148 |
| 6 (5.83) | 2.574 ± 0.303 | 2.587 ± 0.104 | 1.112 ± 0.073 | 1.143 ± 0.045 | 0.924 ± 0.037 | **0.472 ± 0.014** | 0.445 |

Last-10% means are in `summary_nash.json` and `summary_cluster.json` (`launch/collect_matched.py`). Each summary holds the per-seed finals and the seed-mean curves.

## Notes
- **POLA at N = 4 has no checkpoints in its main run.** `m_pola_n4` (nash) trained to the end and logged every result. It then failed while merging its unequal vmap chunks (3 + 3 + 3 + 1), which is now fixed in the branch's scripts; `goofspiel_ppo.py` keeps the old line. `m_pola_n4_ckpt` (cluster) reruns the same config for the checkpoints.
- **Where jobs ran:** N = 4 PPO, ReMAPPO and POLA ran on nash (RTX 3080). Everything else ran on the cluster, on RTX 3090, A5000 or A6000 GPUs. Wall-clock times are therefore not comparable.
- **Tuning is uneven:** POLA and MMD had two or three screening rounds; the PPO-family methods used fixed hyperparameters.
