# Goofspiel: plain PQN at the matched comparison's settings, 23–24 November 2026

Plain PQN in self-play (`bluffjax/examples/goofspiel/goofspiel_pqn.py`, new): PQN-NFSP's Q-learner without the NFSP
average policy, reservoir or policy mixing. Every decision is the epsilon-greedy Q-network's, so the Q(lambda) trace
is never cut. Exploitability is that of the greedy policy (argmax Q over the cards in hand, ties to the lowest card).

Settings (`config_pqn.yaml`): PQN-NFSP's best-response settings (`config_pqn_nfsp_leduc_best.yaml`: lr 2.5e-4
annealed to 0.1x, RAdam, 4 epochs x 32 minibatches, lambda 0.65, gamma 1, grad norm 10, epsilon 0.06 -> 0 over the
run) with the matched comparison's run settings (`runs/2026-11-03_goofspiel_matched`): 1e9 env samples, 10 seeds,
N = 4, 5, 6, exact exploitability every 50 updates, half/full checkpoints.

## Exploration screen (`launch/jobs_screen.txt`, nash RTX 3080, N = 4, 3 seeds, 1e8 samples; `screen.json`)

| Schedule | start_e 0.06 -> 0 over the run (default) | 0.2 -> 0 | 1.0 -> 0.005 over 10% | constant 0.06 |
|---|---|---|---|---|
| Last-10% exploitability | 2.629 | 2.686 | 2.672 | 2.672 |

All four are within noise and above uniform play (2.50); the default was kept.

## Full runs (`launch/jobs_full.txt`)

Ran on the RAI Determined cluster (one A100-80GB each, about 35 min; context `~/det_ctx/pqn` on nash: a `git archive`
of `bluffjax/`, `run.sh`, `extract.py`). Results are on `/pfss/.../ar44ruca/bluffjax-runs/2026-11-23_goofspiel_pqn`;
each command printed the downloader's extractor output (exact-evaluation rows) between `BEGIN_JSON` and `END_JSON`,
kept in `det_logs/`. The same jobs were also submitted to SLURM (228536–228538, cancelled after ~18 min) and N = 4 ran
on nash as a cross-check (final 2.767 ± 0.324; N = 5 was stopped).

Final exploitability of the greedy policy, mean ± std over 10 seeds:

| N (uniform) | PQN | PQN-NFSP (average policy) |
|---|---|---|
| 4 (2.50) | 2.704 ± 0.280 | see `2026-11-02_goofspiel_exploitability` |
| 5 (4.00) | 5.919 ± 0.124 | |
| 6 (5.83) | 10.545 ± 0.612 | |

The greedy policy is deterministic, and a deterministic strategy in Goofspiel is exploitable by more than uniform
play from N = 5 on. Its exploitability barely moves after the first ~1% of training. The data is in
`recursive-reasoning-marl/.../evaluation_new/goofspiel/n{4,5,6}/data/PQN*.npy` (`download_results.py PQN`, reading
`data/PQN_det.json`).
