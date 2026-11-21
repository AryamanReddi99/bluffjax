# Goofspiel: CoPPO and extragradient PPO at the matched budget, 21 November 2026

Two more baselines for the comparison in `runs/2026-11-03_goofspiel_matched`, with the same rules: lr 1e-3, entropy 0.1, 10 seeds, N = 4, 5, 6, exact exploitability every 50 updates, half/full checkpoints, PPO's 244,128 policy backward passes.
- **CoPPO** (`goofspiel_reppo.py`: `reppo_k=1 level1_weight=current clip_mode=product nonself_eps2=<eps2>`): one backward pass per minibatch step, 1e9 samples.
- **Extragradient PPO** (`goofspiel_egppo.py`): two backward passes per minibatch step, 5e8 samples.

Each method's own knob was screened first at N = 4, 3 seeds, at the screens' budget (the backward passes of 1e8 PPO samples).

Results: pending.
