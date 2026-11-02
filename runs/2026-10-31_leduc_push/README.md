# Leduc push, phases A and D (CFR variants, Deep CFR, Single Deep CFR)

The code is on branch `leduc/cfr`. Exploitability is NashConv/2 in chips, measured exactly over the full game tree.

- **`A/<game>/<variant>/<run>/`**: `{kuhn,leduc}_cfr.py --variant <variant> --iterations 2000 --eval_interval 100`.
  - The variants are cfr, cfr_plus, linear and dcfr. They are deterministic, so each has one run.
  - These ran on nash CPU.
  - The Leduc runs used commit `a61e5a9`, which weights regrets by the chance probability so they match OpenSpiel bit for bit. The Kuhn runs used `5c9c429`.
- **`D/<game>/adv<steps>/seed<s>/<run>/`**: `{kuhn,leduc}_deep_cfr.py --sd_cfr --iterations 1000 --eval_interval 250 --traversals 1500 --advantage_train_steps <steps> --batch_size 2048 --policy_train_steps 5000 --memory_capacity 2000000 --strategy_memory_capacity 10000000 --hidden_sizes 128 128 --seed <s> --num_seeds 1`.
  - All 5 seeds of a configuration ran as parallel processes on one cluster GPU (`launch/deep_cfr_job.sh`), with `XLA_FLAGS=--xla_gpu_deterministic_ops=true`.
  - Exception: Leduc adv3000 seed 1 failed at start-up with a shared-filesystem error and was rerun on its own on nash (RTX 3080).
  - Each run has `avg_policy_0` (Deep CFR's policy network) and `sd_cfr_policy_0` (SD-CFR's average strategy table) under `half/` and `full/`.
  - `results.jsonl.gz` logs Deep CFR as `exploitability` and SD-CFR as `sd_cfr/exploitability`.

Each run directory keeps `config.yaml`, `results.jsonl.gz` (plus `results.json.gz` for Deep CFR), and the `half/` and `full/` checkpoints.

`launch/collect.py` builds the summary JSON. The offline wandb runs are in `nash:~/bluffjax-dev/cfr-push/wandb` and `cluster:~/bluffjax-dev/cfr/wandb`, under project `bluffjax-leduc-push`.
