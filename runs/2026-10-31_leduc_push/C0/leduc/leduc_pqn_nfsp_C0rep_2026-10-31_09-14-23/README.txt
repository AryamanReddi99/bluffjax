Reproducibility check: an exact rerun of C0 (same config and seeds) on nash.
Its exact exploitability matches C0 seed for seed (0.1107, 0.1619, 0.1435, 0.1238, 0.1322 at 1e8 env steps), which shows that the gap between C0 (0.134) and the published runs2 run (0.183) on the same config comes from floating-point divergence between runs, not from a code change.
