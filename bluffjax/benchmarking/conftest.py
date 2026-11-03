# The benchmarking folder holds scripts and results, not tests. Some scripts
# match pytest's test_*.py pattern (e.g. speed_memory/benchmark_speed/
# multi_gpu/test_d.py) and read hard-coded paths at import, so keep pytest
# from collecting anything here.
collect_ignore_glob = ["*"]
