"""Load a pickle file and print its contents (e.g. benchmark_results_*.pkl)."""

from __future__ import annotations

import argparse
import pickle
import pprint
import numpy as np

def main() -> None:
    path = "/home/aryaman/Desktop/project-bluff/bluff/bluffjax/benchmarking/cpu_speed/benchmark_cpu_results_20260506_211910.pkl"
    #path = "/home/aryaman/Desktop/project-bluff/bluff/bluffjax/benchmarking/speed_memory/benchmark_speed/single_gpu/benchmark_results_texas_nolimit_holdem_20260304_190537.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    pprint.pprint(data)


if __name__ == "__main__":
    main()
