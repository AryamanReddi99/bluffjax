import pickle
import pprint
import numpy as np

f_bluffjax_kuhn = "/home/aryaman/Desktop/project-bluff/bluff/bluffjax/benchmarking/speed_memory/benchmark_speed/other_libraries/benchmark_results_kuhn_20260304_184410.pkl"
f_bluffjax_leduc = "/home/aryaman/Desktop/project-bluff/bluff/bluffjax/benchmarking/speed_memory/benchmark_speed/other_libraries/benchmark_results_leduc_20260304_184523.pkl"
f_bluffjax_texas_limit = "/home/aryaman/Desktop/project-bluff/bluff/bluffjax/benchmarking/speed_memory/benchmark_speed/other_libraries/benchmark_results_texas_limit_holdem_20260304_184814.pkl"
f_bluffjax_texas_no_limit = "/home/aryaman/Desktop/project-bluff/bluff/bluffjax/benchmarking/speed_memory/benchmark_speed/other_libraries/benchmark_results_texas_nolimit_holdem_20260304_190537.pkl"
f_pgx_kuhn = "/home/aryaman/Desktop/project-bluff/bluff/bluffjax/benchmarking/speed_memory/benchmark_speed/other_libraries/benchmark_results_pgx_kuhn_poker_20260506_225007.pkl"
f_pgx_leduc = "/home/aryaman/Desktop/project-bluff/bluff/bluffjax/benchmarking/speed_memory/benchmark_speed/other_libraries/benchmark_results_pgx_leduc_holdem_20260506_225104.pkl"
f_cpu = "/home/aryaman/Desktop/project-bluff/bluff/bluffjax/benchmarking/speed_memory/benchmark_speed/other_libraries/benchmark_cpu_results_20260506_211910.pkl"

with open(f_pgx_kuhn, "rb") as f:
    d_pgx_kuhn = pickle.load(f)
with open(f_pgx_leduc, "rb") as f:
    d_pgx_leduc = pickle.load(f)
with open(f_cpu, "rb") as f:
    d_cpu = pickle.load(f)
with open(f_bluffjax_kuhn, "rb") as f:
    d_bluffjax_kuhn = pickle.load(f)
with open(f_bluffjax_leduc, "rb") as f:
    d_bluffjax_leduc = pickle.load(f)
with open(f_bluffjax_texas_limit, "rb") as f:
    d_bluffjax_texas_limit = pickle.load(f)
with open(f_bluffjax_texas_no_limit, "rb") as f:
    d_bluffjax_texas_no_limit = pickle.load(f)

pprint.pprint(d_bluffjax_kuhn)
pprint.pprint(d_pgx_kuhn)
pprint.pprint(d_cpu)

mean_bluffjax_kuhn = [np.mean(v) for v in d_bluffjax_kuhn.values()]
mean_pgx_kuhn = [np.mean(v) for v in d_pgx_kuhn.values()]
mean_openspiel_kuhn = [np.mean(v) for v in d_cpu["kuhn_poker"].values()]

print(mean_bluffjax_kuhn)
print(mean_pgx_kuhn)
print(mean_openspiel_kuhn)
