import pickle

fn = "/home/aryaman/Desktop/project-bluff/bluff/bluffjax/benchmark_speed/multi_gpu/benchmark_results_pmap_kemps_20260305_155421.pkl"

with open(fn, "rb") as f:
    d = pickle.load(f)
print(d)
