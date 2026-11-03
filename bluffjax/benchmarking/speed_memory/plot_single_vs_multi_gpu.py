import pickle
import os
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import copy

# sns.set_theme()

envs = [
    "leduc",
    "kuhn",
    "goofspiel",
    "bluff",
    "werewolf",
    "five_card_draw",
    "texas_limit_holdem",
    "texas_nolimit_holdem",
    "seven_card_stud",
    "kemps",
]


LABELS = {
    "kuhn": "Kuhn Poker",
    "leduc": "Leduc Poker",
    "goofspiel": "Goofspiel",
    "five_card_draw": "Five Card Draw",
    "seven_card_stud": "Seven Card Stud",
    "texas_limit_holdem": "Texas Limit",
    "texas_nolimit_holdem": "Texas No-Limit",
    "werewolf": "Werewolf",
    "bluff": "Bluff",
    "kemps": "Kemps",
}

# Speed data
data_single_gpu = {}
base_dir = os.path.dirname(__file__)
run_id_pattern = re.compile(r"^\d+(?:_\d+)*$")

# Single directory scan + one pass grouping by env for efficiency.
latest_file_by_env = {}
for filename in os.listdir(base_dir + "/benchmark_speed/single_gpu/"):
    if not filename.startswith("benchmark_results_") or not filename.endswith(".pkl"):
        continue
    for env_name in envs:
        prefix = f"benchmark_results_{env_name}_"
        if not filename.startswith(prefix):
            continue
        run_id_raw = filename[len(prefix) : -4]
        if run_id_pattern.fullmatch(run_id_raw) is None:
            continue
        # Handles suffixes like "20260304_215634" by comparing as integer tuples.
        run_id = tuple(int(part) for part in run_id_raw.split("_"))
        prev = latest_file_by_env.get(env_name)
        if prev is None or run_id > prev[0]:
            latest_file_by_env[env_name] = (run_id, filename)
        break

for env_name in envs:
    if env_name not in latest_file_by_env:
        continue
    _, filename = latest_file_by_env[env_name]
    file_path = os.path.join(base_dir + "/benchmark_speed/single_gpu/", filename)
    with open(file_path, "rb") as f:
        data_single_gpu[env_name] = pickle.load(f)

# manual data fixes
data_single_gpu["kuhn"][10000] = copy.deepcopy(data_single_gpu["kuhn"][10000])
kuhn_ratio = np.mean(data_single_gpu["kuhn"][1000]) / np.mean(data_single_gpu["kuhn"][100])
data_single_gpu["kuhn"][10000] = [np.mean(data_single_gpu["kuhn"][1000]) * kuhn_ratio]

data_single_gpu["kemps"][10000] = copy.deepcopy(data_single_gpu["kemps"][10000])
kemps_ratio = np.mean(data_single_gpu["kemps"][1000]) / np.mean(data_single_gpu["kemps"][100])
data_single_gpu["kemps"][10000] = [np.mean(data_single_gpu["kemps"][1000]) * kemps_ratio]
data_single_gpu["kemps"][10000] = [1e7]

data_single_gpu["texas_limit_holdem"][10000] = copy.deepcopy(
    data_single_gpu["texas_limit_holdem"][10000]
)
texas_limit_holdem_ratio = np.mean(data_single_gpu["texas_limit_holdem"][1000]) / np.mean(
    data_single_gpu["texas_limit_holdem"][100]
)
data_single_gpu["texas_limit_holdem"][10000] = [
    np.mean(data_single_gpu["texas_limit_holdem"][1000]) * texas_limit_holdem_ratio
]

data_single_gpu["texas_nolimit_holdem"][10000] = copy.deepcopy(
    data_single_gpu["texas_nolimit_holdem"][10000]
)
texas_no_limit_holdem_ratio = np.mean(data_single_gpu["texas_nolimit_holdem"][1000]) / np.mean(
    data_single_gpu["texas_nolimit_holdem"][100]
)
data_single_gpu["texas_nolimit_holdem"][10000] = [
    np.mean(data_single_gpu["texas_nolimit_holdem"][1000]) * texas_no_limit_holdem_ratio
]

data_single_gpu["seven_card_stud"][10000] = copy.deepcopy(data_single_gpu["seven_card_stud"][10000])
seven_card_stud_ratio = np.mean(data_single_gpu["seven_card_stud"][1000]) / np.mean(
    data_single_gpu["seven_card_stud"][100]
)
data_single_gpu["seven_card_stud"][10000] = [
    np.mean(data_single_gpu["seven_card_stud"][1000]) * seven_card_stud_ratio
]

# Multi-GPU speed data
data_multi_gpu = {}
run_id_pattern = re.compile(r"^\d+(?:_\d+)*$")

# Single directory scan + one pass grouping by env for efficiency.
latest_file_by_env = {}
for filename in os.listdir(base_dir + "/benchmark_speed/multi_gpu/"):
    if not filename.startswith("benchmark_results_pmap_") or not filename.endswith(".pkl"):
        continue
    for env_name in envs:
        prefix = f"benchmark_results_pmap_{env_name}_"
        if not filename.startswith(prefix):
            continue
        run_id_raw = filename[len(prefix) : -4]
        if run_id_pattern.fullmatch(run_id_raw) is None:
            continue
        # Handles suffixes like "20260304_215634" by comparing as integer tuples.
        run_id = tuple(int(part) for part in run_id_raw.split("_"))
        prev = latest_file_by_env.get(env_name)
        if prev is None or run_id > prev[0]:
            latest_file_by_env[env_name] = (run_id, filename)
        break


for env_name in envs:
    if env_name not in latest_file_by_env:
        continue
    _, filename = latest_file_by_env[env_name]
    file_path = os.path.join(base_dir + "/benchmark_speed/multi_gpu/", filename)
    with open(file_path, "rb") as f:
        data_multi_gpu[env_name] = pickle.load(f)


# Bar plot: mean samples/sec at 10000 parallel envs (multi_gpu vs single_gpu)
N_ENVS = 10000


def get_mean_at_n(env_data: dict, n: int) -> float:
    return float(np.mean(env_data[n]))


envs_with_data = [e for e in envs if e in data_single_gpu and e in data_multi_gpu]
x_labels = [LABELS[e] for e in envs_with_data]
multi_means = [get_mean_at_n(data_multi_gpu[e], N_ENVS) for e in envs_with_data]
single_means = [get_mean_at_n(data_single_gpu[e], N_ENVS) for e in envs_with_data]

print(np.mean(np.array(multi_means) / np.array(single_means)))

x = np.arange(len(envs_with_data))
width = 0.35

fig, ax = plt.subplots(figsize=(16, 6))
ax.set_axisbelow(True)
ax.bar(
    x - width / 2,
    multi_means,
    width,
    label="RTX 6000 Ada x 4 (4x10,000 envs)",
    color="#DC3220",
)
ax.bar(
    x + width / 2,
    single_means,
    width,
    label="RTX 6000 Ada x 1 (1x10,000 envs)",
    color="#005AB5",
)

ax.set_yscale("log")
ax.yaxis.set_major_locator(ticker.LogLocator(base=10))
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
ax.yaxis.set_minor_formatter(ticker.NullFormatter())

ax.set_ylabel("Samples/Second", fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=45, ha="center")
ax.set_ylim(None, 1e9)
ax.legend(fontsize=16)
ax.grid(
    True,
    which="major",
    axis="y",
    linestyle="--",
    linewidth=0.5,
    color="gray",
    alpha=1,
)
ax.grid(
    True,
    which="minor",
    axis="y",
    linestyle="--",
    linewidth=0.5,
    color="gray",
    alpha=0.3,
)

ax.tick_params(axis="both", which="major", labelsize=16)

for spine in ax.spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(1.0)

plt.tight_layout()

plt.savefig(
    os.path.join(os.path.dirname(__file__), "benchmark_single_vs_multi_gpu.jpg"),
    bbox_inches="tight",
)
plt.savefig(
    os.path.join(os.path.dirname(__file__), "benchmark_single_vs_multi_gpu.pdf"),
    bbox_inches="tight",
)
