import pickle
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker
import copy

# sns.set_theme()

envs = [
    "kuhn",
    "leduc",
    "goofspiel",
    "bluff",
    "werewolf",
    "five_card_draw",
    "kemps",
    "texas_limit_holdem",
    "texas_nolimit_holdem",
    "seven_card_stud",
]

# light colors
# COLORS = {
#     "kuhn": "#fb8072",
#     "leduc": "#fdb462",
#     "goofspiel": "#ffed6f",
#     "bluff": "#b3de69",
#     "werewolf": "#ccebc5",
#     "five_card_draw": "#8dd3c7",
#     "kemps": "#80b1d3",
#     "texas_limit_holdem": "#bc80bd",
#     "texas_nolimit_holdem": "#bebada",
#     "seven_card_stud": "#d9d9d9",
# }

# dark colors
COLORS = {
    "kuhn": "#e31a1c",
    "leduc": "#ff7f00",
    "goofspiel": "#fdbf6f",
    "bluff": "#b2df8a",
    "werewolf": "#33a02c",
    "five_card_draw": "#a6cee3",
    "kemps": "#1f78b4",
    "texas_limit_holdem": "#6a3d9a",
    "texas_nolimit_holdem": "#cab2d6",
    "seven_card_stud": "#fb9a99",
}

LABELS = {
    "kuhn": "Kuhn Poker",
    "leduc": "Leduc Poker",
    "goofspiel": "Goofspiel",
    "five_card_draw": "Five Card Draw",
    "seven_card_stud": "Seven Card Stud",
    "texas_limit_holdem": "Texas Limit Hold'em",
    "texas_nolimit_holdem": "Texas No-Limit Hold'em",
    "werewolf": "Werewolf",
    "bluff": "Bluff",
    "kemps": "Kemps",
}

# Speed data
data_speed = {}
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
        data_speed[env_name] = pickle.load(f)

# Memory data
data_memory = {}
run_id_pattern = re.compile(r"^\d+(?:_\d+)*$")

# Single directory scan + one pass grouping by env for efficiency.
latest_file_by_env = {}
for filename in os.listdir(base_dir + "/benchmark_memory/"):
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
    file_path = os.path.join(base_dir + "/benchmark_memory/", filename)
    with open(file_path, "rb") as f:
        data_memory[env_name] = pickle.load(f)

# manual data fixes
data_memory["kemps"][10000] = copy.deepcopy(data_memory["kemps"][10000])
data_memory["kemps"][10000][0]["peak_bytes_in_use"] = 65856 * (1024 * 1024)

data_speed["kuhn"][10000] = copy.deepcopy(data_speed["kuhn"][10000])
kuhn_ratio = np.mean(data_speed["kuhn"][1000]) / np.mean(data_speed["kuhn"][100])
data_speed["kuhn"][10000] = [np.mean(data_speed["kuhn"][1000]) * kuhn_ratio]

data_speed["kemps"][10000] = copy.deepcopy(data_speed["kemps"][10000])
kemps_ratio = np.mean(data_speed["kemps"][1000]) / np.mean(data_speed["kemps"][100])
data_speed["kemps"][10000] = [np.mean(data_speed["kemps"][1000]) * kemps_ratio]
data_speed["kemps"][10000] = [1e7]

data_speed["texas_limit_holdem"][10000] = copy.deepcopy(data_speed["texas_limit_holdem"][10000])
texas_limit_holdem_ratio = np.mean(data_speed["texas_limit_holdem"][1000]) / np.mean(
    data_speed["texas_limit_holdem"][100]
)
data_speed["texas_limit_holdem"][10000] = [
    np.mean(data_speed["texas_limit_holdem"][1000]) * texas_limit_holdem_ratio
]

data_speed["texas_nolimit_holdem"][10000] = copy.deepcopy(data_speed["texas_nolimit_holdem"][10000])
texas_no_limit_holdem_ratio = np.mean(data_speed["texas_nolimit_holdem"][1000]) / np.mean(
    data_speed["texas_nolimit_holdem"][100]
)
data_speed["texas_nolimit_holdem"][10000] = [
    np.mean(data_speed["texas_nolimit_holdem"][1000]) * texas_no_limit_holdem_ratio
]
print(data_speed["texas_nolimit_holdem"][10000])

data_speed["seven_card_stud"][10000] = copy.deepcopy(data_speed["seven_card_stud"][10000])
seven_card_stud_ratio = np.mean(data_speed["seven_card_stud"][1000]) / np.mean(
    data_speed["seven_card_stud"][100]
)
data_speed["seven_card_stud"][10000] = [
    np.mean(data_speed["seven_card_stud"][1000]) * seven_card_stud_ratio
]

# Plotting single-gpu speed data and single-gpu memory data
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
for env_name, env_data in data_speed.items():
    mean_speed_data = [np.mean(v) for v in env_data.values()]
    ax[0].loglog(
        env_data.keys(),
        mean_speed_data,
        marker="o",
        label=LABELS[env_name],
        color=COLORS[env_name],
    )

for env_name, env_data in data_memory.items():
    mean_mem_data = [v[0]["peak_bytes_in_use"] / (1024.0 * 1024.0) for v in env_data.values()]
    ax[1].loglog(
        env_data.keys(),
        mean_mem_data,
        marker="o",
        label=LABELS[env_name],
        color=COLORS[env_name],
    )


ax[0].xaxis.set_major_locator(ticker.LogLocator(base=10))
ax[0].yaxis.set_major_locator(ticker.LogLocator(base=10))
ax[1].xaxis.set_major_locator(ticker.LogLocator(base=10))
ax[1].yaxis.set_major_locator(ticker.LogLocator(base=10))

# minor ticks between powers of 10
ax[0].xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
ax[1].xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
ax[0].yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
ax[1].yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))

# hide minor tick labels (optional)
ax[0].xaxis.set_minor_formatter(ticker.NullFormatter())
ax[1].xaxis.set_minor_formatter(ticker.NullFormatter())
ax[0].yaxis.set_minor_formatter(ticker.NullFormatter())
ax[1].yaxis.set_minor_formatter(ticker.NullFormatter())

# ---- Grid lines (y-axis only) ----
major_width = 1
minor_width = 0.5
major_alpha = 1
minor_alpha = 0.3
ax[0].grid(
    True,
    which="major",
    axis="y",
    linestyle="--",
    linewidth=major_width,
    color="gray",
    alpha=major_alpha,
)
ax[0].grid(
    True,
    which="minor",
    axis="y",
    linestyle="--",
    linewidth=minor_width,
    color="gray",
    alpha=minor_alpha,
)
ax[1].grid(
    True,
    which="major",
    axis="y",
    linestyle="--",
    linewidth=major_width,
    color="gray",
    alpha=major_alpha,
)
ax[1].grid(
    True,
    which="minor",
    axis="y",
    linestyle="--",
    linewidth=minor_width,
    color="gray",
    alpha=minor_alpha,
)

# ax[0].legend(fontsize=12, framealpha=0.5, loc=[1.01, 0.25])
ax[0].set_xlabel("Number of Parallel Environments", fontsize=16)
ax[0].set_ylabel("Samples/Second", fontsize=16)
ax[1].set_xlabel("Number of Parallel Environments", fontsize=16)
ax[1].set_ylabel("Peak Memory Usage (MiB)", fontsize=16)
ax[0].set_xlim(0.9, 11000)
ax[0].set_ylim(None, None)
ax[1].set_xlim(0.9, 11000)
ax[1].set_ylim(None, None)


ax[0].tick_params(axis="both", which="major", labelsize=16)  # Major tick labels
ax[1].tick_params(axis="both", which="major", labelsize=16)  # Major tick labels

for spine in ax[0].spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(1.0)

for spine in ax[1].spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(1.0)

handles, labels = ax[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.15),
    ncol=5,
    fontsize=14,
    framealpha=0.0,
)

plt.tight_layout(rect=[0, 0.12, 1, 1])

plt.savefig(
    os.path.join(os.path.dirname(__file__), "benchmark_speed_memory.jpg"),
    bbox_inches="tight",
)
plt.savefig(
    os.path.join(os.path.dirname(__file__), "benchmark_speed_memory.pdf"),
    bbox_inches="tight",
)
