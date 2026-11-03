import pickle
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker
import copy

sns.set_theme()

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

COLORS = {
    "kuhn": "#FF0000",
    "leduc": "#FF7F00",
    "goofspiel": "#FFD700",
    "five_card_draw": "#00BFFF",
    "seven_card_stud": "#FF007F",
    "texas_limit_holdem": "#7F00FF",
    "texas_nolimit_holdem": "#C000FF",
    "werewolf": "#00FF7F",
    "bluff": "#7FFF00",
    "kemps": "#007BFF",
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

data = {}
base_dir = os.path.dirname(__file__)
run_id_pattern = re.compile(r"^\d+(?:_\d+)*$")

# Single directory scan + one pass grouping by env for efficiency.
latest_file_by_env = {}
for filename in os.listdir(base_dir):
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
    file_path = os.path.join(base_dir, filename)
    with open(file_path, "rb") as f:
        data[env_name] = pickle.load(f)

d = data
d["kemps"][10000] = copy.deepcopy(d["kemps"][10000])
d["kemps"][10000][0]["peak_bytes_in_use"] = 65856 * (1024 * 1024)

fig, ax = plt.subplots(figsize=(10, 6))
for env_name, env_data in d.items():
    mean_data = [
        v[0]["peak_bytes_in_use"] / (1024.0 * 1024.0) for v in env_data.values()
    ]
    ax.loglog(
        env_data.keys(),
        mean_data,
        marker="o",
        label=LABELS[env_name],
        color=COLORS[env_name],
    )
ax.xaxis.set_major_locator(ticker.LogLocator(base=10))
ax.yaxis.set_major_locator(ticker.LogLocator(base=10))

# minor ticks between powers of 10
ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))

# hide minor tick labels (optional)
ax.xaxis.set_minor_formatter(ticker.NullFormatter())
ax.yaxis.set_minor_formatter(ticker.NullFormatter())

# ---- Grid lines ----
ax.grid(True, which="major", linestyle="--", linewidth=0.8, color="gray")
ax.grid(True, which="minor", linestyle="--", linewidth=0.5, color="gray", alpha=0.5)

ax.legend(fontsize=12, framealpha=0.5, loc=[1.01, 0.25])
ax.set_xlabel("Number of Parallel Environments", fontsize=16)
ax.set_ylabel("Peak Memory Usage (MiB)", fontsize=16)
ax.set_xlim(0.9, 11000)
# ax.set_ylim(1e3, 2e8)

ax.tick_params(axis="both", which="major", labelsize=16)  # Major tick labels

for spine in ax.spines.values():
    spine.set_edgecolor("gray")
    spine.set_linewidth(1.0)

fig.set_size_inches(22 / 2.54, 12 / 2.54)  # Set figure size to 20cm by 20cm
plt.tight_layout()

for spine in ax.spines.values():
    spine.set_edgecolor("gray")
    spine.set_linewidth(1.0)

plt.savefig(
    os.path.join(os.path.dirname(__file__), "benchmark_memory.jpg"),
    bbox_inches="tight",
)
plt.savefig(
    os.path.join(os.path.dirname(__file__), "benchmark_memory.pdf"),
    bbox_inches="tight",
)
