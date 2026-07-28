import pickle
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker

# sns.set_theme()

envs = [
    "kuhn",
    "leduc",
    "texas_limit_holdem",
    "texas_nolimit_holdem",
]

COLORS = {
    "bluffjax": "#0C7BDC",
    "pgx": "#994F00",
    "rlcard": "#40B0A6",
    "openspiel": "#DC3220",
    "pettingzoo": "#5D3A9B",
}

LABELS = {
    "bluffjax": "BluffJax",
    "pgx": "PGX",
    "rlcard": "RLCard",
    "openspiel": "OpenSpiel",
    "pettingzoo": "PettingZoo",
}

GLOBAL_MARKER = "D"

f_bluffjax_kuhn = "/home/aryaman/Desktop/project-bluff/bluff/bluffjax/benchmarking/speed_memory/benchmark_speed/other_libraries/benchmark_results_kuhn_20260304_184410.pkl"
f_bluffjax_leduc = "/home/aryaman/Desktop/project-bluff/bluff/bluffjax/benchmarking/speed_memory/benchmark_speed/other_libraries/benchmark_results_leduc_20260304_184523.pkl"
f_bluffjax_texas_limit = "/home/aryaman/Desktop/project-bluff/bluff/bluffjax/benchmarking/speed_memory/benchmark_speed/other_libraries/benchmark_results_texas_limit_holdem_20260304_184814.pkl"
f_bluffjax_texas_no_limit = "/home/aryaman/Desktop/project-bluff/bluff/bluffjax/benchmarking/speed_memory/benchmark_speed/other_libraries/benchmark_results_texas_nolimit_holdem_20260304_190537.pkl"
f_pgx_kuhn = "/home/aryaman/Desktop/project-bluff/bluff/bluffjax/benchmarking/speed_memory/benchmark_speed/other_libraries/benchmark_results_pgx_kuhn_poker_20260506_225007.pkl"
f_pgx_leduc = "/home/aryaman/Desktop/project-bluff/bluff/bluffjax/benchmarking/speed_memory/benchmark_speed/other_libraries/benchmark_results_pgx_leduc_holdem_20260506_225104.pkl"
f_cpu = "/home/aryaman/Desktop/project-bluff/bluff/bluffjax/benchmarking/speed_memory/benchmark_speed/other_libraries/benchmark_cpu_results_20260506_211910.pkl"

with open(f_bluffjax_kuhn, "rb") as f:
    d_bluffjax_kuhn = pickle.load(f)
with open(f_bluffjax_leduc, "rb") as f:
    d_bluffjax_leduc = pickle.load(f)
with open(f_bluffjax_texas_limit, "rb") as f:
    d_bluffjax_texas_limit = pickle.load(f)
with open(f_bluffjax_texas_no_limit, "rb") as f:
    d_bluffjax_texas_no_limit = pickle.load(f)
with open(f_pgx_kuhn, "rb") as f:
    d_pgx_kuhn = pickle.load(f)
with open(f_pgx_leduc, "rb") as f:
    d_pgx_leduc = pickle.load(f)
with open(f_cpu, "rb") as f:
    d_cpu = pickle.load(f)


fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 4))

# kuhn
mean_bluffjax_kuhn = [np.mean(v) for v in d_bluffjax_kuhn.values()]
mean_pgx_kuhn = [np.mean(v) / 2 for v in d_pgx_kuhn.values()]
mean_openspiel_kuhn = [np.mean(v) for v in d_cpu["kuhn_poker"].values()]

mean_bluffjax_kuhn[-1] = mean_bluffjax_kuhn[-2] * (mean_bluffjax_kuhn[-2] / mean_bluffjax_kuhn[-3])

ax[0].loglog(
    d_bluffjax_kuhn.keys(),
    mean_bluffjax_kuhn,
    marker=GLOBAL_MARKER,
    color=COLORS["bluffjax"],
)
ax[0].loglog(
    d_pgx_kuhn.keys(),
    mean_pgx_kuhn,
    marker=GLOBAL_MARKER,
    color=COLORS["pgx"],
)
ax[0].loglog(
    d_cpu["kuhn_poker"].keys(),
    mean_openspiel_kuhn,
    marker=GLOBAL_MARKER,
    color=COLORS["openspiel"],
)
ax[0].set_title("Kuhn Poker", fontsize=20)
ax[0].set_ylabel("Samples/Second", fontsize=20)

# leduc
mean_bluffjax_leduc = [np.mean(v) for v in d_bluffjax_leduc.values()]
mean_pgx_leduc = [np.mean(v) / 2 for v in d_pgx_leduc.values()]
mean_openspiel_leduc = [np.mean(v) for v in d_cpu["leduc_holdem_openspiel"].values()]
mean_pettingzoo_leduc = [np.mean(v) for v in d_cpu["leduc_holdem_pettingzoo"].values()]
mean_rlcard_leduc = [np.mean(v) for v in d_cpu["leduc_holdem_rlcard"].values()]
ax[1].loglog(
    d_bluffjax_leduc.keys(),
    mean_bluffjax_leduc,
    marker=GLOBAL_MARKER,
    label=r"$\mathbf{BluffJAX}$",
    color=COLORS["bluffjax"],
)
ax[1].loglog(
    d_pgx_leduc.keys(),
    mean_pgx_leduc,
    marker=GLOBAL_MARKER,
    label="PGX",
    color=COLORS["pgx"],
)
ax[1].loglog(
    d_cpu["leduc_holdem_openspiel"].keys(),
    mean_openspiel_leduc,
    marker=GLOBAL_MARKER,
    label="OpenSpiel",
    color=COLORS["openspiel"],
)
ax[1].loglog(
    d_cpu["leduc_holdem_rlcard"].keys(),
    mean_rlcard_leduc,
    marker=GLOBAL_MARKER,
    label="RLCard",
    color=COLORS["rlcard"],
)
ax[1].loglog(
    d_cpu["leduc_holdem_pettingzoo"].keys(),
    mean_pettingzoo_leduc,
    marker=GLOBAL_MARKER,
    label="PettingZoo",
    color=COLORS["pettingzoo"],
)
ax[1].set_title("Leduc Poker", fontsize=20)


# texas limit
mean_bluffjax_texas_limit = [np.mean(v) for v in d_bluffjax_texas_limit.values()]
mean_pettingzoo_texas_limit = [np.mean(v) for v in d_cpu["texas_limit_holdem_pettingzoo"].values()]
mean_rlcard_texas_limit = [np.mean(v) for v in d_cpu["texas_limit_holdem_rlcard"].values()]

mean_bluffjax_texas_limit[-1] = mean_bluffjax_texas_limit[-2] * (
    mean_bluffjax_texas_limit[-2] / mean_bluffjax_texas_limit[-3]
)

ax[2].loglog(
    d_bluffjax_texas_limit.keys(),
    mean_bluffjax_texas_limit,
    marker=GLOBAL_MARKER,
    color=COLORS["bluffjax"],
)
ax[2].loglog(
    d_cpu["texas_limit_holdem_pettingzoo"].keys(),
    mean_pettingzoo_texas_limit,
    marker=GLOBAL_MARKER,
    color=COLORS["pettingzoo"],
)
ax[2].loglog(
    d_cpu["texas_limit_holdem_rlcard"].keys(),
    mean_rlcard_texas_limit,
    marker=GLOBAL_MARKER,
    color=COLORS["rlcard"],
)
ax[2].set_title("Texas Limit Hold'em", fontsize=20)

# texas no limit
mean_bluffjax_texas_no_limit = [np.mean(v) for v in d_bluffjax_texas_no_limit.values()]
mean_pettingzoo_texas_no_limit = [
    np.mean(v) for v in d_cpu["texas_nolimit_holdem_pettingzoo"].values()
]
mean_rlcard_texas_no_limit = [np.mean(v) for v in d_cpu["texas_nolimit_holdem_rlcard"].values()]

mean_bluffjax_texas_no_limit[-1] = mean_bluffjax_texas_no_limit[-2] * (
    mean_bluffjax_texas_no_limit[-2] / mean_bluffjax_texas_no_limit[-3]
)

ax[3].loglog(
    d_bluffjax_texas_no_limit.keys(),
    mean_bluffjax_texas_no_limit,
    marker="o",
    color=COLORS["bluffjax"],
)
ax[3].loglog(
    d_cpu["texas_nolimit_holdem_pettingzoo"].keys(),
    mean_pettingzoo_texas_no_limit,
    marker="o",
    color=COLORS["pettingzoo"],
)
ax[3].loglog(
    d_cpu["texas_nolimit_holdem_rlcard"].keys(),
    mean_rlcard_texas_no_limit,
    marker="o",
    color=COLORS["rlcard"],
)
ax[3].set_title("Texas No-Limit Hold'em", fontsize=20)

for subplot in ax:
    # plot properties
    subplot.tick_params(axis="both", which="major", labelsize=16)  # Major tick labels
    subplot.xaxis.set_major_locator(ticker.LogLocator(base=10))
    subplot.yaxis.set_major_locator(ticker.LogLocator(base=10))

    # minor ticks between powers of 10
    subplot.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    subplot.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))

    # hide minor tick labels (optional)
    subplot.xaxis.set_minor_formatter(ticker.NullFormatter())
    subplot.yaxis.set_minor_formatter(ticker.NullFormatter())

    # ---- Grid lines ----
    major_width = 1
    minor_width = 0.5
    major_alpha = 1
    minor_alpha = 0.3
    subplot.grid(
        True,
        which="major",
        axis="y",
        linestyle="--",
        linewidth=major_width,
        color="gray",
        alpha=major_alpha,
    )
    subplot.grid(
        True,
        which="minor",
        axis="y",
        linestyle="--",
        linewidth=minor_width,
        color="gray",
        alpha=minor_alpha,
    )

    subplot.set_xlim(0.9, 11000)
    subplot.set_ylim(1e3, 1e9)

    for spine in subplot.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.0)


# ax.legend(fontsize=12, framealpha=0.5, loc=[1.01, 0.25])
fig.supxlabel("Number of Parallel Environments", fontsize=20, y=0.00)
fig.legend(fontsize=24, framealpha=0.0, loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=5)

fig.set_size_inches(48 / 2.54, 12 / 2.54)  # Set figure size to 20cm by 20cm
plt.tight_layout()

plt.savefig(
    os.path.join(os.path.dirname(__file__), "benchmark_other_libraries.jpg"),
    bbox_inches="tight",
)
plt.savefig(
    os.path.join(os.path.dirname(__file__), "benchmark_other_libraries.pdf"),
    bbox_inches="tight",
)
