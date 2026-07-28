"""
Plot head-to-head evaluation results for the 'simple' checkpoints (see
head2head_avg_simple.py / head2head_br_simple.py), restricted to the
50%-trained and 100%-trained checkpoints only (matching the halfway/full
comparison used in the main plot_head2head.py). Layout is identical to
plot_head2head.py's win-rate heatmap; draw rates are not plotted.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GAME_NAME = os.path.basename(HERE)

# Only the halfway (50%) and full (100%) checkpoints, in the same
# full-then-half ordering as the main plot_head2head.py's AGENT_TYPES.
SIMPLE_AGENT_TYPES = ["ppo_1", "ppo_0p5", "pqn_1", "pqn_0p5", "random"]

DISPLAY_NAMES = {
    "ppo_1": "PPO-NFSP\n(100%)",
    "ppo_0p5": "PPO-NFSP\n(50%)",
    "pqn_1": "PQN-NFSP\n(100%)",
    "pqn_0p5": "PQN-NFSP\n(50%)",
    "random": "Random",
}


def build_matrices(results):
    agent_types = SIMPLE_AGENT_TYPES
    idx = {name: i for i, name in enumerate(agent_types)}
    n = len(agent_types)

    win_matrix = np.full((n, n), np.nan)

    for m in results["matchups"]:
        if m["agent_a"] not in idx or m["agent_b"] not in idx:
            continue
        i, j = idx[m["agent_a"]], idx[m["agent_b"]]
        win_a = m["win_rate_a"]["avg"] * 100
        win_b = m["win_rate_b"]["avg"] * 100

        if i == j:
            win_matrix[i, j] = (win_a + win_b) / 2
        else:
            win_matrix[i, j] = win_a
            win_matrix[j, i] = win_b

    return agent_types, win_matrix


def plot_heatmap(matrix, agent_types, cbar_label, output_path):
    labels = [DISPLAY_NAMES.get(a, a) for a in agent_types]
    n = len(agent_types)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=100)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    for i in range(n):
        for j in range(n):
            value = matrix[i, j]
            text_color = "white" if value < 50 else "black"
            ax.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(f"{output_path}.jpg", format="jpg", dpi=200)
    fig.savefig(f"{output_path}.pdf", format="pdf")
    plt.close(fig)
    print(f"Saved {output_path}.jpg and {output_path}.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(HERE, "head2head_results_avg_simple.json"),
    )
    args = parser.parse_args()

    with open(args.input, "r") as f:
        results = json.load(f)

    agent_types, win_matrix = build_matrices(results)

    base = os.path.splitext(os.path.basename(args.input))[0]
    suffix = base.replace("head2head_results", "")

    plot_heatmap(
        win_matrix,
        agent_types,
        cbar_label="Win rate (row vs column) (%)",
        output_path=os.path.join(HERE, f"{GAME_NAME}_win_rate{suffix}"),
    )


if __name__ == "__main__":
    main()
