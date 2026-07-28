"""
Plot head-to-head evaluation results (see head2head.py) as two 5x5 heatmaps:
win rate and draw rate, each symmetric-filled (all 25 cells populated).
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

DISPLAY_NAMES = {
    "ppo_full": "PPO-NFSP\n(1e7)",
    "ppo_half": "PPO-NFSP\n(5e6)",
    "pqn_full": "PQN-NFSP\n(1e7)",
    "pqn_half": "PQN-NFSP\n(5e6)",
    "random": "Random",
}


def build_matrices(results):
    agent_types = results["agent_types"]
    idx = {name: i for i, name in enumerate(agent_types)}
    n = len(agent_types)

    win_matrix = np.full((n, n), np.nan)
    draw_matrix = np.full((n, n), np.nan)

    for m in results["matchups"]:
        i, j = idx[m["agent_a"]], idx[m["agent_b"]]
        win_a = m["win_rate_a"]["avg"] * 100
        win_b = m["win_rate_b"]["avg"] * 100
        draw = m["draw_rate"]["avg"] * 100

        if i == j:
            win_matrix[i, j] = (win_a + win_b) / 2
        else:
            win_matrix[i, j] = win_a
            win_matrix[j, i] = win_b

        draw_matrix[i, j] = draw
        draw_matrix[j, i] = draw

    return agent_types, win_matrix, draw_matrix


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
        default=os.path.join(HERE, "head2head_results.json"),
    )
    args = parser.parse_args()

    with open(args.input, "r") as f:
        results = json.load(f)

    agent_types, win_matrix, draw_matrix = build_matrices(results)

    base = os.path.splitext(os.path.basename(args.input))[0]
    suffix = base.replace("head2head_results", "")

    plot_heatmap(
        win_matrix,
        agent_types,
        cbar_label="Win rate (row vs column) (%)",
        output_path=os.path.join(HERE, f"{GAME_NAME}_win_rate{suffix}"),
    )
    plot_heatmap(
        draw_matrix,
        agent_types,
        cbar_label="Draw rate (%)",
        output_path=os.path.join(HERE, f"{GAME_NAME}_draw_rate{suffix}"),
    )


if __name__ == "__main__":
    main()
