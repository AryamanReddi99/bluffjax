"""
Combine win-rate head-to-head heatmaps for 5 games into a single figure
(2 rows x 3 cols, one shared colorbar).
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))

DISPLAY_NAMES = {
    "ppo_full": "PPO-NFSP\n(1e7)",
    "ppo_half": "PPO-NFSP\n(5e6)",
    "pqn_full": "PQN-NFSP\n(1e7)",
    "pqn_half": "PQN-NFSP\n(5e6)",
    "random": "Random",
}

# (game dir, subtitle, results-file suffix, show_row_labels, show_col_labels)
GAMES = [
    ("bluff", "(a) Bluff", "avg", True, True),
    ("kemps", "(b) Kemps", "avg", False, True),
    ("werewolf", "(c) Werewolf", "br", False, True),
    ("five_card_draw", "(d) Five Card Draw", "br", True, False),
    ("seven_card_stud", "(e) Seven Card Stud", "avg", False, False),
]


def build_win_matrix(results):
    agent_types = results["agent_types"]
    idx = {name: i for i, name in enumerate(agent_types)}
    n = len(agent_types)

    win_matrix = np.full((n, n), np.nan)
    for m in results["matchups"]:
        i, j = idx[m["agent_a"]], idx[m["agent_b"]]
        win_a = m["win_rate_a"]["avg"] * 100
        win_b = m["win_rate_b"]["avg"] * 100
        if i == j:
            win_matrix[i, j] = (win_a + win_b) / 2
        else:
            win_matrix[i, j] = win_a
            win_matrix[j, i] = win_b

    return agent_types, win_matrix


def plot_panel(ax, matrix, agent_types, title, show_row_labels, show_col_labels):
    labels = [DISPLAY_NAMES.get(a, a) for a in agent_types]
    n = len(agent_types)

    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=100)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels if show_col_labels else [], fontsize=8)
    ax.set_yticklabels(labels if show_row_labels else [], fontsize=8)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left")

    for i in range(n):
        for j in range(n):
            value = matrix[i, j]
            text_color = "white" if value < 50 else "black"
            ax.text(j, i, f"{value:.1f}", ha="center", va="center",
                     color=text_color, fontsize=8)

    ax.set_title(title, fontsize=13, pad=10, y=-0.16)
    return im


def main():
    fig = plt.figure(figsize=(13, 8.5))
    gs = GridSpec(2, 3, figure=fig, wspace=0.15, hspace=0.2)

    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    im = None
    seven_cs_ax = None

    for (row, col), (game_dir, subtitle, suffix, show_row, show_col) in zip(positions, GAMES):
        ax = fig.add_subplot(gs[row, col])
        path = os.path.join(HERE, game_dir, f"head2head_results_{suffix}.json")
        with open(path, "r") as f:
            results = json.load(f)
        agent_types, win_matrix = build_win_matrix(results)
        im = plot_panel(ax, win_matrix, agent_types, subtitle, show_row, show_col)
        if game_dir == "seven_card_stud":
            seven_cs_ax = ax

    # Colorbar: a vertical bar within the empty (1, 2) cell, height-matched
    # to the Seven Card Stud panel.
    cell_ax = fig.add_subplot(gs[1, 2])
    cell_ax.set_axis_off()
    cell_pos = cell_ax.get_position()
    fig.canvas.draw()
    seven_cs_pos = seven_cs_ax.get_position()
    cbar_ax = fig.add_axes([
        cell_pos.x0 + 0.15 * cell_pos.width,
        seven_cs_pos.y0,
        0.22 * cell_pos.width,
        seven_cs_pos.height,
    ])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Win rate (row vs column) (%)", fontsize=11)
    # fig.suptitle("Head-to-Head Win Rates for Model-Free RL on New Games", y=0.02, fontsize=24)

    output_path = os.path.join(HERE, "all_games_win_rate")
    fig.savefig(f"{output_path}.jpg", format="jpg", dpi=200, bbox_inches="tight")
    fig.savefig(f"{output_path}.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}.jpg and {output_path}.pdf")


if __name__ == "__main__":
    main()
