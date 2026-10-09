"""
ReBeL (Recursive Belief-based Learning) for heads-up limit Texas hold'em.

Self-play RL with depth-limited search over public belief states: Linear
CFR-D subgames to the end of each betting round, a value network over the
1,326-hand beliefs of both players at the leaves, and the same search when
acting. The implementation is shared with No-Limit in
bluffjax/examples/holdem_rebel/.

    uv run --extra baselines python bluffjax/examples/HUL/hul_rebel.py
    # evaluation opponent of the paper (5e6 samples):
    uv run --extra baselines python bluffjax/examples/HUL/hul_rebel.py \
        --config-name config_rebel_opponent
"""

import hydra
from omegaconf import OmegaConf

from bluffjax.examples.holdem_rebel.game import make_game
from bluffjax.examples.holdem_rebel.train import run_training
from bluffjax.utils.paths import register_resolvers


@hydra.main(version_base=None, config_path="./", config_name="config_rebel")
def main(config: dict) -> None:
    config = OmegaConf.to_container(config, resolve=True)
    run_training(make_game("hul"), config)


if __name__ == "__main__":
    register_resolvers()
    main()
