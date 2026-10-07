"""
Repository paths for the training scripts in bluffjax/examples.

Training outputs go to the repository root no matter where a script is
launched from: checkpoints to checkpoints/<game>/, Hydra configs and logs to
outputs/. Configs refer to the root as ${repo_root:}.
"""

from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]


def register_resolvers() -> None:
    """Registers the ${repo_root:} resolver used by the example configs."""
    OmegaConf.register_new_resolver("repo_root", lambda: str(REPO_ROOT), replace=True)
