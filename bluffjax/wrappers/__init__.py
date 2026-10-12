"""
Wrappers that expose BluffJAX games through the JaxMARL and PettingZoo APIs.

- JaxMARLEnv (jaxmarl_wrapper.py): JaxMARL MultiAgentEnv, pure JAX.
- PettingZooAECEnv, PettingZooParallelEnv (pettingzoo_wrapper.py): PettingZoo
  AECEnv for the turn-based games and ParallelEnv for the simultaneous-move
  games.

Install their dependencies with `uv sync --extra wrappers`. Each wrapper is
imported on first use, so the JaxMARL wrapper doesn't need PettingZoo and the
PettingZoo wrappers don't need JaxMARL. See demo.py for how to use them.
"""

import importlib

_WRAPPER_MODULES = {
    "JaxMARLEnv": "bluffjax.wrappers.jaxmarl_wrapper",
    "PettingZooAECEnv": "bluffjax.wrappers.pettingzoo_wrapper",
    "PettingZooParallelEnv": "bluffjax.wrappers.pettingzoo_wrapper",
}

__all__ = list(_WRAPPER_MODULES)


def __getattr__(name: str):
    if name not in _WRAPPER_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(_WRAPPER_MODULES[name]), name)
