import flax.linen as nn
import jax.numpy as jnp
from bluffjax.utils.typing import Any, FloatArray


class ScaledInput(nn.Module):
    """Applies `net` to the observation with the features [start, stop)
    multiplied by `factor`, the others unchanged (e.g. raw chip counts divided
    by the starting stack). `net` is a submodule, so its parameters sit under
    "net" and a checkpoint of the unwrapped network doesn't load into it."""

    net: nn.Module
    start: int
    stop: int
    factor: float

    @nn.compact
    def __call__(self, x: FloatArray) -> Any:
        scale = jnp.ones(x.shape[-1], dtype=jnp.float32)
        scale = scale.at[self.start : self.stop].set(self.factor)
        return self.net(x * scale)
