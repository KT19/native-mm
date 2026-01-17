import jax
from flax import linen as nn

from nmm.models.config import ModelConfig


class SwiGLU(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        config = self.config
        hidden = int(config.d_model * config.mlp_ratio)
        gate = nn.Dense(hidden, dtype=config.dtype, use_bias=False, name="gate")(x)
        up = nn.Dense(hidden, dtype=config.dtype, use_bias=False, name="up")(x)
        act = nn.silu(gate) * up
        out = nn.Dense(config.d_model, dtype=config.dtype, use_bias=False, name="down")(act)

        return out
