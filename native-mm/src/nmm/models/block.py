import jax
from flax import linen as nn

from nmm.models.config import ModelConfig
from nmm.models.mlp import SwiGLU
from nmm.models.rms_norm import RMSNorm
from nmm.models.self_attn import SelfAttention


class TransformerBlock(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x: jax.Array, attn_mask: jax.Array, positions: jax.Array, train: bool) -> jax.Array:
        config = self.config

        h = RMSNorm(config.d_model, dtype=config.dtype, name="rms_attn")(x)
        h = SelfAttention(config, name="attn")(h, attn_mask=attn_mask, positions=positions, train=train)
        if config.dropout > 0.0:
            h = nn.Dropout(rate=config.dropout)(h, deterministic=not train)
        x = x + h.astype(x.dtype)

        h = RMSNorm(config.d_model, dtype=config.dtype, name="rms_mlp")(x)
        h = SwiGLU(config, name="mlp")(h)
        if config.dropout > 0.0:
            h = nn.Dropout(rate=config.dropout)(h, deterministic=not train)
        x = x + h.astype(x.dtype)

        return x
