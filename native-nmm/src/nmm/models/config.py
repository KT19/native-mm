from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class ModelConfig:
    # text
    vocab_size: int
    max_text_len: int  # e.g., 384 or 512
    max_seq_len: int  # max_text_len + n_patches
    # model
    d_model: int
    n_heads: int
    n_layers: int
    mlp_ratio: float
    dropout: float
    attn_dropout: float
    dtype: jnp.dtype = jnp.bfloat16
    # RoPE
    rope_theta: float = 10000.0

    # vision tokenization
    image_size: int = 224
    patch_size: int = 16

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads

    @property
    def n_patches(self) -> int:
        g = self.image_size // self.patch_size

        return g * g
