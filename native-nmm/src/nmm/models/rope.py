from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class RoPECache:
    cos: jax.Array
    sin: jax.Array

    @staticmethod
    def _build_freqs(head_dim: int, theta: float) -> jax.Array:
        rope_dim = head_dim
        i = jnp.arange(0, rope_dim, 2, dtype=jnp.float32)
        freqs = 1.0 / (theta ** (i / rope_dim))

        return freqs

    @classmethod
    def build(cls, max_seq_len: int, head_dim: int, theta: float) -> "RoPECache":
        assert head_dim % 2 == 0
        freqs = cls._build_freqs(head_dim=head_dim, theta=theta)
        pos = jnp.arange(max_seq_len, dtype=jnp.float32)
        angles = pos[:, None] * freqs[None, :]

        return cls(cos=jnp.cos(angles), sin=jnp.sin(angles))

    def apply(self, x: jax.Array, positions: jax.Array) -> jax.Array:
        """
        Apply RoPE to x
        x: (B, H, T, D)
        """
        _, _, _, D = x.shape
        assert D % 2 == 0

        x_even = x[..., 0::2]  # (B, H, T, half)
        x_odd = x[..., 1::2]  # (B, H, T, half)

        cos = self.cos[positions]
        sin = self.sin[positions]

        if positions.ndim == 1:
            # Broadcast to (B, H, T, half)
            cos = cos[None, None, :, :]
            sin = sin[None, None, :, :]
        elif positions.ndim == 2:
            cos = cos[:, None, :, :]
            sin = sin[:, None, :, :]

        y_even = x_even * cos - x_odd * sin
        y_odd = x_even * sin + x_odd * cos

        y = jnp.empty_like(x)
        y = y.at[..., 0::2].set(y_even.astype(x.dtype))
        y = y.at[..., 1::2].set(y_odd.astype(x.dtype))

        return y
