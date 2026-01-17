import jax
import jax.numpy as jnp
from flax import linen as nn

from nmm.models.config import ModelConfig
from nmm.models.rope import RoPECache


def make_causal_mask(t: int) -> jax.Array:
    m = jnp.tril(jnp.ones((t, t), dtype=bool))

    return m[None, None, :, :]


class SelfAttention(nn.Module):
    config: ModelConfig

    def setup(self) -> None:
        head_dim = self.config.d_model // self.config.n_heads
        self.qkv_proj = nn.Dense(3 * self.config.d_model, dtype=self.config.dtype, use_bias=False, name="qkv")
        self.rope = RoPECache.build(
            max_seq_len=self.config.max_seq_len, head_dim=head_dim, theta=self.config.rope_theta
        )
        self.dropout = nn.Dropout(rate=self.config.attn_dropout)
        self.out_proj = nn.Dense(self.config.d_model, dtype=self.config.dtype, use_bias=False, name="out")

    def __call__(self, x: jax.Array, attn_mask: jax.Array, positions: jax.Array, train: bool) -> jax.Array:
        """
        x: (B, T, d_model)
        attn_mask: (B, 1, 1, T) or (B, 1, T, T)
        returns: (B, T, d_model)
        """

        config = self.config
        head_dim = config.d_model // config.n_heads
        B, T, D = x.shape

        qkv = self.qkv_proj(x)  # (B, L, 3D)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = q.reshape(B, T, config.n_heads, head_dim)
        k = k.reshape(B, T, config.n_heads, head_dim)
        v = v.reshape(B, T, config.n_heads, head_dim)

        q = jnp.swapaxes(q, 1, 2)
        k = jnp.swapaxes(k, 1, 2)
        v = jnp.swapaxes(v, 1, 2)

        q = self.rope.apply(q, positions=positions)
        k = self.rope.apply(k, positions=positions)

        scale = 1.0 / jnp.sqrt(jnp.array(head_dim, dtype=jnp.float32))
        scores = jnp.einsum("bhld,bhmd->bhlm", q.astype(jnp.float32), k.astype(jnp.float32)) * scale
        neg = jnp.array(jnp.finfo(jnp.float32).min, dtype=jnp.float32)
        scores = jnp.where(attn_mask, scores, neg)

        attn = jax.nn.softmax(scores, axis=-1).astype(config.dtype)
        if config.attn_dropout > 0.0:
            attn = self.dropout(attn, deterministic=not train)

        out = jnp.einsum("bhlm,bhmd->bhld", attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)

        out = self.out_proj(out)

        return out
