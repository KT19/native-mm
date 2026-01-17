import jax
import jax.numpy as jnp
from flax import linen as nn


class RMSNorm(nn.Module):
    d_model: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        w = self.param("weight", nn.initializers.ones, (self.d_model,)).astype(self.dtype)
        x_f32 = x.astype(jnp.float32)

        rms = jnp.sqrt(jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True) + self.eps)
        y = (x_f32 / rms).astype(self.dtype)

        return y * w
