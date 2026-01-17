import jax
import jax.numpy as jnp
from flax import linen as nn

from nmm.models.rms_norm import RMSNorm


class PatchEmbed(nn.Module):
    d_model: int
    patch_size: int
    image_size: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, images: jax.Array) -> jax.Array:
        """
        images: (B, H, W, C)
        returns: (B, N, d_model)
        """
        b, h, w, c = images.shape
        p = self.patch_size
        assert h % p == 0 and w % p == 0

        gh, gw = h // p, w // p
        n_patches = gh * gw

        # Flatten patches and project
        x = images.reshape(b, gh, p, gw, p, c).transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(b, n_patches, p * p * c)

        x = nn.Dense(self.d_model, dtype=self.dtype, name="patch_in")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.d_model, dtype=self.dtype, name="proj_out")(x)

        pos_emb = self.param(
            "pos_emb",
            nn.initializers.normal(stddev=0.02),
            (1, n_patches, self.d_model),
        )
        x = x + pos_emb.astype(self.dtype)

        x = RMSNorm(self.d_model, name="patch_norm")(x)

        return x