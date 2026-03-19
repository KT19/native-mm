import jax
import jax.numpy as jnp


def masked_ce_loss(logits: jax.Array, targets: jax.Array, mask: jax.Array) -> jax.Array:
    """
    logits: (B, T, V)
    targets: (B, T)
    mask: (B, T)
    """
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, targets[..., None], axis=-1).squeeze(-1)  # (B, T)
    nll = jnp.where(mask, nll, 0.0)
    denom = jnp.maximum(mask.sum(), 1)

    return nll.sum() / denom
