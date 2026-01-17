import jax
import jax.numpy as jnp
from flax import linen as nn

from nmm.models.block import TransformerBlock
from nmm.models.config import ModelConfig
from nmm.models.patch_embed import PatchEmbed
from nmm.models.rms_norm import RMSNorm
from nmm.models.self_attn import make_causal_mask


class NativeMultimodalLM(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(
        self,
        text_ids: jax.Array,  # (B, T_text)
        images: jax.Array | None,  # (B, H, W, C) or None
        text_attention_mask: jax.Array,  # (B, T_text)
        image_attention_mask: jax.Array,  # (B,T_img)
        train: bool,
    ) -> jax.Array:
        config = self.config
        B = text_ids.shape[0]

        tok_emb = nn.Embed(config.vocab_size, config.d_model, dtype=config.dtype, name="tok_emb")
        text_x = tok_emb(text_ids)  # (B, T_text, d_model)

        if images is None:
            img_x = jnp.zeros((B, 0, config.d_model), dtype=config.dtype)
            n_img = 0
        else:
            img_x = PatchEmbed(config.d_model, config.patch_size, config.image_size, dtype=config.dtype, name="patch")(images)
            n_img = img_x.shape[1]

        x = jnp.concatenate([img_x, text_x], axis=1)
        t_total = x.shape[1]
        assert t_total <= config.max_seq_len

        # modality embedding
        type_emb = nn.Embed(2, config.d_model, dtype=config.dtype, name="type_emb")
        img_types = jnp.ones((B, n_img), dtype=jnp.int32)
        text_types = jnp.zeros((B, text_ids.shape[1]), dtype=jnp.int32)
        combined_types = jnp.concatenate([img_types, text_types], axis=1)

        x = x + type_emb(combined_types)

        if config.dropout > 0.0:
            x = nn.Dropout(rate=config.dropout)(x, deterministic=not train)

        # masks
        keep = jnp.concatenate([image_attention_mask, text_attention_mask], axis=1)  # (B, T)
        causal = make_causal_mask(t_total)  # (1, 1, T, T)
        if n_img > 0:
            causal = causal.at[:, :, :n_img, :n_img].set(True)
        attn_mask = causal & keep[:, None, None, :]  # broadcast to (B, 1, T, T)

        img_positions = jnp.arange(n_img, dtype=jnp.int32)
        text_positions = jnp.arange(n_img, n_img + text_ids.shape[1], dtype=jnp.int32)
        positions = jnp.concatenate([img_positions, text_positions], axis=0)

        for i in range(config.n_layers):
            x = TransformerBlock(config, name=f"block_{i}")(x, attn_mask=attn_mask, positions=positions, train=train)

        x = RMSNorm(config.d_model, dtype=config.dtype, name="ln_f")(x)
        # tied LM head
        logits = x @ tok_emb.embedding.T  # (B, T, V)

        return logits
