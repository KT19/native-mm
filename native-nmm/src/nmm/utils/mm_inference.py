from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from PIL import Image
from tokenizers import Tokenizer

from nmm.models.native_model import NativeMultimodalLM
from nmm.tokenizer.tokenizer_io import SpecialTokenIds
from nmm.utils.utils import preprocess_image, softmax_sample


@partial(jax.jit, static_argnums=(1,))
def forward_logits_mm(
    params: Any,
    model: NativeMultimodalLM,
    images: jax.Array,  # (1, n_patches)
    image_attention_mask: jax.Array,  # (1, n_patches)
    text_ids: jax.Array,  # (1, T_text)
    text_attention_mask: jax.Array,  # (1, T_text)
) -> jax.Array:
    """
    Returns logits for all positions: (1, T, V)
    """
    logits = model.apply(
        {"params": params},
        text_ids=text_ids,
        images=images,
        text_attention_mask=text_attention_mask,
        image_attention_mask=image_attention_mask,
        train=False,
    )

    return logits  # type: ignore


def generate_mm(
    params: Any,
    model: NativeMultimodalLM,
    tokenizer: Tokenizer,
    st: SpecialTokenIds,
    image: Image.Image,
    prompt: str,
    image_size: int,
    n_patches: int,
    max_text_len: int,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
    rng: jax.Array | None = None,
) -> str:
    """
    Assume image is given
    Autoregressive generation
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    # Prepare image + masks
    img_np = preprocess_image(image, image_size=image_size)
    img = jnp.asarray(img_np[None, ...], dtype=jnp.float32)
    imask = jnp.ones((1, n_patches), dtype=bool)

    initial_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    # fixed-size text buffer
    curr_text_ids = jnp.full((1, max_text_len), st.pad, dtype=jnp.int32)

    # fill in initial prompt
    init_len = min(len(initial_ids), max_text_len)
    curr_text_ids = curr_text_ids.at[0, :init_len].set(jnp.array(initial_ids[:init_len]))

    generated_ids = list(initial_ids)

    # Generate
    for _ in range(max_new_tokens):
        current_len = len(generated_ids)
        if current_len >= max_text_len:
            break

        # Attention mask (the shape is fixed)
        tmask = curr_text_ids != st.pad

        logits = forward_logits_mm(
            params,
            model,
            images=img,
            image_attention_mask=imask,
            text_ids=curr_text_ids,
            text_attention_mask=tmask,
        )  # (1, n_patches + T, V)

        next_token_idx = n_patches + (current_len - 1)
        next_logits = logits[0, next_token_idx]

        if temperature == 0.0:
            next_id = int(jnp.argmax(next_logits))
        else:
            next_id, rng = softmax_sample(rng, next_logits, temperature=temperature, top_k=top_k)

        generated_ids.append(next_id)
        curr_text_ids = curr_text_ids.at[0, current_len].set(next_id)

        if next_id == st.eos:
            break

    return tokenizer.decode(generated_ids[len(list(initial_ids)) :])
