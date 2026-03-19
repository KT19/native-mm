from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from tokenizers import Tokenizer

from nmm.models.native_model import NativeMultimodalLM
from nmm.tokenizer.tokenizer_io import SpecialTokenIds
from nmm.utils.utils import softmax_sample


@partial(jax.jit, static_argnums=(1,))
def forward_logits_text_only(
    params: Any,
    model: NativeMultimodalLM,
    input_ids: jax.Array,
    attn_mask: jax.Array,
) -> jax.Array:
    """
    Returns logits for all positions: (1, T, V)
    """
    B, T = input_ids.shape
    imask = jnp.zeros((B, 0), dtype=bool)

    logits = model.apply(
        {"params": params},
        text_ids=input_ids,
        images=None,
        text_attention_mask=attn_mask,
        image_attention_mask=imask,
        train=False,
    )

    return logits  # type: ignore


def generate_text(
    params: Any,
    model: NativeMultimodalLM,
    tokenizer: Tokenizer,
    st: SpecialTokenIds,
    prompt: str,
    max_text_len: int,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int = 50,
    rng: jax.Array | None = None,
) -> str:
    """
    Autoregressive generation
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    ids = tokenizer.encode(prompt).ids
    if len(ids) == 0:
        ids = [st.bos]

    if ids[0] != st.bos:
        ids = [st.bos] + ids

    generated = []
    # Generate
    for _ in range(max_new_tokens):
        current_len = len(ids)
        if current_len >= max_text_len:
            input_ids = ids[-max_text_len:]
        else:
            input_ids = ids

        padded_input = jnp.full((1, max_text_len), st.pad, dtype=jnp.int32)
        padded_input = padded_input.at[0, : len(input_ids)].set(input_ids)

        attn_mask = padded_input != st.pad
        logits = forward_logits_text_only(params, model, padded_input, attn_mask)  # (1, T, V)

        last_token_index = len(input_ids) - 1
        next_logits = logits[0, last_token_index]

        if temperature == 0.0:
            next_id = int(jnp.argmax(next_logits))
        else:
            next_id, rng = softmax_sample(rng, next_logits, temperature=temperature, top_k=top_k)

        ids.append(next_id)
        generated.append(next_id)
        if next_id == st.eos:
            break

    return tokenizer.decode(generated)
