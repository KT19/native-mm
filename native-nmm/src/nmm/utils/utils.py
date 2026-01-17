from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image


def count_params(params: Any) -> int:
    leaves = jax.tree_util.tree_leaves(params)
    total = sum(x.size for x in leaves)

    return total


# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(img: Image.Image, image_size: int) -> np.ndarray:
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2

    img = img.crop((left, top, left + s, top + s))
    img = img.resize((image_size, image_size), resample=Image.BICUBIC)  # type: ignore

    x = np.asarray(img).astype(np.float32) / 255.0
    # Apply normalization
    x = (x - IMAGENET_MEAN) / IMAGENET_STD

    return x


def softmax_sample(
    rng: jax.Array,
    logits: jax.Array,
    temperature: float,
    top_k: int,
) -> tuple[int, jax.Array]:
    if temperature <= 0.0:
        return int(jnp.argmax(logits)), rng

    logits = logits.astype(jnp.float32) / jnp.asarray(temperature, jnp.float32)

    if top_k > 0:
        kth = jnp.sort(logits)[-top_k]
        logits = jnp.where(logits >= kth, logits, jnp.finfo(jnp.float32).min)

    probs = jax.nn.softmax(logits, axis=-1)
    rng, sub = jax.random.split(rng)
    idx = jax.random.choice(sub, a=probs.shape[0], p=probs)

    return int(idx), rng
