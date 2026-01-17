import os

import jax
import jax.numpy as jnp
import optax
from flax.training import checkpoints, train_state

from nmm.models.config import ModelConfig
from nmm.models.native_model import NativeMultimodalLM


class TrainState(train_state.TrainState):
    step_rng: jax.Array


def create_state(
    rng: jax.Array,
    config: ModelConfig,
    lr: float,
    weight_decay: float,
    warmup_steps: int,
    total_steps: int,
    prev_ckpt_dir: str | None,
    accum_steps: int = 1,
) -> TrainState:
    model = NativeMultimodalLM(config)

    # learning rate schedule
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=lr * 0.1,
    )

    # dummy init
    dummy_text = jnp.zeros((1, config.max_text_len), dtype=jnp.int32)
    dummy_tmask = jnp.ones((1, config.max_text_len), dtype=bool)

    dummy_image = jnp.zeros((1, config.image_size, config.image_size, 3), dtype=jnp.float32)
    n_patches = (config.image_size // config.patch_size) ** 2
    dummy_imask = jnp.zeros((1, n_patches), dtype=bool)

    variables = model.init(
        rng,
        text_ids=dummy_text,
        images=dummy_image,
        text_attention_mask=dummy_tmask,
        image_attention_mask=dummy_imask,
        train=True,
    )
    params = variables["params"]
    if prev_ckpt_dir is not None:
        # restore params
        # load checkpoint
        restored = checkpoints.restore_checkpoint(
            ckpt_dir=os.path.abspath(prev_ckpt_dir), target={"params": params}, step=None
        )

        params = restored["params"]
        print(f"\nparameter restored from {prev_ckpt_dir}\n")

    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay))
    tx = optax.MultiSteps(tx, every_k_schedule=accum_steps)

    return TrainState.create(apply_fn=model.apply, params=params, tx=tx, step_rng=rng)
