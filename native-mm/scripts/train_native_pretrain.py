import os
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import yaml
from flax.training import checkpoints

from nmm.data.fineweb_edu_stream import make_batch, pack_tokens_to_blocks, stream_fineweb_edu_text
from nmm.data.llava_mid_train_local_stream import (
    collate_llava_onevision,
    llava_local_stream,
    make_batch_llava_onevision,
)
from nmm.models.config import ModelConfig
from nmm.tokenizer.tokenizer_io import load_tokenizer
from nmm.utils.create_state import TrainState, create_state
from nmm.utils.losses import masked_ce_loss
from nmm.utils.utils import count_params


class TextBatch(NamedTuple):
    input_ids: jax.Array  # (B, T_text)
    target_ids: jax.Array  # (B, T_text)
    text_attention_mask: jax.Array  # (B, T_text)
    text_loss_mask: jax.Array  # (B, T_text)


class MMBatch(NamedTuple):
    images: jax.Array  # (B, H, W, 3)
    image_attention_mask: jax.Array  # (B, n_patches) bool
    text_ids: jax.Array  # (B, T_text)
    text_attention_mask: jax.Array  # (B, T_Text)
    text_loss_mask: jax.Array  # (B, T_Text)


class UnifiedBatch(NamedTuple):
    # text field
    text_batch: TextBatch | None
    # multi modal
    mm_batch: MMBatch | None


@partial(jax.jit, static_argnums=(2,))
def train_step(state: TrainState, batch: UnifiedBatch, is_mm: bool) -> tuple[TrainState, dict[str, jax.Array]]:
    rng, step_rng = jax.random.split(state.step_rng)  # default split = 2

    def loss_fn(params: Any) -> tuple[jax.Array, dict[str, jax.Array]]:
        if is_mm:
            assert batch.mm_batch
            logits = state.apply_fn(
                {"params": params},
                text_ids=batch.mm_batch.text_ids,
                images=batch.mm_batch.images,
                text_attention_mask=batch.mm_batch.text_attention_mask,
                image_attention_mask=batch.mm_batch.image_attention_mask,
                train=True,
                rngs={"dropout": step_rng},
            )  # (B, T, V)

            n_img = batch.mm_batch.image_attention_mask.shape[1]

            preds = logits[:, n_img:-1, :]
            targets = batch.mm_batch.text_ids[:, 1:]

            # compute loss where the target token is part of the answer
            mask = batch.mm_batch.text_attention_mask[:, 1:] & batch.mm_batch.text_loss_mask[:, 1:]
        else:
            assert batch.text_batch
            # text only
            B, T = batch.text_batch.input_ids.shape
            logits = state.apply_fn(
                {"params": params},
                text_ids=batch.text_batch.input_ids,
                images=None,
                text_attention_mask=batch.text_batch.text_attention_mask,
                image_attention_mask=jnp.zeros((B, 0), dtype=bool),
                train=True,
                rngs={"dropout": step_rng},
            )
            preds = logits
            targets = batch.text_batch.target_ids
            mask = batch.text_batch.text_loss_mask

        loss = masked_ce_loss(preds, targets, mask)

        return loss, {"loss": loss}

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(step_rng=rng)
    metrics["loss"] = loss

    return state, metrics


def main() -> None:
    available_devices = jax.devices()
    print(f"Available devices: {available_devices}")

    tokenizer_path = "saved_tokenizer/tokenizer.json"
    ckpt_dir = "saved_checkpoints/native"
    os.makedirs(ckpt_dir, exist_ok=True)
    # -- hyper parameters --
    # load config
    with open("configs/model.yaml", "r") as f:
        model_config = yaml.safe_load(f)

    micro_batch_size = 8
    accum_steps = 4
    lr = 3e-4
    weight_decay = 0.1
    total_steps = 20000
    warmup_steps = 2000
    save_every = 2000
    ratio = 3  # i.e., text : mm = 3 : 1
    total_micro_steps = total_steps * accum_steps

    tok, st = load_tokenizer(tokenizer_path)

    config = ModelConfig(**model_config)

    n_patches = (config.image_size // config.patch_size) ** 2
    assert config.max_seq_len == n_patches + config.max_text_len

    rng = jax.random.PRNGKey(0)
    state = create_state(
        rng,
        config,
        lr=lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        prev_ckpt_dir=None,
        accum_steps=accum_steps,
    )

    total_params = count_params(state.params)
    print(f"Model Parameters: {total_params}")
    # for text
    text_iter = stream_fineweb_edu_text(name="CC-MAIN-2025-05", split="train", streaming=True)
    blocks = pack_tokens_to_blocks(text_iter=text_iter, tokenizer=tok, seq_len=config.max_text_len)
    text_batches = make_batch(blocks, batch_size=micro_batch_size)

    # for multi-modal
    stream = llava_local_stream(data_path="./local_llava_data")
    mm_batches = make_batch_llava_onevision(stream, batch_size=micro_batch_size)

    loss_acc = 0.0
    print("--- Start training ---")
    for step in range(1, total_micro_steps + 1):
        is_mm = step % (ratio + 1) == 0
        if is_mm:
            try:
                raw_batch = next(mm_batches)
            except StopIteration:
                print("Create new stream for multi-modal")
                stream = llava_local_stream(data_path="./local_llava_data")
                mm_batches = make_batch_llava_onevision(stream, batch_size=micro_batch_size)
                raw_batch = next(mm_batches)
            np_batch = collate_llava_onevision(
                batch=raw_batch,
                tokenizer=tok,
                st=st,
                image_size=config.image_size,
                n_patches=n_patches,
                max_text_len=config.max_text_len,
            )
            mm_batch = MMBatch(
                images=jnp.asarray(np_batch["images"]),
                image_attention_mask=jnp.asarray(np_batch["image_attention_mask"]),
                text_ids=jnp.asarray(np_batch["text_ids"]),
                text_attention_mask=jnp.asarray(np_batch["text_attention_mask"]),
                text_loss_mask=jnp.asarray(np_batch["text_loss_mask"]),
            )
            batch = UnifiedBatch(
                text_batch=None,
                mm_batch=mm_batch,
            )

        else:
            try:
                x_np, y_np = next(text_batches)
            except StopIteration:
                text_iter = stream_fineweb_edu_text(name="CC-MAIN-2025-05", split="train", streaming=True)
                blocks = pack_tokens_to_blocks(text_iter=text_iter, tokenizer=tok, seq_len=config.max_text_len)
                text_batches = make_batch(blocks, batch_size=micro_batch_size)
                x_np, y_np = next(text_batches)

            x = jnp.asarray(x_np)
            y = jnp.asarray(y_np)
            # Loss mask based on target validity (non-pad targets)
            text_loss_mask = y != st.pad
            text_batch = TextBatch(
                input_ids=x,
                target_ids=y,
                text_attention_mask=(x != st.pad),
                text_loss_mask=text_loss_mask,
            )
            batch = UnifiedBatch(text_batch=text_batch, mm_batch=None)

        state, metrics = train_step(state, batch, is_mm=is_mm)

        loss_acc += float(metrics["loss"])

        if step % accum_steps == 0:
            global_step = step // accum_steps
            avg_loss = loss_acc / accum_steps
            loss_acc = 0.0

            if global_step % 50 == 0:
                print(f"global step={global_step} | loss={avg_loss:.4f}")

            if global_step % save_every == 0:
                checkpoints.save_checkpoint(
                    ckpt_dir=os.path.abspath(ckpt_dir),
                    target={"params": state.params},
                    step=global_step,
                    overwrite=True,
                )
                print(f"Saved checkpoint at step={global_step} to {ckpt_dir}")

    checkpoints.save_checkpoint(
        ckpt_dir=os.path.abspath(ckpt_dir),
        target={"params": state.params},
        step=total_steps,
        overwrite=True,
    )
    print(f"Saved checkpoint at {total_steps}")
    del stream
    import gc

    gc.collect()


if __name__ == "__main__":
    main()
