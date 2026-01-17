import os
from collections.abc import Iterator
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import yaml
from flax.training import checkpoints

from nmm.data.collate_sft import collate_mm_sft, collate_text_sft
from nmm.data.llava_instruct_sft import llava_instruct
from nmm.data.ultrachat_sft import ultrachat_sft
from nmm.models.config import ModelConfig
from nmm.tokenizer.tokenizer_io import load_tokenizer
from nmm.utils.create_state import TrainState, create_state
from nmm.utils.losses import masked_ce_loss
from nmm.utils.utils import count_params


class TextBatch(NamedTuple):
    text_ids: jax.Array
    text_attention_mask: jax.Array
    text_loss_mask: jax.Array


class MMBatch(NamedTuple):
    images: jax.Array
    image_attention_mask: jax.Array
    text_ids: jax.Array
    text_attention_mask: jax.Array
    text_loss_mask: jax.Array


@partial(jax.jit, static_argnums=(2,))
def train_step(state: TrainState, batch: MMBatch | TextBatch, is_mm: bool) -> tuple[TrainState, dict[str, jax.Array]]:
    rng, step_rng = jax.random.split(state.step_rng)

    def loss_fn(params):
        if is_mm:
            assert isinstance(batch, MMBatch)
            # Multi-modal logic
            logits = state.apply_fn(
                {"params": params},
                text_ids=batch.text_ids,
                images=batch.images,
                text_attention_mask=batch.text_attention_mask,
                image_attention_mask=batch.image_attention_mask,
                train=True,
                rngs={"dropout": step_rng},
            )
            n_img = batch.image_attention_mask.shape[1]

            preds = logits[:, n_img:-1, :]
            targets = batch.text_ids[:, 1:]
            mask = batch.text_attention_mask[:, 1:] & batch.text_loss_mask[:, 1:]
        else:
            assert isinstance(batch, TextBatch)
            # Text-only logic
            B, T = batch.text_ids.shape
            logits = state.apply_fn(
                {"params": params},
                text_ids=batch.text_ids,
                images=None,
                text_attention_mask=batch.text_attention_mask,
                image_attention_mask=jnp.zeros((B, 0), dtype=bool),
                train=True,
                rngs={"dropout": step_rng},
            )
            preds = logits[:, :-1, :]
            targets = batch.text_ids[:, 1:]
            mask = batch.text_attention_mask[:, 1:] & batch.text_loss_mask[:, 1:]

        loss = masked_ce_loss(preds, targets, mask)
        return loss, {"loss": loss}

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    state = state.apply_gradients(grads=grads)
    state = state.replace(step_rng=rng)
    metrics["loss"] = loss

    return state, metrics


def batched(it: Iterator[Any], batch_size: int) -> Iterator[list[Any]]:
    buf: list[Any] = []

    for x in it:
        buf.append(x)
        if len(buf) == batch_size:
            yield buf
            buf = []


def main() -> None:
    available_devices = jax.devices()
    print(f"Available devices: {available_devices}")

    tokenizer_path = "saved_tokenizer/tokenizer.json"
    mm_ckpt_dir = "saved_checkpoints/native"
    sft_ckpt_dir = "saved_checkpoints/sft_chat"
    os.makedirs(sft_ckpt_dir, exist_ok=True)
    # -- hyper parameters --
    micro_batch_size = 8
    accum_steps = 8
    lr = 3e-5
    weight_decay = 0.1
    total_steps = 10000
    warmup_steps = 1000
    total_micro_steps = total_steps * accum_steps
    save_every = 1000

    tok, st = load_tokenizer(tokenizer_path)
    with open("configs/model.yaml") as f:
        model_config = yaml.safe_load(f)

    config = ModelConfig(**model_config)
    n_patches = (config.image_size // config.patch_size) ** 2

    detected_ckpt = checkpoints.latest_checkpoint(mm_ckpt_dir)
    print(f"Detected latest checkpoint at: {detected_ckpt}")
    rng = jax.random.PRNGKey(0)
    state = create_state(
        rng,
        config,
        lr=lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        prev_ckpt_dir=mm_ckpt_dir,
        accum_steps=accum_steps,
    )

    total_params = count_params(state.params)
    print(f"Model Parameters: {total_params}")

    # dataset
    text_dataset = ultrachat_sft()
    mm_data = llava_instruct(subsets=["visual_chat", "coco", "image_textualization"])

    text_batches = batched(text_dataset, micro_batch_size)
    mm_batches = batched(mm_data, micro_batch_size)

    loss_acc = 0.0
    print("--- Start training ---")
    for step in range(1, total_micro_steps + 1):
        is_mm = (step % 5) != 0  # weight mm
        # --- make batch ---
        try:
            if is_mm:
                raw = next(mm_batches)
                npb = collate_mm_sft(
                    batch=raw,
                    tokenizer=tok,
                    st=st,
                    image_size=config.image_size,
                    n_patches=n_patches,
                    max_text_len=config.max_text_len,
                )
                batch = MMBatch(
                    images=jnp.asarray(npb["images"]),
                    image_attention_mask=jnp.asarray(npb["image_attention_mask"]),
                    text_ids=jnp.asarray(npb["text_ids"]),
                    text_attention_mask=jnp.asarray(npb["text_attention_mask"]),
                    text_loss_mask=jnp.asarray(npb["text_loss_mask"]),
                )
            else:
                raw = next(text_batches)
                npb = collate_text_sft(batch_messages=raw, tokenizer=tok, st=st, max_text_len=config.max_text_len)
                batch = TextBatch(
                    text_ids=jnp.asarray(npb["text_ids"]),
                    text_attention_mask=jnp.asarray(npb["text_attention_mask"]),
                    text_loss_mask=jnp.asarray(npb["text_loss_mask"]),
                )
        except StopIteration:
            if is_mm:
                print("Create new mm batch")
                mm_batches = batched(
                    llava_instruct(subsets=["visual_chat", "coco", "image_textualization"]), micro_batch_size
                )
                raw = next(mm_batches)

                npb = collate_mm_sft(
                    batch=raw,
                    tokenizer=tok,
                    st=st,
                    image_size=config.image_size,
                    n_patches=n_patches,
                    max_text_len=config.max_text_len,
                )
                batch = MMBatch(
                    images=jnp.asarray(npb["images"]),
                    image_attention_mask=jnp.asarray(npb["image_attention_mask"]),
                    text_ids=jnp.asarray(npb["text_ids"]),
                    text_attention_mask=jnp.asarray(npb["text_attention_mask"]),
                    text_loss_mask=jnp.asarray(npb["text_loss_mask"]),
                )

            else:
                print("Create new text batch")
                text_batches = batched(ultrachat_sft(), micro_batch_size)
                raw = next(text_batches)

                npb = collate_text_sft(batch_messages=raw, tokenizer=tok, st=st, max_text_len=config.max_text_len)
                batch = TextBatch(
                    text_ids=jnp.asarray(npb["text_ids"]),
                    text_attention_mask=jnp.asarray(npb["text_attention_mask"]),
                    text_loss_mask=jnp.asarray(npb["text_loss_mask"]),
                )

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
                    ckpt_dir=os.path.abspath(sft_ckpt_dir),
                    target={"params": state.params},
                    step=global_step,
                    overwrite=True,
                )
                print(f"Saved checkpoint at step={global_step} to {sft_ckpt_dir}")

    checkpoints.save_checkpoint(
        ckpt_dir=os.path.abspath(sft_ckpt_dir),
        target={"params": state.params},
        step=total_steps,
        overwrite=True,
    )
    print(f"Saved checkpoint at {total_steps}")

    import gc

    gc.collect()


if __name__ == "__main__":
    main()
