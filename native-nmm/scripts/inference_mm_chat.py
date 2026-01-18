import os

import jax
import jax.numpy as jnp
import yaml
from flax.training import checkpoints
from PIL import Image

from nmm.models.config import ModelConfig
from nmm.models.native_model import NativeMultimodalLM
from nmm.tokenizer.tokenizer_io import load_tokenizer
from nmm.utils.chat_template import build_chat_prompt_mm, build_chat_prompt_text
from nmm.utils.mm_inference import generate_mm
from nmm.utils.text_inference import generate_text


def render_response(prompt: str, response: str) -> None:
    print("=== PROMPT ===")
    print(prompt)
    print("===RESPONSE")
    print(response)


def main() -> None:
    tokenizer_path = "saved_tokenizer/tokenizer.json"
    ckpt_dir = "saved_checkpoints/sft_chat"
    ckpt_step = None

    tok, st = load_tokenizer(tokenizer_path)
    print(f"Vocab size: {tok.get_vocab_size()}")

    with open("configs/model.yaml", "r") as f:
        model_config = yaml.safe_load(f)
    config = ModelConfig(**model_config)
    n_patches = (config.image_size // config.patch_size) ** 2

    model = NativeMultimodalLM(config)
    rng = jax.random.PRNGKey(0)

    print("Setup...")
    dummy_text = jnp.zeros((1, config.max_text_len), dtype=jnp.int32)
    dummy_image = jnp.zeros((1, config.image_size, config.image_size, 3), dtype=jnp.float32)

    dummy_tmask = jnp.ones((1, config.max_text_len), dtype=bool)
    dummy_imask = jnp.zeros((1, n_patches), dtype=bool)

    variables = model.init(
        rng,
        text_ids=dummy_text,
        images=dummy_image,
        text_attention_mask=dummy_tmask,
        image_attention_mask=dummy_imask,
        train=False,
    )

    initial_params = variables["params"]
    detected_ckpt = checkpoints.latest_checkpoint(ckpt_dir)
    print(f"Detected latest checkpoint at: {detected_ckpt}")

    # load checkpoint
    restored = checkpoints.restore_checkpoint(
        ckpt_dir=os.path.abspath(ckpt_dir), target={"params": initial_params}, step=ckpt_step
    )
    print("\nrestored\n")
    params = restored["params"]
    if params is None:
        raise RuntimeError(f"Failed to restore params from {ckpt_dir}.")

    user_texts = [
        ["What do you see?", "Describe the image."],
        ["Where is this?", "Describe the image."],
        ["What sports?", "Describe the image."],
    ]

    for i in range(len(user_texts)):
        for j in range(2):
            user_text = user_texts[i][j]
            img_path = f"gemini{i + 1}.png"

            print(f"== Process: {img_path}==")
            # test image
            image = Image.open(f"test_images/{img_path}")
            prompt = build_chat_prompt_mm(user_text)
            out = generate_mm(
                params=params,
                model=model,
                tokenizer=tok,
                st=st,
                image=image,
                prompt=prompt,
                image_size=config.image_size,
                n_patches=n_patches,
                max_text_len=config.max_text_len,
                max_new_tokens=256,
                temperature=0.90,
                top_k=50,
                rng=jax.random.PRNGKey(42),
            )

            render_response(prompt, out)
            print("--" * 10)
    print("TEXT ONLY")

    prompt = build_chat_prompt_text("The future of AI?")
    out = generate_text(
        params=params,
        model=model,
        tokenizer=tok,
        st=st,
        prompt=prompt,
        max_text_len=config.max_text_len,
        max_new_tokens=256,
        temperature=0.90,
        top_k=50,
        rng=jax.random.PRNGKey(42),
    )

    render_response(prompt, out)
    print("--" * 10)


if __name__ == "__main__":
    main()
