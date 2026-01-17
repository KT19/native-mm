import io
import json
import os
from typing import Any

import jax
import jax.numpy as jnp
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from flax.training import checkpoints
from PIL import Image
from tokenizers import Tokenizer

from nmm.models.config import ModelConfig
from nmm.models.native_model import NativeMultimodalLM
from nmm.tokenizer.tokenizer_io import SpecialTokenIds, load_tokenizer
from nmm.utils.mm_inference import generate_mm
from nmm.utils.text_inference import generate_text

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- GLOBAL MODEL --
tokenizer, st = load_tokenizer("saved_tokenizer/tokenizer.json")
with open("configs/model.yaml") as f:
    model_config = yaml.safe_load(f)

config = ModelConfig(**model_config)
n_patches = (config.image_size // config.patch_size) ** 2
model = NativeMultimodalLM(config)
rng = jax.random.PRNGKey(0)
dummy_text = jnp.zeros((1, config.max_text_len), dtype=jnp.int32)
dummy_image = jnp.zeros((1, config.image_size, config.image_size, 3), dtype=jnp.float32)
dummy_type = jnp.zeros((1, config.max_seq_len), dtype=jnp.int32)
dummy_tmask = jnp.ones((1, config.max_text_len), dtype=bool)
dummy_imask = jnp.zeros((1, n_patches), dtype=bool)

variables = model.init(
    rng,
    text_ids=dummy_text,
    images=dummy_image,
    text_attention_mask=dummy_tmask,
    image_attention_mask=dummy_imask,
    token_type_ids=dummy_type,
    train=False,
)

initial_params = variables["params"]
detected_ckpt = checkpoints.latest_checkpoint("saved_checkpoints/sft_chat")
restored = checkpoints.restore_checkpoint(
    ckpt_dir=os.path.abspath("saved_checkpoints/sft_chat"), target={"params": initial_params}, step=None
)
print("\nrestored\n")
params = restored["params"]


def generate_chat(
    params: Any,
    model: NativeMultimodalLM,
    tokenizer: Tokenizer,
    st: SpecialTokenIds,
    image: Image.Image | None,
    prompt: str,
    image_size: int,
    n_patches: int,
    max_text_len: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    rng: jax.Array | None = None,
) -> str:
    if image is not None:
        # multi-modal
        response = generate_mm(
            params=params,
            model=model,
            tokenizer=tokenizer,
            st=st,
            image=image,
            prompt=prompt,
            image_size=image_size,
            n_patches=n_patches,
            max_text_len=max_text_len,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            rng=rng,
        )
    else:
        # Text only
        response = generate_text(
            params=params,
            model=model,
            tokenizer=tokenizer,
            st=st,
            prompt=prompt,
            max_text_len=max_text_len,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            rng=rng,
        )

    # filter out (prompt is eliminated)
    response = response[len(prompt) :]
    return response


def prepare_chat_input(image: UploadFile | None, chat_history: Any, user_input: str) -> str:
    prompt = ""

    prompt += "<bos>"

    if image is not None:
        prompt += "<img>"

    for msg in chat_history:
        role = msg["role"]
        content = msg["user"]

        if role == "user":
            prompt += "<|user|>"
        else:
            prompt += "<|assistant|>"

        # Add content
        prompt += content
        prompt += "\n"  # always assuming line break

    # add user prompt
    prompt += "<|user|>"
    prompt += user_input
    prompt += "\n"

    return prompt


@app.post("/chat")
async def chat_endpoint(image: UploadFile = File(None), history: str = Form(...), user_text: str = Form()):
    try:
        # 1. Process History
        chat_history = json.loads(history)
        prompt = prepare_chat_input(image, chat_history, user_text)
        pil_image = None
        if image:
            img_bytes = await image.read()
            pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # inference
        output_text = generate_chat(
            params=params,
            model=model,
            tokenizer=tokenizer,
            st=st,
            image=pil_image,
            prompt=prompt,
            image_size=config.image_size,
            n_patches=n_patches,
            max_text_len=config.max_text_len,
            max_new_tokens=128,
            temperature=0.8,
            top_k=50,
            rng=jax.random.PRNGKey(42),
        )

        return {"response": output_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
