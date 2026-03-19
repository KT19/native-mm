import glob
from collections.abc import Iterator
import random
import numpy as np
import webdataset as wds
from PIL import Image
from tokenizers import Tokenizer

from nmm.tokenizer.tokenizer_io import SpecialTokenIds
from nmm.utils.utils import preprocess_image

Sample = tuple[Image.Image, str]


def llava_local_stream(data_path: str) -> Iterator[Sample]:
    """
    (pil_image, target_text)
    """
    shards = glob.glob(data_path + "/*.tar")
    if not shards:
        raise FileNotFoundError(f"Not .tar files found in {data_path}")
    ds = wds.WebDataset(shards, shardshuffle=True).shuffle(1000).decode("pil").to_tuple("jpg", "txt")  # type: ignore

    for image, text in ds:
        yield image, text


# Diverse prompts to avoid overfitting during pretraining
CAPTION_PROMPTS = [
    "Describe the image.",
    "What do you see in this image?",
    "Can you describe what's in the picture?",
    "What is shown in this image?",
    "Provide a detailed description of this image.",
]


def build_prompt_for_caption() -> str:
    """
    Build prompt that matches SFT chat format for image captioning.
    """
    user_prompt = random.choice(CAPTION_PROMPTS) 
    return f"<|bos|><|img|><|user|>{user_prompt}\n<|assistant|>"


def pack_prompt_answer(
    tokenizer: Tokenizer,
    prompt: str,
    answer: str,
    st: SpecialTokenIds,
    max_text_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        text_ids: (T,) padded
        text_attention_mask: (T,) bool
        text_loss_mask: (T,) bool
    """
    p_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    a_ids = tokenizer.encode(answer, add_special_tokens=False).ids

    ids = p_ids + a_ids + [st.eos]
    ans_start = len(p_ids)

    loss_mask = [False] * len(ids)
    for i in range(ans_start, len(ids)):
        loss_mask[i] = True

    ids = ids[:max_text_len]
    loss_mask = loss_mask[:max_text_len]

    pad_len = max_text_len - len(ids)
    if pad_len > 0:
        ids = ids + [st.pad] * pad_len
        loss_mask = loss_mask + [False] * pad_len

    text_ids = np.asarray(ids, dtype=np.int32)
    text_attention_mask = text_ids != st.pad
    text_loss_mask = np.asarray(loss_mask, dtype=np.bool_)

    return text_ids, text_attention_mask, text_loss_mask


def collate_llava_onevision(
    batch: list[tuple[Image.Image, str]],
    tokenizer: Tokenizer,
    st: SpecialTokenIds,
    image_size: int,
    n_patches: int,
    max_text_len: int,
) -> dict[str, np.ndarray]:
    B = len(batch)

    images = np.zeros((B, image_size, image_size, 3), dtype=np.float32)
    image_attention_mask = np.ones((B, n_patches), dtype=np.bool_)

    text_ids = np.zeros((B, max_text_len), dtype=np.int32)
    text_attention_mask = np.zeros((B, max_text_len), dtype=np.bool_)
    text_loss_mask = np.zeros((B, max_text_len), dtype=np.bool_)

    for i, (im, tgt) in enumerate(batch):
        images[i] = preprocess_image(im, image_size=image_size)
        prompt = build_prompt_for_caption()
        t_ids, t_attn, t_loss = pack_prompt_answer(
            tokenizer=tokenizer, prompt=prompt, answer=tgt, st=st, max_text_len=max_text_len
        )
        text_ids[i] = t_ids
        text_attention_mask[i] = t_attn
        text_loss_mask[i] = t_loss

    return dict(
        images=images,
        image_attention_mask=image_attention_mask,
        text_ids=text_ids,
        text_attention_mask=text_attention_mask,
        text_loss_mask=text_loss_mask,
    )


def make_batch_llava_onevision(stream: Iterator[Sample], batch_size: int) -> Iterator[list[Sample]]:
    buf: list[Sample] = []

    for item in stream:
        buf.append(item)
        if len(buf) == batch_size:
            yield buf
            buf = []
