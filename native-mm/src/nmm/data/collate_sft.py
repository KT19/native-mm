import numpy as np
from PIL import Image
from tokenizers import Tokenizer

from nmm.tokenizer.tokenizer_io import SpecialTokenIds
from nmm.utils.chat_template import pack_chat_sft
from nmm.utils.utils import preprocess_image


def collate_text_sft(
    batch_messages: list[list[dict[str, str]]],
    tokenizer: Tokenizer,
    st: SpecialTokenIds,
    max_text_len: int,
) -> dict[str, np.ndarray]:
    B = len(batch_messages)
    text_ids = np.zeros((B, max_text_len), dtype=np.int32)
    text_attention_mask = np.zeros((B, max_text_len), dtype=np.bool_)
    text_loss_mask = np.zeros((B, max_text_len), dtype=np.bool_)

    for i, msgs in enumerate(batch_messages):
        ids, attn, loss = pack_chat_sft(
            tokenizer=tokenizer,
            st=st,
            messages=msgs,
            max_text_len=max_text_len,
            has_image=False,
        )
        text_ids[i] = ids
        text_attention_mask[i] = attn
        text_loss_mask[i] = loss

    return dict(text_ids=text_ids, text_attention_mask=text_attention_mask, text_loss_mask=text_loss_mask)


def collate_mm_sft(
    batch: list[tuple[Image.Image, list[dict[str, str]]]],
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

    for i, (im, msgs) in enumerate(batch):
        images[i] = preprocess_image(im, image_size=image_size)

        ids, attn, loss = pack_chat_sft(
            tokenizer=tokenizer,
            st=st,
            messages=msgs,
            max_text_len=max_text_len,
            has_image=True,
        )

        text_ids[i] = ids
        text_attention_mask[i] = attn
        text_loss_mask[i] = loss

    return dict(
        images=images,
        image_attention_mask=image_attention_mask,
        text_ids=text_ids,
        text_attention_mask=text_attention_mask,
        text_loss_mask=text_loss_mask,
    )
