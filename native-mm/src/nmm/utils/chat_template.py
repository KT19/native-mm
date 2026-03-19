import numpy as np
from tokenizers import Tokenizer

from nmm.tokenizer.tokenizer_io import SpecialTokenIds


def build_chat_prompt_mm(user_text: str) -> str:
    return f"<|bos|><|img|><|user|>{user_text.strip()}\n<|assistant|>"


def build_chat_prompt_text(user_text: str) -> str:
    return f"<|bos|><|user|>{user_text.strip()}\n<|assistant|>"


def pack_chat_sft(
    tokenizer: Tokenizer,
    st: SpecialTokenIds,
    messages: list[dict[str, str]],
    max_text_len: int,
    has_image: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        text_ids: (T)
        text_attention_mask: (T) bool
        text_loss_mask: (T) bool
    Format:
        <|bos|><|img|><|user|>{...}\n<|assistant|>
    """
    ids: list[int] = []
    loss: list[bool] = []

    def add(piece: str, is_loss: bool) -> None:
        p = tokenizer.encode(piece, add_special_tokens=False).ids
        ids.extend(p)
        loss.extend([is_loss] * len(p))

    # BOS
    add("<|bos|>", False)

    # Image marker
    if has_image:
        add("<|img|>", False)

    for m in messages:
        role = m.get("role", "")
        content = (m.get("content", "") or "").strip()

        if role == "user":
            add("<|user|>", False)
            add(content + "\n", False)
            continue

        if role == "assistant":
            add("<|assistant|>", False)
            add(content, True)
            add("<|eos|>", True)
            add("\n", False)
            continue

        raise ValueError(f"Unknown role '{role}'")

    # truncate/pad
    ids = ids[:max_text_len]
    loss = loss[:max_text_len]

    pad_len = max_text_len - len(ids)
    if pad_len > 0:
        ids += [st.pad] * pad_len
        loss += [False] * pad_len

    text_ids = np.asarray(ids, dtype=np.int32)
    text_attention_mask = text_ids != st.pad
    text_loss_mask = np.asarray(loss, dtype=np.bool_)

    return text_ids, text_attention_mask, text_loss_mask
