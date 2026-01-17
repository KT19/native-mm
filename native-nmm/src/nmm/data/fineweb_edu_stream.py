from collections.abc import Iterator

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer


def stream_fineweb_edu_text(name: str, split: str, streaming: bool = True) -> Iterator[str]:
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name=name, split=split, streaming=streaming)
    for ex in ds:
        txt = ex.get("text", None)  # type: ignore
        if isinstance(txt, str) and txt.strip():
            yield txt.strip()


def pack_tokens_to_blocks(
    text_iter: Iterator[str], tokenizer: Tokenizer, seq_len: int, max_docs: int | None = None
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    buf: list[int] = []
    n_docs = 0

    for doc in text_iter:
        ids = tokenizer.encode(doc).ids
        if len(ids) == 0:
            continue
        buf.extend(ids)

        while len(buf) >= seq_len + 1:
            x = np.asarray(buf[:seq_len], dtype=np.int32)
            y = np.asarray(buf[1 : seq_len + 1], dtype=np.int32)
            yield x, y
            buf.clear()

        n_docs += 1
        if max_docs is not None and n_docs >= max_docs:
            break


def make_batch(
    blocks: Iterator[tuple[np.ndarray, np.ndarray]], batch_size: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for x, y in blocks:
        xs.append(x)
        ys.append(y)
        if len(xs) == batch_size:
            yield np.stack(xs, axis=0), np.stack(ys, axis=0)
            xs.clear()
            ys.clear()
