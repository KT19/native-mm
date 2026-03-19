from collections.abc import Iterator

from datasets import load_dataset


def ultrachat_sft(
    split: str = "train_sft",
    seed: int = 0,
) -> Iterator[list[dict[str, str]]]:
    """
    Below 2GB. No streaming.
    """

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)
    ds = ds.shuffle(seed=seed)

    for ex in ds:
        msgs = ex.get("messages", None)  # type: ignore

        if isinstance(msgs, list) and len(msgs) >= 2:
            yield msgs
