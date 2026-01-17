import random
from collections.abc import Iterator

from datasets import load_dataset
from PIL import Image

Sample = tuple[Image.Image, list[dict[str, str]]]


def _parse_conversation(turns: list) -> list[dict[str, str]] | None:
    """
    Parse conversation.
    """
    if not isinstance(turns, list) or len(turns) < 2:
        return None

    msgs: list[dict[str, str]] = []
    for t in turns:
        if not isinstance(t, dict):
            continue

        # Handle both conversation formats
        role = t.get("from") or t.get("role")
        content = t.get("value") or t.get("content")

        if not isinstance(role, str) or content is None:
            continue

        # Normalize role names
        if role in ("human", "user"):
            role = "user"
        elif role in ("gpt", "assistant"):
            role = "assistant"
        else:
            raise ValueError(f"Unexpected role '{role}'")

        # Clean content
        content = content.removeprefix("<image>\n")
        msgs.append({"role": role, "content": content})

    return msgs if len(msgs) >= 2 else None


def _single_subset(subset: str, split: str, seed: int) -> Iterator[Sample]:
    """Samples from a single subset."""
    ds = load_dataset(
        "mvp-lab/LLaVA-OneVision-1.5-Instruct-Data",
        subset,
        split=split,
    )
    ds = ds.shuffle(seed=seed)

    for ex in ds:
        image = ex.get("image")  # type: ignore
        turns = ex.get("conversations")  # type: ignore

        if not isinstance(image, Image.Image):
            continue

        msgs = _parse_conversation(turns)  # type: ignore
        if msgs is not None:
            yield image, msgs


def llava_instruct(
    subsets: list[str] = ["visual_chat", "coco"],
    split: str = "train",
    seed: int = 0,
) -> Iterator[Sample]:
    """
    Multimodal instruction data from multiple subsets.
    """
    # Create iterators for each subset
    iterators = [_single_subset(s, split, seed + i) for i, s in enumerate(subsets)]

    # Round-robin interleaving
    active = list(range(len(iterators)))
    rng = random.Random(seed)

    while active:
        rng.shuffle(active)
        exhausted = []
        for idx in active:
            try:
                yield next(iterators[idx])
            except StopIteration:
                exhausted.append(idx)

        for idx in exhausted:
            active.remove(idx)
