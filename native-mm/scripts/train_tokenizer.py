import os
from collections.abc import Iterable, Iterator

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from nmm.tokenizer.tokenizer_io import SpecialTokens


def iter_text_samples(
    dataset_name: str, name: str, split: str, text_field: str, streaming: bool = True, max_samples: int | None = None
) -> Iterator[str]:
    ds = load_dataset(dataset_name, name=name, split=split, streaming=streaming)
    n = 0
    for ex in ds:
        txt = ex.get(text_field, None)  # type:ignore
        if not isinstance(txt, str):
            continue
        txt = txt.strip()
        if not txt:
            continue
        yield txt
        n += 1
        if max_samples is not None and n >= max_samples:
            break


def train_bpe_tokenizer(
    iterator: Iterable[str], vocab_size: int, special_tokens: SpecialTokens, min_frequency: int = 2
) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token=special_tokens.unk))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens.all(),
    )

    tokenizer.train_from_iterator(iterator, trainer=trainer)

    return tokenizer


def main() -> None:
    st = SpecialTokens()

    dataset_name = "HuggingFaceFW/fineweb-edu"
    name = "CC-MAIN-2025-05"
    split = "train"
    text_field = "text"

    it = iter_text_samples(
        dataset_name=dataset_name,
        name=name,
        split=split,
        text_field=text_field,
        streaming=True,
        max_samples=100000,
    )

    tok = train_bpe_tokenizer(iterator=it, vocab_size=32000, special_tokens=st, min_frequency=2)
    tokenizer_dir = "saved_tokenizer"
    os.makedirs(tokenizer_dir, exist_ok=True)
    tok.save(tokenizer_dir + "/tokenizer.json")
    for s in st.all():
        print(s, tok.token_to_id(s))


if __name__ == "__main__":
    main()
