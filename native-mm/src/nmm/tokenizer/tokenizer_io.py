from dataclasses import dataclass

from tokenizers import Tokenizer


@dataclass(frozen=True)
class SpecialTokenIds:
    pad: int
    bos: int
    eos: int
    unk: int
    user: int
    assistant: int
    system: int
    img: int
    text: int


@dataclass(frozen=True)
class SpecialTokens:
    pad: str = "<|pad|>"
    bos: str = "<|bos|>"
    eos: str = "<|eos|>"
    unk: str = "<|unk|>"
    user: str = "<|user|>"
    assistant: str = "<|assistant|>"
    system: str = "<|system|>"
    img: str = "<|img|>"
    text: str = "<|text|>"

    def all(self) -> list[str]:
        return [self.pad, self.bos, self.eos, self.unk, self.user, self.assistant, self.system, self.img, self.text]


def load_tokenizer(tokenizer_json_path: str) -> tuple[Tokenizer, SpecialTokenIds]:
    tok = Tokenizer.from_file(tokenizer_json_path)

    def tid(s: str) -> int:
        t = tok.token_to_id(s)
        if t is None:
            raise ValueError(f"Special token '{s}' not found")
        return int(t)

    st = SpecialTokenIds(
        pad=tid("<|pad|>"),
        bos=tid("<|bos|>"),
        eos=tid("<|eos|>"),
        unk=tid("<|unk|>"),
        user=tid("<|user|>"),
        assistant=tid("<|assistant|>"),
        system=tid("<|system|>"),
        img=tid("<|img|>"),
        text=tid("<|text|>"),
    )

    return tok, st
