"""Character-level tokenizer."""

from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    """Map characters to integer IDs and back."""

    def __init__(self, chars: list[str]) -> None:
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        """Build a sorted vocabulary from text."""
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[ch] for ch in text]
        except KeyError as e:
            bad = e.args[0]
            raise ValueError(
                f"character {bad!r} (U+{ord(bad):04X}) is not in the vocabulary. "
                f"A char tokenizer can only encode the {self.vocab_size} characters it was "
                f"built from -- this usually means a sampling prompt used a character the "
                f"corpus never contained."
            ) from None

    def decode(self, ids: list[int]) -> str:
        try:
            return "".join(self.itos[i] for i in ids)
        except KeyError as e:
            raise ValueError(
                f"id {e.args[0]} is out of range for a vocabulary of size {self.vocab_size}"
            ) from None

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"chars": self.chars}, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> CharTokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["chars"])


if __name__ == "__main__":
    text = Path("data/input.txt").read_text(encoding="utf-8")
    tok = CharTokenizer.from_text(text)

    print(f"corpus     {len(text):,} characters")
    print(f"vocab_size {tok.vocab_size}")
    print(f"vocab      {''.join(tok.chars)!r}")

    sample = "Hello there, sweet Juliet!"
    print(f"\nencode({sample!r})\n  -> {tok.encode(sample)}")
    print(f"decode(...) -> {tok.decode(tok.encode(sample))!r}")

    assert tok.decode(tok.encode(text)) == text, "roundtrip failed on the full corpus"
    print(f"\nroundtrip OK on all {len(text):,} characters")

    tok.save("data/vocab.json")
    print("wrote data/vocab.json")
