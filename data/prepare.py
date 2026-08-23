"""Prepare text data for training."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tokenizer import CharTokenizer


def read_corpus(path: str | Path) -> str:
    """Read a non-empty UTF-8 text file."""
    corpus_path = Path(path)
    text = corpus_path.read_text(encoding="utf-8")

    if not text:
        raise ValueError(f"corpus is empty: {corpus_path}")

    return text


def encode_corpus(text: str, tokenizer: CharTokenizer) -> list[int]:
    """Encode a corpus with the given tokenizer."""
    if not text:
        raise ValueError("cannot encode an empty corpus")

    return tokenizer.encode(text)


def split_token_ids(
    token_ids: list[int], train_fraction: float = 0.9
) -> tuple[list[int], list[int]]:
    """Split token IDs into contiguous train and validation sets."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")

    split_index = int(len(token_ids) * train_fraction)
    if split_index == 0 or split_index == len(token_ids):
        raise ValueError("not enough token IDs to create non-empty train and validation sets")

    return token_ids[:split_index], token_ids[split_index:]


def to_uint16_array(token_ids: list[int]) -> NDArray[np.uint16]:
    """Convert token IDs to a uint16 array."""
    if not token_ids:
        raise ValueError("cannot convert an empty token ID list")
    if any(token_id < 0 or token_id > np.iinfo(np.uint16).max for token_id in token_ids):
        raise ValueError("token IDs must be between 0 and 65535 for uint16 storage")

    return np.asarray(token_ids, dtype=np.uint16)


def save_token_ids(token_ids: NDArray[np.uint16], path: str | Path) -> None:
    """Write a one-dimensional uint16 token array as raw binary data."""
    if token_ids.dtype != np.uint16:
        raise ValueError("token array must have dtype uint16")
    if token_ids.ndim != 1:
        raise ValueError("token array must be one-dimensional")
    if token_ids.size == 0:
        raise ValueError("cannot save an empty token array")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    token_ids.tofile(output_path)


def prepare_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    train_fraction: float = 0.9,
) -> None:
    """Write the tokenizer and train/validation files for a corpus."""
    text = read_corpus(input_path)
    tokenizer = CharTokenizer.from_text(text)
    token_ids = encode_corpus(text, tokenizer)
    train_ids, validation_ids = split_token_ids(token_ids, train_fraction)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    tokenizer.save(destination / "vocab.json")
    save_token_ids(to_uint16_array(train_ids), destination / "train.bin")
    save_token_ids(to_uint16_array(validation_ids), destination / "val.bin")


if __name__ == "__main__":
    prepare_dataset("data/input.txt", "data")
    print("wrote data/vocab.json, data/train.bin, and data/val.bin")
