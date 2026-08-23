from pathlib import Path

import numpy as np
import pytest

from data.prepare import (
    encode_corpus,
    prepare_dataset,
    read_corpus,
    save_token_ids,
    split_token_ids,
    to_uint16_array,
)
from tokenizer import CharTokenizer


def test_read_corpus_returns_utf8_text(tmp_path: Path) -> None:
    path = tmp_path / "corpus.txt"
    path.write_text("Hello, Juliet! ✨\n", encoding="utf-8")

    assert read_corpus(path) == "Hello, Juliet! ✨\n"


def test_read_corpus_accepts_string_path(tmp_path: Path) -> None:
    path = tmp_path / "corpus.txt"
    path.write_text("hello", encoding="utf-8")

    assert read_corpus(str(path)) == "hello"


def test_read_corpus_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="corpus is empty"):
        read_corpus(path)


def test_read_corpus_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_corpus(tmp_path / "missing.txt")


def test_read_corpus_rejects_non_utf8_file(tmp_path: Path) -> None:
    path = tmp_path / "corpus.txt"
    path.write_bytes(b"\xff")

    with pytest.raises(UnicodeDecodeError):
        read_corpus(path)


def test_encode_corpus_converts_each_character_to_its_id() -> None:
    text = "cabca"
    tokenizer = CharTokenizer.from_text(text)

    # Vocabulary order is a, b, c.
    assert encode_corpus(text, tokenizer) == [2, 0, 1, 2, 0]


def test_encode_corpus_can_be_decoded_back_to_the_original_text() -> None:
    text = "To be, or not to be\n"
    tokenizer = CharTokenizer.from_text(text)

    token_ids = encode_corpus(text, tokenizer)

    assert tokenizer.decode(token_ids) == text


def test_encode_corpus_rejects_empty_text() -> None:
    tokenizer = CharTokenizer.from_text("abc")

    with pytest.raises(ValueError, match="empty corpus"):
        encode_corpus("", tokenizer)


def test_split_token_ids_uses_90_10_by_default() -> None:
    token_ids = list(range(100))

    train_ids, validation_ids = split_token_ids(token_ids)

    assert train_ids == list(range(90))
    assert validation_ids == list(range(90, 100))


def test_split_token_ids_preserves_every_token_and_order() -> None:
    token_ids = list(range(11))

    train_ids, validation_ids = split_token_ids(token_ids, train_fraction=0.8)

    assert train_ids + validation_ids == token_ids
    assert len(train_ids) == 8
    assert len(validation_ids) == 3


@pytest.mark.parametrize("train_fraction", [0, 1, -0.1, 1.1])
def test_split_token_ids_rejects_invalid_fraction(train_fraction: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        split_token_ids(list(range(10)), train_fraction)


@pytest.mark.parametrize("token_ids", [[], [42]])
def test_split_token_ids_requires_enough_data(token_ids: list[int]) -> None:
    with pytest.raises(ValueError, match="not enough token IDs"):
        split_token_ids(token_ids)


def test_to_uint16_array_preserves_ids_with_compact_dtype() -> None:
    token_ids = [0, 2, 1, 65_535]

    array = to_uint16_array(token_ids)

    assert array.dtype == np.uint16
    assert array.tolist() == token_ids
    assert array.nbytes == len(token_ids) * 2


@pytest.mark.parametrize("token_ids", [[], [-1], [65_536]])
def test_to_uint16_array_rejects_values_that_cannot_be_stored(token_ids: list[int]) -> None:
    with pytest.raises(ValueError):
        to_uint16_array(token_ids)


def test_save_token_ids_writes_raw_ids_that_can_be_read_back(tmp_path: Path) -> None:
    token_ids = np.array([2, 0, 1, 2, 0], dtype=np.uint16)
    output_path = tmp_path / "prepared" / "train.bin"

    save_token_ids(token_ids, output_path)

    saved_ids = np.fromfile(output_path, dtype=np.uint16)
    assert np.array_equal(saved_ids, token_ids)
    assert output_path.stat().st_size == token_ids.nbytes


def test_save_token_ids_accepts_string_path(tmp_path: Path) -> None:
    output_path = tmp_path / "val.bin"

    save_token_ids(np.array([1, 2], dtype=np.uint16), str(output_path))

    assert output_path.exists()


def test_save_token_ids_rejects_wrong_dtype(tmp_path: Path) -> None:
    token_ids = np.array([1, 2], dtype=np.int64)

    with pytest.raises(ValueError, match="dtype uint16"):
        save_token_ids(token_ids, tmp_path / "train.bin")  # type: ignore[arg-type]


def test_save_token_ids_rejects_non_vector_array(tmp_path: Path) -> None:
    token_ids = np.array([[1, 2]], dtype=np.uint16)

    with pytest.raises(ValueError, match="one-dimensional"):
        save_token_ids(token_ids, tmp_path / "train.bin")


def test_save_token_ids_rejects_empty_array(tmp_path: Path) -> None:
    token_ids = np.array([], dtype=np.uint16)

    with pytest.raises(ValueError, match="empty token array"):
        save_token_ids(token_ids, tmp_path / "train.bin")


def test_prepare_dataset_produces_vocab_train_and_validation_files(tmp_path: Path) -> None:
    text = "abcdefghij"
    input_path = tmp_path / "input.txt"
    output_dir = tmp_path / "prepared"
    input_path.write_text(text, encoding="utf-8")

    prepare_dataset(input_path, output_dir)

    tokenizer = CharTokenizer.load(output_dir / "vocab.json")
    train_ids = np.fromfile(output_dir / "train.bin", dtype=np.uint16)
    validation_ids = np.fromfile(output_dir / "val.bin", dtype=np.uint16)

    assert tokenizer.decode(train_ids.tolist()) == "abcdefghi"
    assert tokenizer.decode(validation_ids.tolist()) == "j"
    assert np.concatenate((train_ids, validation_ids)).tolist() == tokenizer.encode(text)
