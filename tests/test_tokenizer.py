import random
from pathlib import Path

import pytest

from tokenizer import CharTokenizer

CORPUS = Path("data/input.txt")


@pytest.fixture(scope="module")
def text() -> str:
    if not CORPUS.exists():
        pytest.skip(f"{CORPUS} not found")
    return CORPUS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tok(text: str) -> CharTokenizer:
    return CharTokenizer.from_text(text)


def test_vocab_is_sorted_unique_chars(tok, text):
    assert tok.chars == sorted(set(text))
    assert tok.vocab_size == 65, "TinyShakespeare has 65 distinct characters"


def test_roundtrip_full_corpus(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_roundtrip_random_slices(tok, text):
    rng = random.Random(1337)
    for _ in range(200):
        i = rng.randrange(0, len(text) - 512)
        chunk = text[i : i + rng.randrange(1, 512)]
        assert tok.decode(tok.encode(chunk)) == chunk


def test_ids_are_in_range(tok, text):
    ids = tok.encode(text[:10_000])
    assert all(0 <= i < tok.vocab_size for i in ids)


def test_encode_is_one_token_per_character(tok):
    s = "To be, or not to be"
    assert len(tok.encode(s)) == len(s)


def test_stoi_and_itos_are_inverses(tok):
    for ch, i in tok.stoi.items():
        assert tok.itos[i] == ch


def test_unknown_character_raises_helpful_error(tok):
    with pytest.raises(ValueError, match="not in the vocabulary"):
        tok.encode("naïve")  # 'ï' never appears in TinyShakespeare


def test_out_of_range_id_raises(tok):
    with pytest.raises(ValueError, match="out of range"):
        tok.decode([0, tok.vocab_size])


def test_save_load_roundtrip(tok, tmp_path, text):
    p = tmp_path / "vocab.json"
    tok.save(p)
    loaded = CharTokenizer.load(p)
    assert loaded.chars == tok.chars
    assert loaded.encode(text[:5000]) == tok.encode(text[:5000])


def test_determinism(text):
    assert CharTokenizer.from_text(text).stoi == CharTokenizer.from_text(text).stoi
