from pathlib import Path

import numpy as np
import pytest
import torch

from dataset import get_batch, load_token_ids


def test_load_token_ids_returns_memory_mapped_values(tmp_path: Path) -> None:
    path = tmp_path / "train.bin"
    expected = np.array([4, 1, 8, 2], dtype=np.uint16)
    expected.tofile(path)

    token_ids = load_token_ids(path)

    assert isinstance(token_ids, np.memmap)
    assert token_ids.dtype == np.uint16
    assert token_ids.tolist() == expected.tolist()


def test_load_token_ids_accepts_string_path(tmp_path: Path) -> None:
    path = tmp_path / "val.bin"
    np.array([1, 2], dtype=np.uint16).tofile(path)

    assert load_token_ids(str(path)).tolist() == [1, 2]


def test_load_token_ids_is_read_only(tmp_path: Path) -> None:
    path = tmp_path / "train.bin"
    np.array([1, 2], dtype=np.uint16).tofile(path)
    token_ids = load_token_ids(path)

    assert not token_ids.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        token_ids[0] = 9


def test_load_token_ids_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_token_ids(tmp_path / "missing.bin")


def test_load_token_ids_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(IsADirectoryError, match="not a file"):
        load_token_ids(tmp_path)


def test_load_token_ids_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.bin"
    path.touch()

    with pytest.raises(ValueError, match="empty"):
        load_token_ids(path)


def test_load_token_ids_rejects_incomplete_uint16(tmp_path: Path) -> None:
    path = tmp_path / "malformed.bin"
    path.write_bytes(b"\x01\x00\x02")

    with pytest.raises(ValueError, match="divisible by 2 bytes"):
        load_token_ids(path)


def test_get_batch_returns_expected_shapes_and_types() -> None:
    data = np.arange(100, dtype=np.uint16)

    x, y = get_batch(data, batch_size=4, block_size=8, device="cpu")

    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert x.dtype == torch.long
    assert y.dtype == torch.long
    assert x.device.type == "cpu"
    assert y.device.type == "cpu"


def test_get_batch_targets_are_inputs_shifted_by_one() -> None:
    data = np.arange(100, dtype=np.uint16)

    x, y = get_batch(data, batch_size=16, block_size=8, device="cpu")

    assert torch.equal(x[:, 1:], y[:, :-1])
    assert torch.equal(y, x + 1)


def test_get_batch_can_be_reproduced_with_seeded_generator() -> None:
    data = np.arange(100, dtype=np.uint16)

    first = get_batch(
        data,
        batch_size=4,
        block_size=8,
        device="cpu",
        generator=torch.Generator().manual_seed(42),
    )
    second = get_batch(
        data,
        batch_size=4,
        block_size=8,
        device="cpu",
        generator=torch.Generator().manual_seed(42),
    )

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


@pytest.mark.parametrize(
    ("batch_size", "block_size", "message"),
    [(0, 8, "batch_size"), (4, 0, "block_size"), (4, -1, "block_size")],
)
def test_get_batch_rejects_invalid_sizes(
    batch_size: int, block_size: int, message: str
) -> None:
    data = np.arange(100, dtype=np.uint16)

    with pytest.raises(ValueError, match=message):
        get_batch(data, batch_size, block_size, "cpu")


@pytest.mark.parametrize("data_size", [7, 8])
def test_get_batch_requires_room_for_target_token(data_size: int) -> None:
    data = np.arange(data_size, dtype=np.uint16)

    with pytest.raises(ValueError, match="more tokens"):
        get_batch(data, batch_size=1, block_size=8, device="cpu")
