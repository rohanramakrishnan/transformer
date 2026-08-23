"""Read prepared token datasets for model training."""

from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from tokenizer import CharTokenizer


def load_token_ids(path: str | Path) -> np.memmap:
    """Open a uint16 token file as a read-only memory map."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"token file does not exist: {dataset_path}")
    if not dataset_path.is_file():
        raise IsADirectoryError(f"token path is not a file: {dataset_path}")

    byte_count = dataset_path.stat().st_size
    bytes_per_token = np.dtype(np.uint16).itemsize
    if byte_count == 0:
        raise ValueError(f"token file is empty: {dataset_path}")
    if byte_count % bytes_per_token != 0:
        raise ValueError(
            f"token file size must be divisible by {bytes_per_token} bytes: {dataset_path}"
        )

    return np.memmap(dataset_path, dtype=np.uint16, mode="r")


def get_batch(
    data: NDArray[np.uint16],
    batch_size: int,
    block_size: int,
    device: str | torch.device,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample input sequences and their next-token targets."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if len(data) <= block_size:
        raise ValueError("dataset must contain more tokens than block_size")

    offsets = torch.randint(
        0,
        len(data) - block_size,
        (batch_size,),
        generator=generator,
    ).tolist()

    x_array = np.stack([data[i : i + block_size] for i in offsets]).astype(np.int64)
    y_array = np.stack([data[i + 1 : i + block_size + 1] for i in offsets]).astype(np.int64)

    x = torch.from_numpy(x_array).to(device)
    y = torch.from_numpy(y_array).to(device)
    return x, y


if __name__ == "__main__":
    train_data = load_token_ids("data/train.bin")
    tokenizer = CharTokenizer.load("data/vocab.json")
    generator = torch.Generator().manual_seed(42)

    x, y = get_batch(
        train_data,
        batch_size=4,
        block_size=64,
        device="cpu",
        generator=generator,
    )

    print(f"batch shape: {tuple(x.shape)}")
    print(f"x: {tokenizer.decode(x[0].tolist())!r}")
    print(f"y: {tokenizer.decode(y[0].tolist())!r}")
