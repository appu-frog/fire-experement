from __future__ import annotations

import numpy as np


def encode(mask: np.ndarray) -> str:
    """Encode a 2D binary mask in the competition row-major 1-indexed RLE."""
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
    padded = np.pad(flat, (1, 1))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    if changes.size == 0:
        return ""
    starts = changes[::2] + 1
    lengths = changes[1::2] - changes[::2]
    return " ".join(f"{start} {length}" for start, length in zip(starts, lengths))


def decode(rle: str, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    if not rle:
        return mask.reshape(shape)
    values = np.fromstring(rle, sep=" ", dtype=np.int64)
    if values.size % 2:
        raise ValueError("RLE must contain start-length pairs")
    for start, length in values.reshape(-1, 2):
        if start < 1 or length < 1 or start - 1 + length > mask.size:
            raise ValueError("RLE run is outside mask bounds")
        mask[start - 1 : start - 1 + length] = 1
    return mask.reshape(shape)
