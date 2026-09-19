from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter

from .io import chip_paths, read_raster


def af_features(data_dir: Path, chip_id: str) -> tuple[np.ndarray, np.ndarray]:
    paths = chip_paths(data_dir, chip_id, "af")
    viirs, aux = read_raster(paths["main"]), read_raster(paths["aux"])
    i4, i5 = viirs[3], viirs[4]
    local_i4 = uniform_filter(np.nan_to_num(i4, nan=0), size=9, mode="nearest")
    local_diff = uniform_filter(np.nan_to_num(i4 - i5, nan=0), size=9, mode="nearest")
    stack = np.concatenate([viirs[:5], aux, (i4 - i5)[None], (i4 - local_i4)[None], (i4 - i5 - local_diff)[None]])
    valid = (viirs[7] > 0) & np.isfinite(stack).all(axis=0)
    return stack.reshape(stack.shape[0], -1).T.astype(np.float32), valid.reshape(-1)


def bs_features(data_dir: Path, chip_id: str) -> tuple[np.ndarray, np.ndarray]:
    paths = chip_paths(data_dir, chip_id, "bs")
    pre, post = read_raster(paths["pre"]).astype(np.float32), read_raster(paths["post"]).astype(np.float32)
    s1pre, s1post, aux = read_raster(paths["s1pre"]).astype(np.float32), read_raster(paths["s1post"]).astype(np.float32), read_raster(paths["aux"]).astype(np.float32)
    nbr_pre = (pre[6] - pre[8]) / np.maximum(pre[6] + pre[8], 1.0)
    nbr_post = (post[6] - post[8]) / np.maximum(post[6] + post[8], 1.0)
    dnbr = nbr_pre - nbr_post
    rdnbr = dnbr / np.sqrt(np.maximum(np.abs(nbr_pre), 1e-3))
    stack = np.concatenate([
        pre[:9] / 10000.0, post[:9] / 10000.0, (post[:9] - pre[:9]) / 10000.0,
        pre[9:10], post[9:10], s1pre / 100.0, s1post / 100.0, (s1post - s1pre) / 100.0,
        aux, nbr_pre[None], nbr_post[None], dnbr[None], rdnbr[None],
    ])
    valid = np.isfinite(stack).all(axis=0)
    return stack.reshape(stack.shape[0], -1).T.astype(np.float32), valid.reshape(-1)
