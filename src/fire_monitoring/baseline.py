from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter

from .io import chip_paths, read_raster


def _local_mean(a: np.ndarray, size: int = 9) -> np.ndarray:
    return uniform_filter(a.astype(np.float32), size=size, mode="nearest")


def predict_af(data_dir: Path, chip_id: str) -> np.ndarray:
    """Conservative context-based active-fire baseline.

    I4 is the MIR fire channel. A candidate must be warm relative to its
    neighborhood and have a large I4-I5 contrast. Invalid pixels are excluded.
    Thresholds are deliberately conservative because AF F1 is FP-sensitive.
    """
    paths = chip_paths(data_dir, chip_id, "af")
    viirs, aux = read_raster(paths["main"]), read_raster(paths["aux"])
    i4, i5, valid = viirs[3], viirs[4], viirs[7] > 0
    contrast = i4 - i5
    local_i4 = _local_mean(np.nan_to_num(i4, nan=0.0))
    landcover = aux[0]
    thermal = (i4 > 315.0) & (contrast > 12.0) & ((i4 - local_i4) > 5.0)
    # Water and dense built-up surroundings often produce unstable reflection;
    # retain only very strong candidates there.
    strict_surface = np.isin(landcover, [50, 80])
    thermal &= (~strict_surface) | ((i4 > 330.0) & (contrast > 20.0))
    return (thermal & valid & np.isfinite(i4) & np.isfinite(i5)).astype(np.uint8)


def _landcover_threshold(landcover: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initial dNBR thresholds: forest is stricter than grass/cropland."""
    low = np.full(landcover.shape, 0.10, np.float32)
    mid = np.full(landcover.shape, 0.22, np.float32)
    high = np.full(landcover.shape, 0.38, np.float32)
    forest = np.isin(landcover, [10, 20, 30, 90, 95])
    low[forest], mid[forest], high[forest] = 0.10, 0.27, 0.44
    return low, mid, high


def predict_bs(data_dir: Path, chip_id: str) -> np.ndarray:
    """Land-cover-aware dNBR baseline with SCL quality mask and SAR support."""
    paths = chip_paths(data_dir, chip_id, "bs")
    pre, post = read_raster(paths["pre"]).astype(np.float32), read_raster(paths["post"]).astype(np.float32)
    s1pre, s1post, aux = read_raster(paths["s1pre"]), read_raster(paths["s1post"]), read_raster(paths["aux"])
    nbr_pre = (pre[6] - pre[8]) / np.maximum(pre[6] + pre[8], 1.0)
    nbr_post = (post[6] - post[8]) / np.maximum(post[6] + post[8], 1.0)
    dnbr = nbr_pre - nbr_post
    low, mid, high = _landcover_threshold(aux[2])
    result = np.zeros(dnbr.shape, dtype=np.uint8)
    result[dnbr >= low] = 1
    result[dnbr >= mid] = 2
    result[dnbr >= high] = 3
    # SCL cloud/shadow/no-data: lower confidence rather than infer severity from optics.
    bad_scl = np.isin(pre[9], [0, 1, 3, 8, 9, 10, 11]) | np.isin(post[9], [0, 1, 3, 8, 9, 10, 11])
    sar_change = np.abs((s1post[0] - s1pre[0]).astype(np.float32)) + np.abs((s1post[1] - s1pre[1]).astype(np.float32))
    result[bad_scl & (sar_change < 150.0)] = 0
    return result
