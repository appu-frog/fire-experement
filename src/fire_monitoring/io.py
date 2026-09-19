from __future__ import annotations

from pathlib import Path
import numpy as np
import rasterio


def read_raster(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read()


def chip_paths(data_dir: Path, chip_id: str, kind: str) -> dict[str, Path]:
    """Return data paths using only the published folder convention."""
    prefix = "AF" if kind == "af" else "BS"
    suffix = chip_id.replace(prefix + "_te", prefix + "_tr") if "train" in str(data_dir) else chip_id
    if kind == "af":
        return {
            "main": data_dir / "af" / "viirs" / f"{suffix}_VIIRS_I1-I5.tif",
            "aux": data_dir / "af" / "_aux" / f"{suffix}_AUX.tif",
        }
    return {
        "pre": data_dir / "bs" / "sentinel2_pre" / f"{suffix}_Sentinel-2_pre.tif",
        "post": data_dir / "bs" / "sentinel2_post" / f"{suffix}_Sentinel-2_post.tif",
        "s1pre": data_dir / "bs" / "sentinel1_pre" / f"{suffix}_Sentinel-1_pre.tif",
        "s1post": data_dir / "bs" / "sentinel1_post" / f"{suffix}_Sentinel-1_post.tif",
        "aux": data_dir / "bs" / "_aux" / f"{suffix}_AUX.tif",
    }
