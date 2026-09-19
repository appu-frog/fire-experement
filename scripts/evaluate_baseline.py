from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fire_monitoring.baseline import predict_af, predict_bs
from src.fire_monitoring.metrics import f1_binary, iou


def read_mask(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing train/")
    parser.add_argument("--output", type=Path, default=Path("artifacts/baseline_metrics.json"))
    args = parser.parse_args()
    af_meta, bs_meta = (pd.read_csv(args.data_dir / k / "meta.csv") for k in ("af", "bs"))
    af_true, af_pred, bs_true, bs_pred = [], [], [], []
    for row in af_meta.itertuples():
        truth = read_mask(args.data_dir / "af" / "masks" / f"{row.chip_id}_MASK.tif")
        valid = truth != 255
        af_true.append(truth[valid]); af_pred.append(predict_af(args.data_dir, row.chip_id)[valid])
    for row in bs_meta.itertuples():
        truth = read_mask(args.data_dir / "bs" / "masks" / f"{row.chip_id}_MASK.tif")
        valid = truth != 255
        bs_true.append(truth[valid]); bs_pred.append(predict_bs(args.data_dir, row.chip_id)[valid])
    af_true, af_pred = np.concatenate(af_true), np.concatenate(af_pred)
    bs_true, bs_pred = np.concatenate(bs_true), np.concatenate(bs_pred)
    result = {
        "f1_af": f1_binary(af_true, af_pred),
        "iou_burn": iou(bs_true > 0, bs_pred > 0, 1),
        "miou_sev": float(np.mean([iou(bs_true, bs_pred, c) for c in (1, 2, 3)])),
    }
    result["score"] = .35 * result["f1_af"] + .35 * result["iou_burn"] + .30 * result["miou_sev"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import json
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
