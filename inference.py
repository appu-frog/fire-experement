from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
from joblib import load

from src.fire_monitoring.baseline import predict_af, predict_bs
from src.fire_monitoring.features import af_features, bs_features
from src.fire_monitoring.rle import encode


def main() -> None:
    parser = argparse.ArgumentParser(description="Fire-monitoring baseline inference")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing test/ data")
    parser.add_argument("--output", type=Path, required=True, help="Output submission.csv")
    parser.add_argument("--models-dir", type=Path, default=None, help="Optional directory from train_models.py")
    args = parser.parse_args()
    template_path = args.data_dir / "sample_submission.csv"
    if not template_path.exists():
        raise FileNotFoundError(f"Missing template: {template_path}")
    template = pd.read_csv(template_path, keep_default_na=False)
    models = {}
    if args.models_dir:
        for kind in ("af", "bs"):
            path = args.models_dir / f"{kind}_histgb.joblib"
            if path.exists():
                models[kind] = load(path)
    settings = {}
    if args.models_dir and (args.models_dir / "settings.json").exists():
        settings = json.loads((args.models_dir / "settings.json").read_text(encoding="utf-8"))
    masks: dict[str, object] = {}
    for chip_id in template.chip_id.unique():
        if chip_id.startswith("AF_"):
            if "af" in models:
                x, valid = af_features(args.data_dir, chip_id)
                threshold = settings.get("af_probability_threshold", .5)
                result = (models["af"].predict_proba(x)[:, 1] >= threshold).reshape(256, 256).astype("uint8")
                masks[chip_id] = result * valid.reshape(256, 256)
            else:
                masks[chip_id] = predict_af(args.data_dir, chip_id)
        else:
            if "bs" in models:
                x, valid = bs_features(args.data_dir, chip_id)
                result = models["bs"].predict(x).reshape(512, 512).astype("uint8")
                masks[chip_id] = result * valid.reshape(512, 512)
            else:
                masks[chip_id] = predict_bs(args.data_dir, chip_id)
    rows = []
    for row in template.itertuples(index=False):
        mask = masks[row.chip_id]
        binary = (mask == int(row.class_id)).astype("uint8")
        rows.append({"chip_id": row.chip_id, "class_id": int(row.class_id), "rle": encode(binary)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
