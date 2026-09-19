"""Train compact tree models from public train chips.

The script holds out complete fire events, reports validation metrics, then fits
production models on all labelled chips. It never uses target-derived metadata.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from joblib import dump
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit

from src.fire_monitoring.features import af_features, bs_features
from src.fire_monitoring.metrics import f1_binary, iou


def read_target(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).reshape(-1)


def collect(data_dir: Path, kind: str, per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    meta = pd.read_csv(data_dir / kind / "meta.csv")
    xs, ys, groups = [], [], []
    feature_fn = af_features if kind == "af" else bs_features
    for row in meta.itertuples(index=False):
        x, valid = feature_fn(data_dir, row.chip_id)
        y = read_target(data_dir / kind / "masks" / f"{row.chip_id}_MASK.tif")
        valid &= y != 255
        picks = []
        for cls in np.unique(y[valid]):
            indices = np.flatnonzero(valid & (y == cls))
            picks.append(rng.choice(indices, size=min(len(indices), per_class), replace=False))
        if not picks:
            continue
        indices = np.concatenate(picks)
        group = row.fire_event_id if pd.notna(row.fire_event_id) else f"unlinked_{kind}_{row.chip_id}"
        xs.append(x[indices]); ys.append(y[indices]); groups.extend([group] * len(indices))
    return np.concatenate(xs), np.concatenate(ys), np.asarray(groups)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    # Sampling keeps the experiment runnable on a laptop; full rasters are
    # still used during inference and metric calculation.
    for kind, max_per_class in (("af", 250), ("bs", 180)):
        x, y, groups = collect(args.data_dir, kind, max_per_class, args.seed)
        train_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=args.seed).split(x, y, groups))
        model = HistGradientBoostingClassifier(max_iter=80, learning_rate=.10, max_leaf_nodes=24, l2_regularization=1.0, random_state=args.seed)
        model.fit(x[train_idx], y[train_idx])
        pred = model.predict(x[val_idx])
        if kind == "af":
            results["af_f1_event_holdout"] = f1_binary(y[val_idx], pred)
        else:
            results["bs_iou_burn_event_holdout"] = iou(y[val_idx] > 0, pred > 0, 1)
            results["bs_miou_severity_event_holdout"] = float(np.mean([iou(y[val_idx], pred, c) for c in (1, 2, 3)]))
        final = HistGradientBoostingClassifier(max_iter=80, learning_rate=.10, max_leaf_nodes=24, l2_regularization=1.0, random_state=args.seed)
        final.fit(x, y)
        dump(final, args.models_dir / f"{kind}_histgb.joblib")
    results["estimated_score"] = .35 * results["af_f1_event_holdout"] + .35 * results["bs_iou_burn_event_holdout"] + .30 * results["bs_miou_severity_event_holdout"]
    (args.models_dir / "validation_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
