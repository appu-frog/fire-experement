from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fire_monitoring.rle import decode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    args = parser.parse_args()
    sub = pd.read_csv(args.submission, keep_default_na=False)
    template = pd.read_csv(args.template, keep_default_na=False)
    meta = pd.read_csv(args.meta)
    expected = set(map(tuple, template[["chip_id", "class_id"]].to_numpy()))
    actual = set(map(tuple, sub[["chip_id", "class_id"]].to_numpy()))
    if len(sub) != len(template) or expected != actual or sub.rle.isna().any():
        raise ValueError("Rows or (chip_id, class_id) pairs differ from template")
    dimensions = meta.set_index("chip_id")[["height", "width"]].to_dict("index")
    by_chip: dict[str, list] = {}
    for row in sub.itertuples(index=False):
        size = dimensions[row.chip_id]
        mask = decode(row.rle, (int(size["height"]), int(size["width"])))
        by_chip.setdefault(row.chip_id, []).append(mask)
    for chip_id, masks in by_chip.items():
        if len(masks) > 1 and sum(masks).max() > 1:
            raise ValueError(f"Overlapping classes in {chip_id}")
    print(f"Valid submission: {len(sub)} rows")


if __name__ == "__main__":
    main()
