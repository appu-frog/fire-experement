"""Write data facts used by the modelling and report decisions."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import rasterio


def mask_counts(paths: list[Path]) -> dict[int, int]:
    totals: dict[int, int] = {}
    for path in paths:
        with rasterio.open(path) as src:
            values = src.read(1)
        for value, count in zip(*__import__("numpy").unique(values, return_counts=True)):
            if value != 255:
                totals[int(value)] = totals.get(int(value), 0) + int(count)
    return totals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing train/")
    parser.add_argument("--output", type=Path, default=Path("report/EDA.md"))
    args = parser.parse_args()
    af_dir, bs_dir = args.data_dir / "af", args.data_dir / "bs"
    af_meta, bs_meta = pd.read_csv(af_dir / "meta.csv"), pd.read_csv(bs_dir / "meta.csv")
    af_counts = mask_counts(list((af_dir / "masks").glob("*.tif")))
    bs_counts = mask_counts(list((bs_dir / "masks").glob("*.tif")))
    af_total = sum(af_counts.values())
    bs_total = sum(bs_counts.values())
    rows = [
        "# EDA: выданная обучающая выборка",
        "",
        "Этот файл генерируется `python scripts/eda.py --data-dir <train>`; цифры не переносятся вручную.",
        "",
        "## Объём и дисбаланс",
        "",
        f"- AF: {len(af_meta)} чипов; пикселей огня: {af_counts.get(1, 0):,} ({af_counts.get(1,0)/af_total:.4%}).",
        f"- BS: {len(bs_meta)} чипов; площадь классов по пикселям: "
        + ", ".join(f"{c}={bs_counts.get(c,0):,} ({bs_counts.get(c,0)/bs_total:.2%})" for c in range(4)) + ".",
        f"- AF-чипов с огнём: {(af_meta.n_fire_px > 0).sum()} из {len(af_meta)}; медиана среди положительных: {af_meta.loc[af_meta.n_fire_px > 0, 'n_fire_px'].median():.0f} пикселей.",
        f"- BS cloud_frac: медиана {bs_meta.cloud_frac.median():.3f}, p90 {bs_meta.cloud_frac.quantile(.9):.3f}, максимум {bs_meta.cloud_frac.max():.3f}.",
        "",
        "## Следствия для модели",
        "",
        "- AF оценивается по полным чипам, поэтому accuracy не используется; порог выбирается по микро-F1 и проверяется на ложных термоаномалиях.",
        "- Для BS dNBR является прозрачной точкой отсчёта, но пороги следует калибровать отдельно по land cover и проверять влияние SCL/SAR абляцией.",
        "- Разбиение проводится по `fire_event_id`; поля `n_fire_px`, `burn_area_ha`, `sev*_px` не подаются в модель, так как это сведения об эталоне.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
