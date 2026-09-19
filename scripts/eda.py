"""Write data facts used by the modelling and report decisions."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio


def mask_counts(paths: list[Path]) -> dict[int, int]:
    totals: dict[int, int] = {}
    for path in paths:
        with rasterio.open(path) as src:
            values = src.read(1)
        for value, count in zip(*np.unique(values, return_counts=True)):
            if value != 255:
                totals[int(value)] = totals.get(int(value), 0) + int(count)
    return totals


def severity_by_landcover(bs_dir: Path) -> pd.DataFrame:
    """Aggregate labelled BS pixels by the supplied WorldCover class."""
    totals: dict[tuple[int, int], int] = {}
    for mask_path in (bs_dir / "masks").glob("*.tif"):
        chip = mask_path.name.replace("_MASK.tif", "")
        with rasterio.open(mask_path) as src:
            severity = src.read(1)
        with rasterio.open(bs_dir / "_aux" / f"{chip}_AUX.tif") as src:
            landcover = src.read(3)
        valid = severity != 255
        codes = landcover[valid].astype(int)
        values = severity[valid].astype(int)
        pairs, counts = np.unique(np.column_stack([codes, values]), axis=0, return_counts=True)
        for (code, cls), count in zip(pairs, counts):
            key = int(code), int(cls)
            totals[key] = totals.get(key, 0) + int(count)
    rows = [{"landcover": lc, "severity": cls, "pixels": count} for (lc, cls), count in totals.items()]
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    if frame.empty:
        return ["нет данных"]
    values = frame[columns].astype(str).values.tolist()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in values]
    return [header, separator, *body]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing train/")
    parser.add_argument("--output", type=Path, default=Path("report/EDA.md"))
    args = parser.parse_args()
    af_dir, bs_dir = args.data_dir / "af", args.data_dir / "bs"
    af_meta, bs_meta = pd.read_csv(af_dir / "meta.csv"), pd.read_csv(bs_dir / "meta.csv")
    af_counts = mask_counts(list((af_dir / "masks").glob("*.tif")))
    bs_counts = mask_counts(list((bs_dir / "masks").glob("*.tif")))
    landcover = severity_by_landcover(bs_dir)
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
        "",
        "## Пространственно-временная структура",
        "",
        f"- Идентификаторы событий AF: {'не предоставлены' if af_meta.fire_event_id.nunique(dropna=True) == 0 else af_meta.fire_event_id.nunique(dropna=True)}; BS-пожаров: {bs_meta.fire_event_id.nunique(dropna=True)}.",
        f"- AF `n_fire_px`: min {af_meta.n_fire_px[af_meta.n_fire_px > 0].min():.0f}, p90 {af_meta.n_fire_px[af_meta.n_fire_px > 0].quantile(.9):.0f}, max {af_meta.n_fire_px.max():.0f}.",
        f"- BS площадь гари: медиана {bs_meta.burn_area_ha.median():.1f} га, p90 {bs_meta.burn_area_ha.quantile(.9):.1f} га, максимум {bs_meta.burn_area_ha.max():.1f} га.",
        "",
        "### AF: чипы по спутнику",
        "",
    ]
    rows.extend(markdown_table(af_meta.groupby("satellite", dropna=False).agg(chips=("chip_id", "count"), fire_pixels=("n_fire_px", "sum")).reset_index(), ["satellite", "chips", "fire_pixels"]))
    rows.extend(["", "### BS: облачность и валидное покрытие", ""])
    cloud = pd.cut(bs_meta.cloud_frac, bins=[-.001, .01, .10, .25, 1.0], labels=["≤1%", "1–10%", "10–25%", ">25%"])
    cloud_summary = bs_meta.assign(cloud_bin=cloud).groupby("cloud_bin", observed=False).agg(chips=("chip_id", "count"), median_burn_ha=("burn_area_ha", "median"), median_valid=("valid_frac", "median")).reset_index()
    cloud_summary["median_burn_ha"] = cloud_summary.median_burn_ha.round(1)
    cloud_summary["median_valid"] = cloud_summary.median_valid.round(3)
    rows.extend(markdown_table(cloud_summary, ["cloud_bin", "chips", "median_burn_ha", "median_valid"]))
    rows.extend(["", "### BS: доля гари по WorldCover", ""])
    pivot = landcover.pivot(index="landcover", columns="severity", values="pixels").fillna(0)
    severity_columns = [column for column in (0, 1, 2, 3) if column in pivot.columns]
    pivot["all_pixels"] = pivot[severity_columns].sum(axis=1)
    pivot["burn_pixels"] = pivot.get(1, 0) + pivot.get(2, 0) + pivot.get(3, 0)
    pivot["burn_share"] = (pivot["burn_pixels"] / pivot["all_pixels"]).map(lambda x: f"{x:.2%}")
    summary = pivot.reset_index().sort_values("burn_pixels", ascending=False).head(10)
    rows.extend(markdown_table(summary, ["landcover", "all_pixels", "burn_pixels", "burn_share"]))
    top_two_share = summary.head(2).burn_pixels.sum() / pivot.burn_pixels.sum()
    rows.extend([
        "",
        "## Дополнительные выводы",
        "",
        f"- Два наиболее представленных типа покрова дают {top_two_share:.1%} размеченных пикселей гари; land cover нужен и для порогов, и как входной признак модели.",
        "- При облачности выше 25% медианная валидная доля падает: оптические индексы должны сопровождаться SCL и проверяться по Sentinel-1.",
        "- В AF нет event-id и region, поэтому для честной валидации этого модуля используется holdout по году съёмки, а не по случайным чипам.",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
