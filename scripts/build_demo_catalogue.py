"""Build a small georeferenced service catalogue from labelled train chips."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
import rasterio
from rasterio.features import shapes
from rasterio.warp import transform, transform_geom
from shapely.geometry import shape, mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("service/catalogue.geojson"))
    parser.add_argument("--max-af", type=int, default=30)
    parser.add_argument("--max-bs", type=int, default=20)
    args = parser.parse_args()
    features: list[dict] = []

    af_meta = pd.read_csv(args.data_dir / "af" / "meta.csv")
    for row in af_meta.loc[af_meta.n_fire_px > 0].head(args.max_af).itertuples(index=False):
        path = args.data_dir / "af" / "masks" / f"{row.chip_id}_MASK.tif"
        with rasterio.open(path) as src:
            mask = src.read(1)
            rows, cols = (mask == 1).nonzero()
            xs, ys = rasterio.transform.xy(src.transform, rows, cols, offset="center")
            lon, lat = transform(src.crs, "EPSG:4326", xs, ys)
        for index, (x, y) in enumerate(zip(lon, lat)):
            features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [x, y]}, "properties": {"id": f"{row.chip_id}_{index}", "kind": "fire_point", "observed_at": str(row.acq_datetime)}})

    bs_meta = pd.read_csv(args.data_dir / "bs" / "meta.csv")
    for row in bs_meta.head(args.max_bs).itertuples(index=False):
        path = args.data_dir / "bs" / "masks" / f"{row.chip_id}_MASK.tif"
        with rasterio.open(path) as src:
            mask = src.read(1)
            for geometry, value in shapes(mask, mask=(mask > 0) & (mask < 255), transform=src.transform):
                cls = int(value)
                native = shape(geometry)
                if native.area < src.res[0] * src.res[1] * 4:
                    continue
                simplified = native.simplify(src.res[0], preserve_topology=True)
                wgs84 = transform_geom(src.crs, "EPSG:4326", mapping(simplified), precision=6)
                features.append({"type": "Feature", "geometry": wgs84, "properties": {"id": f"{row.chip_id}_{len(features)}", "kind": "burn_polygon", "observed_at": str(row.date_post), "severity_class": cls, "area_ha": native.area / 10000.0}})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(features)} features to {args.output}")


if __name__ == "__main__":
    main()
