from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
PARAMS = ROOT / "data/raw/rdc_rocks/derived/georef/transform_params_v1.json"
OUT = ROOT / "data/raw/rdc_rocks/derived/georef/derived_raster_extents_v1.csv"

IMAGES = {
    "3863": "data/raw/rdc_rocks/derived/georef/render/RDC_3863_p001_upright.png",
    "3856": "data/raw/rdc_rocks/derived/ocr/pages/RDC_3856_p001.png",
}


def main() -> None:
    params = json.loads(PARAMS.read_text(encoding="utf-8"))
    rows = []
    for map_id, rel in IMAGES.items():
        t = params[map_id]["transform"]
        width, height = Image.open(ROOT / rel).size
        for corner, (x, y) in {
            "top_left": (0, 0),
            "top_right": (width, 0),
            "bottom_left": (0, height),
            "bottom_right": (width, height),
        }.items():
            rows.append({
                "map_id": map_id,
                "image_path": rel,
                "width_px": width,
                "height_px": height,
                "corner": corner,
                "x_px": x,
                "y_px": y,
                "E_grid_m": f"{(x - t['b']) / t['a']:.1f}",
                "N_grid_m": f"{(y - t['d']) / t['c']:.1f}",
                "crs_status": "map-grid metres only; datum and projection UNRESOLVED; not WGS84",
                "note": "Raster extent computed from documented GCP transform; no geology digitised.",
            })
    with OUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} corner rows -> {OUT}")


if __name__ == "__main__":
    main()
