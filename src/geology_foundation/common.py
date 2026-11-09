from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import ndimage as ndi


ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return target


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    rows = list(rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {target}")
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return target


def software_versions() -> dict[str, str | None]:
    from osgeo import gdal
    import geopandas as gpd
    import shapely

    try:
        from qgis.core import Qgis

        qgis_version = Qgis.QGIS_VERSION
    except Exception:
        qgis_version = None
    return {
        "python": sys.version.split()[0],
        "qgis": qgis_version,
        "gdal": gdal.__version__,
        "geopandas": gpd.__version__,
        "shapely": shapely.__version__,
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
    }


def disk(radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def keep_components(
    mask: np.ndarray, min_area: int, min_side: int = 1
) -> tuple[np.ndarray, int]:
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros_like(mask, dtype=bool), 0
    sizes = np.bincount(labels.ravel())
    keep = np.zeros(count + 1, dtype=bool)
    for label, window in enumerate(ndi.find_objects(labels), start=1):
        if window is None:
            continue
        height = window[0].stop - window[0].start
        width = window[1].stop - window[1].start
        if sizes[label] >= min_area and min(height, width) >= min_side:
            keep[label] = True
    return keep[labels], int(keep.sum())


def relative_posix(path: str | Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def file_record(path: str | Path, *, role: str) -> dict[str, Any]:
    path = Path(path)
    return {
        "path": relative_posix(path),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
