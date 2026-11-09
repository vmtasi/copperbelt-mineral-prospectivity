"""Shared utilities for the Stage 4 support, occurrence and model pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely


ROOT = Path(__file__).resolve().parents[2]
PROJECT_CRS = "EPSG:32735"


def abs_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_config(path: str | Path = "configs/stage4_v1.json") -> dict[str, Any]:
    with abs_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with abs_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: str | Path, role: str | None = None) -> dict[str, Any]:
    absolute = abs_path(path)
    record: dict[str, Any] = {
        "path": relative_posix(absolute),
        "bytes": absolute.stat().st_size,
        "sha256": sha256_file(absolute),
    }
    if role is not None:
        record["role"] = role
    return record


def relative_posix(path: str | Path) -> str:
    absolute = abs_path(path).resolve()
    try:
        return absolute.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return absolute.as_posix()


def verify_expected_hash(path: str | Path, expected: str, label: str) -> dict[str, Any]:
    observed = sha256_file(path)
    if observed.lower() != expected.lower():
        raise RuntimeError(
            f"Protected input hash mismatch for {label}: expected {expected}, observed {observed}"
        )
    return {
        "path": relative_posix(path),
        "expected_sha256": expected.lower(),
        "observed_sha256": observed.lower(),
        "hash_match": True,
    }


def verify_protected(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    protected_files = config["protected_files"]
    protected_hashes = config["protected_hashes"]
    if set(protected_files) != set(protected_hashes):
        raise ValueError("Protected file names and protected hashes do not match")
    return {
        key: verify_expected_hash(protected_files[key], protected_hashes[key], key)
        for key in sorted(protected_files)
    }


def read_gdf(path: str | Path, layer: str | None = None) -> gpd.GeoDataFrame:
    frame = gpd.read_file(abs_path(path), layer=layer, engine="pyogrio")
    if frame.crs is None:
        raise ValueError(f"Layer has no CRS: {path} / {layer}")
    if frame.crs.to_string() != PROJECT_CRS:
        frame = frame.to_crs(PROJECT_CRS)
    return frame


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize value of type {type(value).__name__}")


def write_json(payload: dict[str, Any], path: str | Path) -> Path:
    destination = abs_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)
        handle.write("\n")
    os.replace(temporary, destination)
    return destination


def write_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = abs_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, destination)
    return destination


def write_gpkg(layers: Iterable[tuple[str, gpd.GeoDataFrame]], path: str | Path) -> Path:
    destination = abs_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp-{os.getpid()}.gpkg")
    if temporary.exists():
        temporary.unlink()
    first = True
    try:
        for layer_name, frame in layers:
            if frame.crs is None or frame.crs.to_string() != PROJECT_CRS:
                raise ValueError(f"Layer {layer_name} is not in {PROJECT_CRS}")
            pyogrio.write_dataframe(
                frame,
                temporary,
                layer=layer_name,
                driver="GPKG",
                append=not first,
            )
            first = False
        if first:
            raise ValueError("No layers supplied for GeoPackage")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "geopandas": gpd.__version__,
        "shapely": shapely.__version__,
        "pyogrio": pyogrio.__version__,
    }


def ensure_parent(path: str | Path) -> Path:
    destination = abs_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def output_paths(config: dict[str, Any]) -> dict[str, str]:
    processed = abs_path(config["outputs"]["processed_dir"])
    figure_dir = abs_path(config["outputs"]["figure_dir"])
    processed.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    excluded = {"processed_dir", "figure_dir", "figures"}
    names = {
        key: value
        for key, value in config["outputs"].items()
        if isinstance(value, str) and key not in excluded
    }
    paths = {key: processed / value for key, value in names.items()}
    paths["figure_dir"] = figure_dir
    paths["processed_dir"] = processed
    return {key: str(value) for key, value in paths.items()}


def np_json(value: Any) -> Any:
    """Convert common NumPy scalars for JSON-compatible summary dictionaries."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
