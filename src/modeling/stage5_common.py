"""Shared, read-only validation and I/O helpers for the Stage 5 audit."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modeling.stage4_common import (  # noqa: E402
    PROJECT_CRS,
    abs_path,
    file_record,
    read_gdf,
    relative_posix,
    sha256_file,
    software_versions,
    verify_protected,
    write_csv,
    write_gpkg,
    write_json,
)


def load_config(path: str | Path = "configs/stage5_v1.json") -> dict[str, Any]:
    with abs_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_json(path: str | Path) -> dict[str, Any]:
    with abs_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _assert_output_record(path: str | Path, record: dict[str, Any], label: str) -> dict[str, Any]:
    observed = sha256_file(path)
    expected = str(record["sha256"]).lower()
    if observed.lower() != expected:
        raise RuntimeError(
            f"Stage 5 input {label} does not match its Stage 4 manifest: "
            f"expected {expected}, observed {observed}"
        )
    return {
        "label": label,
        "path": relative_posix(path),
        "sha256": observed.lower(),
        "bytes": abs_path(path).stat().st_size,
    }


def load_stage4_bundle(config: dict[str, Any]) -> dict[str, Any]:
    """Load Stage 4 data and model artefacts only after their manifests verify.

    Stage 4 is the accepted reference. No Stage 5 operation writes to these
    paths. The returned frames are in-memory copies when later mutated.
    """

    stage4_paths = config["stage4"]
    stage4_config = _load_json(stage4_paths["config"])
    protected = verify_protected(stage4_config)

    data_manifest = _load_json(stage4_paths["data_manifest"])
    data_validation = _load_json(stage4_paths["data_validation"])
    model_manifest = _load_json(stage4_paths["model_manifest"])
    model_validation = _load_json(stage4_paths["model_validation"])

    if data_manifest.get("status") != "verified":
        raise RuntimeError("Stage 4 data manifest is not verified")
    if data_validation.get("status") != "pass" or int(data_validation.get("n_failed", 1)) != 0:
        raise RuntimeError("Stage 4 data validation is not passing")
    if model_validation.get("status") != "pass" or int(model_validation.get("n_failed", 1)) != 0:
        raise RuntimeError("Stage 4 model validation is not passing")
    if model_manifest.get("validation", {}).get("status") != "pass":
        raise RuntimeError("Stage 4 model manifest does not record a passing validation")
    if model_manifest.get("data_build_id") != data_manifest.get("build_id"):
        raise RuntimeError("Stage 4 model and data builds do not agree")

    data_outputs = {
        "model_table": stage4_paths["model_table"],
        "occurrences_csv": stage4_paths["occurrences"],
        "group_components_csv": stage4_paths["group_components"],
        "positional_weights_csv": stage4_paths["positional_weights"],
        "support_gpkg": stage4_paths["support_gpkg"],
    }
    model_outputs = {
        "model_predictions": stage4_paths["model_predictions"],
        "model_metrics": stage4_paths["model_metrics"],
        "fold_metrics": stage4_paths["fold_metrics"],
    }
    verified_outputs = [
        _assert_output_record(path, data_manifest["outputs"][key], f"data/{key}")
        for key, path in data_outputs.items()
    ]
    verified_outputs.extend(
        _assert_output_record(path, model_manifest["outputs"][key], f"model/{key}")
        for key, path in model_outputs.items()
    )

    model_table = pd.read_csv(abs_path(stage4_paths["model_table"]))
    occurrences = pd.read_csv(abs_path(stage4_paths["occurrences"]))
    components = pd.read_csv(abs_path(stage4_paths["group_components"]))
    weights = pd.read_csv(abs_path(stage4_paths["positional_weights"]))
    predictions = pd.read_csv(abs_path(stage4_paths["model_predictions"]))
    model_metrics = pd.read_csv(abs_path(stage4_paths["model_metrics"]))
    fold_metrics = pd.read_csv(abs_path(stage4_paths["fold_metrics"]))

    return {
        "stage4_config": stage4_config,
        "data_manifest": data_manifest,
        "data_validation": data_validation,
        "model_manifest": model_manifest,
        "model_validation": model_validation,
        "protected": protected,
        "verified_outputs": verified_outputs,
        "model_table": model_table,
        "occurrences": occurrences,
        "components": components,
        "weights": weights,
        "predictions": predictions,
        "model_metrics": model_metrics,
        "fold_metrics": fold_metrics,
    }


def load_geometries(config: dict[str, Any]) -> dict[str, Any]:
    tracts = read_gdf(config["inputs"]["tracts"]["path"])
    tracts = tracts.copy()
    invalid = ~tracts.geometry.is_valid
    if invalid.any():
        tracts.loc[invalid, "geometry"] = tracts.loc[invalid, "geometry"].make_valid()
    tract_union = unary_union(list(tracts.geometry))
    if not shapely.is_valid(tract_union):
        tract_union = shapely.make_valid(tract_union)

    nominal_hexagons = read_gdf(
        config["inputs"]["nominal_hexagons"]["path"],
        layer=config["inputs"]["nominal_hexagons"]["layer"],
    )
    domains = read_gdf(
        config["inputs"]["daly_domains"]["path"],
        layer=config["inputs"]["daly_domains"]["layer"],
    )
    domain_union = unary_union(list(domains.geometry))
    if not shapely.is_valid(domain_union):
        domain_union = shapely.make_valid(domain_union)

    stage4_paths = config["stage4"]
    catchments = read_gdf(
        stage4_paths["support_gpkg"], layer=stage4_paths["support_layer"]
    )
    if catchments.crs is None or catchments.crs.to_string() != PROJECT_CRS:
        raise ValueError("Stage 4 catchment layer is not in the project CRS")
    return {
        "tracts": tracts,
        "tract_union": tract_union,
        "nominal_hexagons": nominal_hexagons,
        "domains": domains,
        "domain_union": domain_union,
        "catchments": catchments,
    }


def distance_to_boundary_km(points: gpd.GeoSeries, polygon: Any) -> np.ndarray:
    boundary = polygon.boundary
    return np.asarray([shapely.distance(point, boundary) / 1000.0 for point in points], dtype=float)


def shannon_entropy(probability_rows: pd.DataFrame, record_ids: pd.Series) -> pd.Series:
    """Return base-e entropy of each record's categorical positional distribution."""
    values: dict[str, float] = {}
    for record_id, group in probability_rows.groupby("record_id", sort=False):
        probabilities = group["probability"].astype(float).to_numpy()
        probabilities = probabilities[probabilities > 0.0]
        values[str(record_id)] = float(-np.sum(probabilities * np.log(probabilities)))
    return record_ids.astype(str).map(values)


def format_ci(estimate: float, low: float, high: float, decimals: int = 3) -> str:
    if not all(np.isfinite(value) for value in (estimate, low, high)):
        return "NA"
    return f"{estimate:.{decimals}f} ({low:.{decimals}f}, {high:.{decimals}f})"


def markdown_table(frame: pd.DataFrame, columns: list[str], display_columns: list[str] | None = None) -> str:
    if display_columns is None:
        display_columns = columns
    lines = [
        "| " + " | ".join(display_columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in frame[columns].itertuples(index=False):
        values: list[str] = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append("NA" if not np.isfinite(value) else f"{float(value):.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_text(text: str, path: str | Path) -> Path:
    destination = abs_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")
    os.replace(temporary, destination)
    return destination


def software_stack() -> dict[str, str]:
    import scipy

    return {
        **software_versions(),
        "scipy": scipy.__version__,
    }
