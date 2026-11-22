"""Shared data, geometry and model helpers for V15.

The module deliberately reads V14 only. Every writer in the V15 package calls
``ensure_v15_path`` before creating an output.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform
import shapely
import pytensor.tensor as pt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation_strategies import get_along_belt_folds  # noqa: E402
from src.v15 import config  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_v15_path(path: Path) -> Path:
    path = Path(path).resolve()
    root = config.V15.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Refusing to write outside V15: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, value: Any) -> None:
    path = ensure_v15_path(path)
    path.write_text(json.dumps(value, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serialisable: {type(value)!r}")


def domain_code(value: object) -> str:
    text = str(value).lower()
    if "3a" in text:
        return "NRB_3a"
    if "3b" in text:
        return "NRB_3b"
    if "crz" in text:
        return "CRZ"
    if "srb" in text:
        return "SRB"
    if "nkb" in text:
        return "NKB"
    if "mmsb" in text:
        return "MMSB"
    return "Unknown"


def load_v14_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    full = pd.read_csv(config.V14_TABLE)
    subset = [
        "centroid_x",
        "centroid_y",
        config.LITHO_COL,
        "domain",
        "distance_to_lithology_contact",
        "distance_to_fault",
        "bouguer",
    ]
    frame = full.dropna(subset=subset).copy()
    frame["daly_domain"] = frame["domain"].map(domain_code)
    frame = frame[frame["daly_domain"] != "Unknown"].copy()
    frame = frame.dropna(subset=[config.Y_COL]).copy()
    frame["fold"] = get_along_belt_folds(frame, n_folds=4)
    frame = frame.reset_index(drop=True)
    return full, frame


def build_area_table() -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    if not config.GRID_GPKG.is_file():
        raise FileNotFoundError(config.GRID_GPKG)
    if not config.TRACTS_GPKG.is_file():
        raise FileNotFoundError(config.TRACTS_GPKG)
    grid = gpd.read_file(config.GRID_GPKG)
    tracts = gpd.read_file(config.TRACTS_GPKG)
    grid = grid.to_crs(32735)
    tracts = tracts.to_crs(32735)
    union = tracts.geometry.union_all()
    cells = grid.dissolve(by="id").reset_index()
    cells["id"] = cells["id"].astype(int)
    cells["area_km2"] = cells.geometry.area / 1e6
    bounds = (
        grid.groupby("id")
        .agg(xmin=("left", "min"), ymin=("bottom", "min"), xmax=("right", "max"), ymax=("top", "max"))
        .reset_index()
    )
    bounds["id"] = bounds["id"].astype(int)
    cells = cells.merge(bounds, on="id", how="left", validate="one_to_one")
    full_box = shapely.box(
        cells["xmin"].to_numpy(float),
        cells["ymin"].to_numpy(float),
        cells["xmax"].to_numpy(float),
        cells["ymax"].to_numpy(float),
    )
    cells["full_area_km2"] = shapely.area(full_box) / 1e6
    cells["tract_intersection_km2"] = shapely.area(shapely.intersection(full_box, union)) / 1e6
    cells["distance_to_tract_boundary_m"] = cells.geometry.distance(union.boundary)
    actual_minus_intersection = np.abs(cells["area_km2"] - cells["tract_intersection_km2"])
    if actual_minus_intersection.max() > 1e-6:
        raise ValueError(f"Cell area is not the tract intersection: max {actual_minus_intersection.max()}")
    if (cells["area_km2"] <= 0).any():
        raise ValueError("Non-positive cell area present")
    if (cells["area_km2"] > 25.000001).any():
        raise ValueError("Cell area exceeds a 5 km square")
    audit = {
        "grid_path": str(config.GRID_GPKG),
        "tract_path": str(config.TRACTS_GPKG),
        "grid_sha256": sha256(config.GRID_GPKG),
        "tract_sha256": sha256(config.TRACTS_GPKG),
        "crs": "EPSG:32735",
        "unit": "km2",
        "cells": int(len(cells)),
        "unique_ids": int(cells["id"].nunique()),
        "full_cell_area_min_km2": float(cells["full_area_km2"].min()),
        "full_cell_area_max_km2": float(cells["full_area_km2"].max()),
        "clipped_area_sum_km2": float(cells["area_km2"].sum()),
        "max_abs_clipped_minus_tract_intersection_km2": float(actual_minus_intersection.max()),
        "definition": "5 km cell footprint intersected with the union of the five USGS permissive tracts",
    }
    return cells, audit


def sample_raster_classes(frame: pd.DataFrame) -> pd.Series:
    if not config.LITHO_RASTER.is_file():
        return pd.Series(np.nan, index=frame.index, dtype=float)
    with rasterio.open(config.LITHO_RASTER) as raster:
        xs, ys = warp_transform(
            "EPSG:32735",
            raster.crs,
            frame["centroid_x"].to_numpy(float),
            frame["centroid_y"].to_numpy(float),
        )
        values = np.array([int(next(raster.sample([(x, y)]))[0]) for x, y in zip(xs, ys)], dtype=float)
        values[values == raster.nodata] = np.nan
    return pd.Series(values, index=frame.index, name="litho_class_raster_centroid")


def build_model_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    full, frame = load_v14_frame()
    cells, area_audit = build_area_table()
    area = cells[["id", "area_km2", "full_area_km2", "distance_to_tract_boundary_m"]]
    model = frame.merge(area, on="id", how="left", validate="one_to_one")
    if model["area_km2"].isna().any():
        raise ValueError("Some complete-case cells have no polygon area")
    model["log_area_km2"] = np.log(model["area_km2"])
    model["small_cell"] = model["area_km2"] < 1.0
    model["litho_class_raster_centroid"] = sample_raster_classes(model)
    model = model[
        [
            "id",
            config.Y_COL,
            "fold",
            "daly_domain",
            "centroid_x",
            "centroid_y",
            "distance_to_fault",
            "distance_to_lithology_contact",
            "bouguer",
            config.LITHO_COL,
            "litho_class_raster_centroid",
            "area_km2",
            "log_area_km2",
            "small_cell",
            "full_area_km2",
            "distance_to_tract_boundary_m",
        ]
    ].copy()
    audit = {
        **area_audit,
        "v14_table": str(config.V14_TABLE),
        "v14_table_sha256": sha256(config.V14_TABLE),
        "full_v14_rows": int(len(full)),
        "complete_case_rows": int(len(model)),
        "positive_cells": int(model[config.Y_COL].sum()),
        "fold_counts": {str(k): int(v) for k, v in model.groupby("fold").size().items()},
        "fold_positives": {str(k): int(v) for k, v in model.groupby("fold")[config.Y_COL].sum().items()},
    }
    return model.reset_index(drop=True), audit


def load_frozen_oof() -> pd.DataFrame:
    frame = pd.read_csv(config.V14_OOF)
    frame["fold"] = frame["fold"].astype(int)
    return frame


def add_lithology_dummies(frame: pd.DataFrame, source: str = "v14") -> pd.DataFrame:
    column = config.LITHO_COL if source == "v14" else "litho_class_raster_centroid"
    values = frame[column].astype("Int64").astype(str)
    dummies = pd.get_dummies(values, prefix="rock", dtype=float)
    out = pd.concat([frame.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    out.attrs["rock_columns"] = dummies.columns.tolist()
    return out


def prepare_fold(
    frame: pd.DataFrame,
    fold: int,
    spec: dict[str, Any],
    litho_source: str = "v14",
) -> dict[str, Any]:
    """Build training/test arrays with all transformations fitted on training only."""

    dummied = add_lithology_dummies(frame, source=litho_source)
    rock_columns = dummied.attrs["rock_columns"]
    train = dummied[dummied["fold"] != fold].copy()
    test = dummied[dummied["fold"] == fold].copy()
    y_train = train[config.Y_COL].to_numpy(dtype=int)
    y_test = test[config.Y_COL].to_numpy(dtype=int)

    valid_rocks = [
        column
        for column in rock_columns
        if ((train[column] == 1) & (y_train == 1)).sum() > 0
        and ((train[column] == 1) & (y_train == 0)).sum() > 0
    ]

    arrays: dict[str, Any] = {
        "train": {},
        "test": {},
        "valid_rocks": valid_rocks,
        "n_rocks": len(valid_rocks),
        "n_domains": len(config.DOMAINS),
        "domains": config.DOMAINS,
        "y_train": y_train,
        "y_test": y_test,
        "train_size": int(len(train)),
        "test_size": int(len(test)),
        "p_ref_train": float((y_train.sum() + 0.5) / (len(y_train) + 1.0)),
        "train_ids": train["id"].to_numpy(dtype=int),
        "test_ids": test["id"].to_numpy(dtype=int),
    }

    for key, data in (("train", train), ("test", test)):
        for name, column in config.CONTINUOUS.items():
            short = "grav" if name == "gravity" else name
            values = data[column].to_numpy(dtype=float)
            if key == "train":
                mean = float(np.mean(values))
                scale = float(np.std(values, ddof=0))
                if not np.isfinite(scale) or scale <= 0:
                    raise ValueError(f"Non-positive training scale for {name}")
                arrays[f"{short}_mean"] = mean
                arrays[f"{short}_scale"] = scale
            z = (values - arrays[f"{short}_mean"]) / arrays[f"{short}_scale"]
            arrays[key][f"{short}_z"] = z
            arrays[key][f"{short}_z2"] = z ** 2
        arrays[key]["rocks"] = data[valid_rocks].to_numpy(dtype=float) if valid_rocks else np.empty((len(data), 0))
        arrays[key]["domain_idx"] = pd.Categorical(data["daly_domain"], categories=config.DOMAINS).codes.astype(int)
        arrays[key]["log_area"] = data["log_area_km2"].to_numpy(dtype=float)
        log_area = data["log_area_km2"].to_numpy(dtype=float)
        if key == "train":
            arrays["log_area_mean"] = float(np.mean(log_area))
            arrays["log_area_scale"] = float(np.std(log_area, ddof=0))
        arrays[key]["log_area_z"] = (log_area - arrays["log_area_mean"]) / arrays["log_area_scale"]
        arrays[key]["ids"] = data["id"].to_numpy(dtype=int)
        arrays[key]["centroid_x"] = data["centroid_x"].to_numpy(dtype=float)
        arrays[key]["centroid_y"] = data["centroid_y"].to_numpy(dtype=float)
        arrays[key]["y"] = data[config.Y_COL].to_numpy(dtype=int)

    arrays["area_train_sum"] = float(np.exp(arrays["train"]["log_area"]).sum())
    arrays["area_train_mean"] = float(np.exp(arrays["train"]["log_area"]).mean())
    return arrays


def cloglog_probability(eta: np.ndarray) -> np.ndarray:
    return -np.expm1(-np.exp(eta))


def logistic_probability(eta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-eta))


def _samples(trace: Any, name: str) -> np.ndarray:
    return np.asarray(trace.posterior[name].values).reshape(-1, *trace.posterior[name].shape[2:])


def linear_predictor_samples(
    trace: Any,
    spec: dict[str, Any],
    arrays: dict[str, Any],
    split: str,
) -> np.ndarray:
    data = arrays[split]
    n_domains = arrays["n_domains"]
    family = spec["family"]
    alpha = _samples(trace, "alpha")[:, None] if family == "global" else _samples(trace, "alpha_dom")[:, data["domain_idx"]]
    eta = alpha.copy()

    if family in {"global", "domain_intercept"}:
        eta = eta + _samples(trace, "beta_f_lin")[:, None] * data["fault_z"][None, :]
        eta = eta + _samples(trace, "beta_f_sq")[:, None] * data["fault_z2"][None, :]
        eta = eta + _samples(trace, "beta_l_lin")[:, None] * data["lith_z"][None, :]
        eta = eta + _samples(trace, "beta_l_sq")[:, None] * data["lith_z2"][None, :]
        gravity = _samples(trace, "beta_grav")[:, None]
    else:
        eta = eta + _samples(trace, "beta_f_lin")[:, data["domain_idx"]] * data["fault_z"][None, :]
        eta = eta + _samples(trace, "beta_f_sq")[:, data["domain_idx"]] * data["fault_z2"][None, :]
        eta = eta + _samples(trace, "beta_l_lin")[:, data["domain_idx"]] * data["lith_z"][None, :]
        eta = eta + _samples(trace, "beta_l_sq")[:, data["domain_idx"]] * data["lith_z2"][None, :]
        if family == "domain_distance_gravity":
            gravity = _samples(trace, "beta_grav")[:, data["domain_idx"]]
        else:
            gravity = _samples(trace, "beta_grav")[:, None]
    eta = eta + gravity * data["grav_z"][None, :]

    rocks = _samples(trace, "beta_rocks")
    if arrays["n_rocks"]:
        eta = eta + rocks @ data["rocks"].T
    if spec["area"] == "offset":
        eta = eta + data["log_area"][None, :]
    elif spec["area"] == "covariate":
        eta = eta + _samples(trace, "beta_area")[:, None] * data["log_area_z"][None, :]
    return eta


def predict_probabilities(
    trace: Any,
    spec: dict[str, Any],
    arrays: dict[str, Any],
    split: str,
) -> np.ndarray:
    eta = linear_predictor_samples(trace, spec, arrays, split)
    if spec["link"] == "cloglog":
        probability = cloglog_probability(eta)
    else:
        probability = logistic_probability(eta)
    return np.clip(probability, config.PCELL, 1.0 - config.PCELL)


def summarize_draws(probability: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "prob": probability.mean(axis=0),
        "prob_sd": probability.std(axis=0),
        "prob_lower": np.percentile(probability, 2.5, axis=0),
        "prob_upper": np.percentile(probability, 97.5, axis=0),
    }


