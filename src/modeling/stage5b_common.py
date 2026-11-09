"""Shared support-construction and diagnostic helpers for Stage 5B."""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy import stats
from shapely.ops import unary_union
from shapely.strtree import STRtree


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modeling.fit_stage4_models import (  # noqa: E402
    Experiment,
    fit_experiment,
)
from src.modeling.stage4_common import (  # noqa: E402
    PROJECT_CRS,
    abs_path,
    file_record,
    relative_posix,
    sha256_file,
    write_csv,
    write_json,
)
from src.modeling.stage5_common import (  # noqa: E402
    distance_to_boundary_km,
    load_config as load_stage5_config,
    load_geometries,
    load_stage4_bundle,
)


def load_config(path: str | Path = "configs/stage5b_v1.json") -> dict[str, Any]:
    with abs_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def output_paths(config: dict[str, Any]) -> dict[str, str]:
    processed = abs_path(config["outputs"]["processed_dir"])
    figures = abs_path(config["outputs"]["figure_dir"])
    processed.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    figure_keys = {"scenario_figure", "retention_figure", "metric_figure", "diagnostic_figure", "report"}
    for key, value in config["outputs"].items():
        if key == "processed_dir":
            paths[key] = str(processed)
        elif key == "figure_dir":
            paths[key] = str(figures)
        elif key in figure_keys:
            paths[key] = str(figures / value)
        else:
            paths[key] = str(processed / value)
    return paths


def validate_stage5a(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    stage5_config = load_stage5_config(config["stage5"]["config"])
    manifest_path = abs_path(config["stage5"]["inventory_manifest"])
    validation_path = abs_path(config["stage5"]["inventory_validation"])
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    with validation_path.open("r", encoding="utf-8") as handle:
        validation = json.load(handle)
    if manifest.get("status") != "verified":
        raise RuntimeError("Stage 5A inventory manifest is not verified")
    if validation.get("status") != "pass" or int(validation.get("n_failed", 1)) != 0:
        raise RuntimeError("Stage 5A independent verification did not pass")
    if manifest.get("validation", {}).get("sha256") != sha256_file(validation_path):
        raise RuntimeError("Stage 5A validation file does not match the manifest")
    bundle = load_stage4_bundle(stage5_config)
    geometries = load_geometries(stage5_config)
    return bundle, geometries, stage5_config


def selected_hexagons(hexagons: gpd.GeoDataFrame, threshold: float) -> gpd.GeoDataFrame:
    selected = hexagons.loc[
        hexagons["frame_overlap_fraction"].astype(float).ge(float(threshold))
    ].copy()
    if selected.empty:
        raise ValueError(f"No nominal hexagons satisfy overlap threshold {threshold}")
    if selected["support_id"].duplicated().any():
        raise ValueError("Selected nominal hexagons contain duplicate support identifiers")
    if selected["frame_overlap_area_km2"].le(0).any():
        raise ValueError("Selected nominal hexagons contain zero tract exposure")
    return selected


def assign_points_to_hexagons(
    points: gpd.GeoSeries, hexagons: gpd.GeoDataFrame
) -> pd.Series:
    """Assign each point to one selected hexagon, resolving boundaries by distance."""
    geometries = list(hexagons.geometry)
    tree = STRtree(geometries)
    centers = np.column_stack([hexagons.geometry.centroid.x, hexagons.geometry.centroid.y])
    identifiers = hexagons["support_id"].astype(str).to_numpy()
    output: list[str | None] = []
    for point in points:
        candidates = np.asarray(tree.query(point, predicate="intersects"), dtype=int)
        if candidates.size == 0:
            output.append(None)
            continue
        if candidates.size == 1:
            output.append(str(identifiers[int(candidates[0])]))
            continue
        distances = np.linalg.norm(centers[candidates] - np.array([point.x, point.y]), axis=1)
        order = np.lexsort((identifiers[candidates], distances))
        output.append(str(identifiers[int(candidates[int(order[0])])]))
    return pd.Series(output, index=points.index, dtype=object)


def _empty_counts(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "n_records", "n_deposits", "n_prospects", "n_groups",
        "n_expanded_records", "n_deposits_plus_prospects",
        "record_positional_outside_probability_sum",
    ]
    for column in columns:
        frame[column] = 0.0
    return frame


def build_support_scenario(
    model_table: pd.DataFrame,
    occurrences: pd.DataFrame,
    hexagons: gpd.GeoDataFrame,
    catchments: gpd.GeoDataFrame,
    tract_union: Any,
    scenario: dict[str, Any],
) -> tuple[pd.DataFrame, gpd.GeoDataFrame, pd.DataFrame, dict[str, Any]]:
    """Build one support frame and its record assignment table."""
    family = str(scenario["family"])
    threshold = scenario.get("record_filter_threshold")
    hex_threshold = scenario.get("hex_threshold")
    point_series = gpd.GeoSeries(
        gpd.points_from_xy(occurrences["x_m"], occurrences["y_m"]), crs=PROJECT_CRS
    )

    if family == "hex":
        if hex_threshold is None:
            raise ValueError("Hex support requires a hex threshold")
        selected = selected_hexagons(hexagons, float(hex_threshold))
        support = model_table.loc[
            model_table["support_id"].isin(selected["support_id"])
        ].copy()
        support = support.merge(
            selected[["support_id", "frame_overlap_area_km2", "frame_overlap_fraction"]],
            on="support_id", how="left", validate="one_to_one", suffixes=("", "_selected"),
        )
        area_column = (
            "frame_overlap_area_km2_selected"
            if "frame_overlap_area_km2_selected" in support.columns
            else "frame_overlap_area_km2"
        )
        support["support_area_km2"] = support[area_column].astype(float)
        geometry_lookup = selected.set_index("support_id").geometry
        support["_geometry"] = geometry_lookup.loc[support["support_id"]].to_numpy()
        retained_ids = assign_points_to_hexagons(point_series, selected)
    elif family == "voronoi":
        threshold_hex = 0.0 if threshold is None else float(threshold)
        selected = selected_hexagons(hexagons, threshold_hex)
        contained = assign_points_to_hexagons(point_series, selected)
        if threshold is None:
            retained_ids = occurrences["assigned_support_id"].astype(object)
        else:
            retained_ids = contained
        support = model_table.copy()
        support["support_area_km2"] = support["tract_area_km2"].astype(float)
        catchment_lookup = catchments.set_index("support_id").geometry
        support["_geometry"] = catchment_lookup.loc[support["support_id"]].to_numpy()
    else:
        raise ValueError(f"Unknown support family: {family}")

    support = support.sort_values("support_id").reset_index(drop=True)
    support = _empty_counts(support)
    support["tract_area_km2"] = support["support_area_km2"].astype(float)
    support["log_tract_area_km2"] = np.log(support["tract_area_km2"].clip(lower=1e-12))
    support["BASE_LOG_AREA_KM2"] = np.log(support["tract_area_km2"].clip(lower=1e-12))

    records = occurrences.copy()
    records["point_support_id"] = retained_ids.to_numpy()
    records["in_tract"] = records["in_tract"].astype(bool)
    records["retained_by_support"] = records["in_tract"] & records["point_support_id"].notna()
    records = records.loc[records["retained_by_support"]].copy()
    records["scenario_support_id"] = records["point_support_id"].astype(str)
    invalid = ~records["scenario_support_id"].isin(set(support["support_id"]))
    if invalid.any():
        raise RuntimeError(f"{scenario['id']} assigns records to absent support cells")

    index = pd.Series(np.arange(len(support)), index=support["support_id"].to_numpy())
    assigned_index = records["scenario_support_id"].map(index)
    if assigned_index.isna().any():
        raise RuntimeError("A retained record does not map to its support cell")
    assigned_index = assigned_index.to_numpy(dtype=int)
    weights = records["record_expansion"].astype(float).to_numpy()
    support.loc[:, "n_records"] = np.bincount(
        assigned_index, minlength=len(support)
    ).astype(float)
    for status, column in (
        ("deposit", "n_deposits"), ("prospect", "n_prospects"), ("group", "n_groups")
    ):
        mask = records["SiteStatus"].eq(status).to_numpy()
        support.loc[:, column] = np.bincount(
            assigned_index[mask], minlength=len(support)
        ).astype(float)
    support.loc[:, "n_expanded_records"] = np.bincount(
        assigned_index, weights=weights, minlength=len(support)
    ).astype(float)
    support.loc[:, "n_deposits_plus_prospects"] = (
        support["n_deposits"] + support["n_prospects"]
    )
    support.loc[:, "record_positional_outside_probability_sum"] = np.bincount(
        assigned_index,
        weights=records["positional_outside_probability"].astype(float).to_numpy(),
        minlength=len(support),
    )
    support["any_record"] = support["n_records"].gt(0).astype(int)
    support["any_expanded_record"] = support["n_expanded_records"].gt(0).astype(int)

    if int(round(support["n_records"].sum())) != len(records):
        raise RuntimeError(f"{scenario['id']} count total disagrees with retained records")
    if not np.isclose(
        support["n_expanded_records"].sum(), records["record_expansion"].sum(), atol=1e-9
    ):
        raise RuntimeError(f"{scenario['id']} expanded total disagrees with records")

    selected_union = unary_union(list(selected.geometry))
    filter_coverage_union = selected_union.intersection(tract_union)
    coverage_union = (
        tract_union if family == "voronoi" else filter_coverage_union
    )
    support_area_sum = float(support["support_area_km2"].sum())
    union_area = float(coverage_union.area / 1e6)
    support_context = {
        "scenario_id": scenario["id"],
        "family": family,
        "hex_threshold": hex_threshold,
        "record_filter_threshold": threshold,
        "n_support": int(len(support)),
        "support_area_sum_km2": support_area_sum,
        "coverage_union_area_km2": union_area,
        "coverage_fraction_of_tract": union_area / float(tract_union.area / 1e6),
        "record_filter_coverage_fraction_of_tract": float(
            filter_coverage_union.area / tract_union.area
        ),
        "support_overlap_area_km2": support_area_sum - union_area,
        "n_in_tract_records": int(occurrences["in_tract"].astype(bool).sum()),
        "n_retained_records": int(len(records)),
        "n_dropped_records": int(occurrences["in_tract"].astype(bool).sum() - len(records)),
        "retained_fraction": float(len(records) / max(int(occurrences["in_tract"].astype(bool).sum()), 1)),
        "n_positive_cells": int(support["any_record"].sum()),
        "prevalence": float(support["any_record"].mean()),
        "n_expanded_records": float(support["n_expanded_records"].sum()),
        "n_deposits": int(support["n_deposits"].sum()),
        "n_prospects": int(support["n_prospects"].sum()),
        "n_groups": int(support["n_groups"].sum()),
    }
    geometry_frame = gpd.GeoDataFrame(
        support[["support_id", "support_area_km2", "new_fold"]].copy(),
        geometry=support["_geometry"].to_numpy(),
        crs=PROJECT_CRS,
    )
    if geometry_frame.geometry.isna().any():
        raise RuntimeError("Support geometry was not attached")
    support = support.drop(columns=["_geometry"])
    return support, geometry_frame, records, support_context


def positional_retention(
    occurrences: pd.DataFrame,
    tract_union: Any,
    hexagons: gpd.GeoDataFrame,
    stage4_config: dict[str, Any],
    thresholds: list[float],
) -> pd.DataFrame:
    """Propagate each record's 1 km positional model through each support coverage."""
    accuracy_m = float(stage4_config["inputs"]["occurrences"]["positional_accuracy_m"])
    draws = int(stage4_config["inputs"]["occurrences"]["positional_monte_carlo_draws"])
    seed = int(stage4_config["inputs"]["occurrences"]["positional_seed"])
    variants = [("VORONOI_ALL", None)] + [
        (f"HEX_{threshold:g}", float(threshold)) for threshold in thresholds
    ]
    unions: dict[str, Any] = {}
    for label, threshold in variants:
        if threshold is not None:
            selected = selected_hexagons(hexagons, threshold)
            unions[label] = unary_union(list(selected.geometry))
    rows: list[dict[str, Any]] = []
    for row in occurrences.itertuples():
        rng = np.random.default_rng([seed, int(row.source_fid)])
        radius = accuracy_m * np.sqrt(rng.random(draws))
        angle = 2.0 * np.pi * rng.random(draws)
        draw_x = float(row.x_m) + radius * np.cos(angle)
        draw_y = float(row.y_m) + radius * np.sin(angle)
        draw_points = shapely.points(draw_x, draw_y)
        inside_tract = np.asarray(shapely.covers(tract_union, draw_points), dtype=bool)
        for label, threshold in variants:
            if not bool(row.in_tract):
                probability = 0.0
            elif threshold is None:
                probability = float(np.mean(inside_tract))
            else:
                inside_support = np.asarray(
                    shapely.covers(unions[label], draw_points), dtype=bool
                )
                probability = float(np.mean(inside_tract & inside_support))
            rows.append({
                "coverage_variant": label,
                "record_id": row.record_id,
                "site_status": row.SiteStatus,
                "in_tract": bool(row.in_tract),
                "source_fold": int(row.assigned_fold) if pd.notna(row.assigned_fold) else math.nan,
                "point_retained": bool(row.in_tract) if threshold is None else None,
                "p_retained_mc": probability,
                "p_lost_mc": 1.0 - probability,
                "n_draws": draws,
            })
    frame = pd.DataFrame(rows)
    if not np.allclose(
        frame.groupby(["coverage_variant", "record_id"]).size(), 1
    ):
        raise RuntimeError("Positional retention rows are not unique per variant/record")
    return frame


def s0_experiment(stage4_config: dict[str, Any]) -> Experiment:
    reference_id = "S0_NO_GRAVITY"
    entries = [
        entry for entry in stage4_config["models"]["sensitivity_experiments"]
        if entry["id"] == reference_id
    ]
    if len(entries) != 1:
        raise ValueError("Stage 4 configuration does not contain exactly one S0_NO_GRAVITY")
    entry = entries[0]
    return Experiment(
        id=reference_id,
        label=entry["label"],
        predictors=tuple(entry["predictors"]),
        outcome=entry.get("outcome", stage4_config["models"]["outcome"]),
        set_name="support_sensitivity",
    )


def fit_s0(frame: pd.DataFrame, stage4_config: dict[str, Any]) -> tuple[pd.DataFrame, ...]:
    return fit_experiment(frame.sort_values("support_id").reset_index(drop=True), s0_experiment(stage4_config), stage4_config)


def area_elasticity(frame: pd.DataFrame, stage4_config: dict[str, Any]) -> dict[str, Any]:
    """Fit a diagnostic NB2 model with log-area as a free coefficient."""
    diagnostic = copy.deepcopy(stage4_config)
    diagnostic["models"]["ridge_lambda"] = 0.0
    diagnostic["models"]["continuous"]["BASE_LOG_AREA_KM2"] = {
        "transform": "identity", "standardize": True,
    }
    experiment = Experiment(
        id="AREA_ELASTICITY",
        label="Free log-area elasticity diagnostic",
        predictors=("BASE_LOG_AREA_KM2",),
        outcome="n_records",
        set_name="area_diagnostic",
    )
    working = frame.sort_values("support_id").reset_index(drop=True).copy()
    working["BASE_LOG_AREA_KM2"] = np.log(working["support_area_km2"].clip(lower=1e-12))
    _, _, coefficients, fold_metrics = fit_experiment(working, experiment, diagnostic)
    full = coefficients.loc[
        coefficients["fit_scope"].eq("full_data_descriptive")
        & coefficients["term"].eq("BASE_LOG_AREA_KM2")
    ].iloc[0]
    folds = coefficients.loc[
        coefficients["fit_scope"].str.startswith("fold_")
        & coefficients["term"].eq("BASE_LOG_AREA_KM2")
    ]
    natural = folds["estimate_natural"].dropna().to_numpy(dtype=float)
    return {
        "term": "BASE_LOG_AREA_KM2",
        "delta_log_rate_per_log_area": float(full["estimate_natural"]),
        "ci_low": float(full["ci_low_natural"]),
        "ci_high": float(full["ci_high_natural"]),
        "implied_area_exponent": float(1.0 + full["estimate_natural"]),
        "fold_mean_delta": float(np.mean(natural)) if natural.size else math.nan,
        "fold_min_delta": float(np.min(natural)) if natural.size else math.nan,
        "fold_max_delta": float(np.max(natural)) if natural.size else math.nan,
        "n_folds_finite": int(natural.size),
        "n_folds_positive": int(np.sum(natural > 0)),
        "n_folds_negative": int(np.sum(natural < 0)),
        "n_support": int(len(frame)),
        "n_records": float(working["n_records"].sum()),
        "ridge_lambda": 0.0,
        "fold_metrics_n": int(len(fold_metrics)),
    }


def residual_area_diagnostics(predictions: pd.DataFrame, frame: pd.DataFrame) -> dict[str, Any]:
    area = frame["support_area_km2"].astype(float)
    residual = predictions["pearson_residual"].astype(float)
    observed = predictions["observed_count"].astype(float)
    predicted = predictions["predicted_count"].astype(float)
    rate = predictions["predicted_rate_per_km2"].astype(float)
    return {
        "n_support": int(len(frame)),
        "n_records": float(observed.sum()),
        "mean_pearson_residual": float(residual.mean()),
        "median_pearson_residual": float(residual.median()),
        "area_vs_record_count_spearman_r": float(stats.spearmanr(area, observed).statistic),
        "area_vs_record_count_spearman_p": float(stats.spearmanr(area, observed).pvalue),
        "area_vs_pearson_residual_spearman_r": float(stats.spearmanr(area, residual).statistic),
        "area_vs_pearson_residual_spearman_p": float(stats.spearmanr(area, residual).pvalue),
        "area_vs_predicted_rate_spearman_r": float(stats.spearmanr(area, rate).statistic),
        "area_vs_predicted_rate_spearman_p": float(stats.spearmanr(area, rate).pvalue),
        "observed_over_predicted_total": float(observed.sum() / predicted.sum()) if predicted.sum() > 0 else math.nan,
    }


def boundary_diagnostics(predictions: pd.DataFrame, frame: pd.DataFrame, tract_union: Any) -> dict[str, Any]:
    if "distance_to_tract_boundary_km" not in frame:
        points = gpd.GeoSeries(
            gpd.points_from_xy(frame["x_m"], frame["y_m"]), crs=PROJECT_CRS
        )
        distance = distance_to_boundary_km(points, tract_union)
    else:
        distance = frame["distance_to_tract_boundary_km"].to_numpy(dtype=float)
    residual = predictions["pearson_residual"].to_numpy(dtype=float)
    near = distance <= 2.0
    return {
        "n_support": int(len(distance)),
        "distance_to_tract_boundary_mean_km": float(np.mean(distance)),
        "distance_to_tract_boundary_median_km": float(np.median(distance)),
        "n_support_within_2km_boundary": int(np.sum(near)),
        "mean_residual_within_2km": float(np.mean(residual[near])) if near.any() else math.nan,
        "mean_residual_beyond_2km": float(np.mean(residual[~near])) if (~near).any() else math.nan,
        "distance_vs_residual_spearman_r": float(stats.spearmanr(distance, residual).statistic),
        "distance_vs_residual_spearman_p": float(stats.spearmanr(distance, residual).pvalue),
        "n_records_within_2km": int(predictions.loc[near, "observed_count"].sum()),
        "n_records_beyond_2km": int(predictions.loc[~near, "observed_count"].sum()),
    }
