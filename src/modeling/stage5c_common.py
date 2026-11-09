"""Shared spatial weights, partitions and residual diagnostics for Stage 5C."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modeling.stage4_common import (  # noqa: E402
    PROJECT_CRS,
    abs_path,
    sha256_file,
    write_csv,
    write_json,
)
from src.modeling.stage5_common import (  # noqa: E402
    distance_to_boundary_km,
)
from src.modeling.stage5b_common import validate_stage5a  # noqa: E402


@dataclass(frozen=True)
class WeightSpec:
    id: str
    kind: str
    value: float
    label: str


def load_config(path: str | Path = "configs/stage5c_v1.json") -> dict[str, Any]:
    with abs_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def output_paths(config: dict[str, Any]) -> dict[str, str]:
    processed = abs_path(config["outputs"]["processed_dir"])
    figures = abs_path(config["outputs"]["figure_dir"])
    processed.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    figure_keys = {
        "partition_figure", "partition_transfer_figure", "moran_figure",
        "local_cluster_figure", "residual_diagnostic_figure",
        "spatial_model_figure", "residual_maps_figure", "report",
    }
    paths: dict[str, str] = {}
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


def validate_stage5b(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = abs_path(config["stage5b"]["manifest"])
    validation_path = abs_path(config["stage5b"]["validation"])
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    with validation_path.open("r", encoding="utf-8") as handle:
        validation = json.load(handle)
    if manifest.get("status") != "verified":
        raise RuntimeError("Stage 5B manifest is not verified")
    if validation.get("status") != "pass" or int(validation.get("n_failed", 1)) != 0:
        raise RuntimeError("Stage 5B independent verification did not pass")
    if manifest.get("validation", {}).get("sha256") != sha256_file(validation_path):
        raise RuntimeError("Stage 5B validation file does not match its manifest")
    return {"manifest": manifest, "validation": validation}


def load_verified_inputs(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    stage5b = validate_stage5b(config)
    bundle, geometries, stage5_config = validate_stage5a(config)
    return bundle, geometries, {"stage5b": stage5b, "stage5": stage5_config}


def normalise_axis(axis: np.ndarray, reference: np.ndarray) -> np.ndarray:
    vector = np.asarray(axis, dtype=float)
    vector = vector / np.linalg.norm(vector)
    reference = np.asarray(reference, dtype=float)
    reference = reference / np.linalg.norm(reference)
    if float(np.dot(vector, reference)) < 0.0:
        vector = -vector
    return vector


def add_spatial_features(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    output = frame.copy()
    axis = normalise_axis(
        output[["belt_axis_x", "belt_axis_y"]].iloc[0].to_numpy(dtype=float),
        np.asarray([1.0, -1.0], dtype=float),
    )
    cross = np.asarray([-axis[1], axis[0]], dtype=float)
    coordinates = output[["x_m", "y_m"]].to_numpy(dtype=float)
    belt = coordinates @ axis
    across = coordinates @ cross
    center = coordinates.mean(axis=0)
    x = (coordinates[:, 0] - center[0]) / 100000.0
    y = (coordinates[:, 1] - center[1]) / 100000.0
    belt_scaled = (belt - belt.mean()) / 100000.0
    across_scaled = (across - across.mean()) / 100000.0
    output["SPATIAL_X"] = x
    output["SPATIAL_Y"] = y
    output["SPATIAL_X2"] = x**2
    output["SPATIAL_XY"] = x * y
    output["SPATIAL_Y2"] = y**2
    output["SPATIAL_BELT"] = belt_scaled
    output["SPATIAL_BELT2"] = belt_scaled**2
    output["SPATIAL_BELT3"] = belt_scaled**3
    output["SPATIAL_CROSS"] = across_scaled

    rbf = config["rbf"]
    bandwidth = float(rbf["bandwidth_km"]) / 100.0
    belt_centres = np.quantile(
        belt_scaled, np.linspace(0.10, 0.90, int(rbf["n_belt_centres"]))
    )
    cross_centres = np.quantile(
        across_scaled, np.linspace(0.15, 0.85, int(rbf["n_cross_centres"]))
    )
    rows: list[dict[str, Any]] = []
    feature_names: list[str] = []
    index = 0
    for belt_center in belt_centres:
        for cross_center in cross_centres:
            index += 1
            name = f"SPATIAL_RBF_{index:02d}"
            distance_sq = (belt_scaled - belt_center) ** 2 + (across_scaled - cross_center) ** 2
            output[name] = np.exp(-0.5 * distance_sq / bandwidth**2)
            feature_names.append(name)
            rows.append({
                "feature": name,
                "belt_center_scaled_100km": float(belt_center),
                "cross_center_scaled_100km": float(cross_center),
                "bandwidth_km": float(rbf["bandwidth_km"]),
                "basis": "fixed Gaussian radial basis",
            })
    diagnostics = pd.DataFrame(rows)
    metadata = {
        "belt_axis": axis.tolist(),
        "cross_axis": cross.tolist(),
        "coordinate_center_m": center.tolist(),
        "rbf_feature_count": len(feature_names),
        "rbf_bandwidth_km": float(rbf["bandwidth_km"]),
        "rbf_centers": diagnostics.to_dict(orient="records"),
    }
    return output, diagnostics, metadata


def contiguous_axis_folds(
    frame: pd.DataFrame, axis: np.ndarray, n_folds: int
) -> np.ndarray:
    coordinates = frame[["x_m", "y_m"]].to_numpy(dtype=float)
    projection = coordinates @ normalise_axis(axis, np.asarray([1.0, -1.0], dtype=float))
    order = np.lexsort((coordinates[:, 0], coordinates[:, 1], projection))
    sizes = np.full(n_folds, len(frame) // n_folds, dtype=int)
    sizes[: len(frame) % n_folds] += 1
    fold = np.empty(len(frame), dtype=int)
    start = 0
    for fold_number, size in enumerate(sizes, start=1):
        fold[order[start : start + size]] = fold_number
        start += size
    return fold


def checkerboard_folds(
    frame: pd.DataFrame, axis: np.ndarray, block_size_km: float, n_folds: int
) -> np.ndarray:
    coordinates = frame[["x_m", "y_m"]].to_numpy(dtype=float)
    along = normalise_axis(axis, np.asarray([1.0, -1.0], dtype=float))
    across_axis = np.asarray([-along[1], along[0]], dtype=float)
    u = coordinates @ along
    v = coordinates @ across_axis
    block_m = float(block_size_km) * 1000.0
    i = np.floor((u - np.min(u)) / block_m).astype(int)
    j = np.floor((v - np.min(v)) / block_m).astype(int)
    fold = 1 + ((i + 2 * j) % n_folds)
    if sorted(np.unique(fold).tolist()) != list(range(1, n_folds + 1)):
        raise RuntimeError("Checkerboard partition did not populate every fold")
    return fold


def add_training_masks(
    frame: pd.DataFrame, n_folds: int, buffer_km: float
) -> pd.DataFrame:
    output = frame.copy()
    coordinates = output[["x_m", "y_m"]].to_numpy(dtype=float)
    holdout_fold = output["new_fold"].to_numpy(dtype=int)
    for fold in range(1, n_folds + 1):
        heldout = holdout_fold == fold
        if not np.any(heldout):
            raise RuntimeError(f"Partition fold {fold} has no holdout cells")
        tree = cKDTree(coordinates[heldout])
        distance, _ = tree.query(coordinates, k=1)
        output[f"distance_to_fold_{fold}_m"] = distance
        output[f"train_included_fold_{fold}"] = (~heldout) & (
            distance > float(buffer_km) * 1000.0
        )
        if not np.any(output[f"train_included_fold_{fold}"]):
            raise RuntimeError(f"Partition fold {fold} has no training cells after buffering")
    return output


def build_partition_frame(
    frame: pd.DataFrame, partition: dict[str, Any], reference_fold: pd.Series | None = None
) -> pd.DataFrame:
    output = frame.copy()
    family = partition["family"]
    n_folds = int(partition["n_folds"])
    axis = output[["belt_axis_x", "belt_axis_y"]].iloc[0].to_numpy(dtype=float)
    if family == "belt_axis":
        generated = contiguous_axis_folds(output, axis, n_folds)
        if reference_fold is not None:
            reference = output["support_id"].map(reference_fold).to_numpy(dtype=int)
            if not np.array_equal(generated, reference):
                raise RuntimeError("Rebuilt belt-axis folds do not reproduce Stage 4")
            fold = reference
        else:
            fold = generated
    elif family == "checkerboard":
        fold = checkerboard_folds(
            output, axis, float(partition["block_size_km"]), n_folds
        )
    else:
        raise ValueError(f"Unknown partition family: {family}")
    output["new_fold"] = fold
    output["new_fold_label"] = [f"Fold {value}" for value in fold]
    output = add_training_masks(output, n_folds, float(partition["buffer_km"]))
    output["partition_id"] = partition["id"]
    output["partition_label"] = partition["label"]
    output["partition_family"] = family
    output["partition_buffer_km"] = float(partition["buffer_km"])
    return output


def partition_definition(
    frame: pd.DataFrame, partition: dict[str, Any], informative_min: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in range(1, int(partition["n_folds"]) + 1):
        holdout = frame["new_fold"].eq(fold)
        train = frame[f"train_included_fold_{fold}"].astype(bool)
        buffer = (~holdout) & (~train)
        rows.append({
            "partition_id": partition["id"],
            "partition_label": partition["label"],
            "partition_family": partition["family"],
            "partition_role": partition["role"],
            "buffer_km": float(partition["buffer_km"]),
            "fold": fold,
            "fold_label": f"Fold {fold}",
            "n_holdout": int(holdout.sum()),
            "holdout_area_km2": float(frame.loc[holdout, "tract_area_km2"].sum()),
            "holdout_records": int(frame.loc[holdout, "n_records"].sum()),
            "holdout_positive_cells": int(frame.loc[holdout, "n_records"].gt(0).sum()),
            "holdout_prevalence": float(frame.loc[holdout, "n_records"].gt(0).mean()),
            "n_train": int(train.sum()),
            "train_records": int(frame.loc[train, "n_records"].sum()),
            "train_positive_cells": int(frame.loc[train, "n_records"].gt(0).sum()),
            "n_buffer_excluded": int(buffer.sum()),
            "buffer_excluded_records": int(frame.loc[buffer, "n_records"].sum()),
            "informative_fold": bool(
                int(frame.loc[holdout, "n_records"].gt(0).sum()) >= informative_min
            ),
        })
    return pd.DataFrame(rows)


def weight_specs(config: dict[str, Any]) -> list[WeightSpec]:
    specs: list[WeightSpec] = []
    for value in config["residuals"]["k_neighbors"]:
        specs.append(WeightSpec(
            id=f"knn_{int(value)}",
            kind="knn",
            value=float(value),
            label=f"{int(value)} nearest neighbours",
        ))
    for value in config["residuals"]["distance_bands_km"]:
        specs.append(WeightSpec(
            id=f"distance_{float(value):g}km",
            kind="distance",
            value=float(value),
            label=f"Within {float(value):g} km",
        ))
    return specs


def spatial_weights(
    coordinates: np.ndarray, spec: WeightSpec
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    coordinates = np.asarray(coordinates, dtype=float)
    n = len(coordinates)
    tree = cKDTree(coordinates)
    if spec.kind == "knn":
        k = int(spec.value)
        if k < 1 or k >= n:
            raise ValueError(f"Invalid k-neighbour weight: {k}")
        _, neighbours = tree.query(coordinates, k=k + 1)
        neighbours = np.asarray(neighbours[:, 1:], dtype=int)
        rows = np.repeat(np.arange(n), k)
        cols = neighbours.reshape(-1)
        data = np.ones(n * k, dtype=float)
    elif spec.kind == "distance":
        radius_m = float(spec.value) * 1000.0
        neighbours = tree.query_ball_point(coordinates, r=radius_m)
        row_parts: list[np.ndarray] = []
        col_parts: list[np.ndarray] = []
        data_parts: list[np.ndarray] = []
        for index, row in enumerate(neighbours):
            row_values = np.asarray(sorted(row), dtype=int)
            row_parts.append(np.full(len(row_values), index, dtype=int))
            col_parts.append(row_values)
            data_parts.append(np.ones(len(row_values), dtype=float))
        rows = np.concatenate(row_parts)
        cols = np.concatenate(col_parts)
        data = np.concatenate(data_parts)
    else:
        raise ValueError(f"Unknown spatial weight kind: {spec.kind}")
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    matrix.sum_duplicates()
    matrix.setdiag(0.0)
    matrix.eliminate_zeros()
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    isolated = np.flatnonzero(row_sums <= 0.0)
    if len(isolated):
        _, nearest = tree.query(coordinates[isolated], k=2)
        fallback = np.asarray(nearest[:, 1], dtype=int)
        matrix = matrix.tolil()
        for row, column in zip(isolated, fallback):
            matrix[row, column] = 1.0
        matrix = matrix.tocsr()
        row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    matrix = sparse.diags(1.0 / row_sums) @ matrix
    matrix = matrix.tocsr()
    degree = np.asarray(matrix.getnnz(axis=1), dtype=int)
    return matrix, {
        "spatial_weight": spec.id,
        "spatial_weight_label": spec.label,
        "weight_kind": spec.kind,
        "weight_parameter": float(spec.value),
        "mean_neighbours": float(np.mean(degree)),
        "min_neighbours": int(np.min(degree)),
        "max_neighbours": int(np.max(degree)),
        "n_isolates": 0,
        "n_nearest_neighbour_fallback": int(len(isolated)),
    }


def moran_test(
    residuals: np.ndarray,
    weights: sparse.csr_matrix,
    n_permutations: int,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(residuals, dtype=float)
    valid = np.isfinite(values)
    values = values[valid]
    matrix = weights[valid][:, valid].tocsr()
    n = len(values)
    if n < 4:
        return {
            "n": n, "moran_i": math.nan, "expected_i": math.nan,
            "permutation_sd": math.nan, "z_permutation": math.nan,
            "p_positive": math.nan, "p_negative": math.nan, "p_two_sided": math.nan,
        }
    centered = values - np.mean(values)
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        raise RuntimeError("Zero residual variance in Moran test")
    observed = float(np.dot(centered, matrix @ centered) / denominator)
    rng = np.random.default_rng(seed)
    permuted = np.empty((n, int(n_permutations)), dtype=float)
    for column in range(int(n_permutations)):
        permuted[:, column] = centered[rng.permutation(n)]
    lags = matrix @ permuted
    numerators = np.einsum("ij,ij->j", permuted, lags)
    denominators = np.einsum("ij,ij->j", permuted, permuted)
    draws = numerators / denominators
    finite = draws[np.isfinite(draws)]
    positive_count = int(np.sum(finite >= observed))
    negative_count = int(np.sum(finite <= observed))
    p_positive = (positive_count + 1.0) / (len(finite) + 1.0)
    p_negative = (negative_count + 1.0) / (len(finite) + 1.0)
    sd = float(np.std(finite, ddof=1))
    return {
        "n": n,
        "moran_i": observed,
        "expected_i": -1.0 / (n - 1.0),
        "permutation_sd": sd,
        "z_permutation": float((observed - np.mean(finite)) / sd) if sd > 0 else math.nan,
        "p_positive": p_positive,
        "p_negative": p_negative,
        "p_two_sided": min(1.0, 2.0 * min(p_positive, p_negative)),
    }


def bh_qvalues(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    n = len(values)
    order = np.argsort(values)
    ranked = values[order]
    q_values = ranked * n / np.arange(1, n + 1)
    q_values = np.minimum.accumulate(q_values[::-1])[::-1]
    output = np.empty(n, dtype=float)
    output[order] = np.clip(q_values, 0.0, 1.0)
    return output


def local_moran(
    frame: pd.DataFrame,
    weights: sparse.csr_matrix,
    n_permutations: int,
    seed: int,
    fdr: float,
) -> pd.DataFrame:
    residuals = frame["pearson_residual"].to_numpy(dtype=float)
    valid = np.isfinite(residuals)
    if not np.all(valid):
        raise RuntimeError("Local Moran requires complete residual values")
    matrix = weights.tocsr()
    centered = residuals - np.mean(residuals)
    m2 = float(np.dot(centered, centered) / len(centered))
    lag = matrix @ centered
    local_i = centered * lag / m2
    rng = np.random.default_rng(seed)
    permuted = np.empty((len(centered), int(n_permutations)), dtype=float)
    for column in range(int(n_permutations)):
        permuted[:, column] = centered[rng.permutation(len(centered))]
    permuted_local = permuted * (matrix @ permuted) / m2
    p_high = (np.sum(permuted_local >= local_i[:, None], axis=1) + 1.0) / (n_permutations + 1.0)
    p_low = (np.sum(permuted_local <= local_i[:, None], axis=1) + 1.0) / (n_permutations + 1.0)
    p_two = np.minimum(1.0, 2.0 * np.minimum(p_high, p_low))

    n = len(centered)
    other_count = n - 1
    total = float(np.sum(centered))
    total_sq = float(np.dot(centered, centered))
    other_mean = (total - centered) / other_count
    other_ss = np.maximum(total_sq - centered**2, 0.0)
    other_variance = np.maximum(other_ss / other_count - other_mean**2, 1e-15)
    weight_sq_sum = np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
    lag_variance = (
        other_variance
        * other_count
        / max(other_count - 1.0, 1.0)
        * np.maximum(weight_sq_sum - 1.0 / other_count, 0.0)
    )
    expected_i = centered * other_mean / m2
    sd_i = np.abs(centered) * np.sqrt(lag_variance) / m2
    z_score = np.divide(
        local_i - expected_i, sd_i,
        out=np.zeros_like(local_i), where=sd_i > 0.0,
    )
    p_high_normal = stats.norm.sf(z_score)
    p_low_normal = stats.norm.cdf(z_score)
    p_two_normal = np.minimum(1.0, 2.0 * np.minimum(p_high_normal, p_low_normal))
    q_two_normal = bh_qvalues(p_two_normal)
    significant = (q_two_normal < float(fdr)) & (p_two < 0.05)
    cluster = np.full(len(frame), "not_significant", dtype=object)
    cluster[significant & (local_i > 0) & (centered > 0)] = "high_high"
    cluster[significant & (local_i > 0) & (centered < 0)] = "low_low"
    cluster[significant & (local_i < 0) & (centered > 0)] = "high_low"
    cluster[significant & (local_i < 0) & (centered < 0)] = "low_high"
    output = frame[[
        "support_id", "x_m", "y_m", "new_fold", "domain_abbreviation",
        "tract_area_km2", "observed_count", "predicted_count", "pearson_residual",
    ]].copy()
    output = output.rename(columns={"observed_count": "n_records"})
    output["residual_z"] = centered
    output["spatial_lag_z"] = lag
    output["local_moran_i"] = local_i
    output["p_high"] = p_high
    output["p_low"] = p_low
    output["p_two_sided"] = p_two
    output["p_high_normal"] = p_high_normal
    output["p_low_normal"] = p_low_normal
    output["p_two_sided_normal"] = p_two_normal
    output["q_two_sided_bh"] = q_two_normal
    output["significant_fdr"] = significant
    output["cluster_class"] = cluster
    return output


def randomized_quantile_residuals(
    observed: np.ndarray,
    mu: np.ndarray,
    theta: np.ndarray,
    uniform_draws: np.ndarray,
) -> np.ndarray:
    y = np.asarray(observed, dtype=float)
    mean = np.asarray(mu, dtype=float)
    dispersion = np.asarray(theta, dtype=float)
    probability = dispersion / (dispersion + mean)
    lower = stats.nbinom.cdf(y - 1.0, dispersion, probability)
    upper = stats.nbinom.cdf(y, dispersion, probability)
    lower = np.where(y <= 0.0, 0.0, lower)
    widths = np.maximum(upper - lower, 0.0)
    uniform = lower + np.asarray(uniform_draws, dtype=float) * widths
    uniform = np.clip(uniform, 1e-10, 1.0 - 1e-10)
    return stats.norm.ppf(uniform)


def residual_summaries(
    predictions: pd.DataFrame,
    tract_union: Any,
    config: dict[str, Any],
) -> pd.DataFrame:
    frame = predictions.copy()
    points = gpd.GeoSeries(
        gpd.points_from_xy(frame["x_m"], frame["y_m"]), crs=PROJECT_CRS
    )
    frame["tract_boundary_distance_km"] = distance_to_boundary_km(points, tract_union)
    boundary_thresholds = [-0.001] + [
        float(value) for value in config["residuals"]["boundary_thresholds_km"]
    ] + [math.inf]
    boundary_labels = [
        "0-1 km", "1-2 km", "2-5 km", "5-10 km", ">=10 km"
    ]
    frame["tract_boundary_class"] = pd.cut(
        frame["tract_boundary_distance_km"],
        bins=boundary_thresholds,
        labels=boundary_labels,
        right=False,
        include_lowest=True,
    ).astype(str)
    area_thresholds = [float(value) for value in config["residuals"]["area_classes_km2"]]
    area_labels = ["0-5 km2", "5-10 km2", "10-15 km2", "15-25 km2"]
    frame["support_area_class"] = pd.cut(
        frame["tract_area_km2"],
        bins=area_thresholds,
        labels=area_labels,
        right=False,
        include_lowest=True,
    ).astype(str)
    frame["mapped_coverage"] = np.where(
        frame["domain_abbreviation"].eq("unassigned"),
        "outside_Daly_envelope",
        "inside_Daly_envelope",
    )
    grouping_specs = {
        "fold": "new_fold",
        "domain": "domain_abbreviation",
        "tract_boundary_distance": "tract_boundary_class",
        "support_area": "support_area_class",
        "mapped_coverage": "mapped_coverage",
    }
    rows: list[dict[str, Any]] = []
    for grouping, column in grouping_specs.items():
        for value, group in frame.groupby(column, observed=True, sort=True):
            if group.empty:
                continue
            observed = float(group["observed_count"].sum())
            predicted = float(group["predicted_count"].sum())
            rows.append({
                "grouping": grouping,
                "group_value": str(value),
                "n_support_cells": len(group),
                "area_km2": float(group["tract_area_km2"].sum()),
                "n_positive_cells": int(group["observed_any"].sum()),
                "observed_records": observed,
                "predicted_records": predicted,
                "observed_over_predicted": predicted / observed if observed > 0 else math.nan,
                "observed_records_per_1000_km2": 1000.0 * observed / group["tract_area_km2"].sum(),
                "predicted_records_per_1000_km2": 1000.0 * predicted / group["tract_area_km2"].sum(),
                "mean_pearson_residual": float(group["pearson_residual"].mean()),
                "sd_pearson_residual": float(group["pearson_residual"].std(ddof=1)),
                "mean_rqr": float(group["rqr_residual"].mean()),
                "sd_rqr": float(group["rqr_residual"].std(ddof=1)),
                "median_tract_boundary_distance_km": float(
                    group["tract_boundary_distance_km"].median()
                ),
            })
    return pd.DataFrame(rows)


def informative_fold_mean(fold_metrics: pd.DataFrame, min_positive_cells: int) -> dict[str, float]:
    informative = fold_metrics.loc[
        fold_metrics["holdout_positive_cells"].ge(int(min_positive_cells))
    ]
    metrics = [
        "roc_auc", "average_precision", "ap_over_prevalence", "brier_skill",
        "calibration_slope", "nb_deviance",
    ]
    return {
        f"informative_{metric}_mean": float(informative[metric].mean(skipna=True))
        for metric in metrics
    }


def safe_write_json(payload: dict[str, Any], path: str | Path) -> Path:
    return write_json(payload, path)


def safe_write_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    return write_csv(frame, path)
