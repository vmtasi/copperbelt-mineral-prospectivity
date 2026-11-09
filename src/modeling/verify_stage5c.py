"""Independently verify the Stage 5C residual and spatial-transfer experiment.

The verifier reconstructs the partitions, fixed spatial basis and residual
diagnostics, and refits the accepted reference and partition models. Stage 5C
implementation helpers are not imported by this audit.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import struct
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import shapely
from scipy import sparse, stats
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modeling.fit_stage4_models import (  # noqa: E402
    Experiment,
    binary_calibration,
    compute_point_metrics,
    fit_experiment,
)
from src.modeling.stage4_common import (  # noqa: E402
    abs_path,
    file_record,
    relative_posix,
    sha256_file,
    verify_protected,
    write_json,
)
from src.modeling.stage5_common import (  # noqa: E402
    load_config as load_stage5_config,
    load_geometries,
    load_stage4_bundle,
    software_stack,
)


POINT_METRICS = [
    "roc_auc", "average_precision", "ap_over_prevalence", "brier_score",
    "brier_skill", "binary_log_loss", "nb_deviance", "mae", "rmse",
    "predicted_total", "observed_total", "total_count_ratio",
    "calibration_intercept", "calibration_slope", "pi_coverage",
    "mean_pi_width",
]
PREDICTION_COLUMNS = [
    "observed_count", "observed_any", "predicted_count",
    "predicted_rate_per_km2", "predicted_any_probability", "pi_low_count",
    "pi_high_count", "pearson_residual", "theta", "baseline_train_probability",
]


def record_check(checks: list[dict[str, Any]], name: str, passed: bool, details: Any) -> None:
    checks.append({"name": name, "status": "pass" if passed else "fail", "details": details})


def capture(checks: list[dict[str, Any]], name: str, function: Callable[[], Any]) -> Any:
    try:
        details = function()
        record_check(checks, name, True, details)
        return details
    except Exception as error:  # noqa: BLE001 - report every independent check
        record_check(checks, name, False, {"error": f"{type(error).__name__}: {error}"})
        return None


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.DataFrame):
        return {"rows": int(len(value)), "columns": [str(column) for column in value.columns]}
    if isinstance(value, pd.Series):
        return {"length": int(len(value)), "name": str(value.name)}
    if isinstance(value, pd.Index):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        item = value.item()
        if not isinstance(item, type(value)):
            return json_safe(item)
    return value


def read_json(path: str | Path) -> dict[str, Any]:
    with abs_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def assert_close(observed: float, expected: float, tolerance: float = 1e-8, label: str = "value") -> None:
    observed = float(observed)
    expected = float(expected)
    if math.isnan(observed) and math.isnan(expected):
        return
    if not math.isclose(observed, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(
            f"{label} differs: observed={observed:.17g}, expected={expected:.17g}"
        )


def verify_records(records: dict[str, Any]) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    for name, record in records.items():
        path = abs_path(record["path"])
        if not path.exists():
            mismatches.append({"name": name, "path": record["path"], "error": "missing"})
            continue
        observed = sha256_file(path)
        if observed.lower() != str(record["sha256"]).lower():
            mismatches.append({"name": name, "path": record["path"], "error": "sha256 mismatch",
                               "expected": record["sha256"], "observed": observed})
        if int(record.get("bytes", -1)) != path.stat().st_size:
            mismatches.append({"name": name, "path": record["path"], "error": "byte-size mismatch",
                               "expected": int(record.get("bytes", -1)), "observed": path.stat().st_size})
    if mismatches:
        raise RuntimeError(f"{len(mismatches)} file records failed verification")
    return {"checked": len(records), "mismatches": mismatches}


def verify_png(path: str | Path) -> dict[str, Any]:
    absolute = abs_path(path)
    with absolute.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Not a readable PNG: {absolute}")
    width, height = struct.unpack(">II", header[16:24])
    if width < 500 or height < 500:
        raise ValueError(f"PNG dimensions are implausibly small: {absolute} ({width}x{height})")
    return {"path": relative_posix(absolute), "width": width, "height": height,
            "bytes": absolute.stat().st_size}


def normalise_axis(axis: np.ndarray, reference: np.ndarray) -> np.ndarray:
    vector = np.asarray(axis, dtype=float)
    vector = vector / np.linalg.norm(vector)
    reference = np.asarray(reference, dtype=float)
    reference = reference / np.linalg.norm(reference)
    if float(np.dot(vector, reference)) < 0.0:
        vector = -vector
    return vector


def contiguous_folds(frame: pd.DataFrame, n_folds: int) -> np.ndarray:
    axis = normalise_axis(frame[["belt_axis_x", "belt_axis_y"]].iloc[0].to_numpy(dtype=float),
                          np.asarray([1.0, -1.0]))
    coordinates = frame[["x_m", "y_m"]].to_numpy(dtype=float)
    projection = coordinates @ axis
    order = np.lexsort((coordinates[:, 0], coordinates[:, 1], projection))
    sizes = np.full(n_folds, len(frame) // n_folds, dtype=int)
    sizes[: len(frame) % n_folds] += 1
    fold = np.empty(len(frame), dtype=int)
    start = 0
    for number, size in enumerate(sizes, start=1):
        fold[order[start : start + size]] = number
        start += size
    return fold


def checkerboard_folds(frame: pd.DataFrame, block_size_km: float, n_folds: int) -> np.ndarray:
    axis = normalise_axis(frame[["belt_axis_x", "belt_axis_y"]].iloc[0].to_numpy(dtype=float),
                          np.asarray([1.0, -1.0]))
    cross = np.asarray([-axis[1], axis[0]], dtype=float)
    coordinates = frame[["x_m", "y_m"]].to_numpy(dtype=float)
    along = coordinates @ axis
    across = coordinates @ cross
    block_m = float(block_size_km) * 1000.0
    i = np.floor((along - np.min(along)) / block_m).astype(int)
    j = np.floor((across - np.min(across)) / block_m).astype(int)
    fold = 1 + ((i + 2 * j) % n_folds)
    if sorted(np.unique(fold).tolist()) != list(range(1, n_folds + 1)):
        raise AssertionError("Checkerboard did not populate every fold")
    return fold


def partition_frame(frame: pd.DataFrame, specification: dict[str, Any],
                    reference_fold: np.ndarray) -> pd.DataFrame:
    output = frame.copy()
    n_folds = int(specification["n_folds"])
    if specification["family"] == "belt_axis":
        fold = reference_fold.copy()
    elif specification["family"] == "checkerboard":
        fold = checkerboard_folds(output, float(specification["block_size_km"]), n_folds)
    else:
        raise ValueError(f"Unknown partition family: {specification['family']}")
    output["new_fold"] = fold
    output["new_fold_label"] = [f"Fold {value}" for value in fold]
    coordinates = output[["x_m", "y_m"]].to_numpy(dtype=float)
    for number in range(1, n_folds + 1):
        heldout = fold == number
        tree = cKDTree(coordinates[heldout])
        distance, _ = tree.query(coordinates, k=1)
        output[f"distance_to_fold_{number}_m"] = distance
        output[f"train_included_fold_{number}"] = (~heldout) & (
            distance > float(specification["buffer_km"]) * 1000.0
        )
    output["partition_id"] = specification["id"]
    output["partition_label"] = specification["label"]
    output["partition_family"] = specification["family"]
    output["partition_buffer_km"] = float(specification["buffer_km"])
    return output


def partition_definition(frame: pd.DataFrame, specification: dict[str, Any],
                         informative_min: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for number in range(1, int(specification["n_folds"]) + 1):
        holdout = frame["new_fold"].eq(number)
        train = frame[f"train_included_fold_{number}"].astype(bool)
        buffer = (~holdout) & (~train)
        rows.append({
            "partition_id": specification["id"], "partition_label": specification["label"],
            "partition_family": specification["family"], "partition_role": specification["role"],
            "buffer_km": float(specification["buffer_km"]), "fold": number,
            "fold_label": f"Fold {number}", "n_holdout": int(holdout.sum()),
            "holdout_area_km2": float(frame.loc[holdout, "tract_area_km2"].sum()),
            "holdout_records": int(frame.loc[holdout, "n_records"].sum()),
            "holdout_positive_cells": int(frame.loc[holdout, "n_records"].gt(0).sum()),
            "holdout_prevalence": float(frame.loc[holdout, "n_records"].gt(0).mean()),
            "n_train": int(train.sum()),
            "train_records": int(frame.loc[train, "n_records"].sum()),
            "train_positive_cells": int(frame.loc[train, "n_records"].gt(0).sum()),
            "n_buffer_excluded": int(buffer.sum()),
            "buffer_excluded_records": int(frame.loc[buffer, "n_records"].sum()),
            "informative_fold": bool(int(frame.loc[holdout, "n_records"].gt(0).sum()) >= informative_min),
        })
    return pd.DataFrame(rows)


def independent_spatial_features(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    output = frame.copy()
    axis = normalise_axis(output[["belt_axis_x", "belt_axis_y"]].iloc[0].to_numpy(dtype=float),
                          np.asarray([1.0, -1.0]))
    cross = np.asarray([-axis[1], axis[0]], dtype=float)
    coordinates = output[["x_m", "y_m"]].to_numpy(dtype=float)
    belt = coordinates @ axis
    across = coordinates @ cross
    center = coordinates.mean(axis=0)
    x = (coordinates[:, 0] - center[0]) / 100000.0
    y = (coordinates[:, 1] - center[1]) / 100000.0
    belt_scaled = (belt - belt.mean()) / 100000.0
    across_scaled = (across - across.mean()) / 100000.0
    output["SPATIAL_X"], output["SPATIAL_Y"] = x, y
    output["SPATIAL_X2"], output["SPATIAL_XY"], output["SPATIAL_Y2"] = x**2, x * y, y**2
    output["SPATIAL_BELT"], output["SPATIAL_BELT2"], output["SPATIAL_BELT3"] = belt_scaled, belt_scaled**2, belt_scaled**3
    output["SPATIAL_CROSS"] = across_scaled
    rbf = config["rbf"]
    bandwidth = float(rbf["bandwidth_km"]) / 100.0
    belt_centres = np.quantile(belt_scaled, np.linspace(0.10, 0.90, int(rbf["n_belt_centres"])))
    cross_centres = np.quantile(across_scaled, np.linspace(0.15, 0.85, int(rbf["n_cross_centres"])))
    diagnostics: list[dict[str, Any]] = []
    index = 0
    for belt_center in belt_centres:
        for cross_center in cross_centres:
            index += 1
            name = f"SPATIAL_RBF_{index:02d}"
            distance_sq = (belt_scaled - belt_center) ** 2 + (across_scaled - cross_center) ** 2
            output[name] = np.exp(-0.5 * distance_sq / bandwidth**2)
            diagnostics.append({"feature": name, "belt_center_scaled_100km": float(belt_center),
                                "cross_center_scaled_100km": float(cross_center),
                                "bandwidth_km": float(rbf["bandwidth_km"])})
    metadata = {"belt_axis": axis.tolist(), "cross_axis": cross.tolist(),
                "coordinate_center_m": center.tolist(), "rbf_feature_count": index,
                "rbf_bandwidth_km": float(rbf["bandwidth_km"])}
    return output, pd.DataFrame(diagnostics), metadata


def spatial_feature_table(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    features = sorted({term for model in config["spatial_models"] for term in model["terms"]})
    correlation = frame[features].corr().abs()
    rows: list[dict[str, Any]] = []
    for feature in features:
        values = frame[feature].to_numpy(dtype=float)
        others = correlation.loc[feature].drop(labels=[feature])
        rows.append({"feature": feature, "n": len(values), "min": float(np.min(values)),
                     "q01": float(np.quantile(values, 0.01)), "median": float(np.median(values)),
                     "q99": float(np.quantile(values, 0.99)), "max": float(np.max(values)),
                     "sd": float(np.std(values, ddof=0)),
                     "max_abs_correlation_with_other_spatial_feature": float(others.max()) if len(others) else math.nan})
    return pd.DataFrame(rows)


def make_uniforms(frame: pd.DataFrame, seed: int) -> pd.Series:
    ordered = frame.sort_values("support_id").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    return pd.Series(rng.uniform(size=len(ordered)), index=ordered["support_id"], dtype=float)


def add_rqr(frame: pd.DataFrame, uniforms: pd.Series) -> pd.DataFrame:
    output = frame.copy()
    y = output["observed_count"].to_numpy(dtype=float)
    mean = output["predicted_count"].to_numpy(dtype=float)
    theta = output["theta"].to_numpy(dtype=float)
    probability = theta / (theta + mean)
    lower = stats.nbinom.cdf(y - 1.0, theta, probability)
    upper = stats.nbinom.cdf(y, theta, probability)
    lower = np.where(y <= 0.0, 0.0, lower)
    widths = np.maximum(upper - lower, 0.0)
    draws = uniforms.reindex(output["support_id"]).to_numpy(dtype=float)
    quantile = np.clip(lower + draws * widths, 1e-10, 1.0 - 1e-10)
    output["rqr_residual"] = stats.norm.ppf(quantile)
    return output


def spatial_weights(coordinates: np.ndarray, weight_id: str) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    coordinates = np.asarray(coordinates, dtype=float)
    n = len(coordinates)
    tree = cKDTree(coordinates)
    if weight_id.startswith("knn_"):
        k = int(weight_id.split("_", 1)[1])
        _, neighbours = tree.query(coordinates, k=k + 1)
        neighbours = np.asarray(neighbours[:, 1:], dtype=int)
        rows = np.repeat(np.arange(n), k)
        cols = neighbours.reshape(-1)
        data = np.ones(n * k, dtype=float)
    elif weight_id.startswith("distance_"):
        radius_km = float(weight_id.removeprefix("distance_").removesuffix("km"))
        neighbours = tree.query_ball_point(coordinates, r=radius_km * 1000.0)
        rows_parts, cols_parts, data_parts = [], [], []
        for index, row in enumerate(neighbours):
            values = np.asarray(sorted(row), dtype=int)
            rows_parts.append(np.full(len(values), index, dtype=int))
            cols_parts.append(values)
            data_parts.append(np.ones(len(values), dtype=float))
        rows = np.concatenate(rows_parts)
        cols = np.concatenate(cols_parts)
        data = np.concatenate(data_parts)
    else:
        raise ValueError(f"Unknown spatial weight: {weight_id}")
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    matrix.sum_duplicates()
    matrix.setdiag(0.0)
    matrix.eliminate_zeros()
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    isolated = np.flatnonzero(row_sums <= 0.0)
    if len(isolated):
        _, nearest = tree.query(coordinates[isolated], k=2)
        matrix = matrix.tolil()
        for row, column in zip(isolated, np.asarray(nearest[:, 1], dtype=int)):
            matrix[row, column] = 1.0
        matrix = matrix.tocsr()
        row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    matrix = sparse.diags(1.0 / row_sums) @ matrix
    degree = np.asarray(matrix.getnnz(axis=1), dtype=int)
    return matrix.tocsr(), {
        "mean_neighbours": float(np.mean(degree)),
        "min_neighbours": int(np.min(degree)),
        "max_neighbours": int(np.max(degree)),
        "n_nearest_neighbour_fallback": int(len(isolated)),
    }


def moran_observed(values: np.ndarray, weights: sparse.csr_matrix) -> float:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    values = values[valid]
    matrix = weights[valid][:, valid].tocsr()
    centered = values - np.mean(values)
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        raise ValueError("Zero residual variance in Moran statistic")
    return float(np.dot(centered, matrix @ centered) / denominator)


def moran_permutation(values: np.ndarray, weights: sparse.csr_matrix,
                      n_permutations: int, seed: int) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    values = values[valid]
    matrix = weights[valid][:, valid].tocsr()
    centered = values - np.mean(values)
    denominator = float(np.dot(centered, centered))
    observed = float(np.dot(centered, matrix @ centered) / denominator)
    rng = np.random.default_rng(seed)
    permuted = np.empty((len(values), int(n_permutations)), dtype=float)
    for column in range(int(n_permutations)):
        permuted[:, column] = centered[rng.permutation(len(values))]
    lags = matrix @ permuted
    draws = np.einsum("ij,ij->j", permuted, lags) / np.einsum("ij,ij->j", permuted, permuted)
    finite = draws[np.isfinite(draws)]
    sd = float(np.std(finite, ddof=1))
    return {
        "moran_i": observed,
        "p_positive": (int(np.sum(finite >= observed)) + 1.0) / (len(finite) + 1.0),
        "z_permutation": float((observed - np.mean(finite)) / sd) if sd > 0.0 else math.nan,
    }


def moran_test_independent(values: np.ndarray, weights: sparse.csr_matrix,
                           n_permutations: int, seed: int) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    values = values[valid]
    matrix = weights[valid][:, valid].tocsr()
    n = len(values)
    if n < 4:
        return {key: math.nan for key in (
            "n", "moran_i", "expected_i", "permutation_sd", "z_permutation",
            "p_positive", "p_negative", "p_two_sided",
        )}
    centered = values - np.mean(values)
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        raise ValueError("Zero residual variance in Moran test")
    observed = float(np.dot(centered, matrix @ centered) / denominator)
    if int(n_permutations) == 0:
        return {
            "n": n, "moran_i": observed, "expected_i": -1.0 / (n - 1.0),
            "permutation_sd": math.nan, "z_permutation": math.nan,
            "p_positive": math.nan, "p_negative": math.nan, "p_two_sided": math.nan,
        }
    rng = np.random.default_rng(seed)
    permuted = np.empty((n, int(n_permutations)), dtype=float)
    for column in range(int(n_permutations)):
        permuted[:, column] = centered[rng.permutation(n)]
    lags = matrix @ permuted
    draws = (np.einsum("ij,ij->j", permuted, lags)
             / np.einsum("ij,ij->j", permuted, permuted))
    finite = draws[np.isfinite(draws)]
    sd = float(np.std(finite, ddof=1))
    p_positive = (int(np.sum(finite >= observed)) + 1.0) / (len(finite) + 1.0)
    p_negative = (int(np.sum(finite <= observed)) + 1.0) / (len(finite) + 1.0)
    return {
        "n": n,
        "moran_i": observed,
        "expected_i": -1.0 / (n - 1.0),
        "permutation_sd": sd,
        "z_permutation": float((observed - np.mean(finite)) / sd) if sd > 0.0 else math.nan,
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


def local_clusters(frame: pd.DataFrame, weights: sparse.csr_matrix, n_permutations: int,
                   seed: int, fdr: float) -> pd.DataFrame:
    residuals = frame["pearson_residual"].to_numpy(dtype=float)
    if not np.all(np.isfinite(residuals)):
        raise ValueError("Local Moran residuals must all be finite")
    centered = residuals - np.mean(residuals)
    m2 = float(np.dot(centered, centered) / len(centered))
    lag = weights @ centered
    local_i = centered * lag / m2
    rng = np.random.default_rng(seed)
    permuted = np.empty((len(centered), int(n_permutations)), dtype=float)
    for column in range(int(n_permutations)):
        permuted[:, column] = centered[rng.permutation(len(centered))]
    permuted_local = permuted * (weights @ permuted) / m2
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
    weight_sq_sum = np.asarray(weights.multiply(weights).sum(axis=1)).ravel()
    lag_variance = (other_variance * other_count / max(other_count - 1.0, 1.0)
                    * np.maximum(weight_sq_sum - 1.0 / other_count, 0.0))
    expected_i = centered * other_mean / m2
    sd_i = np.abs(centered) * np.sqrt(lag_variance) / m2
    z_score = np.divide(local_i - expected_i, sd_i, out=np.zeros_like(local_i), where=sd_i > 0.0)
    p_high_normal = stats.norm.sf(z_score)
    p_low_normal = stats.norm.cdf(z_score)
    p_two_normal = np.minimum(1.0, 2.0 * np.minimum(p_high_normal, p_low_normal))
    q_normal = bh_qvalues(p_two_normal)
    significant = (q_normal < float(fdr)) & (p_two < 0.05)
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
    output["q_two_sided_bh"] = q_normal
    output["significant_fdr"] = significant
    output["cluster_class"] = cluster
    return output


def compare_predictions(expected: pd.DataFrame, actual: pd.DataFrame, label: str) -> dict[str, float]:
    expected = expected.sort_values("support_id").reset_index(drop=True)
    actual = actual.sort_values("support_id").reset_index(drop=True)
    if not expected["support_id"].astype(str).equals(actual["support_id"].astype(str)):
        raise AssertionError(f"{label}: support identifiers differ")
    if not expected["new_fold"].astype(int).equals(actual["new_fold"].astype(int)):
        raise AssertionError(f"{label}: fold assignments differ")
    differences: dict[str, float] = {}
    for column in PREDICTION_COLUMNS:
        left = expected[column].to_numpy(dtype=float)
        right = actual[column].to_numpy(dtype=float)
        if not np.allclose(left, right, rtol=1e-8, atol=1e-10, equal_nan=True):
            raise AssertionError(f"{label}/{column}: maximum difference {np.nanmax(np.abs(left-right)):.3e}")
        differences[column] = float(np.nanmax(np.abs(left - right)))
    return differences


def recompute_metrics(subset: pd.DataFrame) -> dict[str, float]:
    return compute_point_metrics(
        subset["observed_count"].to_numpy(dtype=float),
        subset["observed_any"].to_numpy(dtype=float),
        subset["predicted_count"].to_numpy(dtype=float),
        subset["predicted_any_probability"].to_numpy(dtype=float),
        subset["theta"].to_numpy(dtype=float),
        subset["tract_area_km2"].to_numpy(dtype=float),
        subset["predicted_rate_per_km2"].to_numpy(dtype=float),
        subset["baseline_train_probability"].to_numpy(dtype=float),
    )


def compare_metric_row(expected: pd.Series, actual: dict[str, float], label: str) -> float:
    maximum = 0.0
    for metric in POINT_METRICS:
        assert_close(actual[metric], expected[metric], tolerance=1e-8, label=f"{label}/{metric}")
        if np.isfinite(actual[metric]) and np.isfinite(expected[metric]):
            maximum = max(maximum, abs(actual[metric] - float(expected[metric])))
    return maximum


def compare_metric_tables(expected_fold: pd.DataFrame, expected_summary: pd.DataFrame,
                          fitted_fold: pd.DataFrame, fitted_summary: pd.DataFrame, label: str) -> float:
    maximum = 0.0
    for fold in (1, 2, 3, 4):
        left = expected_fold.loc[expected_fold["fold"].eq(fold)].iloc[0]
        right = fitted_fold.loc[fitted_fold["fold"].eq(fold)].iloc[0]
        for metric in POINT_METRICS:
            assert_close(right[metric], left[metric], tolerance=1e-8, label=f"{label}/fold_{fold}/{metric}")
            if np.isfinite(left[metric]) and np.isfinite(right[metric]):
                maximum = max(maximum, abs(left[metric] - right[metric]))
    for aggregation in ("pooled_oof", "fold_mean"):
        left = expected_summary.loc[expected_summary["aggregation"].eq(aggregation)].iloc[0]
        right = fitted_summary.loc[fitted_summary["aggregation"].eq(aggregation)].iloc[0]
        for metric in POINT_METRICS:
            assert_close(right[metric], left[metric], tolerance=1e-8, label=f"{label}/{aggregation}/{metric}")
            if np.isfinite(left[metric]) and np.isfinite(right[metric]):
                maximum = max(maximum, abs(left[metric] - right[metric]))
    return maximum


def residual_summary_independent(predictions: pd.DataFrame, tract_union: shapely.Geometry,
                                 config: dict[str, Any]) -> pd.DataFrame:
    frame = predictions.copy()
    boundary = tract_union.boundary
    frame["tract_boundary_distance_km"] = np.asarray([
        shapely.distance(shapely.Point(float(x), float(y)), boundary) / 1000.0
        for x, y in zip(frame["x_m"], frame["y_m"])
    ], dtype=float)
    frame["tract_boundary_class"] = pd.cut(
        frame["tract_boundary_distance_km"],
        bins=[-0.001] + [float(value) for value in config["residuals"]["boundary_thresholds_km"]] + [math.inf],
        labels=["0-1 km", "1-2 km", "2-5 km", "5-10 km", ">=10 km"],
        right=False, include_lowest=True,
    ).astype(str)
    frame["support_area_class"] = pd.cut(
        frame["tract_area_km2"],
        bins=[float(value) for value in config["residuals"]["area_classes_km2"]],
        labels=["0-5 km2", "5-10 km2", "10-15 km2", "15-25 km2"],
        right=False, include_lowest=True,
    ).astype(str)
    frame["mapped_coverage"] = np.where(frame["domain_abbreviation"].eq("unassigned"),
                                        "outside_Daly_envelope", "inside_Daly_envelope")
    specifications = {"fold": "new_fold", "domain": "domain_abbreviation",
                      "tract_boundary_distance": "tract_boundary_class",
                      "support_area": "support_area_class", "mapped_coverage": "mapped_coverage"}
    rows: list[dict[str, Any]] = []
    for grouping, column in specifications.items():
        for value, group in frame.groupby(column, observed=True, sort=True):
            observed = float(group["observed_count"].sum())
            predicted = float(group["predicted_count"].sum())
            rows.append({
                "grouping": grouping, "group_value": str(value), "n_support_cells": len(group),
                "area_km2": float(group["tract_area_km2"].sum()),
                "n_positive_cells": int(group["observed_any"].sum()),
                "observed_records": observed, "predicted_records": predicted,
                "observed_over_predicted": predicted / observed if observed > 0.0 else math.nan,
                "observed_records_per_1000_km2": 1000.0 * observed / group["tract_area_km2"].sum(),
                "predicted_records_per_1000_km2": 1000.0 * predicted / group["tract_area_km2"].sum(),
                "mean_pearson_residual": float(group["pearson_residual"].mean()),
                "sd_pearson_residual": float(group["pearson_residual"].std(ddof=1)),
                "mean_rqr": float(group["rqr_residual"].mean()),
                "sd_rqr": float(group["rqr_residual"].std(ddof=1)),
                "median_tract_boundary_distance_km": float(group["tract_boundary_distance_km"].median()),
            })
    return pd.DataFrame(rows)


def compare_residual_summary(expected: pd.DataFrame, actual: pd.DataFrame) -> dict[str, Any]:
    keys = ["grouping", "group_value"]
    merged = expected.merge(actual, on=keys, how="outer", suffixes=("_expected", "_actual"), indicator=True)
    if not merged["_merge"].eq("both").all():
        raise AssertionError(f"Residual-summary groups differ: {merged.loc[merged['_merge'].ne('both'), keys].to_dict('records')}")
    numeric = [column for column in expected.columns if column not in keys and pd.api.types.is_numeric_dtype(expected[column])]
    maximum = 0.0
    for column in numeric:
        left = merged[f"{column}_expected"].to_numpy(dtype=float)
        right = merged[f"{column}_actual"].to_numpy(dtype=float)
        if not np.allclose(left, right, rtol=1e-8, atol=1e-10, equal_nan=True):
            raise AssertionError(f"Residual-summary field {column} differs")
        finite = np.isfinite(left) & np.isfinite(right)
        if finite.any():
            maximum = max(maximum, float(np.max(np.abs(left[finite] - right[finite]))))
    return {"groups": int(len(expected)), "maximum_numeric_difference": maximum}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage5c_v1.json")
    args = parser.parse_args()
    config_path = abs_path(args.config)
    config = read_json(config_path)
    processed = abs_path(config["outputs"]["processed_dir"])
    figures = abs_path(config["outputs"]["figure_dir"])
    manifest_path = processed / config["outputs"]["manifest"]
    validation_path = processed / config["outputs"]["validation"]
    manifest = read_json(manifest_path)
    original_manifest_sha = manifest.get("original_manifest_sha256") or sha256_file(manifest_path)
    checks: list[dict[str, Any]] = []

    bundle = load_stage4_bundle(load_stage5_config(config["stage5"]["config"]))
    geometries = load_geometries(load_stage5_config(config["stage5"]["config"]))
    frame = bundle["model_table"].copy().sort_values("support_id").reset_index(drop=True)
    fit_config = copy.deepcopy(bundle["stage4_config"])

    capture(checks, "manifest_output_hashes", lambda: verify_records(manifest["outputs"]))
    capture(checks, "manifest_input_hashes", lambda: verify_records(manifest["inputs"]))
    capture(checks, "config_hash", lambda: verify_records({"config": manifest["config"]}))
    for key in ("partition_figure", "partition_transfer_figure", "moran_figure",
                "local_cluster_figure", "residual_diagnostic_figure",
                "spatial_model_figure", "residual_maps_figure"):
        capture(checks, f"png_{key}", lambda key=key: verify_png(manifest["outputs"][key]["path"]))

    def stage4_protection() -> dict[str, Any]:
        protected = verify_protected(bundle["stage4_config"])
        if not all(record["hash_match"] for record in protected.values()):
            raise AssertionError("A protected Stage 4 or Stage 3 asset changed")
        return {"protected_assets": len(protected), "all_hash_matches": True,
                "build_id": bundle["model_manifest"]["build_id"]}
    capture(checks, "stage4_and_stage3_protection", stage4_protection)

    def stage5b_integrity() -> dict[str, Any]:
        stage5b_manifest = read_json(config["stage5b"]["manifest"])
        stage5b_validation = read_json(config["stage5b"]["validation"])
        if stage5b_manifest.get("status") != "verified":
            raise AssertionError("Stage 5B manifest is not verified")
        if stage5b_validation.get("status") != "pass" or int(stage5b_validation.get("n_failed", 1)) != 0:
            raise AssertionError("Stage 5B validation is not passing")
        if stage5b_manifest.get("validation", {}).get("sha256") != sha256_file(config["stage5b"]["validation"]):
            raise AssertionError("Stage 5B validation hash is stale")
        verify_records(stage5b_manifest["outputs"])
        if stage5b_manifest.get("build_id") != manifest.get("stage5b_reference"):
            raise AssertionError("Stage 5B reference build identifier changed")
        return {"build_id": stage5b_manifest["build_id"],
                "n_outputs_checked": len(stage5b_manifest["outputs"]),
                "validation_checks": stage5b_validation["n_checks"]}
    capture(checks, "stage5b_protection", stage5b_integrity)

    def support_counts() -> dict[str, Any]:
        if len(frame) != 2193 or not frame["support_id"].is_unique:
            raise AssertionError("The Stage 4 support table is not the expected 2,193-cell frame")
        records = int(frame["n_records"].sum())
        positive = int(frame["n_records"].gt(0).sum())
        if records != 184 or positive != 146:
            raise AssertionError(f"Expected 184 records in 146 cells; found {records} in {positive}")
        if frame["tract_area_km2"].isna().any() or frame["tract_area_km2"].le(0).any():
            raise AssertionError("Exposure is not complete and strictly positive")
        if sorted(frame["new_fold"].unique().tolist()) != [1, 2, 3, 4]:
            raise AssertionError("Stage 4 reference folds are not 1-4")
        return {"n_support": len(frame), "n_records": records,
                "n_positive_cells": positive, "prevalence": float(frame["n_records"].gt(0).mean())}
    capture(checks, "support_counts", support_counts)

    features = pd.DataFrame()
    feature_diagnostics = pd.DataFrame()
    feature_metadata: dict[str, Any] = {}

    def build_features() -> dict[str, Any]:
        nonlocal features, feature_diagnostics, feature_metadata
        features, feature_diagnostics, feature_metadata = independent_spatial_features(frame, config)
        spatial_terms = sorted({term for model in config["spatial_models"] for term in model["terms"]})
        for term in spatial_terms:
            fit_config["models"]["continuous"][term] = {"transform": "identity"}
        stored = pd.read_csv(processed / config["outputs"]["spatial_feature_diagnostics"])
        calculated = spatial_feature_table(features, config)
        merged = stored.merge(calculated, on="feature", suffixes=("_stored", "_actual"))
        if len(stored) != len(calculated) or len(merged) != len(stored):
            raise AssertionError("Spatial feature diagnostic rows differ")
        for column in ("min", "q01", "median", "q99", "max", "sd"):
            if not np.allclose(merged[f"{column}_stored"], merged[f"{column}_actual"],
                               rtol=1e-10, atol=1e-12):
                raise AssertionError(f"Spatial feature diagnostic differs: {column}")
        reproduction = read_json(processed / config["outputs"]["reference_reproduction"])
        stored_metadata = reproduction["spatial_basis_metadata"]
        assert_close(feature_metadata["rbf_feature_count"], stored_metadata["rbf_feature_count"],
                     label="RBF feature count")
        for key in ("belt_axis", "cross_axis", "coordinate_center_m"):
            if not np.allclose(feature_metadata[key], stored_metadata[key], rtol=1e-12, atol=1e-9):
                raise AssertionError(f"Spatial basis metadata differs: {key}")
        return {"n_spatial_features": len(stored), "rbf_feature_count": feature_metadata["rbf_feature_count"]}
    capture(checks, "independent_spatial_basis", build_features)

    partition_frames: dict[str, pd.DataFrame] = {}
    partition_definitions = pd.DataFrame()

    def verify_partition_definitions() -> dict[str, Any]:
        nonlocal partition_frames, partition_definitions
        reference = contiguous_folds(frame, 4)
        if not np.array_equal(reference, frame["new_fold"].to_numpy(dtype=int)):
            raise AssertionError("Independent belt-axis folds do not reproduce Stage 4")
        definitions: list[pd.DataFrame] = []
        for specification in config["partitions"]:
            reconstructed = partition_frame(frame, specification, reference)
            partition_frames[specification["id"]] = reconstructed
            definitions.append(partition_definition(
                reconstructed, specification,
                int(config["residuals"]["informative_fold_min_positive_cells"]),
            ))
        partition_definitions = pd.concat(definitions, ignore_index=True)
        stored = pd.read_csv(processed / config["outputs"]["partition_definitions"])
        keys = ["partition_id", "fold"]
        merged = stored.merge(partition_definitions, on=keys, how="outer",
                              suffixes=("_stored", "_actual"), indicator=True)
        if not merged["_merge"].eq("both").all():
            raise AssertionError("Partition-definition rows differ")
        numeric = ["n_holdout", "holdout_area_km2", "holdout_records", "holdout_positive_cells",
                   "holdout_prevalence", "n_train", "train_records", "train_positive_cells",
                   "n_buffer_excluded", "buffer_excluded_records"]
        maximum = 0.0
        for column in numeric:
            left = merged[f"{column}_stored"].to_numpy(dtype=float)
            right = merged[f"{column}_actual"].to_numpy(dtype=float)
            if not np.allclose(left, right, rtol=1e-10, atol=1e-10):
                raise AssertionError(f"Partition-definition field differs: {column}")
            maximum = max(maximum, float(np.max(np.abs(left - right))))
        if not merged["informative_fold_stored"].astype(bool).equals(
                merged["informative_fold_actual"].astype(bool)):
            raise AssertionError("Informative-fold flags differ")
        return {"partition_rows": len(stored), "maximum_numeric_difference": maximum}
    capture(checks, "independent_partition_reconstruction", verify_partition_definitions)

    def refit_reference_and_partitions() -> dict[str, Any]:
        if partition_frames == {} or features.empty:
            raise AssertionError("Feature or partition reconstruction failed")
        partition_predictions = pd.read_csv(processed / config["outputs"]["partition_predictions"])
        partition_fold_metrics = pd.read_csv(processed / config["outputs"]["partition_fold_metrics"])
        partition_summary = pd.read_csv(processed / config["outputs"]["partition_summary"])
        spatial_predictions = pd.read_csv(processed / config["outputs"]["spatial_model_predictions"])
        maximum_prediction = 0.0
        maximum_metric = 0.0
        for specification in config["partitions"]:
            partition_id = specification["id"]
            experiment = Experiment(
                id=f"S0_{partition_id}", label=f"S0 {specification['label']}",
                predictors=tuple(config["reference"]["predictors"]),
                outcome=config["reference"]["outcome"], set_name="stage5c_partition",
            )
            prediction, metrics, _coef, fold_metrics = fit_experiment(
                partition_frames[partition_id], experiment, fit_config)
            expected_prediction = partition_predictions.loc[
                partition_predictions["partition_id"].eq(partition_id)]
            maximum_prediction = max(maximum_prediction,
                                     max(compare_predictions(expected_prediction, prediction,
                                                             f"partition/{partition_id}").values()))
            expected_fold = partition_fold_metrics.loc[
                partition_fold_metrics["partition_id"].eq(partition_id)]
            expected_summary = partition_summary.loc[
                partition_summary["partition_id"].eq(partition_id)]
            maximum_metric = max(maximum_metric, compare_metric_tables(
                expected_fold, expected_summary, fold_metrics, metrics,
                f"partition/{partition_id}"))
        reference = Experiment(
            id="S0_REFERENCE", label="Accepted S0 without spatial terms",
            predictors=tuple(config["reference"]["predictors"]),
            outcome=config["reference"]["outcome"], set_name="stage5c_spatial_comparator",
        )
        prediction, metrics, _coef, fold_metrics = fit_experiment(features, reference, fit_config)
        expected_prediction = spatial_predictions.loc[spatial_predictions["model_id"].eq("S0_REFERENCE")]
        maximum_prediction = max(maximum_prediction,
                                 max(compare_predictions(expected_prediction, prediction,
                                                         "spatial/S0_REFERENCE").values()))
        expected_fold = pd.read_csv(processed / config["outputs"]["spatial_model_fold_metrics"])
        expected_summary = pd.read_csv(processed / config["outputs"]["spatial_model_metrics"])
        maximum_metric = max(maximum_metric, compare_metric_tables(
            expected_fold.loc[expected_fold["model_id"].eq("S0_REFERENCE")],
            expected_summary.loc[expected_summary["model_id"].eq("S0_REFERENCE")],
            fold_metrics, metrics, "spatial/S0_REFERENCE"))
        return {"partition_models_refit": len(config["partitions"]),
                "reference_models_refit": 1,
                "maximum_prediction_difference": maximum_prediction,
                "maximum_metric_difference": maximum_metric}
    capture(checks, "independent_model_refit", refit_reference_and_partitions)

    def serialized_metric_recomputation() -> dict[str, Any]:
        partition_predictions = pd.read_csv(processed / config["outputs"]["partition_predictions"])
        partition_fold_metrics = pd.read_csv(processed / config["outputs"]["partition_fold_metrics"])
        partition_summary = pd.read_csv(processed / config["outputs"]["partition_summary"])
        spatial_predictions = pd.read_csv(processed / config["outputs"]["spatial_model_predictions"])
        spatial_fold_metrics = pd.read_csv(processed / config["outputs"]["spatial_model_fold_metrics"])
        spatial_summary = pd.read_csv(processed / config["outputs"]["spatial_model_metrics"])
        maximum_fold_difference = 0.0
        maximum_summary_difference = 0.0
        partition_models = 0
        spatial_models = 0

        def compare_summary(actual: dict[str, float], expected: pd.Series,
                            label: str) -> float:
            maximum = 0.0
            for metric in POINT_METRICS:
                assert_close(actual[metric], expected[metric], tolerance=1e-5,
                             label=f"{label}/{metric}")
                if np.isfinite(actual[metric]) and np.isfinite(expected[metric]):
                    maximum = max(maximum, abs(actual[metric] - float(expected[metric])))
            return maximum

        for specification in config["partitions"]:
            partition_id = specification["id"]
            model_id = f"S0_{partition_id}"
            predictions = partition_predictions.loc[
                partition_predictions["model_id"].eq(model_id)
            ].copy()
            if len(predictions) != 2193:
                raise AssertionError(f"{partition_id}: expected 2,193 OOF predictions")
            recomputed_folds: list[dict[str, float]] = []
            for fold in (1, 2, 3, 4):
                subset = predictions.loc[predictions["new_fold"].eq(fold)]
                actual = recompute_metrics(subset)
                expected = partition_fold_metrics.loc[
                    partition_fold_metrics["partition_id"].eq(partition_id)
                    & partition_fold_metrics["model_id"].eq(model_id)
                    & partition_fold_metrics["fold"].eq(fold)
                ]
                if len(expected) != 1:
                    raise AssertionError(f"{partition_id}/fold_{fold}: expected one row")
                maximum_fold_difference = max(
                    maximum_fold_difference,
                    compare_summary(actual, expected.iloc[0],
                                    f"partition/{partition_id}/fold_{fold}"),
                )
                recomputed_folds.append(actual)
            pooled = recompute_metrics(predictions)
            expected_pooled = partition_summary.loc[
                partition_summary["partition_id"].eq(partition_id)
                & partition_summary["model_id"].eq(model_id)
                & partition_summary["aggregation"].eq("pooled_oof")
            ]
            expected_mean = partition_summary.loc[
                partition_summary["partition_id"].eq(partition_id)
                & partition_summary["model_id"].eq(model_id)
                & partition_summary["aggregation"].eq("fold_mean")
            ]
            if len(expected_pooled) != 1 or len(expected_mean) != 1:
                raise AssertionError(f"{partition_id}: summary rows are incomplete")
            maximum_summary_difference = max(
                maximum_summary_difference,
                compare_summary(pooled, expected_pooled.iloc[0],
                                f"partition/{partition_id}/pooled_oof"),
            )
            fold_mean = {
                metric: float(pd.Series([row[metric] for row in recomputed_folds]).mean(skipna=True))
                for metric in POINT_METRICS
            }
            maximum_summary_difference = max(
                maximum_summary_difference,
                compare_summary(fold_mean, expected_mean.iloc[0],
                                f"partition/{partition_id}/fold_mean"),
            )
            partition_models += 1

        for specification in config["spatial_models"]:
            model_id = specification["id"]
            predictions = spatial_predictions.loc[
                spatial_predictions["model_id"].eq(model_id)
            ].copy()
            if len(predictions) != 2193:
                raise AssertionError(f"{model_id}: expected 2,193 OOF predictions")
            recomputed_folds = []
            for fold in (1, 2, 3, 4):
                subset = predictions.loc[predictions["new_fold"].eq(fold)]
                actual = recompute_metrics(subset)
                expected = spatial_fold_metrics.loc[
                    spatial_fold_metrics["model_id"].eq(model_id)
                    & spatial_fold_metrics["fold"].eq(fold)
                ]
                if len(expected) != 1:
                    raise AssertionError(f"{model_id}/fold_{fold}: expected one row")
                maximum_fold_difference = max(
                    maximum_fold_difference,
                    compare_summary(actual, expected.iloc[0],
                                    f"spatial/{model_id}/fold_{fold}"),
                )
                recomputed_folds.append(actual)
            pooled = recompute_metrics(predictions)
            expected_pooled = spatial_summary.loc[
                spatial_summary["model_id"].eq(model_id)
                & spatial_summary["aggregation"].eq("pooled_oof")
            ]
            expected_mean = spatial_summary.loc[
                spatial_summary["model_id"].eq(model_id)
                & spatial_summary["aggregation"].eq("fold_mean")
            ]
            if len(expected_pooled) != 1 or len(expected_mean) != 1:
                raise AssertionError(f"{model_id}: summary rows are incomplete")
            maximum_summary_difference = max(
                maximum_summary_difference,
                compare_summary(pooled, expected_pooled.iloc[0],
                                f"spatial/{model_id}/pooled_oof"),
            )
            fold_mean = {
                metric: float(pd.Series([row[metric] for row in recomputed_folds]).mean(skipna=True))
                for metric in POINT_METRICS
            }
            maximum_summary_difference = max(
                maximum_summary_difference,
                compare_summary(fold_mean, expected_mean.iloc[0],
                                f"spatial/{model_id}/fold_mean"),
            )
            spatial_models += 1
        return {
            "partition_models_recomputed": partition_models,
            "spatial_models_recomputed": spatial_models,
            "maximum_fold_metric_difference": maximum_fold_difference,
            "maximum_summary_metric_difference": maximum_summary_difference,
        }
    capture(checks, "serialized_oof_metric_recomputation", serialized_metric_recomputation)

    def moran_replay() -> dict[str, Any]:
        baseline = bundle["predictions"].loc[
            bundle["predictions"]["model_id"].isin(["N0_AREA", "S0_NO_GRAVITY"])
        ].copy()
        uniforms = make_uniforms(frame, int(config["residuals"]["rqr_seed"]))
        baseline = add_rqr(baseline, uniforms)
        spatial = pd.read_csv(processed / config["outputs"]["spatial_model_predictions"])
        spatial = add_rqr(spatial, uniforms)
        stored = pd.read_csv(processed / config["outputs"]["moran_tests"])
        key_columns = ["model_id", "evaluation_scope", "residual_type", "spatial_weight"]
        if stored.duplicated(key_columns).any():
            raise AssertionError("Stored Moran table contains duplicate keys")

        def source_for(model_id: str) -> pd.DataFrame:
            if model_id in {"N0_AREA", "S0_NO_GRAVITY"}:
                return baseline
            return spatial

        def subset_for(row: pd.Series) -> pd.DataFrame:
            source = source_for(str(row["model_id"]))
            subset = source.loc[source["model_id"].eq(str(row["model_id"]))].copy()
            scope = str(row["evaluation_scope"])
            if scope != "pooled_oof":
                fold = int(scope.removeprefix("fold_"))
                subset = subset.loc[subset["new_fold"].eq(fold)].copy()
            return subset

        weight_cache: dict[tuple[str, str, str], tuple[sparse.csr_matrix, dict[str, Any]]] = {}

        def weights_for(row: pd.Series) -> tuple[sparse.csr_matrix, dict[str, Any]]:
            source_name = (
                "baseline" if str(row["model_id"]) in {"N0_AREA", "S0_NO_GRAVITY"}
                else "spatial"
            )
            key = (source_name, str(row["evaluation_scope"]), str(row["spatial_weight"]))
            if key not in weight_cache:
                subset = subset_for(row)
                weight_cache[key] = spatial_weights(
                    subset[["x_m", "y_m"]].to_numpy(dtype=float),
                    str(row["spatial_weight"]),
                )
            return weight_cache[key]

        maximum_moran_difference = 0.0
        maximum_metadata_difference = 0.0
        replay_rows = 0
        maximum_permutation_difference = 0.0
        replay_p: list[dict[str, Any]] = []
        for _, row in stored.sort_values(key_columns).iterrows():
            subset = subset_for(row)
            residual_type = str(row["residual_type"])
            if residual_type == "Pearson count residual":
                column = "pearson_residual"
            elif residual_type == "randomized quantile residual":
                column = "rqr_residual"
            else:
                raise AssertionError(f"Unknown residual type: {residual_type}")
            weights, metadata = weights_for(row)
            observed = moran_test_independent(
                subset[column].to_numpy(dtype=float), weights, 0,
                int(row["seed"]),
            )
            if int(row["n"]) != int(observed["n"]):
                raise AssertionError(f"Moran n differs for {tuple(row[key_columns])}")
            assert_close(observed["moran_i"], row["moran_i"], tolerance=1e-9,
                         label=f"moran/{tuple(row[key_columns])}")
            assert_close(observed["expected_i"], row["expected_i"], tolerance=1e-12,
                         label=f"expected_i/{tuple(row[key_columns])}")
            maximum_moran_difference = max(
                maximum_moran_difference, abs(observed["moran_i"] - float(row["moran_i"]))
            )
            for field in ("mean_neighbours", "min_neighbours", "max_neighbours",
                          "n_nearest_neighbour_fallback"):
                expected = float(row[field])
                actual = float(metadata[field])
                assert_close(actual, expected, tolerance=1e-9,
                             label=f"{field}/{tuple(row[key_columns])}")
                maximum_metadata_difference = max(
                    maximum_metadata_difference, abs(actual - expected)
                )

            if str(row["model_id"]) == "S0_NO_GRAVITY":
                replay = moran_test_independent(
                    subset[column].to_numpy(dtype=float), weights,
                    int(row["n_permutations"]), int(row["seed"]),
                )
                for field in ("p_positive", "p_negative", "p_two_sided",
                              "z_permutation", "permutation_sd"):
                    assert_close(replay[field], row[field], tolerance=1e-7,
                                 label=f"permutation/{field}/{tuple(row[key_columns])}")
                    if np.isfinite(replay[field]) and np.isfinite(float(row[field])):
                        maximum_permutation_difference = max(
                            maximum_permutation_difference,
                            abs(replay[field] - float(row[field])),
                        )
                replay_p.append({
                    "evaluation_scope": str(row["evaluation_scope"]),
                    "residual_type": residual_type,
                    "spatial_weight": str(row["spatial_weight"]),
                    "p_positive": float(replay["p_positive"]),
                    "z_permutation": float(replay["z_permutation"]),
                })
                replay_rows += 1

        if len(stored) != 98:
            raise AssertionError(f"Expected 98 Moran rows; found {len(stored)}")
        if replay_rows != 35:
            raise AssertionError(f"Expected 35 accepted-S0 replay rows; found {replay_rows}")
        return {
            "moran_rows_checked": len(stored),
            "maximum_moran_difference": maximum_moran_difference,
            "maximum_weight_metadata_difference": maximum_metadata_difference,
            "accepted_s0_permutation_rows_replayed": replay_rows,
            "maximum_permutation_difference": maximum_permutation_difference,
            "selected_replays": replay_p,
        }
    capture(checks, "moran_all_rows_and_s0_permutation_replay", moran_replay)

    def local_cluster_recomputation() -> dict[str, Any]:
        spatial = pd.read_csv(processed / config["outputs"]["spatial_model_predictions"])
        uniforms = make_uniforms(frame, int(config["residuals"]["rqr_seed"]))
        spatial = add_rqr(spatial, uniforms)
        local_frame = spatial.loc[
            spatial["model_id"].eq("S0_REFERENCE")
        ].sort_values("support_id").reset_index(drop=True)
        weights, metadata = spatial_weights(
            local_frame[["x_m", "y_m"]].to_numpy(dtype=float),
            str(config["residuals"]["primary_weight"]),
        )
        calculated = local_clusters(
            local_frame, weights,
            int(config["residuals"]["local_permutations"]),
            int(config["residuals"]["rqr_seed"]) + 7777,
            float(config["residuals"]["local_fdr"]),
        )
        stored = pd.read_csv(processed / config["outputs"]["local_moran"])
        if len(stored) != 2193 or len(calculated) != 2193:
            raise AssertionError("Local Moran tables do not contain 2,193 cells")
        if not stored["support_id"].astype(str).equals(calculated["support_id"].astype(str)):
            raise AssertionError("Local Moran support identifiers or order differ")
        numeric = [
            "x_m", "y_m", "tract_area_km2", "n_records", "predicted_count",
            "pearson_residual", "residual_z", "spatial_lag_z", "local_moran_i",
            "p_high", "p_low", "p_two_sided", "p_high_normal", "p_low_normal",
            "p_two_sided_normal", "q_two_sided_bh",
        ]
        maximum = 0.0
        for column in numeric:
            left = stored[column].to_numpy(dtype=float)
            right = calculated[column].to_numpy(dtype=float)
            if not np.allclose(left, right, rtol=1e-8, atol=1e-10, equal_nan=True):
                difference = np.nanmax(np.abs(left - right))
                raise AssertionError(f"Local Moran field differs: {column} ({difference:.3e})")
            finite = np.isfinite(left) & np.isfinite(right)
            if finite.any():
                maximum = max(maximum, float(np.max(np.abs(left[finite] - right[finite]))))
        if not stored["significant_fdr"].astype(bool).equals(
                calculated["significant_fdr"].astype(bool)):
            raise AssertionError("Local Moran significance flags differ")
        if not stored["cluster_class"].astype(str).equals(calculated["cluster_class"].astype(str)):
            raise AssertionError("Local Moran cluster classes differ")
        counts = calculated["cluster_class"].value_counts().to_dict()
        expected_counts = {"high_high": 25, "low_high": 25, "not_significant": 2143}
        if counts != expected_counts:
            raise AssertionError(f"Unexpected local-cluster counts: {counts}")
        if int(counts.get("low_low", 0)) or int(counts.get("high_low", 0)):
            raise AssertionError("Unexpected low-low or high-low cluster present")
        return {
            "n_cells": len(calculated),
            "cluster_counts": {str(key): int(value) for key, value in counts.items()},
            "n_fdr_significant": int(calculated["significant_fdr"].sum()),
            "maximum_numeric_difference": maximum,
            "weight_metadata": metadata,
        }
    capture(checks, "local_cluster_recomputation", local_cluster_recomputation)

    def residual_context_recomputation() -> dict[str, Any]:
        spatial = pd.read_csv(processed / config["outputs"]["spatial_model_predictions"])
        uniforms = make_uniforms(frame, int(config["residuals"]["rqr_seed"]))
        spatial = add_rqr(spatial, uniforms)
        reference = spatial.loc[spatial["model_id"].eq("S0_REFERENCE")].copy()
        calculated = residual_summary_independent(
            reference, geometries["tract_union"], config
        )
        stored = pd.read_csv(processed / config["outputs"]["residual_summary"])
        comparison = compare_residual_summary(stored, calculated)
        observation_stored = pd.read_csv(processed / config["outputs"]["observation_context"])
        observation_calculated = calculated.loc[
            calculated["grouping"].isin(
                ["tract_boundary_distance", "domain", "mapped_coverage"]
            )
        ].reset_index(drop=True)
        observation_comparison = compare_residual_summary(
            observation_stored, observation_calculated
        )
        domain_records = frame.groupby("domain_abbreviation")["n_records"].sum().to_dict()
        expected_domains = {"CRZ": 0, "MMSB": 0, "NKB": 0, "NRB": 183, "unassigned": 1}
        if {str(key): int(value) for key, value in domain_records.items()} != expected_domains:
            raise AssertionError(f"Unexpected domain record counts: {domain_records}")
        outside = calculated.loc[
            calculated["grouping"].eq("mapped_coverage")
            & calculated["group_value"].eq("outside_Daly_envelope")
        ]
        if len(outside) != 1 or int(outside.iloc[0]["observed_records"]) != 1:
            raise AssertionError("Outside-Daly observation count differs from one record")
        return {
            "residual_summary": comparison,
            "observation_context": observation_comparison,
            "domain_record_counts": {str(key): int(value) for key, value in domain_records.items()},
            "outside_daly_records": int(outside.iloc[0]["observed_records"]),
        }
    capture(checks, "residual_context_recomputation", residual_context_recomputation)

    def stage4_reference_reproduction() -> dict[str, Any]:
        stage4 = bundle["predictions"].loc[
            bundle["predictions"]["model_id"].eq(config["reference"]["model_id"])
        ].copy()
        spatial = pd.read_csv(processed / config["outputs"]["spatial_model_predictions"])
        reference = spatial.loc[spatial["model_id"].eq("S0_REFERENCE")].copy()
        partition = pd.read_csv(processed / config["outputs"]["partition_predictions"])
        partition_reference = partition.loc[partition["model_id"].eq("S0_BELT10")].copy()
        stored = read_json(processed / config["outputs"]["reference_reproduction"])
        maximum = 0.0
        for name, candidate in (("spatial", reference), ("partition", partition_reference)):
            joined = candidate.set_index("support_id").join(
                stage4.set_index("support_id"), rsuffix="_stage4"
            )
            if len(joined) != 2193:
                raise AssertionError(f"{name}: reference reproduction does not cover 2,193 cells")
            for column in ("predicted_count", "predicted_any_probability", "pearson_residual"):
                difference = float(np.max(np.abs(
                    joined[column].to_numpy(dtype=float)
                    - joined[f"{column}_stage4"].to_numpy(dtype=float)
                )))
                maximum = max(maximum, difference)
                if difference > 1e-10:
                    raise AssertionError(f"{name}/{column}: difference {difference:.3e}")
                assert_close(stored[name][f"max_abs_{'probability' if column == 'predicted_any_probability' else 'predicted_count' if column == 'predicted_count' else 'pearson_residual'}_difference"],
                             difference, tolerance=1e-10,
                             label=f"stored_reproduction/{name}/{column}")
            if "new_fold_stage4" in joined:
                if not np.array_equal(
                    joined["new_fold"].to_numpy(dtype=int),
                    joined["new_fold_stage4"].to_numpy(dtype=int),
                ):
                    raise AssertionError(f"{name}: reference fold assignment differs")
            if int(stored[name]["n"]) != len(joined):
                raise AssertionError(f"{name}: stored reproduction n differs")
        if not bool(stored["partition"]["same_fold"]):
            raise AssertionError("Stored reference reproduction does not report identical folds")
        return {"n": 2193, "maximum_absolute_difference": maximum,
                "stored_tolerance": float(stored["tolerances"]["max_abs_predicted_count_difference"])}
    capture(checks, "stage4_reference_reproduction", stage4_reference_reproduction)

    def report_consistency() -> dict[str, Any]:
        report_path = figures / config["outputs"]["report"]
        report_path = abs_path(report_path)
        if not report_path.exists():
            raise AssertionError(f"Report is missing: {report_path}")
        text = report_path.read_text(encoding="utf-8")
        links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
        expected_figures = [
            config["outputs"][key] for key in (
                "partition_figure", "partition_transfer_figure", "moran_figure",
                "local_cluster_figure", "residual_diagnostic_figure",
                "spatial_model_figure", "residual_maps_figure",
            )
        ]
        if len(links) != len(expected_figures):
            raise AssertionError(f"Report has {len(links)} image links; expected {len(expected_figures)}")
        for link in links:
            target = (report_path.parent / link).resolve()
            if not target.exists():
                raise AssertionError(f"Report image link is broken: {link}")
        if set(Path(link).name for link in links) != set(expected_figures):
            raise AssertionError("Report image links do not match the canonical figure set")

        spatial_metrics = pd.read_csv(processed / config["outputs"]["spatial_model_metrics"])
        pooled = spatial_metrics.loc[
            spatial_metrics["model_id"].eq("S0_REFERENCE")
            & spatial_metrics["aggregation"].eq("pooled_oof")
        ].iloc[0]
        required = [
            "The response remains the intensity of historically compiled occurrence records per unit tract-supported area. It is not a detection-free mineralisation probability.",
            "The value 0.170 belongs to the exploratory `S0_BASE` comparator and is not the accepted reference.",
            "Only Fold 1 reaches p < 0.05; Folds 2-4 do not.",
            "The `low_high` class is a low-residual cell surrounded by higher-residual neighbours; it is not a low-low cluster.",
            "No low-low or high-low clusters were detected.",
            "No comparator improves ranking, probability accuracy and residual dependence together.",
            "The observation process remains unresolved.",
            "The checkerboard partition tests local interpolation, not regional geological transfer.",
            "Boundary classes show a mixed, non-monotonic pattern",
        ]
        missing = [phrase for phrase in required if phrase not in text]
        if missing:
            raise AssertionError(f"Report is missing required corrected statements: {missing}")
        forbidden = [
            "high-high and low-low clusters",
            "all folds remain significant",
            "the checkerboard result is regional transfer",
        ]
        present = [phrase for phrase in forbidden if phrase.lower() in text.lower()]
        if present:
            raise AssertionError(f"Report contains retracted wording: {present}")
        for value in (
            f"{pooled['roc_auc']:.3f}", f"{pooled['average_precision']:.3f}",
            f"{pooled['ap_over_prevalence']:.2f}", f"{pooled['brier_skill']:.3f}",
            f"{pooled['calibration_slope']:.3f}",
        ):
            if value not in text:
                raise AssertionError(f"Report omits the accepted pooled summary value {value}")
        if "| S0_REFERENCE | 0 | 3 | 0.642 | 1.694 | 0.034 | 0.619 | 0.143 |" not in text:
            raise AssertionError("Report's S0_REFERENCE spatial-comparator row lacks Moran's I")
        moran = pd.read_csv(processed / config["outputs"]["moran_tests"])
        pearson = moran.loc[
            moran["model_id"].eq("S0_NO_GRAVITY")
            & moran["residual_type"].eq("Pearson count residual")
            & moran["evaluation_scope"].str.startswith("fold_")
        ]
        rqr = moran.loc[
            moran["model_id"].eq("S0_NO_GRAVITY")
            & moran["residual_type"].eq("randomized quantile residual")
            & moran["evaluation_scope"].str.startswith("fold_")
        ]
        if f"ranging from {pearson['moran_i'].min():.3f} to {pearson['moran_i'].max():.3f}" not in text:
            raise AssertionError("Report's Pearson fold range does not match the Moran table")
        if f"from {rqr['moran_i'].min():.3f} to {rqr['moran_i'].max():.3f}" not in text:
            raise AssertionError("Report's RQR fold range does not match the Moran table")
        for cluster, count, share in (
            ("not_significant", 2143, "0.977"),
            ("low_high", 25, "0.011"),
            ("high_high", 25, "0.011"),
        ):
            if f"| {cluster} | {count} | {share} |" not in text:
                raise AssertionError(f"Report cluster row differs for {cluster}")
        return {
            "report": relative_posix(report_path),
            "image_links": len(links),
            "required_statements": len(required),
            "forbidden_statements_absent": len(forbidden),
        }
    capture(checks, "report_consistency", report_consistency)

    def manifest_consistency() -> dict[str, Any]:
        expected = {
            "n_support": 2193,
            "n_records": 184,
            "n_positive_cells": 146,
            "n_partitions": 4,
            "n_spatial_models": 5,
            "n_moran_tests": 98,
            "n_local_moran_cells": 2193,
        }
        observed = {key: int(manifest["counts"][key]) for key in expected}
        if observed != expected:
            raise AssertionError(f"Manifest counts differ: {observed}")
        output_keys = set(manifest["outputs"])
        expected_outputs = {
            key for key in config["outputs"]
            if key not in {"processed_dir", "figure_dir", "manifest", "validation"}
        }
        if output_keys != expected_outputs:
            raise AssertionError(
                f"Manifest output keys differ: missing={sorted(expected_outputs - output_keys)}, "
                f"extra={sorted(output_keys - expected_outputs)}"
            )
        return {"counts": observed, "n_outputs": len(output_keys)}
    capture(checks, "manifest_consistency", manifest_consistency)

    n_failed = sum(check["status"] == "fail" for check in checks)
    validation = {
        "schema_version": "1.0.0",
        "status": "pass" if n_failed == 0 else "fail",
        "n_checks": len(checks),
        "n_failed": n_failed,
        "checks": checks,
        "stage5c_build_id": manifest["build_id"],
        "stage5b_build_id": manifest.get("stage5b_reference"),
        "original_manifest_sha256": original_manifest_sha,
        "software": software_stack(),
    }
    validation_path = processed / config["outputs"]["validation"]
    validation = json_safe(validation)
    write_json(validation, validation_path)
    manifest["original_manifest_sha256"] = original_manifest_sha
    manifest["validation"] = {
        **file_record(validation_path, role="independent Stage 5C verification"),
        "status": validation["status"],
        "n_checks": validation["n_checks"],
        "n_failed": validation["n_failed"],
    }
    manifest["status"] = "verified" if n_failed == 0 else "verification_failed"
    write_json(manifest, manifest_path)
    print(json.dumps({
        "status": validation["status"],
        "n_checks": validation["n_checks"],
        "n_failed": n_failed,
        "manifest_status": manifest["status"],
        "validation": relative_posix(validation_path),
    }, indent=2), flush=True)
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
