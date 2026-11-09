"""Independently verify Stage 4 model outputs and their reported metrics."""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modeling.stage4_common import (  # noqa: E402
    abs_path,
    file_record,
    load_config,
    output_paths,
    relative_posix,
    sha256_file,
    verify_protected,
    write_json,
)


RECOMPUTED_METRICS = [
    "roc_auc",
    "average_precision",
    "ap_over_prevalence",
    "brier_score",
    "brier_skill",
    "predicted_total",
    "observed_total",
    "total_count_ratio",
    "pi_coverage",
]


def record_check(checks: list[dict[str, Any]], name: str, passed: bool, details: Any) -> None:
    checks.append({"name": name, "status": "pass" if passed else "fail", "details": details})


def capture(checks: list[dict[str, Any]], name: str, function: Callable[[], Any]) -> Any:
    try:
        details = function()
        record_check(checks, name, True, details)
        return details
    except Exception as error:  # noqa: BLE001 - verifier must report every failure
        record_check(checks, name, False, {"error": f"{type(error).__name__}: {error}"})
        return None


def verify_file_records(records: dict[str, Any]) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    checked = 0
    for name, record in records.items():
        path = abs_path(record["path"])
        checked += 1
        if not path.exists():
            mismatches.append({"name": name, "path": record["path"], "error": "missing"})
            continue
        observed = sha256_file(path)
        if observed.lower() != str(record["sha256"]).lower():
            mismatches.append({
                "name": name, "path": record["path"],
                "expected": record["sha256"], "observed": observed,
            })
        if int(record.get("bytes", -1)) != path.stat().st_size:
            mismatches.append({
                "name": name, "path": record["path"], "error": "byte-size mismatch",
                "expected": int(record.get("bytes", -1)), "observed": path.stat().st_size,
            })
    return {"checked": checked, "mismatches": mismatches}


def verify_png(path: str | Path) -> dict[str, Any]:
    absolute = abs_path(path)
    with absolute.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Not a readable PNG: {absolute}")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid PNG dimensions: {absolute}")
    return {"path": relative_posix(absolute), "width": width, "height": height}


def roc_auc(y_binary: np.ndarray, scores: np.ndarray) -> float:
    positives = int(np.sum(y_binary))
    negatives = int(len(y_binary) - positives)
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = stats.rankdata(scores, method="average")
    rank_sum = float(np.sum(ranks[y_binary == 1]))
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision(y_binary: np.ndarray, scores: np.ndarray) -> float:
    positives = int(np.sum(y_binary))
    if positives == 0:
        return math.nan
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_binary[order].astype(int)
    scores_sorted = scores[order]
    group_ends = np.r_[np.flatnonzero(np.diff(scores_sorted) != 0), len(y_sorted) - 1]
    cumulative_tp = np.cumsum(y_sorted)[group_ends].astype(float)
    recall = cumulative_tp / positives
    precision = cumulative_tp / (group_ends + 1.0)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def recompute_metrics(subset: pd.DataFrame) -> dict[str, float]:
    y_count = subset["observed_count"].to_numpy(dtype=float)
    y_binary = subset["observed_any"].to_numpy(dtype=float)
    mu = subset["predicted_count"].to_numpy(dtype=float)
    p_any = subset["predicted_any_probability"].to_numpy(dtype=float)
    area = subset["tract_area_km2"].to_numpy(dtype=float)
    ranking_rate = subset["predicted_rate_per_km2"].to_numpy(dtype=float)
    baseline = subset["baseline_train_probability"].to_numpy(dtype=float)
    prevalence = float(np.mean(y_binary))
    ap = average_precision(y_binary, ranking_rate)
    brier = float(np.mean((y_binary - p_any) ** 2))
    brier_baseline = float(np.mean((y_binary - baseline) ** 2))
    observed_total = float(np.sum(y_count))
    predicted_total = float(np.sum(mu))
    low = subset["pi_low_count"].to_numpy(dtype=float)
    high = subset["pi_high_count"].to_numpy(dtype=float)
    return {
        "roc_auc": roc_auc(y_binary, ranking_rate),
        "average_precision": ap,
        "ap_over_prevalence": ap / prevalence if prevalence > 0 and np.isfinite(ap) else math.nan,
        "brier_score": brier,
        "brier_skill": 1.0 - brier / brier_baseline if brier_baseline > 0 else math.nan,
        "predicted_total": predicted_total,
        "observed_total": observed_total,
        "total_count_ratio": predicted_total / observed_total if observed_total > 0 else math.nan,
        "pi_coverage": float(np.mean((y_count >= low) & (y_count <= high))),
    }


def expected_experiments(config: dict[str, Any]) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    for set_name, key in (("primary", "primary_experiments"), ("sensitivity", "sensitivity_experiments")):
        for entry in config["models"][key]:
            experiments.append({
                "id": entry["id"],
                "set": set_name,
                "outcome": entry.get("outcome", config["models"]["outcome"]),
                "predictors": entry.get("predictors", []),
            })
    return experiments


def reconstruct_prediction(
    support: pd.DataFrame, coefficients: pd.DataFrame, model_id: str, fold: int,
) -> np.ndarray:
    model_coefficients = coefficients.loc[
        coefficients["model_id"].eq(model_id) & coefficients["fit_scope"].eq(f"fold_{fold}")
    ].copy()
    if model_coefficients.empty:
        raise ValueError(f"No fold coefficients for {model_id}, fold {fold}")
    intercept_rows = model_coefficients.loc[model_coefficients["term_kind"].eq("intercept")]
    if len(intercept_rows) != 1:
        raise ValueError(f"Expected one intercept for {model_id}, fold {fold}")
    linear = np.full(len(support), float(intercept_rows.iloc[0]["estimate_z"]), dtype=float)
    for row in model_coefficients.loc[~model_coefficients["term_kind"].eq("intercept")].itertuples():
        values = support[row.term_raw].astype(float).to_numpy()
        if row.term_kind == "continuous":
            if row.transform.startswith("log1p"):
                values = np.log1p(values)
            design = (values - float(row.center)) / float(row.scale)
        elif row.term_kind == "categorical_contrast":
            design = support[row.term_raw].astype(int).eq(int(row.term_level)).to_numpy(dtype=float)
        else:
            raise ValueError(f"Unsupported coefficient term kind: {row.term_kind}")
        linear += design * float(row.estimate_z)
    return np.exp(np.clip(np.log(support["tract_area_km2"].to_numpy(dtype=float)) + linear, -30.0, 30.0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage4_v1.json")
    args = parser.parse_args()
    config_path = abs_path(args.config)
    config = load_config(config_path)
    paths = output_paths(config)
    checks: list[dict[str, Any]] = []

    manifest_path = abs_path(paths["model_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    support = pd.read_csv(abs_path(paths["model_table"]))
    predictions = pd.read_csv(abs_path(paths["model_predictions"]))
    fold_metrics = pd.read_csv(abs_path(paths["fold_metrics"]))
    model_metrics = pd.read_csv(abs_path(paths["model_metrics"]))
    coefficients = pd.read_csv(abs_path(paths["coefficients_by_fold"]))
    experiments = expected_experiments(config)
    expected_ids = [entry["id"] for entry in experiments]
    expected_rows = len(support) * len(experiments)

    def protected_check() -> dict[str, Any]:
        protected = verify_protected(config)
        if not all(record["hash_match"] for record in protected.values()):
            raise AssertionError("A protected input differs from its declared hash")
        return protected

    capture(checks, "protected_inputs_unchanged", protected_check)

    def manifest_check() -> dict[str, Any]:
        if manifest.get("status") != "fitted":
            raise AssertionError("Model manifest is not in fitted state")
        if manifest["config"]["sha256"].lower() != sha256_file(config_path).lower():
            raise AssertionError("Model manifest config hash does not match the current config")
        data_manifest = abs_path(paths["data_manifest"])
        data_validation = abs_path(paths["data_validation"])
        if file_record(data_manifest)["sha256"] != manifest["data_manifest"]["sha256"]:
            raise AssertionError("Data manifest differs from the model manifest record")
        if file_record(data_validation)["sha256"] != manifest["data_validation"]["sha256"]:
            raise AssertionError("Data validation differs from the model manifest record")
        observed_model_table = sha256_file(paths["model_table"])
        if observed_model_table != manifest["model_table"]["sha256"]:
            raise AssertionError("Model table differs from the model manifest record")
        return {
            "build_id": manifest["build_id"],
            "data_build_id": manifest["data_build_id"],
            "config_sha256": manifest["config"]["sha256"],
            "model_table_sha256": observed_model_table,
        }

    capture(checks, "manifest_lineage", manifest_check)
    capture(checks, "model_output_hashes", lambda: verify_file_records(manifest["outputs"]))

    def prediction_schema_check() -> dict[str, Any]:
        required = {
            "support_id", "new_fold", "tract_area_km2", "observed_count", "observed_any",
            "predicted_count", "predicted_rate_per_km2", "predicted_any_probability",
            "pi_low_count", "pi_high_count", "baseline_train_probability", "model_id",
            "outcome", "experiment_set",
        }
        missing = required.difference(predictions.columns)
        if missing:
            raise AssertionError(f"Prediction table missing columns: {sorted(missing)}")
        if len(predictions) != expected_rows:
            raise AssertionError(f"Prediction rows {len(predictions)} != {expected_rows}")
        if set(predictions["model_id"]) != set(expected_ids):
            raise AssertionError("Prediction model identifiers differ from the configuration")
        duplicate = predictions.duplicated(["model_id", "support_id"]).any()
        if duplicate:
            raise AssertionError("A support cell has more than one prediction for a model")
        counts = predictions.groupby("model_id")["support_id"].nunique()
        if not counts.eq(len(support)).all():
            raise AssertionError(f"Models without complete OOF coverage: {counts.to_dict()}")
        return {"n_rows": len(predictions), "n_models": len(experiments), "n_support": len(support)}

    capture(checks, "prediction_coverage", prediction_schema_check)

    def fold_integrity_check() -> dict[str, Any]:
        expected_fold = support.set_index("support_id")["new_fold"]
        observed_fold = predictions.set_index(["model_id", "support_id"])["new_fold"].unstack("model_id")
        for model_id in expected_ids:
            mismatch = int((observed_fold[model_id] != expected_fold).sum())
            if mismatch:
                raise AssertionError(f"{model_id} has {mismatch} predictions outside their assigned fold")
        fold_records = support.groupby("new_fold")["n_records"].sum().to_dict()
        fold_positive = support.groupby("new_fold")["any_record"].sum().to_dict()
        if int(fold_records.get(4, -1)) != 1 or int(fold_positive.get(4, -1)) != 1:
            raise AssertionError("Fold 4 must contain exactly one record in one positive cell")
        return {
            "holdout_records": {int(k): int(v) for k, v in fold_records.items()},
            "holdout_positive_cells": {int(k): int(v) for k, v in fold_positive.items()},
        }

    capture(checks, "out_of_fold_integrity", fold_integrity_check)

    def exposure_check() -> dict[str, Any]:
        rate_error = np.abs(
            predictions["predicted_count"] / predictions["tract_area_km2"]
            - predictions["predicted_rate_per_km2"]
        )
        max_rate_error = float(rate_error.max())
        if max_rate_error > 1e-10:
            raise AssertionError(f"Predicted count and incidence rate disagree: {max_rate_error}")
        area_sums = support.groupby("new_fold")["tract_area_km2"].sum()
        model_area_sums = predictions.groupby(["model_id", "new_fold"])["tract_area_km2"].sum().unstack()
        max_area_error = float(
            np.max(np.abs(model_area_sums.to_numpy() - area_sums.to_numpy()[None, :]))
        )
        if max_area_error > 1e-6:
            raise AssertionError(f"Model exposure differs across folds: {max_area_error}")
        n0 = predictions.loc[predictions["model_id"].eq("N0_AREA")]
        n0_spread = n0.groupby("new_fold")["predicted_rate_per_km2"].agg(
            lambda values: float(values.max() - values.min())
        )
        if float(n0_spread.max()) > 1e-10:
            raise AssertionError("N0_AREA incidence rate is not constant within a fold")
        return {
            "max_count_rate_identity_error": max_rate_error,
            "max_model_area_sum_error_km2": max_area_error,
            "n0_within_fold_rate_spread_max": float(n0_spread.max()),
            "total_exposure_km2": float(support["tract_area_km2"].sum()),
        }

    capture(checks, "exposure_and_incidence_ranking", exposure_check)

    def observed_values_check() -> dict[str, Any]:
        base = support.set_index("support_id")
        detail: dict[str, Any] = {}
        for experiment in experiments:
            subset = predictions.loc[predictions["model_id"].eq(experiment["id"])].set_index("support_id")
            outcome_column = experiment["outcome"]
            count_mismatch = int(np.sum(subset["observed_count"].to_numpy() != base.loc[subset.index, outcome_column].to_numpy()))
            any_column = "any_expanded_record" if outcome_column == "n_expanded_records" else "any_record"
            binary_mismatch = int(np.sum(subset["observed_any"].to_numpy() != base.loc[subset.index, any_column].to_numpy()))
            area_mismatch = float(np.max(np.abs(subset["tract_area_km2"] - base.loc[subset.index, "tract_area_km2"])))
            if count_mismatch or binary_mismatch or area_mismatch > 1e-8:
                raise AssertionError(f"Observed response or area mismatch for {experiment['id']}")
            detail[experiment["id"]] = {
                "count_mismatches": count_mismatch,
                "binary_mismatches": binary_mismatch,
                "max_area_error_km2": area_mismatch,
            }
        return detail

    capture(checks, "observed_response_alignment", observed_values_check)

    def coefficient_prediction_check() -> dict[str, Any]:
        errors: dict[str, float] = {}
        for experiment in experiments:
            model_id = experiment["id"]
            max_error = 0.0
            for fold in (1, 2, 3, 4):
                prediction_subset = predictions.loc[
                    predictions["model_id"].eq(model_id) & predictions["new_fold"].eq(fold)
                ].copy().sort_values("support_id")
                support_subset = support.loc[
                    support["new_fold"].eq(fold)
                ].copy().sort_values("support_id")
                if prediction_subset["support_id"].tolist() != support_subset["support_id"].tolist():
                    raise AssertionError(f"Support order mismatch for {model_id}, fold {fold}")
                rebuilt = reconstruct_prediction(support_subset, coefficients, model_id, fold)
                error = float(np.max(np.abs(
                    rebuilt - prediction_subset["predicted_count"].to_numpy(dtype=float)
                )))
                max_error = max(max_error, error)
            if max_error > 1e-7:
                raise AssertionError(f"{model_id} predictions do not reproduce from stored coefficients: {max_error}")
            errors[model_id] = max_error
        return errors

    capture(checks, "predictions_reproduce_from_coefficients", coefficient_prediction_check)

    def metric_check() -> dict[str, Any]:
        pooled_rows = model_metrics.loc[model_metrics["aggregation"].eq("pooled_oof")].set_index("model_id")
        fold_mean_rows = model_metrics.loc[model_metrics["aggregation"].eq("fold_mean")].set_index("model_id")
        differences: dict[str, dict[str, float]] = {}
        for experiment in experiments:
            model_id = experiment["id"]
            subset = predictions.loc[predictions["model_id"].eq(model_id)]
            direct = recompute_metrics(subset)
            fold_values = pd.DataFrame([
                recompute_metrics(subset.loc[subset["new_fold"].eq(fold)])
                for fold in (1, 2, 3, 4)
            ])
            differences[model_id] = {}
            for metric in RECOMPUTED_METRICS:
                pooled_value = float(pooled_rows.loc[model_id, metric])
                if not np.isclose(direct[metric], pooled_value, atol=1e-8, rtol=1e-8, equal_nan=True):
                    raise AssertionError(
                        f"Pooled {metric} differs for {model_id}: direct={direct[metric]}, stored={pooled_value}"
                    )
                mean_value = float(fold_values[metric].mean(skipna=True))
                stored_mean = float(fold_mean_rows.loc[model_id, metric])
                if not np.isclose(mean_value, stored_mean, atol=1e-8, rtol=1e-8, equal_nan=True):
                    raise AssertionError(
                        f"Fold-mean {metric} differs for {model_id}: direct={mean_value}, stored={stored_mean}"
                    )
                differences[model_id][metric] = float(abs(direct[metric] - pooled_value))
        return {
            "models_checked": expected_ids,
            "max_absolute_pooled_metric_difference": float(
                max(value for model in differences.values() for value in model.values() if np.isfinite(value))
            ),
        }

    capture(checks, "pooled_and_fold_mean_metrics_reproduce", metric_check)

    def fold_metric_check() -> dict[str, Any]:
        indexed = fold_metrics.set_index(["model_id", "fold"])
        max_difference = 0.0
        for experiment in experiments:
            model_id = experiment["id"]
            subset = predictions.loc[predictions["model_id"].eq(model_id)]
            for fold in (1, 2, 3, 4):
                direct = recompute_metrics(subset.loc[subset["new_fold"].eq(fold)])
                for metric in RECOMPUTED_METRICS:
                    stored = float(indexed.loc[(model_id, fold), metric])
                    difference = abs(direct[metric] - stored)
                    if not np.isclose(direct[metric], stored, atol=1e-8, rtol=1e-8, equal_nan=True):
                        raise AssertionError(
                            f"Fold {metric} differs for {model_id}, fold {fold}: direct={direct[metric]}, stored={stored}"
                        )
                    if np.isfinite(difference):
                        max_difference = max(max_difference, float(difference))
        return {"max_absolute_fold_metric_difference": max_difference}

    capture(checks, "fold_metrics_reproduce", fold_metric_check)

    def calibration_reasonableness_check() -> dict[str, Any]:
        primary = [entry["id"] for entry in experiments if entry["set"] == "primary"]
        pooled = model_metrics.loc[
            model_metrics["aggregation"].eq("pooled_oof") & model_metrics["model_id"].isin(primary)
        ].set_index("model_id")
        slopes = {model_id: float(pooled.loc[model_id, "calibration_slope"]) for model_id in primary}
        invalid = {model_id: slopes[model_id] for model_id in primary if not np.isfinite(slopes[model_id])}
        if invalid:
            raise AssertionError(f"Non-finite calibration slopes: {invalid}")
        return {"calibration_slopes": slopes}

    capture(checks, "calibration_slopes_computable", calibration_reasonableness_check)

    def bootstrap_shape_check() -> dict[str, Any]:
        bootstrap = pd.read_csv(abs_path(paths["metric_bootstrap"]))
        pairwise = pd.read_csv(abs_path(paths["pairwise_deltas"]))
        if bootstrap.empty or bootstrap["ci_low"].isna().any() or bootstrap["ci_high"].isna().any():
            raise AssertionError("Bootstrap table is empty or has missing interval bounds")
        if pairwise.empty or pairwise["ci_low"].isna().any() or pairwise["ci_high"].isna().any():
            raise AssertionError("Pairwise-delta table is empty or has missing interval bounds")
        return {
            "bootstrap_rows": len(bootstrap),
            "pairwise_rows": len(pairwise),
            "bootstrap_scopes": sorted(bootstrap["evaluation_scope"].unique().tolist()),
            "bootstrap_requested": int(bootstrap["n_requested"].min()),
        }

    capture(checks, "bootstrap_outputs_complete", bootstrap_shape_check)

    def figure_check() -> list[dict[str, Any]]:
        figure_dir = abs_path(config["outputs"]["figure_dir"])
        names = [
            config["outputs"]["figures"][key]
            for key in ("support_folds", "occurrence_counts")
        ] + [
            name for name in (
                "stage4_model_comparison.png",
                "stage4_calibration_residuals.png",
                "stage4_coefficients.png",
                "stage4_model_maps.png",
            ) if (figure_dir / name).exists()
        ]
        records = [verify_png(figure_dir / name) for name in names]
        if len(records) < 6:
            raise AssertionError(f"Expected six Stage 4 figures, found {len(records)}")
        return records

    capture(checks, "model_and_data_figures_readable", figure_check)

    def report_check() -> dict[str, Any]:
        report_path = abs_path(manifest["outputs"]["model_report"]["path"])
        text = report_path.read_text(encoding="utf-8")
        required = [
            "Stage 4 initial count-model comparison",
            "documented-record intensity",
            "nearest-centre Voronoi partition",
            "Nominal 10 km² hexagons intersect",
            "literal hexagon support",
            "Record distribution across Daly domains",
            "Fold 4 contains one recorded occurrence",
            "Discrimination: pooled and fold-specific",
            "Limitations carried forward",
        ]
        missing = [phrase for phrase in required if phrase not in text]
        if missing:
            raise AssertionError(f"Model report lacks required caveats: {missing}")
        return {"path": relative_posix(report_path), "bytes": report_path.stat().st_size}

    capture(checks, "model_report_caveats_present", report_check)

    passed = all(check["status"] == "pass" for check in checks)
    validation = {
        "schema_version": "1.0.0",
        "build_id": manifest["build_id"],
        "status": "pass" if passed else "fail",
        "checks": checks,
        "n_checks": len(checks),
        "n_failed": int(sum(check["status"] == "fail" for check in checks)),
        "verified_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "notes": [
            "Predictions were independently reconstructed from stored fold coefficients and catchment exposure.",
            "Pooled, fold and fold-mean discrimination, precision-recall, count and interval metrics were recomputed from OOF predictions.",
            "Bootstrap resampling was checked for completeness but not re-run.",
        ],
    }
    validation_path = abs_path(paths["model_validation"])
    write_json(validation, validation_path)
    if passed:
        manifest["validation"] = {
            "status": "pass",
            "path": relative_posix(validation_path),
            "sha256": sha256_file(validation_path),
        }
        write_json(manifest, manifest_path)

    print(json.dumps({
        "status": validation["status"],
        "build_id": manifest["build_id"],
        "n_checks": validation["n_checks"],
        "n_failed": validation["n_failed"],
        "failed": [check for check in checks if check["status"] == "fail"],
        "validation": relative_posix(validation_path),
    }, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
