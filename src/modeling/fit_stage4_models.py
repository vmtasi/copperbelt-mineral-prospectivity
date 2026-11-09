"""Fit Stage 4 negative-binomial count models and spatial diagnostics.

The outcome is the number of historically compiled occurrence records in each
tract-clipped Voronoi catchment. Catchment area is included as a log exposure
offset. Predictions therefore estimate documented-record intensity, not the
true geological probability of mineralisation.

Validation is out-of-fold under four contiguous belt-axis folds with a 10 km
training buffer. Fits are penalised maximum likelihood, not MCMC.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy import optimize, special, stats
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modeling.stage4_common import (  # noqa: E402
    abs_path,
    build_id,
    file_record,
    load_config,
    output_paths,
    read_gdf,
    relative_posix,
    sha256_file,
    software_versions,
    verify_protected,
    write_csv,
    write_json,
)


Z975 = float(stats.norm.ppf(0.975))
REQUIRED_COLUMNS = {
    "support_id", "x_m", "y_m", "domain_abbreviation", "new_fold",
    "new_fold_label", "tract_area_km2", "n_records", "n_expanded_records",
    "any_record", "any_expanded_record",
}
BOOTSTRAP_METRICS = [
    "roc_auc", "average_precision", "ap_over_prevalence", "brier_score",
    "brier_skill", "nb_deviance", "mae", "rmse", "total_count_ratio",
]
POINT_METRICS = [
    "roc_auc", "average_precision", "ap_over_prevalence", "brier_score",
    "brier_skill", "binary_log_loss", "nb_deviance", "mae", "rmse",
    "predicted_total", "observed_total", "total_count_ratio",
    "calibration_intercept", "calibration_slope", "pi_coverage",
    "mean_pi_width",
]


@dataclass(frozen=True)
class Experiment:
    id: str
    label: str
    predictors: tuple[str, ...]
    outcome: str
    set_name: str


@dataclass
class DesignBlock:
    x_train: np.ndarray
    x_target: np.ndarray
    offset_train: np.ndarray
    offset_target: np.ndarray
    y_train: np.ndarray
    y_target: np.ndarray
    terms: pd.DataFrame


@dataclass
class NBFit:
    beta: np.ndarray
    theta: float
    covariance: np.ndarray
    se: np.ndarray
    objective: float
    converged: bool
    message: str
    n_iterations: int


def experiments_from_config(config: dict[str, Any]) -> list[Experiment]:
    experiments: list[Experiment] = []
    seen: set[str] = set()
    for set_name, key in (("primary", "primary_experiments"), ("sensitivity", "sensitivity_experiments")):
        for entry in config["models"][key]:
            experiment = Experiment(
                id=entry["id"], label=entry["label"],
                predictors=tuple(entry.get("predictors", [])),
                outcome=entry.get("outcome", config["models"]["outcome"]),
                set_name=set_name,
            )
            if experiment.id in seen:
                raise ValueError(f"Duplicate experiment id: {experiment.id}")
            seen.add(experiment.id)
            experiments.append(experiment)
    return experiments


def validate_inputs(
    config: dict[str, Any], paths: dict[str, str], config_path: str | Path
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    protected = verify_protected(config)
    manifest = json.loads(abs_path(paths["data_manifest"]).read_text(encoding="utf-8"))
    validation = json.loads(abs_path(paths["data_validation"]).read_text(encoding="utf-8"))
    if manifest.get("status") != "verified":
        raise RuntimeError("Stage 4 data manifest is not verified")
    if validation.get("status") != "pass" or validation.get("n_failed", 1) != 0:
        raise RuntimeError("Stage 4 data validation is not passing")
    observed = file_record(paths["model_table"])
    if observed["sha256"] != manifest["outputs"]["model_table"]["sha256"]:
        raise RuntimeError("Model table does not match the verified data manifest")
    frame = pd.read_csv(abs_path(paths["model_table"]))
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Model table missing columns: {sorted(missing)}")
    if len(frame) != int(config["support"]["expected_n"]):
        raise ValueError("Unexpected support-cell count")
    if not frame["support_id"].is_unique:
        raise ValueError("support_id is not unique")
    if frame["tract_area_km2"].isna().any() or frame["tract_area_km2"].le(0).any():
        raise ValueError("Exposure must be complete and strictly positive")
    if sorted(frame["new_fold"].unique().tolist()) != [1, 2, 3, 4]:
        raise ValueError("Expected folds 1-4")
    for fold in (1, 2, 3, 4):
        column = f"train_included_fold_{fold}"
        holdout = frame["new_fold"].eq(fold)
        if column not in frame:
            raise ValueError(f"Missing training indicator: {column}")
        if frame.loc[holdout, column].any():
            raise ValueError(f"Fold {fold} holdout is marked as training")
        if not frame.loc[~holdout, column].any():
            raise ValueError(f"Fold {fold} has no training cells")
    for experiment in experiments_from_config(config):
        for predictor in experiment.predictors:
            if predictor not in frame.columns:
                raise ValueError(f"Missing predictor: {predictor}")
            if predictor in config["models"]["continuous"] and frame[predictor].isna().any():
                raise ValueError(f"Missing predictor values: {predictor}")
    frame = frame.sort_values("support_id").reset_index(drop=True)
    return frame, manifest, {
        "protected": protected,
        "config_sha256": sha256_file(config_path),
        "model_table_sha256": observed["sha256"],
    }


def _continuous_values(series: pd.Series, transform: str) -> np.ndarray:
    values = series.astype(float).to_numpy()
    if transform == "log1p":
        if np.any(values < 0):
            raise ValueError("log1p transform received negative values")
        return np.log1p(values)
    if transform == "identity":
        return values
    raise ValueError(f"Unknown continuous transform: {transform}")


def build_design(
    train: pd.DataFrame,
    target: pd.DataFrame,
    experiment: Experiment,
    config: dict[str, Any],
) -> DesignBlock:
    continuous = config["models"]["continuous"]
    categorical = config["models"]["categorical"]
    columns: list[np.ndarray] = [np.ones(len(train), dtype=float)]
    target_columns: list[np.ndarray] = [np.ones(len(target), dtype=float)]
    term_rows: list[dict[str, Any]] = [{
        "term": "intercept", "term_raw": "intercept", "term_kind": "intercept",
        "term_level": "", "reference_level": "", "transform": "identity",
        "center": 0.0, "scale": 1.0, "n_nonzero_train": len(train),
        "identifiable": True,
    }]
    for predictor in experiment.predictors:
        if predictor in continuous:
            spec = continuous[predictor]
            transform = spec.get("transform", "identity")
            train_values = _continuous_values(train[predictor], transform)
            target_values = _continuous_values(target[predictor], transform)
            center = float(np.mean(train_values))
            scale = float(np.std(train_values, ddof=0))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"Zero or invalid training spread: {predictor}")
            columns.append((train_values - center) / scale)
            target_columns.append((target_values - center) / scale)
            term_rows.append({
                "term": predictor, "term_raw": predictor, "term_kind": "continuous",
                "term_level": "", "reference_level": "",
                "transform": f"{transform};training_zscore", "center": center,
                "scale": scale, "n_nonzero_train": int(np.count_nonzero(train_values)),
                "identifiable": True,
            })
        elif predictor in categorical:
            spec = categorical[predictor]
            reference = int(spec["reference"])
            levels = [int(level) for level in spec["levels"]]
            if reference not in levels:
                raise ValueError(f"Reference level absent for {predictor}")
            unknown_train = set(train[predictor].astype(int).unique()) - set(levels)
            unknown_target = set(target[predictor].astype(int).unique()) - set(levels)
            if unknown_train or unknown_target:
                raise ValueError(f"Unexpected levels for {predictor}")
            for level in levels:
                if level == reference:
                    continue
                train_dummy = train[predictor].astype(int).eq(level).to_numpy(dtype=float)
                target_dummy = target[predictor].astype(int).eq(level).to_numpy(dtype=float)
                columns.append(train_dummy)
                target_columns.append(target_dummy)
                count = int(train_dummy.sum())
                term_rows.append({
                    "term": f"{predictor}[{level}]", "term_raw": predictor,
                    "term_kind": "categorical_contrast", "term_level": str(level),
                    "reference_level": str(reference), "transform": "dummy_vs_reference",
                    "center": 0.0, "scale": 1.0, "n_nonzero_train": count,
                    "identifiable": count > 0,
                })
        else:
            raise ValueError(f"Predictor has no declared transform: {predictor}")
    offset_train = train["tract_area_km2"].astype(float).to_numpy()
    offset_target = target["tract_area_km2"].astype(float).to_numpy()
    if np.any(offset_train <= 0) or np.any(offset_target <= 0):
        raise ValueError("Non-positive exposure in design block")
    return DesignBlock(
        x_train=np.column_stack(columns), x_target=np.column_stack(target_columns),
        offset_train=np.log(offset_train), offset_target=np.log(offset_target),
        y_train=train[experiment.outcome].astype(float).to_numpy(),
        y_target=target[experiment.outcome].astype(float).to_numpy(),
        terms=pd.DataFrame(term_rows),
    )


def nb2_nll(
    params: np.ndarray, y: np.ndarray, offset: np.ndarray, design: np.ndarray,
    ridge: float, penalty: np.ndarray,
) -> float:
    beta = params[:-1]
    theta = float(np.exp(np.clip(params[-1], -10.0, 10.0)))
    eta = np.clip(offset + design @ beta, -30.0, 30.0)
    mu = np.exp(eta)
    loglik = (
        special.gammaln(y + theta) - special.gammaln(theta) - special.gammaln(y + 1.0)
        + theta * (np.log(theta) - np.log(theta + mu))
        + y * (eta - np.log(theta + mu))
    )
    return float(-np.sum(loglik) + 0.5 * ridge * np.dot(beta * penalty, beta))


def numeric_hessian(function: Any, point: np.ndarray) -> np.ndarray:
    n_parameters = len(point)
    hessian = np.empty((n_parameters, n_parameters), dtype=float)
    steps = 1e-4 * (1.0 + np.abs(point))
    central = function(point)
    for i in range(n_parameters):
        step_i = np.zeros(n_parameters)
        step_i[i] = steps[i]
        hessian[i, i] = (
            function(point + step_i) - 2.0 * central + function(point - step_i)
        ) / steps[i] ** 2
        for j in range(i + 1, n_parameters):
            step_j = np.zeros(n_parameters)
            step_j[j] = steps[j]
            value = (
                function(point + step_i + step_j) - function(point + step_i - step_j)
                - function(point - step_i + step_j) + function(point - step_i - step_j)
            ) / (4.0 * steps[i] * steps[j])
            hessian[i, j] = value
            hessian[j, i] = value
    return 0.5 * (hessian + hessian.T)


def fit_nb2(
    y: np.ndarray, offset: np.ndarray, design: np.ndarray, ridge: float,
    terms: pd.DataFrame,
) -> NBFit:
    n_parameters = design.shape[1]
    penalty = np.ones(n_parameters, dtype=float)
    penalty[0] = 0.0
    total_exposure = float(np.sum(np.exp(offset)))
    intercept_start = math.log(max((float(np.sum(y)) + 0.5) / total_exposure, 1e-12))
    bounds = [(None, None)] * n_parameters + [(-10.0, 10.0)]
    best: optimize.OptimizeResult | None = None
    for theta_start in (0.2, 1.0, 5.0):
        initial = np.zeros(n_parameters + 1, dtype=float)
        initial[0] = intercept_start
        initial[-1] = math.log(theta_start)
        result = optimize.minimize(
            nb2_nll, initial, args=(y, offset, design, ridge, penalty),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 3000, "maxls": 100, "ftol": 1e-11, "gtol": 1e-7},
        )
        if best is None or result.fun < best.fun:
            best = result
    if best is None or not np.isfinite(best.fun):
        raise RuntimeError("Negative-binomial fit failed")
    point = np.asarray(best.x, dtype=float)
    objective = lambda value: nb2_nll(value, y, offset, design, ridge, penalty)  # noqa: E731
    hessian = numeric_hessian(objective, point)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    floor = max(float(np.max(eigenvalues)) * 1e-10, 1e-10)
    inverse_eigenvalues = 1.0 / np.maximum(eigenvalues, floor)
    covariance = (eigenvectors * inverse_eigenvalues) @ eigenvectors.T
    covariance = 0.5 * (covariance + covariance.T)
    standard_errors = np.sqrt(np.clip(np.diag(covariance)[:-1], 0.0, None))
    standard_errors[~terms["identifiable"].to_numpy(dtype=bool)] = np.nan
    return NBFit(
        beta=point[:-1].copy(), theta=float(np.exp(point[-1])),
        covariance=covariance[:-1, :-1].copy(), se=standard_errors,
        objective=float(best.fun), converged=bool(best.success),
        message=str(best.message), n_iterations=int(best.nit),
    )


def predict_nb2(
    fit: NBFit, design: np.ndarray, offset: np.ndarray, y: np.ndarray
) -> dict[str, np.ndarray]:
    eta = np.clip(offset + design @ fit.beta, -30.0, 30.0)
    mu = np.exp(eta)
    rate = np.exp(np.clip(design @ fit.beta, -30.0, 30.0))
    theta = float(fit.theta)
    probability_zero = (theta / (theta + mu)) ** theta
    probability = theta / (theta + mu)
    variance = mu + mu**2 / theta
    return {
        "mu": mu,
        "rate": rate,
        "p_any": np.clip(1.0 - probability_zero, 0.0, 1.0),
        "pi_low": np.asarray(stats.nbinom.ppf(0.025, theta, probability), dtype=float),
        "pi_high": np.asarray(stats.nbinom.ppf(0.975, theta, probability), dtype=float),
        "pearson_residual": (y - mu) / np.sqrt(variance),
    }


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


def binary_calibration(y_binary: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    positives = int(np.sum(y_binary))
    negatives = int(len(y_binary) - positives)
    if positives < 2 or negatives < 2:
        return math.nan, math.nan
    p = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    x = np.log(p / (1.0 - p))
    if np.std(x) <= 1e-12:
        return math.nan, math.nan
    design = np.column_stack([np.ones(len(x)), x])

    def objective(beta: np.ndarray) -> float:
        eta = np.clip(design @ beta, -30.0, 30.0)
        return float(
            -np.sum(y_binary * eta - np.logaddexp(0.0, eta))
            + 0.5e-4 * beta[1] ** 2
        )

    result = optimize.minimize(objective, np.zeros(2), method="L-BFGS-B")
    beta = np.asarray(result.x, dtype=float)
    if not result.success or not np.all(np.isfinite(beta)) or abs(beta[1]) > 25.0:
        return math.nan, math.nan
    return float(beta[0]), float(beta[1])


def nb_deviance(y: np.ndarray, mu: np.ndarray, theta: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    theta = np.asarray(theta, dtype=float)
    terms = np.zeros_like(y, dtype=float)
    positive = y > 0
    terms[positive] = y[positive] * np.log(y[positive] / mu[positive])
    terms -= (y + theta) * np.log((y + theta) / (mu + theta))
    return 2.0 * terms


def compute_point_metrics(
    y_count: np.ndarray, y_binary: np.ndarray, mu: np.ndarray, p_any: np.ndarray,
    theta: np.ndarray, area: np.ndarray, ranking_rate: np.ndarray,
    baseline_probability: np.ndarray | float,
) -> dict[str, float]:
    prevalence = float(np.mean(y_binary))
    ap = average_precision(y_binary, ranking_rate)
    brier = float(np.mean((y_binary - p_any) ** 2))
    baseline = np.asarray(baseline_probability, dtype=float)
    brier_baseline = float(np.mean((y_binary - baseline) ** 2))
    clipped = np.clip(p_any, 1e-12, 1.0 - 1e-12)
    observed_total = float(np.sum(y_count))
    predicted_total = float(np.sum(mu))
    calibration_intercept, calibration_slope = binary_calibration(y_binary, p_any)
    probability = np.clip(theta / (theta + mu), 1e-12, 1.0 - 1e-12)
    low = stats.nbinom.ppf(0.025, theta, probability)
    high = stats.nbinom.ppf(0.975, theta, probability)
    return {
        "roc_auc": roc_auc(y_binary, ranking_rate),
        "average_precision": ap,
        "ap_over_prevalence": ap / prevalence if prevalence > 0 and np.isfinite(ap) else math.nan,
        "brier_score": brier,
        "brier_skill": 1.0 - brier / brier_baseline if brier_baseline > 0 else math.nan,
        "binary_log_loss": float(-np.mean(y_binary * np.log(clipped) + (1 - y_binary) * np.log(1 - clipped))),
        "nb_deviance": float(np.mean(nb_deviance(y_count, mu, theta))),
        "mae": float(np.mean(np.abs(y_count - mu))),
        "rmse": float(np.sqrt(np.mean((y_count - mu) ** 2))),
        "predicted_total": predicted_total,
        "observed_total": observed_total,
        "total_count_ratio": predicted_total / observed_total if observed_total > 0 else math.nan,
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
        "pi_coverage": float(np.mean((y_count >= low) & (y_count <= high))),
        "mean_pi_width": float(np.mean(high - low)),
        "prevalence": prevalence,
    }


def compute_bootstrap_metrics(
    y_count: np.ndarray, y_binary: np.ndarray, mu: np.ndarray, p_any: np.ndarray,
    theta: np.ndarray, area: np.ndarray, ranking_rate: np.ndarray,
    baseline_probability: np.ndarray,
) -> dict[str, float]:
    prevalence = float(np.mean(y_binary))
    ap = average_precision(y_binary, ranking_rate)
    brier = float(np.mean((y_binary - p_any) ** 2))
    brier_baseline = float(np.mean((y_binary - baseline_probability) ** 2))
    observed_total = float(np.sum(y_count))
    predicted_total = float(np.sum(mu))
    return {
        "roc_auc": roc_auc(y_binary, ranking_rate),
        "average_precision": ap,
        "ap_over_prevalence": ap / prevalence if prevalence > 0 and np.isfinite(ap) else math.nan,
        "brier_score": brier,
        "brier_skill": 1.0 - brier / brier_baseline if brier_baseline > 0 else math.nan,
        "nb_deviance": float(np.mean(nb_deviance(y_count, mu, theta))),
        "mae": float(np.mean(np.abs(y_count - mu))),
        "rmse": float(np.sqrt(np.mean((y_count - mu) ** 2))),
        "total_count_ratio": predicted_total / observed_total if observed_total > 0 else math.nan,
    }


def coefficient_rows(
    experiment: Experiment, fit: NBFit, design: DesignBlock, train: pd.DataFrame,
    fit_scope: str, fold: int,
) -> pd.DataFrame:
    terms = design.terms.copy()
    terms["estimate_z"] = fit.beta
    terms["se_z"] = fit.se
    terms["ci_low_z"] = terms["estimate_z"] - Z975 * terms["se_z"]
    terms["ci_high_z"] = terms["estimate_z"] + Z975 * terms["se_z"]
    terms["estimate_natural"] = terms["estimate_z"] / terms["scale"]
    terms["se_natural"] = terms["se_z"] / terms["scale"]
    terms["ci_low_natural"] = terms["estimate_natural"] - Z975 * terms["se_natural"]
    terms["ci_high_natural"] = terms["estimate_natural"] + Z975 * terms["se_natural"]
    terms["experiment_set"] = experiment.set_name
    terms["model_id"] = experiment.id
    terms["model_label"] = experiment.label
    terms["outcome"] = experiment.outcome
    terms["fit_scope"] = fit_scope
    terms["fold"] = fold
    terms["n_train"] = len(train)
    terms["train_records"] = float(train[experiment.outcome].sum())
    terms["train_positive_cells"] = int(train[experiment.outcome].gt(0).sum())
    terms["theta"] = fit.theta
    terms["dispersion_alpha"] = 1.0 / fit.theta
    terms["objective"] = fit.objective
    terms["converged"] = fit.converged
    terms["optimizer_message"] = fit.message
    terms["n_iterations"] = fit.n_iterations
    terms["uncertainty_method"] = "penalised-likelihood Hessian Laplace approximation"
    return terms


def fold_training_masks(frame: pd.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    holdout = frame["new_fold"].eq(fold).to_numpy()
    train = frame[f"train_included_fold_{fold}"].astype(bool).to_numpy()
    buffer = (~holdout) & (~train)
    if np.any(holdout & train):
        raise RuntimeError(f"Training mask overlaps holdout fold {fold}")
    return holdout, train, buffer


def fit_experiment(
    frame: pd.DataFrame, experiment: Experiment, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ridge = float(config["models"]["ridge_lambda"])
    prediction_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    for fold in (1, 2, 3, 4):
        holdout, train_mask, buffer = fold_training_masks(frame, fold)
        train = frame.loc[train_mask].copy()
        target = frame.loc[holdout].copy()
        design = build_design(train, target, experiment, config)
        fit = fit_nb2(design.y_train, design.offset_train, design.x_train, ridge, design.terms)
        predictions = predict_nb2(fit, design.x_target, design.offset_target, design.y_target)
        training_any = (
            train["n_expanded_records"].gt(0) if experiment.outcome == "n_expanded_records"
            else train["any_record"].astype(bool)
        )
        holdout_any = (
            target["n_expanded_records"].gt(0) if experiment.outcome == "n_expanded_records"
            else target["any_record"].astype(bool)
        )
        baseline_probability = float(training_any.mean())
        metrics = compute_point_metrics(
            design.y_target, holdout_any.to_numpy(dtype=float), predictions["mu"],
            predictions["p_any"], np.full(len(target), fit.theta),
            target["tract_area_km2"].to_numpy(dtype=float), predictions["rate"],
            baseline_probability,
        )
        row: dict[str, Any] = {
            "experiment_set": experiment.set_name, "model_id": experiment.id,
            "model_label": experiment.label, "outcome": experiment.outcome,
            "fold": fold, "fold_label": f"Fold {fold}", "n_train": len(train),
            "n_holdout": len(target), "n_buffer_excluded": int(np.sum(buffer)),
            "buffer_excluded_records": float(frame.loc[buffer, experiment.outcome].sum()),
            "buffer_excluded_positive_cells": int(frame.loc[buffer, experiment.outcome].gt(0).sum()),
            "train_records": float(train[experiment.outcome].sum()),
            "holdout_records": float(target[experiment.outcome].sum()),
            "holdout_expanded_records": float(target["n_expanded_records"].sum()),
            "train_positive_cells": int(train[experiment.outcome].gt(0).sum()),
            "holdout_positive_cells": int(target[experiment.outcome].gt(0).sum()),
            "train_cell_prevalence": float(train["any_record"].mean()),
            "holdout_cell_prevalence": float(holdout_any.mean()),
            "train_record_rate_per_km2": float(train[experiment.outcome].sum() / train["tract_area_km2"].sum()),
            "holdout_record_rate_per_km2": float(target[experiment.outcome].sum() / target["tract_area_km2"].sum()),
            "holdout_outside_daly": int(target["domain_abbreviation"].eq("unassigned").sum()),
            "holdout_domains": json.dumps(target["domain_abbreviation"].value_counts().to_dict(), sort_keys=True),
            "train_domains": json.dumps(train["domain_abbreviation"].value_counts().to_dict(), sort_keys=True),
            "theta": fit.theta, "dispersion_alpha": 1.0 / fit.theta,
            "converged": fit.converged, "optimizer_message": fit.message,
            "metric_note": (
                "Only one holdout record: discrimination is analytically unstable"
                if fold == 4 else ""
            ),
        }
        row.update(metrics)
        fold_rows.append(row)
        prediction_frame = target[[
            "support_id", "x_m", "y_m", "domain_abbreviation", "new_fold",
            "new_fold_label", "tract_area_km2", "any_record", "n_expanded_records",
        ]].copy()
        prediction_frame["experiment_set"] = experiment.set_name
        prediction_frame["model_id"] = experiment.id
        prediction_frame["model_label"] = experiment.label
        prediction_frame["outcome"] = experiment.outcome
        prediction_frame["observed_count"] = design.y_target
        prediction_frame["observed_any"] = holdout_any.astype(int).to_numpy()
        prediction_frame["predicted_count"] = predictions["mu"]
        prediction_frame["predicted_rate_per_km2"] = predictions["rate"]
        prediction_frame["predicted_any_probability"] = predictions["p_any"]
        prediction_frame["pi_low_count"] = predictions["pi_low"]
        prediction_frame["pi_high_count"] = predictions["pi_high"]
        prediction_frame["pearson_residual"] = predictions["pearson_residual"]
        prediction_frame["theta"] = fit.theta
        prediction_frame["baseline_train_probability"] = baseline_probability
        prediction_frames.append(prediction_frame)
        coefficient_frames.append(
            coefficient_rows(experiment, fit, design, train, f"fold_{fold}", fold)
        )

    full_design = build_design(frame, frame, experiment, config)
    full_fit = fit_nb2(
        full_design.y_train, full_design.offset_train, full_design.x_train,
        ridge, full_design.terms,
    )
    coefficient_frames.append(
        coefficient_rows(experiment, full_fit, full_design, frame, "full_data_descriptive", 0)
    )
    prediction_frame = pd.concat(prediction_frames, ignore_index=True)
    pooled = compute_point_metrics(
        prediction_frame["observed_count"].to_numpy(dtype=float),
        prediction_frame["observed_any"].to_numpy(dtype=float),
        prediction_frame["predicted_count"].to_numpy(dtype=float),
        prediction_frame["predicted_any_probability"].to_numpy(dtype=float),
        prediction_frame["theta"].to_numpy(dtype=float),
        prediction_frame["tract_area_km2"].to_numpy(dtype=float),
        prediction_frame["predicted_rate_per_km2"].to_numpy(dtype=float),
        prediction_frame["baseline_train_probability"].to_numpy(dtype=float),
    )
    pooled_row: dict[str, Any] = {
        "experiment_set": experiment.set_name, "model_id": experiment.id,
        "model_label": experiment.label, "outcome": experiment.outcome,
        "aggregation": "pooled_oof", "n_support": len(prediction_frame),
        "n_positive_cells": int(prediction_frame["observed_any"].sum()),
        "observed_records": float(prediction_frame["observed_count"].sum()),
        "observed_rate_per_km2": float(
            prediction_frame["observed_count"].sum() / prediction_frame["tract_area_km2"].sum()
        ),
        "n_folds": 4,
        "metric_note": "Pooled out-of-fold predictions; records are dominated by Fold 1",
    }
    pooled_row.update(pooled)
    fold_metrics = pd.DataFrame(fold_rows)
    mean_row: dict[str, Any] = {
        "experiment_set": experiment.set_name, "model_id": experiment.id,
        "model_label": experiment.label, "outcome": experiment.outcome,
        "aggregation": "fold_mean", "n_support": math.nan,
        "n_positive_cells": math.nan, "observed_records": math.nan,
        "observed_rate_per_km2": math.nan, "n_folds": 4,
        "metric_note": "Descriptive equal-weight mean; Fold 4 is unstable",
    }
    for metric in POINT_METRICS:
        mean_row[metric] = float(fold_metrics[metric].mean(skipna=True))
        mean_row[f"{metric}_fold_sd"] = float(fold_metrics[metric].std(skipna=True, ddof=1))
    metrics_output = pd.DataFrame([pooled_row, mean_row])
    return prediction_frame, metrics_output, pd.concat(coefficient_frames, ignore_index=True), fold_metrics


def predictor_diagnostics(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    continuous = config["models"]["continuous"]
    rows: list[dict[str, Any]] = []
    transformed: dict[str, np.ndarray] = {}
    for variable, spec in continuous.items():
        values = _continuous_values(frame[variable], spec.get("transform", "identity"))
        transformed[variable] = values
        for metric, value in {
            "n": float(len(values)), "missing": float(np.isnan(values).sum()),
            "min": float(np.min(values)), "q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)), "mean": float(np.mean(values)),
            "q75": float(np.quantile(values, 0.75)), "max": float(np.max(values)),
            "sd": float(np.std(values, ddof=1)),
            "skewness": float(stats.skew(values, bias=False)),
        }.items():
            rows.append({
                "diagnostic_type": "distribution", "variable_1": variable,
                "variable_2": "", "metric": metric, "value": value,
                "n": len(values), "notes": spec.get("transform", "identity"),
            })
    variables = list(transformed)
    for experiment in experiments_from_config(config):
        model_variables = [
            predictor for predictor in experiment.predictors
            if predictor in continuous
        ]
        if len(model_variables) < 2:
            continue
        matrix = np.column_stack([transformed[variable] for variable in model_variables])
        correlation = np.corrcoef(matrix, rowvar=False)
        vif = np.diag(np.linalg.pinv(correlation, hermitian=True))
        for variable, value in zip(model_variables, vif):
            rows.append({
                "diagnostic_type": "vif", "variable_1": variable, "variable_2": "",
                "metric": "vif_transformed", "value": float(value), "n": len(frame),
                "notes": f"within-model continuous block: {experiment.id}",
            })
    for first, second in combinations(variables, 2):
        rows.append({
            "diagnostic_type": "correlation", "variable_1": first,
            "variable_2": second, "metric": "pearson_transformed",
            "value": float(stats.pearsonr(transformed[first], transformed[second]).statistic),
            "n": len(frame), "notes": "declared transform",
        })
        rows.append({
            "diagnostic_type": "correlation", "variable_1": first,
            "variable_2": second, "metric": "spearman_raw",
            "value": float(stats.spearmanr(frame[first], frame[second]).statistic),
            "n": len(frame), "notes": "untransformed values",
        })
    return pd.DataFrame(rows)


def calibration_bins(predictions: pd.DataFrame, model_ids: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = [("pooled_oof", predictions)] + [
        (f"fold_{fold}", predictions.loc[predictions["new_fold"].eq(fold)])
        for fold in (1, 2, 3, 4)
    ]
    for scope_name, scope in scopes:
        if scope.empty:
            continue
        for model_id in model_ids:
            subset = scope.loc[scope["model_id"].eq(model_id)].copy()
            if subset.empty:
                continue
            order = np.argsort(subset["predicted_any_probability"].to_numpy(), kind="mergesort")
            ranks = np.empty(len(subset), dtype=int)
            ranks[order] = np.arange(len(subset))
            subset["bin"] = np.minimum((ranks * 10) // max(len(subset), 1), 9)
            for bin_id, group in subset.groupby("bin", sort=True):
                rows.append({
                    "model_id": model_id, "evaluation_scope": scope_name,
                    "bin": int(bin_id) + 1, "n": len(group),
                    "predicted_any_mean": float(group["predicted_any_probability"].mean()),
                    "observed_any_rate": float(group["observed_any"].mean()),
                    "predicted_count_mean": float(group["predicted_count"].mean()),
                    "observed_count_mean": float(group["observed_count"].mean()),
                    "predicted_count_sum": float(group["predicted_count"].sum()),
                    "observed_count_sum": float(group["observed_count"].sum()),
                    "observed_rate_per_km2": float(
                        group["observed_count"].sum() / group["tract_area_km2"].sum()
                    ),
                    "predicted_rate_per_km2": float(
                        group["predicted_count"].sum() / group["tract_area_km2"].sum()
                    ),
                })
    return pd.DataFrame(rows)


def moran_permutation_test(
    residuals: np.ndarray, coordinates: np.ndarray, k_neighbors: int,
    n_permutations: int, seed: int,
) -> dict[str, float | int]:
    residuals = np.asarray(residuals, dtype=float)
    valid = np.isfinite(residuals)
    residuals = residuals[valid]
    coordinates = coordinates[valid]
    n = len(residuals)
    if n <= k_neighbors + 2:
        return {
            "n": n, "moran_i": math.nan, "expected_i": math.nan,
            "permutation_sd": math.nan, "z_permutation": math.nan,
            "p_positive": math.nan, "p_two_sided": math.nan,
        }
    tree = cKDTree(coordinates)
    _, neighbours = tree.query(coordinates, k=k_neighbors + 1)
    neighbours = neighbours[:, 1:]
    centered = residuals - np.mean(residuals)

    def statistic(values: np.ndarray) -> float:
        values = values - np.mean(values)
        denominator = float(np.dot(values, values))
        if denominator <= 0:
            return math.nan
        return float(np.dot(values, np.mean(values[neighbours], axis=1)) / denominator)

    observed = statistic(centered)
    rng = np.random.default_rng(seed)
    permuted = np.empty(n_permutations, dtype=float)
    batch_size = 100
    for start in range(0, n_permutations, batch_size):
        stop = min(start + batch_size, n_permutations)
        batch = np.empty((stop - start, n), dtype=float)
        for row in range(stop - start):
            batch[row] = centered[rng.permutation(n)]
        lags = np.mean(batch[:, neighbours], axis=2)
        denominator = np.einsum("ij,ij->i", batch, batch)
        permuted[start:stop] = np.divide(
            np.einsum("ij,ij->i", batch, lags), denominator,
            out=np.full(stop - start, np.nan), where=denominator > 0,
        )
    finite = permuted[np.isfinite(permuted)]
    positive_count = int(np.sum(finite >= observed))
    negative_count = int(np.sum(finite <= observed))
    p_positive = (positive_count + 1.0) / (len(finite) + 1.0)
    p_negative = (negative_count + 1.0) / (len(finite) + 1.0)
    return {
        "n": n, "moran_i": observed, "expected_i": -1.0 / (n - 1),
        "permutation_sd": float(np.std(finite, ddof=1)),
        "z_permutation": float((observed - np.mean(finite)) / np.std(finite, ddof=1))
        if np.std(finite, ddof=1) > 0 else math.nan,
        "p_positive": p_positive, "p_two_sided": min(1.0, 2.0 * min(p_positive, p_negative)),
    }


def residual_moran_table(predictions: pd.DataFrame, model_ids: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = [("pooled_oof", predictions)] + [
        (f"fold_{fold}", predictions.loc[predictions["new_fold"].eq(fold)])
        for fold in (1, 2, 3, 4)
    ]
    for scope_name, scope in scopes:
        for model_index, model_id in enumerate(model_ids):
            subset = scope.loc[scope["model_id"].eq(model_id)].copy()
            if subset.empty:
                continue
            seed = 20261106 + model_index * 101 + (
                0 if scope_name == "pooled_oof" else int(scope_name.split("_")[1])
            )
            test = moran_permutation_test(
                subset["pearson_residual"].to_numpy(dtype=float),
                subset[["x_m", "y_m"]].to_numpy(dtype=float), 6, 999, seed,
            )
            test.update({
                "model_id": model_id, "evaluation_scope": scope_name,
                "residual_type": "Pearson count residual", "k_neighbors": 6,
                "n_permutations": 999, "seed": seed,
            })
            rows.append(test)
    return pd.DataFrame(rows)


def spatial_block_resamples(
    coordinates: np.ndarray, y_binary: np.ndarray, block_size_km: float,
    n_resamples: int, seed: int,
) -> list[np.ndarray]:
    block_size_m = float(block_size_km * 1000.0)
    block_x = np.floor(coordinates[:, 0] / block_size_m).astype(np.int64)
    block_y = np.floor(coordinates[:, 1] / block_size_m).astype(np.int64)
    labels = np.asarray(
        [f"{x}:{y}" for x, y in zip(block_x.tolist(), block_y.tolist())],
        dtype=object,
    )
    unique_blocks, inverse = np.unique(labels, return_inverse=True)
    indices_by_block = [np.flatnonzero(inverse == index) for index in range(len(unique_blocks))]
    rng = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    attempts = 0
    max_attempts = max(1000, n_resamples * 100)
    while len(accepted) < n_resamples and attempts < max_attempts:
        attempts += 1
        draws = rng.integers(0, len(unique_blocks), size=len(unique_blocks))
        sample = np.concatenate([indices_by_block[index] for index in draws])
        positives = int(np.sum(y_binary[sample]))
        if 0 < positives < len(sample):
            accepted.append(sample)
    if len(accepted) < n_resamples:
        raise RuntimeError(
            f"Only {len(accepted)} valid block resamples; requested {n_resamples}"
        )
    return accepted


def bootstrap_and_pairwise(
    frame: pd.DataFrame, predictions: pd.DataFrame, experiments: list[Experiment],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    block_size_km = float(config["models"]["block_bootstrap"]["block_size_km"])
    pooled_resamples = int(config["models"]["block_bootstrap"]["n_resamples"])
    fold_resamples = int(config["models"]["fold_bootstrap"]["n_resamples"])
    seed = int(config["models"]["block_bootstrap"]["seed"])
    all_ids = [experiment.id for experiment in experiments]
    ordered = frame.sort_values("support_id").reset_index(drop=True)
    support_ids = ordered["support_id"].tolist()
    model_arrays: dict[str, dict[str, np.ndarray]] = {}
    for experiment in experiments:
        subset = predictions.loc[predictions["model_id"].eq(experiment.id)].set_index("support_id").loc[support_ids]
        model_arrays[experiment.id] = {
            "y": subset["observed_count"].to_numpy(dtype=float),
            "yb": subset["observed_any"].to_numpy(dtype=float),
            "mu": subset["predicted_count"].to_numpy(dtype=float),
            "rate": subset["predicted_rate_per_km2"].to_numpy(dtype=float),
            "p": subset["predicted_any_probability"].to_numpy(dtype=float),
            "theta": subset["theta"].to_numpy(dtype=float),
            "area": subset["tract_area_km2"].to_numpy(dtype=float),
            "baseline": subset["baseline_train_probability"].to_numpy(dtype=float),
        }
    scopes: list[tuple[str, np.ndarray]] = [("pooled_oof", np.arange(len(ordered)))]
    for fold in (1, 2, 3, 4):
        scopes.append((f"fold_{fold}", np.flatnonzero(ordered["new_fold"].eq(fold).to_numpy())))
    metric_samples: dict[tuple[str, str, str], np.ndarray] = {}
    point_values: dict[tuple[str, str, str], float] = {}
    bootstrap_rows: list[dict[str, Any]] = []
    for scope_index, (scope_name, global_indices) in enumerate(scopes):
        coordinates = ordered.loc[global_indices, ["x_m", "y_m"]].to_numpy(dtype=float)
        requested = pooled_resamples if scope_name == "pooled_oof" else fold_resamples
        accepted_local = spatial_block_resamples(
            coordinates, model_arrays[all_ids[0]]["yb"][global_indices],
            block_size_km, requested, seed + scope_index * 10007,
        )
        accepted_global = [global_indices[sample] for sample in accepted_local]
        for model_id in all_ids:
            arrays = model_arrays[model_id]
            direct = compute_bootstrap_metrics(
                arrays["y"][global_indices], arrays["yb"][global_indices],
                arrays["mu"][global_indices], arrays["p"][global_indices],
                arrays["theta"][global_indices], arrays["area"][global_indices],
                arrays["rate"][global_indices],
                arrays["baseline"][global_indices],
            )
            experiment = next(entry for entry in experiments if entry.id == model_id)
            for metric in BOOTSTRAP_METRICS:
                point_values[(model_id, scope_name, metric)] = direct[metric]
                values = np.array([
                    compute_bootstrap_metrics(
                        arrays["y"][index], arrays["yb"][index], arrays["mu"][index],
                        arrays["p"][index], arrays["theta"][index],
                        arrays["area"][index], arrays["rate"][index],
                        arrays["baseline"][index],
                    )[metric]
                    for index in accepted_global
                ], dtype=float)
                metric_samples[(model_id, scope_name, metric)] = values
                finite = values[np.isfinite(values)]
                bootstrap_rows.append({
                    "experiment_set": experiment.set_name, "model_id": model_id,
                    "outcome": experiment.outcome, "evaluation_scope": scope_name,
                    "metric": metric, "estimate": direct[metric],
                    "ci_low": float(np.quantile(finite, 0.025)) if len(finite) else math.nan,
                    "ci_high": float(np.quantile(finite, 0.975)) if len(finite) else math.nan,
                    "n_valid": int(len(finite)), "n_requested": len(values),
                    "block_size_km": block_size_km, "seed": seed + scope_index * 10007,
                })
    lower_is_better = {"brier_score", "nb_deviance", "mae", "rmse"}
    pairs = [
        ("S0_BASE", "N0_AREA"), ("S1_HM", "S0_BASE"),
        ("S2_DENS50", "S0_BASE"), ("S3_HM_DENS50", "S0_BASE"),
        ("S1_HM", "S2_DENS50"), ("S3_HM_DENS50", "S1_HM"),
        ("S3_HM_DENS50", "S2_DENS50"),
    ]
    pair_rows: list[dict[str, Any]] = []
    for scope_name, _ in scopes:
        for metric in BOOTSTRAP_METRICS:
            for model_a, model_b in pairs:
                values_a = metric_samples[(model_a, scope_name, metric)]
                values_b = metric_samples[(model_b, scope_name, metric)]
                finite = np.isfinite(values_a) & np.isfinite(values_b)
                deltas = values_a[finite] - values_b[finite]
                if not len(deltas):
                    continue
                estimate = point_values[(model_a, scope_name, metric)] - point_values[
                    (model_b, scope_name, metric)
                ]
                p_low = (1.0 + np.sum(deltas <= 0.0)) / (len(deltas) + 1.0)
                p_high = (1.0 + np.sum(deltas >= 0.0)) / (len(deltas) + 1.0)
                direction = "lower_is_better" if metric in lower_is_better else "higher_is_better"
                if np.isfinite(estimate):
                    favoured = (
                        model_a if (estimate < 0 if direction == "lower_is_better" else estimate > 0)
                        else model_b
                    )
                else:
                    favoured = ""
                pair_rows.append({
                    "evaluation_scope": scope_name, "metric": metric,
                    "model_a": model_a, "model_b": model_b,
                    "delta_a_minus_b": estimate,
                    "ci_low": float(np.quantile(deltas, 0.025)),
                    "ci_high": float(np.quantile(deltas, 0.975)),
                    "p_two_sided": float(min(1.0, 2.0 * min(p_low, p_high))),
                    "n_valid": int(len(deltas)), "direction": direction,
                    "favoured_by_point_estimate": favoured,
                    "bootstrap_unit": f"spatial_block_{block_size_km:g}km",
                })
    return pd.DataFrame(bootstrap_rows), pd.DataFrame(pair_rows)


def write_text(text: str, path: str | Path) -> Path:
    destination = abs_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(temporary, destination)
    return destination


MODEL_COLORS = {
    "N0_AREA": "#7f7f7f", "S0_BASE": "#1b9e77", "S1_HM": "#d95f02",
    "S2_DENS50": "#7570b3", "S3_HM_DENS50": "#e7298a",
}


def make_model_comparison_figure(
    predictions: pd.DataFrame, fold_metrics: pd.DataFrame, model_ids: list[str], path: Path
) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    for model_id in model_ids:
        subset = predictions.loc[predictions["model_id"].eq(model_id)].sort_values(
            "predicted_rate_per_km2", ascending=False
        )
        y = subset["observed_any"].to_numpy(dtype=int)
        tp = np.cumsum(y)
        fp = np.cumsum(1 - y)
        tpr = np.r_[0.0, tp / max(tp[-1], 1)]
        fpr = np.r_[0.0, fp / max(fp[-1], 1)]
        axes[0].plot(fpr, tpr, label=model_id, color=MODEL_COLORS.get(model_id), linewidth=1.8)
        recall = tp / max(tp[-1], 1)
        precision = tp / np.arange(1, len(tp) + 1)
        axes[1].plot(recall, precision, label=model_id, color=MODEL_COLORS.get(model_id), linewidth=1.6)
    axes[0].plot([0, 1], [0, 1], color="#bbbbbb", linestyle="--", linewidth=0.8)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="Pooled out-of-fold ROC")
    axes[1].axhline(
        predictions["observed_any"].mean(), color="#bbbbbb", linestyle="--",
        linewidth=0.8, label="Prevalence",
    )
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Pooled out-of-fold precision-recall")
    for axis in axes[:2]:
        axis.legend(frameon=False, fontsize=8)
    primary_fold = fold_metrics.loc[fold_metrics["model_id"].isin(model_ids)]
    for model_id in model_ids:
        subset = primary_fold.loc[primary_fold["model_id"].eq(model_id)].sort_values("fold")
        axes[2].plot(
            subset["fold"], subset["roc_auc"], marker="o", label=model_id,
            color=MODEL_COLORS.get(model_id), linewidth=1.4,
        )
    axes[2].set_xticks([1, 2, 3, 4], ["Fold 1", "Fold 2", "Fold 3", "Fold 4"])
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set(ylabel="ROC-AUC", title="Fold-specific discrimination")
    axes[2].axhline(0.5, color="#bbbbbb", linestyle="--", linewidth=0.8)
    axes[2].text(
        1.03, 0.97, "Fold 4: 1 record\n(estimate unstable)",
        ha="left", va="top", fontsize=7.5, color="#8c2d04", linespacing=1.2,
    )
    axes[2].legend(frameon=False, fontsize=8, loc="lower right")
    for axis in axes:
        axis.grid(alpha=0.2, linewidth=0.5)
    figure.suptitle("Stage 4 count-model comparison", fontsize=13, fontweight="bold")
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def make_calibration_residual_figure(
    bins: pd.DataFrame, residual_moran: pd.DataFrame, fold_metrics: pd.DataFrame,
    model_ids: list[str], path: Path,
) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 9.2))
    pooled = bins.loc[bins["evaluation_scope"].eq("pooled_oof")]
    for model_id in model_ids:
        subset = pooled.loc[pooled["model_id"].eq(model_id)].sort_values("predicted_any_mean")
        axes[0, 0].plot(
            subset["predicted_any_mean"], subset["observed_any_rate"], marker="o",
            markersize=3, linewidth=1.3, label=model_id, color=MODEL_COLORS.get(model_id),
        )
        axes[0, 1].plot(
            subset["predicted_count_mean"], subset["observed_count_mean"], marker="o",
            markersize=3, linewidth=1.3, label=model_id, color=MODEL_COLORS.get(model_id),
        )
    binary_limit = max(pooled["predicted_any_mean"].max(), pooled["observed_any_rate"].max())
    axes[0, 0].plot([0, binary_limit], [0, binary_limit], color="#bbbbbb", linestyle="--", linewidth=0.8)
    axes[0, 0].set(xlabel="Mean predicted probability", ylabel="Observed positive-cell rate", title="Binary calibration")
    count_limit = max(pooled["predicted_count_mean"].max(), pooled["observed_count_mean"].max())
    axes[0, 1].plot([0, count_limit], [0, count_limit], color="#bbbbbb", linestyle="--", linewidth=0.8)
    axes[0, 1].set(xlabel="Mean predicted count", ylabel="Mean observed count", title="Count calibration")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].legend(frameon=False, fontsize=8)
    coverage = fold_metrics.loc[fold_metrics["model_id"].isin(model_ids)]
    for model_id in model_ids:
        subset = coverage.loc[coverage["model_id"].eq(model_id)].sort_values("fold")
        axes[1, 0].plot(
            subset["fold"], subset["pi_coverage"], marker="o", linewidth=1.4,
            label=model_id, color=MODEL_COLORS.get(model_id),
        )
    axes[1, 0].axhline(0.95, color="#bbbbbb", linestyle="--", linewidth=0.8)
    axes[1, 0].set_xticks([1, 2, 3, 4], ["Fold 1", "Fold 2", "Fold 3", "Fold 4"])
    axes[1, 0].set_ylim(0.0, 1.02)
    axes[1, 0].set(ylabel="95% central PI coverage", title="Predictive-interval coverage")
    axes[1, 0].text(
        3.98, 0.55, "Fold 4: 1 record\n(coverage not informative)",
        ha="right", va="center", fontsize=7.5, color="#8c2d04", linespacing=1.2,
    )
    axes[1, 0].legend(frameon=False, fontsize=8, loc="lower right")
    pooled_moran = residual_moran.loc[residual_moran["evaluation_scope"].eq("pooled_oof")]
    positions = np.arange(len(model_ids))
    values = [
        pooled_moran.loc[pooled_moran["model_id"].eq(model_id), "moran_i"].iloc[0]
        for model_id in model_ids
    ]
    axes[1, 1].bar(
        positions, values, color=[MODEL_COLORS.get(model_id) for model_id in model_ids],
        alpha=0.85, width=0.65,
    )
    axes[1, 1].axhline(0.0, color="#555555", linewidth=0.8)
    axes[1, 1].set_xticks(positions, model_ids, rotation=25, ha="right")
    axes[1, 1].set(ylabel="Moran's I", title="Pooled Pearson-residual spatial autocorrelation")
    for axis in axes.ravel():
        axis.grid(alpha=0.2, linewidth=0.5)
    figure.suptitle("Stage 4 calibration, interval and residual diagnostics", fontsize=13, fontweight="bold")
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def make_coefficient_figure(coefficient_summary: pd.DataFrame, path: Path) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 6.0))
    term_labels = {
        "BASE_DIST_FAULT_KM": "Distance to mapped fault (log)",
        "BASE_DIST_CONTACT_KM": "Distance to lithology contact (log)",
        "BASE_BOUGUER_MGAL": "Bouguer gravity",
        "STR_DIST_NAMED_HM_KM": "Distance to HM structures (log)",
        "STR_NAMED_LINE_DENSITY_50KM": "Named-structure density, 50 km",
    }

    def readable_term(term: str) -> str:
        if term.startswith("BASE_CONTACT_CLASS["):
            return f"Contact class {term.split('[', 1)[1].rstrip(']')}"
        return term_labels.get(term, term)

    s3 = coefficient_summary.loc[
        coefficient_summary["model_id"].eq("S3_HM_DENS50")
    ].sort_values("estimate_full").copy()
    s3["term_label"] = s3["term"].map(readable_term)
    positions = np.arange(len(s3))
    axes[0].errorbar(
        s3["estimate_full"], positions,
        xerr=[s3["estimate_full"] - s3["ci_low_full"], s3["ci_high_full"] - s3["estimate_full"]],
        fmt="o", markersize=4, color="#253494", ecolor="#777777", linewidth=1.0, capsize=2,
    )
    axes[0].axvline(0.0, color="#bbbbbb", linestyle="--", linewidth=0.8)
    axes[0].set_yticks(positions, s3["term_label"])
    axes[0].set(xlabel="Natural-scale coefficient (log-rate scale)", title="S3 descriptive full-data coefficients")
    structural_terms = ["STR_DIST_NAMED_HM_KM", "STR_NAMED_LINE_DENSITY_50KM"]
    structural = coefficient_summary.loc[
        coefficient_summary["term"].isin(structural_terms)
        & coefficient_summary["model_id"].isin(["S1_HM", "S2_DENS50", "S3_HM_DENS50"])
    ].copy()
    colors = {"S1_HM": "#d95f02", "S2_DENS50": "#7570b3", "S3_HM_DENS50": "#e7298a"}
    offsets = {"S1_HM": -0.16, "S2_DENS50": 0.0, "S3_HM_DENS50": 0.16}
    y_positions = {term: index for index, term in enumerate(structural_terms)}
    labelled_models: set[str] = set()
    for model_id, subset in structural.groupby("model_id"):
        indexed = subset.set_index("term")
        for term in structural_terms:
            if term not in indexed.index:
                continue
            row = indexed.loc[term]
            add_label = model_id not in labelled_models
            axes[1].errorbar(
                row["estimate_full"], y_positions[term] + offsets[model_id],
                xerr=[[row["estimate_full"] - row["ci_low_full"]], [row["ci_high_full"] - row["estimate_full"]]],
                fmt="o", markersize=4, color=colors[model_id], ecolor=colors[model_id],
                linewidth=1.0, capsize=2,
                label=model_id if add_label else None,
            )
            if add_label:
                labelled_models.add(model_id)
    axes[1].axvline(0.0, color="#bbbbbb", linestyle="--", linewidth=0.8)
    axes[1].set_yticks(list(y_positions.values()), ["HM distance", "50 km named-structure density"])
    axes[1].set(xlabel="Natural-scale coefficient (log-rate scale)", title="Structural coefficients across models")
    axes[1].legend(frameon=False, fontsize=8, title="Model", title_fontsize=8)
    for axis in axes:
        axis.grid(axis="x", alpha=0.2, linewidth=0.5)
    figure.suptitle("Stage 4 coefficient summaries", fontsize=13, fontweight="bold")
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def make_model_map(predictions: pd.DataFrame, support_gpkg: str, path: Path) -> Path:
    catchments = read_gdf(support_gpkg, layer="stage4_support_catchments")
    model_ids = ["S0_BASE", "S1_HM", "S2_DENS50", "S3_HM_DENS50"]
    all_rates = predictions.loc[predictions["model_id"].isin(model_ids), "predicted_rate_per_km2"]
    vmin = max(float(all_rates.quantile(0.01)), 1e-5)
    vmax = max(float(all_rates.quantile(0.995)), vmin * 1.01)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    bounds = catchments.total_bounds
    x_pad = (bounds[2] - bounds[0]) * 0.03
    y_pad = (bounds[3] - bounds[1]) * 0.03
    xlim = (bounds[0] - x_pad, bounds[2] + x_pad)
    ylim = (bounds[1] - y_pad, bounds[3] + y_pad)
    map_ratio = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0])
    panel_width = 4.4
    panel_height = panel_width / map_ratio
    gap_x, gap_y = 0.55, 0.45
    left, right = 0.45, 0.35
    bottom, top = 1.55, 0.95
    figure_width = left + 2 * panel_width + gap_x + right
    figure_height = bottom + 2 * panel_height + gap_y + top
    figure = plt.figure(figsize=(figure_width, figure_height))
    axes = []
    for row in range(2):
        for col in range(2):
            x0 = left + col * (panel_width + gap_x)
            y0 = bottom + (1 - row) * (panel_height + gap_y)
            axes.append(
                figure.add_axes([
                    x0 / figure_width, y0 / figure_height,
                    panel_width / figure_width, panel_height / figure_height,
                ])
            )

    for axis, model_id in zip(axes, model_ids):
        subset = predictions.loc[predictions["model_id"].eq(model_id)].copy()
        plotted = catchments.merge(
            subset[["support_id", "predicted_rate_per_km2", "observed_count"]],
            on="support_id", how="left", validate="one_to_one",
        )
        plotted.plot(
            ax=axis, column="predicted_rate_per_km2", cmap="magma",
            norm=norm, edgecolor="none", linewidth=0.0,
        )
        positives = subset.loc[subset["observed_count"].gt(0)]
        axis.scatter(
            positives["x_m"], positives["y_m"],
            s=np.clip(positives["observed_count"] * 4.0, 8.0, 42.0),
            facecolors="none", edgecolors="#00e5ff", linewidths=0.7,
        )
        pooled_auc = roc_auc(
            subset["observed_any"].to_numpy(dtype=int),
            subset["predicted_rate_per_km2"].to_numpy(dtype=float),
        )
        axis.set_title(
            f"{model_id}  |  pooled OOF AUC {pooled_auc:.3f}\n"
            f"observed positive cells: {int(subset['observed_count'].gt(0).sum())}",
            fontsize=9, pad=4,
        )
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        axis.set_aspect("equal", adjustable="box")
        axis.set_axis_off()

    scalar_map = ScalarMappable(norm=norm, cmap="magma")
    scalar_map.set_array([])
    colorbar_axis = figure.add_axes([
        left / figure_width, 0.95 / figure_height,
        (2 * panel_width + gap_x) / figure_width, 0.14 / figure_height,
    ])
    colorbar = figure.colorbar(scalar_map, cax=colorbar_axis, orientation="horizontal")
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.set_label("Predicted documented records per km² (log scale)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)

    legend_counts = [1, 5, 10]
    legend_handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="", markerfacecolor="none",
            markeredgecolor="#00e5ff", markeredgewidth=0.7,
            markersize=float(np.sqrt(np.clip(count * 4.0, 8.0, 42.0))),
            label=f"{count} record" + ("s" if count != 1 else ""),
        )
        for count in legend_counts
    ]
    figure.legend(
        handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01),
        ncol=3, frameon=False, fontsize=8,
        title="Observed records in catchment", title_fontsize=8,
    )
    figure.suptitle(
        "Out-of-fold predicted documented-record intensity (records km$^{-2}$)",
        fontsize=13, fontweight="bold", y=1.0 - 0.20 / figure_height,
    )
    figure.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)
    return path


def markdown_table(frame: pd.DataFrame, columns: list[str], labels: list[str] | None = None) -> str:
    display = frame[columns].copy()
    if labels is not None:
        display.columns = labels
    formatted = pd.DataFrame(index=display.index)
    for column in display.columns:
        values = display[column]
        if pd.api.types.is_integer_dtype(values.dtype):
            formatted[column] = values.map(lambda value: "NA" if pd.isna(value) else str(int(value)))
        elif pd.api.types.is_float_dtype(values.dtype):
            formatted[column] = values.map(
                lambda value: "NA" if not np.isfinite(value) else f"{float(value):.3f}"
            )
        else:
            formatted[column] = values.astype(str)
    lines = ["| " + " | ".join(formatted.columns) + " |", "|" + "|".join(["---"] * len(formatted.columns)) + "|"]
    for row in formatted.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def nominal_hex_support_audit(
    config: dict[str, Any], paths: dict[str, str], frame: pd.DataFrame
) -> dict[str, Any]:
    hexagons = read_gdf(
        config["inputs"]["support_gpkg"]["path"],
        layer="structural_support_polygons",
    )[["support_id", "geometry"]]
    occurrences = read_gdf(paths["support_gpkg"], layer="stage4_occurrences")
    in_tract_mask = occurrences["in_tract"].astype(str).str.lower().eq("true")
    in_tract = occurrences.loc[in_tract_mask].copy()
    joined = in_tract[["record_id", "geometry"]].sjoin(
        hexagons, how="left", predicate="intersects"
    )
    inside_ids = set(joined.loc[joined["support_id"].notna(), "record_id"].astype(str))
    assigned = in_tract.merge(
        frame[["support_id", "area_overlap_class"]],
        left_on="assigned_support_id", right_on="support_id",
        how="left", validate="many_to_one",
    )
    assigned["inside_hex"] = assigned["record_id"].astype(str).isin(inside_ids)
    by_class = assigned.groupby("area_overlap_class", dropna=False).agg(
        assigned_records=("record_id", "size"),
        inside_hex_records=("inside_hex", "sum"),
    )
    return {
        "hex_area_km2": float(frame["nominal_hex_overlap_area_km2"].sum()),
        "tract_area_km2": float(frame["tract_area_km2"].sum()),
        "n_in_tract_records": int(len(in_tract)),
        "n_outside_hex": int((~assigned["inside_hex"]).sum()),
        "records_inside_by_class": {
            str(key): int(value)
            for key, value in by_class["inside_hex_records"].items()
        },
    }


def write_report(
    path: Path, frame: pd.DataFrame, model_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame, pairwise: pd.DataFrame, calibration: pd.DataFrame,
    residual_moran: pd.DataFrame, coefficient_summary: pd.DataFrame,
    predictor_diagnostics_table: pd.DataFrame, data_manifest: dict[str, Any],
    support_audit: dict[str, Any],
) -> Path:
    primary_ids = ["N0_AREA", "S0_BASE", "S1_HM", "S2_DENS50", "S3_HM_DENS50"]
    pooled = model_metrics.loc[model_metrics["aggregation"].eq("pooled_oof")].set_index("model_id")
    fold_mean = model_metrics.loc[model_metrics["aggregation"].eq("fold_mean")].set_index("model_id")
    pooled_primary = pooled.loc[primary_ids].reset_index()
    fold_mean_primary = fold_mean.loc[primary_ids].reset_index()
    pooled_pairs = pairwise.loc[
        pairwise["evaluation_scope"].eq("pooled_oof")
        & pairwise["metric"].isin(["roc_auc", "average_precision", "brier_score", "nb_deviance"])
    ].copy()
    moran_primary = residual_moran.loc[
        residual_moran["evaluation_scope"].eq("pooled_oof")
        & residual_moran["model_id"].isin(primary_ids)
    ].set_index("model_id").loc[primary_ids].reset_index()
    hm_density = predictor_diagnostics_table.loc[
        predictor_diagnostics_table["diagnostic_type"].eq("correlation")
        & predictor_diagnostics_table["variable_1"].eq("STR_DIST_NAMED_HM_KM")
        & predictor_diagnostics_table["variable_2"].eq("STR_NAMED_LINE_DENSITY_50KM")
        & predictor_diagnostics_table["metric"].eq("pearson_transformed")
    ]
    hm_density_value = float(hm_density["value"].iloc[0]) if len(hm_density) else math.nan
    structural = coefficient_summary.loc[
        coefficient_summary["term"].isin(["STR_DIST_NAMED_HM_KM", "STR_NAMED_LINE_DENSITY_50KM"])
        & coefficient_summary["experiment_set"].eq("primary")
    ].sort_values(["model_id", "term"])
    fold_table = fold_metrics.loc[
        fold_metrics["model_id"].isin(primary_ids)
    ].sort_values(["model_id", "fold"])[[
        "model_id", "fold", "holdout_positive_cells", "holdout_records",
        "holdout_cell_prevalence", "roc_auc", "ap_over_prevalence", "brier_skill",
    ]]
    for column in ("fold", "holdout_positive_cells", "holdout_records"):
        fold_table[column] = fold_table[column].astype(int)
    fold_composition = (
        frame.groupby("new_fold", as_index=False)
        .agg(
            support_cells=("support_id", "size"),
            positive_cells=("any_record", "sum"),
            records=("n_records", "sum"),
            outside_daly=("domain_abbreviation", lambda values: values.eq("unassigned").sum()),
        )
    )
    fold_composition["cell_prevalence"] = (
        fold_composition["positive_cells"] / fold_composition["support_cells"]
    )
    pooled_table = pooled_primary[[
        "model_id", "roc_auc", "average_precision", "ap_over_prevalence", "brier_score",
        "brier_skill", "nb_deviance", "total_count_ratio", "calibration_slope", "pi_coverage",
    ]].copy()
    fold_lookup = fold_mean_primary.set_index("model_id")
    for metric in ["roc_auc", "average_precision", "ap_over_prevalence", "brier_skill", "pi_coverage"]:
        pooled_table[f"fold_mean_{metric}"] = (
            fold_lookup[metric].reindex(pooled_table["model_id"]).to_numpy()
        )
    pairwise_display = pooled_pairs[[
        "metric", "model_a", "model_b", "delta_a_minus_b", "ci_low", "ci_high",
        "p_two_sided", "favoured_by_point_estimate",
    ]]
    structural_display = structural[[
        "model_id", "term", "estimate_full", "ci_low_full", "ci_high_full",
        "fold_mean_natural", "fold_sd_natural", "fold_sign_consistency",
    ]]
    moran_display = moran_primary[["model_id", "moran_i", "p_positive", "n"]]
    hex_percent = 100.0 * support_audit["hex_area_km2"] / support_audit["tract_area_km2"]
    outside_percent = 100.0 * support_audit["n_outside_hex"] / support_audit["n_in_tract_records"]
    inside_99 = support_audit["records_inside_by_class"].get("full_ge_99pct", 0)
    inside_80 = inside_99 + support_audit["records_inside_by_class"].get("overlap_80_to_lt_99pct", 0)
    informative_folds = fold_metrics.loc[
        fold_metrics["model_id"].isin(primary_ids) & fold_metrics["fold"].isin([1, 2, 3])
    ].groupby("model_id").agg(
        mean_auc=("roc_auc", "mean"),
        mean_ap_ratio=("ap_over_prevalence", "mean"),
    ).reindex(primary_ids)
    informative_text = ", ".join(
        f"{model_id} {row['mean_auc']:.3f} (AP/prevalence {row['mean_ap_ratio']:.2f})"
        for model_id, row in informative_folds.iterrows()
    )
    gravity_rows = model_metrics.loc[
        model_metrics["model_id"].isin(["S0_BASE", "S0_NO_GRAVITY"])
        & model_metrics["aggregation"].eq("pooled_oof")
    ].set_index("model_id")
    gravity_text = ""
    if {"S0_BASE", "S0_NO_GRAVITY"}.issubset(gravity_rows.index):
        gravity_text = (
            "The inherited-gravity sensitivity is essentially indistinguishable from S0: "
            f"pooled AUC {float(gravity_rows.loc['S0_NO_GRAVITY', 'roc_auc']):.4f} versus "
            f"{float(gravity_rows.loc['S0_BASE', 'roc_auc']):.4f}, and Brier skill "
            f"{float(gravity_rows.loc['S0_NO_GRAVITY', 'brier_skill']):.4f} versus "
            f"{float(gravity_rows.loc['S0_BASE', 'brier_skill']):.4f}. "
            "Gravity is not driving the baseline, and the simpler no-gravity specification is defensible."
        )
    nrb_records = frame.loc[frame["domain_abbreviation"].eq("NRB")]
    nrb_counts = nrb_records.groupby("domain_name", sort=False)["n_records"].sum().sort_values(ascending=False)
    nrb_total = int(nrb_records["n_records"].sum())
    other_records = int(frame.loc[
        ~frame["domain_abbreviation"].isin(["NRB", "unassigned"]), "n_records"
    ].sum())
    outside_records = int(frame.loc[frame["domain_abbreviation"].eq("unassigned"), "n_records"].sum())
    nrb_count_text = " and ".join(str(int(value)) for value in nrb_counts.values)
    other_text = "none" if other_records == 0 else str(other_records)
    outside_text = (
        "1 record falls" if outside_records == 1
        else f"{outside_records} records fall"
    )
    domain_text = (
        f"Record distribution across Daly domains is highly uneven: {nrb_total} of {int(frame['n_records'].sum())} "
        f"records fall in NRB, with {nrb_count_text} in its mapped subdomains; CRZ, MMSB and NKB contain "
        f"{other_text}, and {outside_text} outside the Daly envelope. The inventory therefore cannot identify "
        "domain-varying relationships outside NRB; any such terms would be prior-driven."
    )
    lines = [
        "# Stage 4 initial count-model comparison", "",
        "## Scientific scope", "",
        "These models estimate the intensity of historically compiled occurrence records per unit tract-supported area. They do not estimate a detection-free probability of geological mineralisation, and a cell without a record is not treated as verified barren ground.", "",
        "The support is a nearest-centre Voronoi partition of the permissive-tract footprint around the 2,193 accepted cell centres. This retains every centre, covers the tract exactly once without gaps or overlap, and avoids discarding records that fall between the original nominal hexagons. Each catchment keeps its actual tract-supported area as a log exposure offset. All 285 catchments outside the reconstructed Daly envelope remain in the analysis as `unassigned`; no domain label was extrapolated. Daly domains are not model predictors in this stage.", "",
        f"Nominal 10 km² hexagons intersect {support_audit['hex_area_km2']:,.1f} km² of the {support_audit['tract_area_km2']:,.1f} km² tract ({hex_percent:.2f}%). {support_audit['n_outside_hex']} of {support_audit['n_in_tract_records']} in-tract records ({outside_percent:.1f}%) fall outside every nominal hexagon; all are nearest-centre assigned to partial-overlap catchments. Under literal hexagon support, only {inside_99} records would remain in the ≥99%-overlap population and {inside_80} in the ≥80%-overlap population. The catchment partition is therefore used so that every tract location and every record has positive support. Nominal hex overlap area remains recorded for audit; it is not the primary exposure.", "",
        "## Data and validation", "",
        f"- Data build: `{data_manifest['build_id']}`; status `{data_manifest['status']}`.",
        f"- Support cells: {len(frame):,}; positive cells: {int(frame['any_record'].gt(0).sum()):,}; records: {int(frame['n_records'].sum()):,}.",
        f"- Tract-supported area: {float(frame['tract_area_km2'].sum()):,.3f} km².",
        "- Primary validation: four contiguous equal-count belt-axis folds with a 10 km training buffer.",
        "- Fold 4 contains one recorded occurrence. Its fold-specific discrimination is shown but must not be treated as a stable estimate.", "",
        markdown_table(fold_composition, list(fold_composition.columns), [
            "Fold", "Support cells", "Positive cells", "Records",
            "Outside Daly envelope", "Cell prevalence",
        ]), "",
        domain_text, "",
        "## Discrimination: pooled and fold-specific", "",
        markdown_table(pooled_table, list(pooled_table.columns), [
            "Model", "Pooled ROC-AUC", "Pooled AP", "Pooled AP / prevalence", "Pooled Brier",
            "Pooled Brier skill", "NB deviance", "Predicted / observed count",
            "Calibration slope", "PI coverage", "Fold-mean ROC-AUC", "Fold-mean AP",
            "Fold-mean AP / prevalence", "Fold-mean Brier skill", "Fold-mean PI coverage",
        ]), "",
        "Pooled out-of-fold ROC-AUC mixes within-sector ranking with between-sector shifts in prevalence and fitted intercept. It is not a clean measure of within-sector transfer here. The area-only N0 reference demonstrates the problem: its per-area rate is constant within each fitted fold, yet its pooled AUC falls below 0.5 because Fold 1 has high occurrence prevalence and a relatively low training intercept. The fold-mean is the more conservative summary, but it is still a four-fold descriptive average and Fold 4 has only one record.", "",
        f"Because Fold 4 contains one record and 280 of the 285 catchments outside the Daly envelope, it is excluded from an informative-fold summary. Over Folds 1–3, mean ROC-AUC is {informative_text}; Fold 4 is reported but not interpreted.", "",
        "The constant area-only model N0 is a reference, not a geological model. Brier skill uses a fold-specific constant baseline derived only from training data. The calibration slope is a binary any-record diagnostic; the separate calibration bins provide the count-level check.", "",
        "### Fold-specific transfer", "",
        markdown_table(fold_table, list(fold_table.columns), [
            "Model", "Fold", "Positive cells", "Records", "Cell prevalence", "ROC-AUC",
            "AP / prevalence", "Brier skill",
        ]), "",
        "## Incremental comparisons", "",
        "Deltas use the same 50 km spatial-block resamples for paired models. The sign convention is `model_a - model_b`. Intervals are percentile intervals from 1,000 pooled block resamples. P-values are descriptive and account for spatial clustering only approximately.", "",
        markdown_table(pairwise_display, list(pairwise_display.columns), [
            "Metric", "Model A", "Model B", "A - B", "95% low", "95% high",
            "p (two-sided)", "Point-estimate preference",
        ]), "",
        "These paired deltas are also pooled across sectors and therefore inherit the between-fold composition effect. The fold-specific deltas are retained in `stage4_pairwise_deltas_v1.csv`; they show where an apparent pooled gain is or is not reproduced within a sector.", "",
        "Nothing here establishes that a larger model is preferable. S1 and S2 test different structural representations. S3 mixes them, and the observed HM-distance / 50 km density Pearson correlation is "
        f"{hm_density_value:.3f} after the declared transforms; the separate coefficients are therefore not independent evidence.", "",
        gravity_text, "",
        "## Structural coefficient summaries", "",
        "Full-data estimates are descriptive and receive a Laplace approximation to the penalised-likelihood Hessian. Fold estimates are fitted independently with training-only standardisation. Natural-scale coefficients refer to a one-unit change in the declared transformed predictor; categorical contrasts are log rate ratios against the reference class.", "",
        markdown_table(structural_display, list(structural_display.columns), [
            "Model", "Term", "Full estimate", "95% low", "95% high",
            "Fold mean", "Fold SD", "Sign consistency",
        ]), "",
        "A coefficient on HM distance or named-structure density is an association with documented-record intensity. It is not by itself evidence of a causal structural control on mineralisation.", "",
        "## Calibration and spatial residuals", "",
        markdown_table(moran_display, list(moran_display.columns), [
            "Model", "Moran's I", "One-sided p (positive)", "n",
        ]), "",
        "Moran's I is computed on pooled out-of-fold Pearson count residuals using six nearest neighbours and 999 residual permutations. Positive values indicate residual spatial clustering beyond the fitted covariates. These are permutation diagnostics, not adjustments for predictor or label uncertainty.", "",
        "## Limitations carried forward", "",
        "- The outcome is a compiled-record inventory, not a complete census of mineralisation; observation and reporting intensity remain unmodelled.",
        "- The gravity covariate is inherited from the nearest V14 cell; `S0_NO_GRAVITY` is the required sensitivity for that provenance limitation.",
        "- Contact class is inherited from the V14 legacy class because the rebuilt nearest-contact class is degenerate at coincident contacts; rebuilt class and tie diagnostics remain in the data table.",
        "- Fold 1 contributes 104 of 184 records, while Fold 4 contributes one. Pooled performance is dominated by the western sectors.",
        "- The 10 km buffer reduces, but does not eliminate, spatial dependence between training and holdout cells.",
        f"- The primary exposure is the tract-supported Voronoi catchment, not the nominal hexagon intersection. This is deliberate: literal hexagon support would discard {support_audit['n_outside_hex']} of {support_audit['n_in_tract_records']} in-tract records and {100.0 - hex_percent:.1f}% of tract area. A hex-only sensitivity would therefore change the response, not merely the offset.",
        "- Predictors are sampled at the original cell centres. Catchment area corrects exposure, but it does not represent within-catchment variation in fault, contact or gravity values.",
        "- These are penalised maximum-likelihood fits, not MCMC posterior samples. Coefficient intervals are local approximations.",
        "- No domain-varying coefficients or detection surface were introduced in Stage 4.", "",
        "## Reproduction", "",
        "The fitter writes its manifest, predictions, fold metrics, coefficient tables, bootstrap tables, calibration bins, residual diagnostics and figures under `outputs/stage4/` and `data/processed/stage4/`. The V14 benchmark, Stage 3 structural layers and manuscript were not modified.", "",
    ]
    return write_text("\n".join(lines), path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage4_v1.json")
    args = parser.parse_args()
    config_path = abs_path(args.config)
    config = load_config(config_path)
    paths = output_paths(config)
    frame, data_manifest, provenance = validate_inputs(config, paths, config_path)
    support_audit = nominal_hex_support_audit(config, paths, frame)
    experiments = experiments_from_config(config)
    primary_ids = [experiment.id for experiment in experiments if experiment.set_name == "primary"]

    predictions_parts: list[pd.DataFrame] = []
    metric_parts: list[pd.DataFrame] = []
    coefficient_parts: list[pd.DataFrame] = []
    fold_metric_parts: list[pd.DataFrame] = []
    for experiment in experiments:
        print(f"Fitting {experiment.id}: {experiment.label}", flush=True)
        prediction, metrics, coefficients, fold_metrics = fit_experiment(frame, experiment, config)
        predictions_parts.append(prediction)
        metric_parts.append(metrics)
        coefficient_parts.append(coefficients)
        fold_metric_parts.append(fold_metrics)
    predictions_frame = pd.concat(predictions_parts, ignore_index=True)
    model_metrics = pd.concat(metric_parts, ignore_index=True)
    coefficients = pd.concat(coefficient_parts, ignore_index=True)
    fold_metrics = pd.concat(fold_metric_parts, ignore_index=True)

    summary_rows: list[dict[str, Any]] = []
    for (model_id, term), group in coefficients.groupby(["model_id", "term"], sort=False):
        full = group.loc[group["fit_scope"].eq("full_data_descriptive")].iloc[0]
        folds = group.loc[
            group["fit_scope"].str.startswith("fold_") & group["identifiable"].astype(bool)
        ]
        natural = folds["estimate_natural"].dropna()
        positive_folds = int((natural > 0).sum())
        negative_folds = int((natural < 0).sum())
        finite_folds = int(natural.size)
        summary_rows.append({
            "experiment_set": full["experiment_set"], "model_id": model_id,
            "model_label": full["model_label"], "outcome": full["outcome"],
            "term": term, "term_raw": full["term_raw"], "term_kind": full["term_kind"],
            "term_level": full["term_level"], "reference_level": full["reference_level"],
            "transform": full["transform"], "estimate_full": full["estimate_natural"],
            "se_full": full["se_natural"], "ci_low_full": full["ci_low_natural"],
            "ci_high_full": full["ci_high_natural"],
            "fold_mean_natural": float(natural.mean()) if finite_folds else math.nan,
            "fold_sd_natural": float(natural.std(ddof=1)) if finite_folds > 1 else math.nan,
            "fold_min_natural": float(natural.min()) if finite_folds else math.nan,
            "fold_max_natural": float(natural.max()) if finite_folds else math.nan,
            "n_folds_finite": finite_folds, "n_folds_positive": positive_folds,
            "n_folds_negative": negative_folds,
            "fold_sign_consistency": max(positive_folds, negative_folds) / finite_folds
            if finite_folds else math.nan,
            "uncertainty_method": full["uncertainty_method"],
        })
    coefficient_summary = pd.DataFrame(summary_rows)
    diagnostics = predictor_diagnostics(frame, config)
    calibration = calibration_bins(predictions_frame, [experiment.id for experiment in experiments])
    residual_moran = residual_moran_table(predictions_frame, [experiment.id for experiment in experiments])
    bootstrap, pairwise = bootstrap_and_pairwise(frame, predictions_frame, experiments, config)

    output_files = {
        "model_predictions": paths["model_predictions"],
        "fold_metrics": paths["fold_metrics"], "model_metrics": paths["model_metrics"],
        "metric_bootstrap": paths["metric_bootstrap"],
        "pairwise_deltas": paths["pairwise_deltas"],
        "coefficients_by_fold": paths["coefficients_by_fold"],
        "coefficient_summary": paths["coefficient_summary"],
        "calibration_bins": paths["calibration_bins"],
        "residual_moran": paths["residual_moran"],
        "predictor_diagnostics": paths["predictor_diagnostics"],
    }
    write_csv(predictions_frame, output_files["model_predictions"])
    write_csv(fold_metrics, output_files["fold_metrics"])
    write_csv(model_metrics, output_files["model_metrics"])
    write_csv(bootstrap, output_files["metric_bootstrap"])
    write_csv(pairwise, output_files["pairwise_deltas"])
    write_csv(coefficients, output_files["coefficients_by_fold"])
    write_csv(coefficient_summary, output_files["coefficient_summary"])
    write_csv(calibration, output_files["calibration_bins"])
    write_csv(residual_moran, output_files["residual_moran"])
    write_csv(diagnostics, output_files["predictor_diagnostics"])

    figure_dir = Path(paths["figure_dir"])
    figure_files = {
        "model_comparison": figure_dir / "stage4_model_comparison.png",
        "calibration_residuals": figure_dir / "stage4_calibration_residuals.png",
        "coefficients": figure_dir / "stage4_coefficients.png",
        "model_maps": figure_dir / "stage4_model_maps.png",
    }
    make_model_comparison_figure(predictions_frame, fold_metrics, primary_ids, figure_files["model_comparison"])
    make_calibration_residual_figure(
        calibration, residual_moran, fold_metrics, primary_ids,
        figure_files["calibration_residuals"],
    )
    make_coefficient_figure(coefficient_summary, figure_files["coefficients"])
    make_model_map(
        predictions_frame.loc[predictions_frame["model_id"].isin(primary_ids)],
        paths["support_gpkg"], figure_files["model_maps"],
    )
    report_path = figure_dir / "stage4_model_report_v1.md"
    write_report(
        report_path, frame, model_metrics, fold_metrics, pairwise, calibration,
        residual_moran, coefficient_summary, diagnostics, data_manifest, support_audit,
    )

    output_records = {name: file_record(path, role=name) for name, path in output_files.items()}
    output_records.update({name: file_record(path, role=name) for name, path in figure_files.items()})
    output_records["model_report"] = file_record(report_path, role="model_report")
    run_id = build_id()
    manifest = {
        "schema_version": "1.0.0", "build_id": run_id, "status": "fitted",
        "purpose": "Stage 4 initial negative-binomial count-model comparison on verified spatial support",
        "config": {"path": relative_posix(config_path), "sha256": provenance["config_sha256"]},
        "data_manifest": file_record(paths["data_manifest"], role="data_manifest"),
        "data_validation": file_record(paths["data_validation"], role="data_validation"),
        "data_build_id": data_manifest["build_id"], "data_status": data_manifest["status"],
        "model_table": {
            "path": relative_posix(paths["model_table"]),
            "sha256": provenance["model_table_sha256"],
        },
        "protected_inputs_unchanged": provenance["protected"],
        "distribution": config["models"]["distribution"],
        "outcome": config["models"]["outcome"], "offset": config["models"]["offset"],
        "ridge_lambda": config["models"]["ridge_lambda"],
        "inference": "penalised maximum likelihood; Laplace Hessian intervals",
        "experiments": [{
            "id": experiment.id, "label": experiment.label, "set": experiment.set_name,
            "outcome": experiment.outcome, "predictors": list(experiment.predictors),
        } for experiment in experiments],
        "fold_design": {
            "scheme": config["folds"]["primary_scheme"],
            "buffer_km": config["folds"]["primary_buffer_km"],
            "folds": {
                str(fold): {
                    "n_holdout": int(frame["new_fold"].eq(fold).sum()),
                    "n_positive_cells": int(frame.loc[frame["new_fold"].eq(fold), "any_record"].gt(0).sum()),
                    "n_records": int(frame.loc[frame["new_fold"].eq(fold), "n_records"].sum()),
                } for fold in (1, 2, 3, 4)
            },
            "warning": "Fold 4 contains one record; its fold-specific discrimination is unstable",
        },
        "software": {
            **software_versions(), "scipy": __import__("scipy").__version__,
            "matplotlib": __import__("matplotlib").__version__,
        },
        "outputs": output_records,
        "validation": {"status": "pending", "path": relative_posix(paths["model_validation"])},
        "notes": [
            "The response is documented-record intensity, not true geological prospectivity.",
            "All 2,193 support catchments are retained; tract-supported area is the exposure.",
            "The 285 cells outside the Daly envelope remain unassigned and are not extrapolated.",
            "Observed HM geometry is primary; bridged HM is sensitivity only.",
            "S3 predictors are correlated and must not be interpreted as independent evidence.",
            "No domain-varying coefficients or detection surface are included in this stage.",
        ],
    }
    manifest_path = write_json(manifest, paths["model_manifest"])
    print(json.dumps({
        "status": "fitted", "build_id": run_id, "models": len(experiments),
        "predictions": len(predictions_frame),
        "fold4_records": int(frame.loc[frame["new_fold"].eq(4), "n_records"].sum()),
        "manifest": relative_posix(manifest_path), "report": relative_posix(report_path),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
