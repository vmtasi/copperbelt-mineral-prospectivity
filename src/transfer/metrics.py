"""Metric summaries and resampling for the transfer analysis."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from src.transfer import config
from src.v15.metrics import (
    clip_probability,
    log_score,
    metric_row,
    roc_auc,
)


def calibration_in_the_large(y: np.ndarray, probability: np.ndarray) -> float:
    """Logistic recalibration intercept with the slope fixed at 1.

    Solves for ``a`` in ``logit P(y=1) = a + logit(p)``. Zero means the mean
    predicted risk matches the observed rate on the logit scale; negative
    values mean the predictions are too high. This is the pre-specified
    calibration-in-the-large. It differs from the intercept of the joint
    (intercept, slope) recalibration reported as ``calibration_intercept``.
    """
    from scipy.optimize import minimize_scalar
    from scipy.special import logit

    y = np.asarray(y, dtype=float)
    offset = logit(clip_probability(np.asarray(probability, dtype=float)))

    def objective(intercept: float) -> float:
        eta = intercept + offset
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    fitted = minimize_scalar(objective, bounds=(-20.0, 20.0), method="bounded")
    return float(fitted.x) if fitted.success else float("nan")


def class_stratified_bootstrap(
    y: np.ndarray,
    probability: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    replicates: int = config.BOOTSTRAP_REPLICATES,
    seed: int = config.BOOTSTRAP_SEED,
) -> tuple[float, float]:
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sample = np.r_[
            rng.choice(positives, size=len(positives), replace=True),
            rng.choice(negatives, size=len(negatives), replace=True),
        ]
        draws[index] = statistic(y[sample], probability[sample])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def spatial_block_ids(frame: pd.DataFrame, size_m: float = config.INNER_BLOCK_SIZE_M) -> np.ndarray:
    x = frame["centroid_x"].to_numpy(dtype=float)
    y = frame["centroid_y"].to_numpy(dtype=float)
    bx = np.floor((x - x.min()) / size_m).astype(int)
    by = np.floor((y - y.min()) / size_m).astype(int)
    return bx * 100000 + by


def spatial_block_bootstrap_metric(
    frame: pd.DataFrame,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    replicates: int = config.BOOTSTRAP_REPLICATES,
    seed: int = config.BOOTSTRAP_SEED,
) -> tuple[float, float]:
    work = frame.reset_index(drop=True).copy()
    work["_block"] = spatial_block_ids(work)
    block_values = work["_block"].unique()
    if len(block_values) < 2:
        return float("nan"), float("nan")
    groups = {block: work.loc[work["_block"] == block] for block in block_values}
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = rng.choice(block_values, size=len(block_values), replace=True)
        merged = pd.concat([groups[block] for block in sampled], ignore_index=True)
        draws[index] = statistic(
            merged["y"].to_numpy(dtype=int),
            merged["prob"].to_numpy(dtype=float),
        )
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def spatial_block_bootstrap_delta(
    frame: pd.DataFrame,
    left_score: np.ndarray,
    right_score: np.ndarray,
    replicates: int = config.BOOTSTRAP_REPLICATES,
    seed: int = config.BOOTSTRAP_SEED,
) -> tuple[float, float]:
    work = frame.reset_index(drop=True).copy()
    work["delta"] = np.asarray(left_score) - np.asarray(right_score)
    work["_block"] = spatial_block_ids(work)
    grouped = work.groupby("_block")["delta"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sample = rng.integers(0, len(grouped), size=len(grouped))
        draws[index] = sums[sample].sum() / counts[sample].sum()
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def metric_summary(
    predictions: pd.DataFrame,
    label: str,
    seed: int = config.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    row = metric_row(predictions, label)
    y = predictions["y"].to_numpy(dtype=int)
    probability = predictions["prob"].to_numpy(dtype=float)
    auc_lower, auc_upper = class_stratified_bootstrap(y, probability, roc_auc, seed=seed)
    log_lower, log_upper = spatial_block_bootstrap_metric(
        predictions,
        lambda labels, values: float(log_score(labels, values).mean()),
        seed=seed,
    )
    row.update(
        {
            "calibration_in_the_large": calibration_in_the_large(y, probability),
            "auc_ci_lower": auc_lower,
            "auc_ci_upper": auc_upper,
            "mean_log_score_block_ci_lower": log_lower,
            "mean_log_score_block_ci_upper": log_upper,
        }
    )
    return row


def metric_fields() -> list[str]:
    return [
        "sample",
        "n",
        "positives",
        "prevalence",
        "mean_log_score",
        "total_log_score",
        "auc",
        "auc_ci_lower",
        "auc_ci_upper",
        "pr_auc",
        "pr_auc_baseline",
        "brier",
        "brier_reference",
        "brier_skill",
        "calibration_in_the_large",
        "calibration_intercept",
        "calibration_slope",
        "mean_probability",
        "observed_rate",
        "mean_log_score_block_ci_lower",
        "mean_log_score_block_ci_upper",
    ]

