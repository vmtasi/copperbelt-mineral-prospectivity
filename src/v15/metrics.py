"""Held-out metrics, calibration and spatial-block uncertainty for V15.

The core metrics are implemented with NumPy/SciPy so they remain usable under
the project's QGIS test shim, which blocks pyarrow and therefore scikit-learn.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata

from src.v15 import config


def clip_probability(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), config.PCELL, 1.0 - config.PCELL)


def log_score(y: np.ndarray, probability: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    p = clip_probability(probability)
    return y * np.log(p) + (1.0 - y) * np.log(1.0 - p)


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney ROC-AUC of any real-valued score (higher = more likely positive).

    The score is ranked as given. It must not be clipped to the probability
    range: clipping a raw covariate (area in km2, gravity in mGal) collapses
    every value above 1 into a tie and silently corrupts the AUC.
    """
    y = np.asarray(y, dtype=int)
    p = np.asarray(score, dtype=float)
    positives = y == 1
    negatives = ~positives
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(p, method="average")
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y: np.ndarray, probability: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    p = clip_probability(probability)
    order = np.argsort(-p, kind="mergesort")
    y_sorted = y[order]
    if y_sorted.sum() == 0:
        return float("nan")
    cumulative = np.cumsum(y_sorted)
    precision = cumulative / np.arange(1, len(y_sorted) + 1)
    return float((precision * y_sorted).sum() / y_sorted.sum())


def brier_score(y: np.ndarray, probability: np.ndarray) -> float:
    return float(np.mean((np.asarray(y, dtype=float) - clip_probability(probability)) ** 2))


def calibration(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype=float)
    x = logit(clip_probability(probability))

    def objective(parameters: np.ndarray) -> float:
        intercept, slope = parameters
        eta = intercept + slope * x
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    fitted = minimize(objective, np.array([0.0, 1.0]), method="BFGS")
    if not fitted.success:
        fitted = minimize(
            objective,
            np.array([0.0, 1.0]),
            method="L-BFGS-B",
            bounds=[(-20.0, 20.0), (-20.0, 20.0)],
        )
    if not fitted.success or not np.isfinite(fitted.x).all():
        return float("nan"), float("nan")
    return float(fitted.x[0]), float(fitted.x[1])


def metric_row(frame: pd.DataFrame, label: str = "pooled") -> dict[str, Any]:
    y = frame["y"].to_numpy(int)
    p = clip_probability(frame["prob"].to_numpy(float))
    reference = clip_probability(frame["p_ref_train"].to_numpy(float))
    scores = log_score(y, p)
    brier = brier_score(y, p)
    brier_ref = brier_score(y, reference)
    intercept, slope = calibration(y, p)
    return {
        "sample": label,
        "n": int(len(frame)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "mean_log_score": float(scores.mean()),
        "total_log_score": float(scores.sum()),
        "auc": roc_auc(y, p),
        "pr_auc": average_precision(y, p),
        "pr_auc_baseline": float(y.mean()),
        "brier": brier,
        "brier_reference": brier_ref,
        "brier_skill": float(1.0 - brier / brier_ref),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "mean_probability": float(p.mean()),
        "observed_rate": float(y.mean()),
    }


def reliability_table(frame: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    y = frame["y"].to_numpy(int)
    p = clip_probability(frame["prob"].to_numpy(float))
    order = np.argsort(p)
    groups = np.array_split(order, bins)
    rows = []
    for index, group in enumerate(groups, start=1):
        if len(group) == 0:
            continue
        rows.append(
            {
                "bin": index,
                "n": int(len(group)),
                "mean_probability": float(p[group].mean()),
                "observed_rate": float(y[group].mean()),
                "lower_probability": float(p[group].min()),
                "upper_probability": float(p[group].max()),
                "positives": int(y[group].sum()),
            }
        )
    return pd.DataFrame(rows)


def assign_spatial_blocks(frame: pd.DataFrame, size_m: float = 25000.0) -> np.ndarray:
    x = frame["centroid_x"].to_numpy(float)
    y = frame["centroid_y"].to_numpy(float)
    bx = np.floor((x - x.min()) / size_m).astype(int)
    by = np.floor((y - y.min()) / size_m).astype(int)
    return bx * 100000 + by


def block_bootstrap_paired(
    left: pd.DataFrame,
    right: pd.DataFrame,
    replicates: int = 2000,
    seed: int = config.BOOTSTRAP_SEED,
    size_m: float = 25000.0,
) -> dict[str, Any]:
    merged = left[["id", "fold", "y", "prob", "centroid_x", "centroid_y"]].merge(
        right[["id", "prob"]], on="id", suffixes=("_left", "_right"), validate="one_to_one"
    )
    left_score = log_score(merged["y"], merged["prob_left"])
    right_score = log_score(merged["y"], merged["prob_right"])
    delta = left_score - right_score
    merged = merged.assign(delta=delta, block=assign_spatial_blocks(merged, size_m=size_m))
    blocks = merged["block"].unique()
    block_stats = merged.groupby("block")["delta"].agg(["sum", "count"]).loc[blocks]
    block_sum = block_stats["sum"].to_numpy(float)
    block_count = block_stats["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sample = rng.integers(0, len(blocks), size=len(blocks))
        draws[index] = float(block_sum[sample].sum() / block_count[sample].sum())
    return {
        "n": int(len(merged)),
        "blocks": int(len(blocks)),
        "mean_delta_log_score": float(delta.mean()),
        "total_delta_log_score": float(delta.sum()),
        "ci_lower": float(np.percentile(draws, 2.5)),
        "ci_upper": float(np.percentile(draws, 97.5)),
        "probability_left_better": float((draws > 0).mean()),
        "replicates": int(replicates),
        "block_size_m": size_m,
    }


def paired_table(predictions: dict[str, pd.DataFrame], bootstrap: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    runs = []
    for left_name, right_name in combinations(sorted(predictions), 2):
        left = predictions[left_name]
        right = predictions[right_name]
        merged = left[["id", "y", "prob"]].merge(
            right[["id", "prob"]], on="id", suffixes=("_left", "_right"), validate="one_to_one"
        )
        delta = log_score(merged["y"], merged["prob_left"]) - log_score(merged["y"], merged["prob_right"])
        row = {
            "left": left_name,
            "right": right_name,
            "n": int(len(merged)),
            "mean_delta_log_score": float(delta.mean()),
            "total_delta_log_score": float(delta.sum()),
            "auc_left": roc_auc(merged["y"], merged["prob_left"]),
            "auc_right": roc_auc(merged["y"], merged["prob_right"]),
            "auc_delta": roc_auc(merged["y"], merged["prob_left"]) - roc_auc(merged["y"], merged["prob_right"]),
        }
        rows.append(row)
        if bootstrap:
            runs.append({"left": left_name, "right": right_name, **block_bootstrap_paired(left, right)})
    return pd.DataFrame(rows), pd.DataFrame(runs)
