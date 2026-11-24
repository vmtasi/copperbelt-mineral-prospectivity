"""Sector definitions and spatial cross-validation helpers."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from src.numerics import PCA
from src.transfer import config


def transfer_axis(frame: pd.DataFrame) -> np.ndarray:
    projection = PCA(n_components=1).fit_transform(
        frame[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    ).ravel()
    axis_km = (projection - projection.min()) / 1000.0
    return axis_km


def gap_partition(frame: pd.DataFrame) -> tuple[np.ndarray, list[float]]:
    axis = transfer_axis(frame)
    positive_positions = np.sort(axis[frame[config.Y_COL].to_numpy(dtype=int) == 1])
    gaps = np.diff(positive_positions)
    order = np.argsort(-gaps, kind="stable")
    maximum = float(axis.max())
    thresholds: list[float] = []
    for index in order:
        candidate = float((positive_positions[index] + positive_positions[index + 1]) / 2.0)
        if candidate <= 100.0 or maximum - candidate < 100.0:
            continue
        trial = np.sort(np.r_[thresholds, candidate])
        bounds = np.r_[0.0, trial, maximum]
        if np.diff(bounds).min() >= 100.0:
            thresholds = trial.tolist()
        if len(thresholds) == 3:
            break
    if len(thresholds) != 3:
        raise ValueError("P-GAP did not produce three feasible cuts")
    thresholds = sorted(thresholds)
    labels = np.digitize(axis, thresholds).astype(int)
    return labels, thresholds


def positive_partition(frame: pd.DataFrame) -> tuple[np.ndarray, list[float]]:
    axis = transfer_axis(frame)
    positive_positions = np.sort(axis[frame[config.Y_COL].to_numpy(dtype=int) == 1])
    if len(positive_positions) != 138:
        raise ValueError("P-POS requires the established 138 positive cells")
    cumulative = [35, 70, 104]  # 35 / 35 / 34 / 34 positives
    thresholds = [
        float((positive_positions[split - 1] + positive_positions[split]) / 2.0)
        for split in cumulative
    ]
    labels = np.digitize(axis, thresholds).astype(int)
    counts = np.bincount(labels, minlength=4)
    positive_counts = np.bincount(
        labels,
        weights=frame[config.Y_COL].to_numpy(dtype=float),
        minlength=4,
    ).astype(int)
    if not np.all((positive_counts == 34) | (positive_counts == 35)):
        raise ValueError(f"P-POS positive counts violate 34-35 rule: {positive_counts}")
    return labels, thresholds


def partition_details(
    frame: pd.DataFrame,
    labels: np.ndarray,
    thresholds: list[float],
) -> pd.DataFrame:
    axis = transfer_axis(frame)
    point_labels = np.digitize(axis[frame[config.Y_COL].to_numpy(dtype=int) == 1], thresholds)
    positive_axis = axis[frame[config.Y_COL].to_numpy(dtype=int) == 1]
    boundaries = np.r_[0.0, thresholds, float(axis.max())]
    rows = []
    for sector in range(4):
        mask = labels == sector
        sector_positive_axis = positive_axis[point_labels == sector]
        within_10 = 0
        for boundary in thresholds:
            within_10 += int(np.sum(np.abs(sector_positive_axis - boundary) <= 10.0))
        domain_counts = frame.loc[mask, "daly_domain"].value_counts()
        rows.append(
            {
                "sector": sector + 1,
                "start_km": float(boundaries[sector]),
                "end_km": float(boundaries[sector + 1]),
                "length_km": float(boundaries[sector + 1] - boundaries[sector]),
                "cells": int(mask.sum()),
                "positives": int(frame.loc[mask, config.Y_COL].sum()),
                "prevalence": float(frame.loc[mask, config.Y_COL].mean()),
                "positives_within_10km_boundaries": within_10,
                "domain_composition": "; ".join(
                    f"{domain}:{int(count)}" for domain, count in domain_counts.items()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_partitions(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = frame[["id", "centroid_x", "centroid_y", config.Y_COL, "daly_domain"]].copy()
    out["axis_km"] = transfer_axis(frame)
    out["fold_primary"] = frame[config.FOLD_COL].to_numpy(dtype=int)
    p_gap, gap_thresholds = gap_partition(frame)
    p_pos, pos_thresholds = positive_partition(frame)
    out["fold_p_gap"] = p_gap
    out["fold_p_pos"] = p_pos
    details = {
        "primary": partition_details(
            frame,
            frame[config.FOLD_COL].to_numpy(dtype=int),
            _canonical_cut_values(out),
        ),
        "p_gap": partition_details(frame, p_gap, gap_thresholds),
        "p_pos": partition_details(frame, p_pos, pos_thresholds),
    }
    metadata = {
        "axis_km_min": 0.0,
        "axis_km_max": float(out["axis_km"].max()),
        "p_gap_cuts": gap_thresholds,
        "p_pos_cuts": pos_thresholds,
    }
    return out, {"details": details, "metadata": metadata}


def _canonical_cut_values(partition: pd.DataFrame) -> list[float]:
    values = []
    for sector in range(3):
        left = partition.loc[partition["fold_primary"] == sector, "axis_km"].max()
        right = partition.loc[partition["fold_primary"] == sector + 1, "axis_km"].min()
        values.append(float((left + right) / 2.0))
    return values


def assign_block_folds(
    frame: pd.DataFrame,
    n_folds: int = 4,
    seed: int = config.BOOTSTRAP_SEED,
    block_size_m: float = config.INNER_BLOCK_SIZE_M,
) -> np.ndarray:
    """Assign intact 25-km blocks to balanced inner folds."""

    work = frame.reset_index(drop=True)
    blocks = block_ids(work, block_size_m=block_size_m)
    block_values = np.unique(blocks)
    if len(block_values) < n_folds:
        raise ValueError("Fewer spatial blocks than requested inner folds")
    rng = np.random.default_rng(seed)
    order = rng.permutation(block_values)
    sizes = {block: int(np.sum(blocks == block)) for block in block_values}
    positive = {
        block: int(work.loc[blocks == block, config.Y_COL].sum()) for block in block_values
    }
    fold_sizes = np.zeros(n_folds, dtype=int)
    positive_sizes = np.zeros(n_folds, dtype=int)
    assignments = np.empty(len(work), dtype=int)
    for block in order:
        score = fold_sizes + 10.0 * positive_sizes
        fold = int(np.argmin(score))
        assignments[blocks == block] = fold
        fold_sizes[fold] += sizes[block]
        positive_sizes[fold] += positive[block]
    for fold in range(n_folds):
        test = assignments == fold
        train = ~test
        if len(np.unique(work.loc[train, config.Y_COL])) < 2:
            raise ValueError(f"Inner fold {fold + 1} has one class in training")
    return assignments


def block_ids(
    frame: pd.DataFrame,
    block_size_m: float = config.INNER_BLOCK_SIZE_M,
) -> np.ndarray:
    x = frame["centroid_x"].to_numpy(dtype=float)
    y = frame["centroid_y"].to_numpy(dtype=float)
    bx = np.floor((x - x.min()) / block_size_m).astype(int)
    by = np.floor((y - y.min()) / block_size_m).astype(int)
    return bx * 100000 + by


def stratified_random_folds(
    y: np.ndarray,
    n_folds: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    folds = np.empty(len(y), dtype=int)
    for label in (0, 1):
        indices = np.flatnonzero(y == label)
        indices = rng.permutation(indices)
        folds[indices] = np.arange(len(indices)) % n_folds
    return folds


def sector_pairs() -> list[tuple[int, int]]:
    return list(combinations(range(4), 2))

