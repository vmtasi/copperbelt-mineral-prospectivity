"""Tests for the Paper 1 geographic transfer pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.transfer import config  # noqa: E402
from src.transfer.analysis import decomposition_gap_values  # noqa: E402
from src.transfer.data import build_design, load_transfer_frame  # noqa: E402
from src.transfer.metrics import calibration_in_the_large  # noqa: E402
from src.transfer.partitions import (  # noqa: E402
    assign_block_folds,
    block_ids,
    build_partitions,
)
from src.v15.metrics import roc_auc  # noqa: E402


def test_canonical_fold_integrity():
    frame = load_transfer_frame()
    assert len(frame) == 1872
    assert int(frame[config.Y_COL].sum()) == 138
    counts = frame.groupby(config.FOLD_COL).agg(
        cells=("id", "size"), positives=(config.Y_COL, "sum")
    )
    assert counts["cells"].tolist() == [468, 468, 468, 468]
    assert counts["positives"].tolist() == [9, 31, 37, 61]


def test_training_only_standardization_and_quadratic_construction():
    frame = load_transfer_frame()
    train_mask = frame[config.FOLD_COL].to_numpy(dtype=int) != 0
    test_mask = ~train_mask
    first = build_design(frame, train_mask, test_mask)
    changed = frame.copy()
    changed.loc[test_mask, ["fault_stored_km", "contact_km", "gravity_mgal"]] *= 10.0
    second = build_design(changed, train_mask, test_mask)
    for name in ("fault", "contact", "gravity"):
        assert first["parameters"][f"{name}_mean"] == second["parameters"][f"{name}_mean"]
        assert first["parameters"][f"{name}_scale"] == second["parameters"][f"{name}_scale"]
    assert abs(float(first["X_train"][:, 0].mean())) < 1e-12
    assert abs(float(first["X_train"][:, 2].mean())) < 1e-12
    assert abs(float(first["X_train"][:, 4].mean())) < 1e-12
    assert np.allclose(first["X_train"][:, 1], first["X_train"][:, 0] ** 2)
    assert np.allclose(first["X_train"][:, 3], first["X_train"][:, 2] ** 2)


def test_no_cell_overlap_outer_or_inner_spatial_cv():
    frame = load_transfer_frame()
    folds = frame[config.FOLD_COL].to_numpy(dtype=int)
    for fold in range(4):
        data = build_design(frame, folds != fold, folds == fold)
        assert not set(data["train_ids"]).intersection(data["test_ids"])
        sector = frame.loc[folds == fold].reset_index(drop=True)
        inner = assign_block_folds(sector, n_folds=4, seed=100 + fold)
        for inner_fold in range(4):
            train_ids = set(sector.loc[inner != inner_fold, "id"])
            test_ids = set(sector.loc[inner == inner_fold, "id"])
            assert not train_ids.intersection(test_ids)


def test_fitted_inner_local_cv_tested_each_cell_once_with_intact_blocks():
    """Check the inner folds the A3 local fits actually used, not a fresh draw."""
    frame = load_transfer_frame()
    folds = frame[config.FOLD_COL].to_numpy(dtype=int)
    paths = sorted(
        (config.OUTPUT / "predictions" / "amendment1").glob("a3_local_sector_*.csv")
    )
    assert len(paths) == 16
    predictions = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    for sector in range(4):
        subset = frame.loc[folds == sector].reset_index(drop=True)
        tested = predictions.loc[predictions["target_sector"] == sector + 1]
        # Every sector cell is predicted exactly once, by exactly one inner fit.
        assert tested["id"].is_unique
        assert set(tested["id"]) == set(subset["id"])
        assert tested["inner_fold"].nunique() == 4
        # No 25-km block is split between inner folds, so no block contributes
        # cells to both the training and the test side of any inner fit.
        blocks = pd.Series(block_ids(subset), index=subset["id"])
        inner = tested.set_index("id")["inner_fold"]
        per_block = inner.groupby(blocks.reindex(inner.index)).nunique()
        assert (per_block == 1).all()


def test_calibration_in_the_large_sign_and_zero_point():
    rng = np.random.default_rng(7)
    probability = rng.uniform(0.01, 0.3, 20000)
    y = rng.binomial(1, probability)
    assert abs(calibration_in_the_large(y, probability)) < 0.05
    assert calibration_in_the_large(y, np.minimum(2 * probability, 0.99)) < -0.5
    assert calibration_in_the_large(y, probability / 2) > 0.5


def test_log_area_is_the_fixed_offset():
    frame = load_transfer_frame()
    data = build_design(
        frame,
        frame[config.FOLD_COL].to_numpy(dtype=int) != 1,
        frame[config.FOLD_COL].to_numpy(dtype=int) == 1,
    )
    assert np.allclose(
        data["offset_train"],
        data["train"][config.AREA_COL].to_numpy(dtype=float),
    )
    assert np.allclose(
        data["offset_test"],
        data["test"][config.AREA_COL].to_numpy(dtype=float),
    )


def test_map_mcmc_gate_record_is_arithmetically_correct():
    path = config.OUTPUT / "diagnostics" / "map_mcmc_validation.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    expected = abs(
        result["mean_log_score_difference_map_minus_mcmc"]
    ) <= result["tolerance"]
    assert result["pass"] is expected


def test_amendment_environment_and_all_mcmc_fits_converged():
    reproduction = json.loads(
        (config.OUTPUT / "diagnostics" / "environment_reproduction.json").read_text(
            encoding="utf-8"
        )
    )
    assert reproduction["max_abs_delta"] < 0.01
    status_path = config.OUTPUT / "diagnostics" / "amendment1_status.csv"
    if status_path.exists():
        status = pd.read_csv(status_path)
        assert len(status) == 102
        assert status["converged"].astype(bool).all()
        assert (status["diverging"] == 0).all()
        assert (status["max_rhat"] < 1.05).all()
        assert (status["min_ess_bulk"] >= 200).all()
        assert (status["min_ess_tail"] >= 200).all()


def test_decomposition_gap_arithmetic():
    transfer = np.array([-1.0, -1.0])
    base = np.array([-0.5, -0.5])
    recal = np.array([-0.2, -0.2])
    local = np.array([0.0, 0.0])
    assert decomposition_gap_values(transfer, base, recal, local) == pytest.approx(
        (0.5, 0.3, 0.2)
    )
    cells = pd.read_csv(config.OUTPUT / "metrics" / "a3_decomposition_cells.csv")
    fitted = pd.read_csv(config.OUTPUT / "metrics" / "a3_decomposition.csv").set_index(
        "sector"
    )
    for sector, group in cells.groupby("sector"):
        assert group["base_minus_transfer"].mean() == pytest.approx(
            fitted.loc[sector, "base_minus_transfer"]
        )
        assert group["recal_minus_base"].mean() == pytest.approx(
            fitted.loc[sector, "recal_minus_base"]
        )
        assert group["local_minus_recal"].mean() == pytest.approx(
            fitted.loc[sector, "local_minus_recal"]
        )


def test_a1_prevalence_skill_arithmetic():
    metrics = pd.read_csv(config.OUTPUT / "metrics" / "a1_transfer_metrics.csv")
    for row in metrics.itertuples(index=False):
        assert row.log_score_skill_vs_prevalence == pytest.approx(
            row.mean_log_score - row.prevalence_baseline_log_score
        )


def test_auc_orientation_for_closer_and_lower_scores():
    y = np.array([1, 1, 0, 0])
    close = np.array([1.0, 2.0, 9.0, 10.0])
    low_gravity = np.array([-20.0, -10.0, 5.0, 20.0])
    assert roc_auc(y, -close) == 1.0
    assert roc_auc(y, -low_gravity) == 1.0


def test_partition_rules_reproduce_registered_cuts_and_counts():
    frame = load_transfer_frame()
    partitions, info = build_partitions(frame)
    assert np.allclose(
        info["metadata"]["p_gap_cuts"],
        [172.2091080459546, 300.0220944062411, 423.4675514405727],
        atol=0.01,
    )
    assert np.allclose(
        info["metadata"]["p_pos_cuts"],
        [248.38854666482626, 441.3300826224843, 490.61474092632346],
        atol=0.01,
    )
    gap = info["details"]["p_gap"]
    assert gap["cells"].tolist() == [531, 386, 351, 604]
    assert gap["positives"].tolist() == [10, 30, 25, 73]
    assert (gap["length_km"] >= 100).all()
    positive = info["details"]["p_pos"]
    assert positive["cells"].tolist() == [787, 550, 236, 299]
    assert positive["positives"].tolist() == [35, 35, 34, 34]
    assert partitions["fold_p_gap"].notna().all()
    assert partitions["fold_p_pos"].notna().all()

