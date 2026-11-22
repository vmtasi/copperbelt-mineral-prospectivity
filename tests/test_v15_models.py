"""Unit checks for the V15 exposure likelihood, design and model nesting."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v15 import config  # noqa: E402
from src.v15.common import cloglog_probability, prepare_fold  # noqa: E402
from src.v15.metrics import log_score  # noqa: E402
from src.v15.models import build_model  # noqa: E402


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return pd.read_csv(ROOT / "paper_1" / "v15" / "data" / "v15_model_frame.csv")


def test_cloglog_equals_at_least_one_poisson_probability():
    area = np.array([0.001, 0.5, 5.0, 25.0])
    intensity = np.array([0.2, 0.02, 0.004, 0.001])
    expected = 1.0 - np.exp(-intensity * area)
    observed = cloglog_probability(np.log(intensity) + np.log(area))
    assert np.allclose(observed, expected, rtol=1e-12, atol=1e-15)


def test_complete_case_population_and_folds(frame):
    assert len(frame) == 1872
    assert int(frame[config.Y_COL].sum()) == 138
    assert sorted(frame["fold"].unique().tolist()) == [0, 1, 2, 3]
    counts = frame.groupby("fold").size().tolist()
    positives = frame.groupby("fold")[config.Y_COL].sum().tolist()
    assert counts == [468, 468, 468, 468]
    assert positives == [9, 31, 37, 61]
    assert frame["id"].is_unique
    assert np.isfinite(frame["log_area_km2"]).all()
    assert (frame["area_km2"] > 0).all()


def test_prepare_fold_has_no_id_overlap_and_train_only_scaling(frame):
    arrays = prepare_fold(frame, 0, config.MODELS["M2_domain_distance"])
    assert len(arrays["train_ids"]) == 1404
    assert len(arrays["test_ids"]) == 468
    assert not set(arrays["train_ids"]).intersection(arrays["test_ids"])
    train_mean = arrays["train"]["fault_z"].mean()
    test_mean = arrays["test"]["fault_z"].mean()
    assert abs(train_mean) < 1e-12
    assert abs(test_mean) > 1e-8
    assert arrays["train"]["rocks"].shape[0] == 1404
    assert arrays["test"]["rocks"].shape[0] == 468
    assert arrays["train"]["rocks"].shape[1] == arrays["test"]["rocks"].shape[1]


def test_model_nesting_and_parameter_shapes(frame):
    arrays = prepare_fold(frame, 0, config.MODELS["M3_domain_distance_gravity"])
    models = {
        name: build_model(name, arrays, config.MODELS[name]) for name in config.PRIMARY_MODELS
    }
    assert "alpha" in models["M0_global"].named_vars
    assert "alpha_dom" not in models["M0_global"].named_vars
    assert "alpha_dom" in models["M1_domain_intercept"].named_vars
    assert models["M1_domain_intercept"]["beta_f_lin"].ndim == 0
    assert models["M2_domain_distance"]["beta_f_lin"].ndim == 1
    assert models["M2_domain_distance"]["beta_grav"].ndim == 0
    assert models["M3_domain_distance_gravity"]["beta_grav"].ndim == 1
    for model in models.values():
        assert model["alpha_dom"].ndim == 1 if "alpha_dom" in model.named_vars else True


def test_log_score_clipping_is_finite():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 1.0, 0.0, 1.0])
    score = log_score(y, p)
    assert np.isfinite(score).all()
    assert score[1] < -20 and score[2] < -20


def test_occurrence_accounting_is_not_one_to_one():
    summary = json.loads(
        (ROOT / "paper_1" / "v15" / "diagnostics" / "occurrence_support_summary.json").read_text()
    )
    assert summary["source_points"] == 187
    assert summary["points_in_model_positive_cells"] == 184
    assert summary["points_in_model_negative_cells"] == 2
    assert summary["points_in_excluded_cells"] == 1
    assert summary["unique_model_positive_cells_with_points"] == 138
    assert summary["max_points_in_one_positive_cell"] > 1
    assert summary["v14_positive_rows_before_complete_case"] == 139
    assert summary["v14_positive_rows_after_complete_case"] == 138


def test_v14_control_uses_global_gravity():
    assert config.MODELS["V14_control"]["family"] == "domain_distance"



def test_roc_auc_ranks_raw_covariates_without_clipping():
    """Regression: roc_auc once clipped scores to [1e-12, 1], tying every value > 1."""
    from src.v15.metrics import roc_auc

    def roc_auc_score(labels, values):  # brute-force pairwise definition
        pos, neg = values[labels == 1], values[labels == 0]
        diff = pos[:, None] - neg[None, :]
        return float(((diff > 0) + 0.5 * (diff == 0)).mean())

    rng = np.random.default_rng(7)
    y = np.r_[np.ones(40, dtype=int), np.zeros(400, dtype=int)]
    for score in (
        rng.uniform(0.0, 25.0, y.size) + 3.0 * y,     # km2-like, mostly > 1
        rng.normal(-15.0, 10.0, y.size) - 8.0 * y,    # mGal-like, negative
        np.round(rng.uniform(0, 5, y.size)) + y,      # heavy ties
    ):
        assert np.isclose(roc_auc(y, score), roc_auc_score(y, score))
