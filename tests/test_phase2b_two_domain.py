"""Invariants for the Phase 2b fixed-domain analysis."""

from __future__ import annotations

import sys
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import phase2b_two_domain as phase  # noqa: E402


@pytest.fixture(scope="module")
def frames():
    return phase.load_frames()


def test_frame_sizes_and_positive_support(frames):
    primary = frames["primary"]
    secondary = frames["secondary"]
    assert len(primary) == 1872
    assert int(primary[phase.Y_COL].sum()) == 138
    assert len(secondary) == 1677
    assert int(secondary[phase.Y_COL].sum()) == 138
    assert set(primary["daly_domain"]) == {
        "CRZ",
        "MMSB",
        "NKB",
        "NRB_3a",
        "NRB_3b",
        "SRB",
    }
    assert set(secondary["daly_domain"]) == {"NRB_3a", "NRB_3b"}


def test_canonical_fold_ids_are_preserved(frames):
    primary = frames["primary"]
    counts = primary.groupby("fold").agg(
        cells=("id", "size"), positives=(phase.Y_COL, "sum")
    )
    assert counts["cells"].tolist() == [468, 468, 468, 468]
    assert counts["positives"].tolist() == [9, 31, 37, 61]

    composition = pd.crosstab(primary["fold"], primary["daly_domain"])
    assert composition.loc[0, "NRB_3a"] == 0
    assert composition.loc[0, "NRB_3b"] == 310
    assert composition.loc[1, "NRB_3a"] == 207
    assert composition.loc[1, "NRB_3b"] == 240
    assert composition.loc[2, "NRB_3a"] == 468
    assert composition.loc[3, "NRB_3a"] == 452


def test_rebuilt_fault_variant_matches_documented_audit(frames):
    primary = frames["primary"]
    assert (primary["distance_to_fault_km"] >= 0).all()
    assert primary["distance_to_fault_km"].notna().all()
    assert int(primary["fault_distance_reproduced_within_1m"].sum()) == 1645
    assert int((~primary["fault_distance_reproduced_within_1m"]).sum()) == 227
    assert (
        int(
            primary.loc[
                ~primary["fault_distance_reproduced_within_1m"], phase.Y_COL
            ].sum()
        )
        == 26
    )


def test_prepare_fold_uses_training_only_scaling(frames):
    arrays = phase.prepare_fold(frames["primary"], 0)
    train = arrays["train"]
    test = arrays["test"]
    assert abs(float(train["fault_z"].mean())) < 1e-12
    assert abs(float(train["lith_z"].mean())) < 1e-12
    assert abs(float(train["grav_z"].mean())) < 1e-12
    assert abs(float(test["fault_z"].mean())) > 1e-8
    assert train["rocks"].shape[0] == arrays["train_size"]
    assert test["rocks"].shape[0] == arrays["test_size"]
    assert train["rocks"].shape[1] == test["rocks"].shape[1]


def test_models_are_nested_fixed_effects_without_hierarchy(frames):
    arrays = phase.prepare_fold(frames["primary"], 0)
    models = {
        name: phase.build_model(name, arrays, include_outside=True)
        for name in phase.MODEL_ORDER
    }
    assert "beta_outside_nrb" in models["N0_global"].named_vars
    assert "beta_3b" not in models["N0_global"].named_vars
    assert "beta_3b" in models["N1_fixed_3b"].named_vars
    assert "beta_3b_f_lin" not in models["N1_fixed_3b"].named_vars
    assert "beta_3b_f_lin" in models["N2_3b_distance_interactions"].named_vars
    assert "beta_3b_grav" not in models["N2_3b_distance_interactions"].named_vars
    assert "beta_3b_grav" in models["N3_3b_gravity_interaction"].named_vars
    forbidden = {"alpha_mu", "alpha_sigma", "mu_beta", "sigma_beta"}
    for model in models.values():
        assert forbidden.isdisjoint(model.named_vars)


def test_secondary_n0_has_no_outside_nrb_term(frames):
    arrays = phase.prepare_fold(frames["secondary"], 0)
    model = phase.build_model("N0_global", arrays, include_outside=False)
    assert "beta_outside_nrb" not in model.named_vars
    assert "beta_3b" not in model.named_vars


def test_fault_column_flag_selects_stored_variant_and_output_root():
    parsed = phase.parser().parse_args(
        ["--fault-column", phase.FAULT_COLUMN_STORED, "build"]
    )
    assert parsed.fault_column == phase.FAULT_COLUMN_STORED
    original = phase.SELECTED_FAULT_COLUMN
    try:
        phase.select_fault_column(phase.FAULT_COLUMN_STORED)
        assert phase.PHASE == phase.PHASE_2C
        primary, _, _ = phase.make_frames()
        assert (
            primary["fault_distance_model_km"]
            == primary["distance_to_fault_stored_km"]
        ).all()
        assert not (
            primary["fault_distance_model_km"]
            == primary["distance_to_fault_km"]
        ).all()
    finally:
        phase.select_fault_column(original)


def test_completed_outputs_are_complete_and_converged(frames):
    status = json.loads(
        (phase.PHASE / "diagnostics" / "fit_status.json").read_text(encoding="utf-8")
    )
    assert len(status) == 32
    assert all(row["status"] == "PASS" for row in status)
    assert all(row["diverging"] == 0 for row in status)
    assert all(row["max_rhat"] < 1.05 for row in status)
    assert all(row["min_ess_bulk"] >= 200 for row in status)
    assert all(row["min_ess_tail"] >= 200 for row in status)
    for frame_name in phase.FRAME_ORDER:
        expected_ids = set(frames[frame_name]["id"])
        for model_name in phase.MODEL_ORDER:
            paths = list(
                (phase.PHASE / "predictions" / frame_name).glob(
                    f"{model_name}_fold_*.csv"
                )
            )
            assert len(paths) == 4
            prediction = pd.concat([pd.read_csv(path) for path in paths])
            assert len(prediction) == len(frames[frame_name])
            assert set(prediction["id"]) == expected_ids
            assert prediction["prob"].between(0, 1).all()


def test_phase2c_stored_fault_outputs_are_complete_and_converged(frames):
    status = json.loads(
        (phase.PHASE_2C / "diagnostics" / "fit_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(status) == 16
    assert all(row["status"] == "PASS" for row in status)
    assert all(row["diverging"] == 0 for row in status)
    assert all(row["max_rhat"] < 1.05 for row in status)
    assert all(row["min_ess_bulk"] >= 200 for row in status)
    assert all(row["min_ess_tail"] >= 200 for row in status)
    expected_ids = set(frames["primary"]["id"])
    for model_name in phase.MODEL_ORDER:
        paths = list(
            (phase.PHASE_2C / "predictions" / "primary").glob(
                f"{model_name}_fold_*.csv"
            )
        )
        assert len(paths) == 4
        prediction = pd.concat([pd.read_csv(path) for path in paths])
        assert len(prediction) == 1872
        assert set(prediction["id"]) == expected_ids

    robustness = pd.read_csv(
        phase.PHASE_2C / "metrics" / "fault_variant_2x2.csv"
    )
    assert len(robustness) == 6
    fixed = robustness.loc[robustness["row_type"] == "fixed-effect ladder"]
    assert set(fixed["fault_variant"]) == {"stored", "rebuilt"}
    assert set(fixed["comparison"]) == {"N1-N2", "N2-N3"}
