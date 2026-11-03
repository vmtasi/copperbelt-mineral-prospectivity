"""Validate V14 traces and their derived V14 out-of-fold predictions.

The checks are deliberately tied to stable grid-cell ids and the exact V14
centroid-contact training table.  A trace is not accepted merely because its
filename contains ``v14``: its observed labels, chain structure, rock-term
dimension, convergence diagnostics, and reproduced OOF probabilities must all
agree with the V14 modeling frame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.runtime_shim  # noqa: F401,E402

import arviz as az
import numpy as np
import pandas as pd

from src.numerics import StandardScaler, average_precision_score, brier_score_loss, roc_auc_score
from src.oof_provenance import (
    V14_DATA_PATH,
    V14_MANIFEST_PATH,
    V14_OOF_PATH,
    load_v14_oof,
)
from src.trace_io import load_trace, resolve_trace_path, trace_exists
from src.validation_strategies import get_along_belt_folds

TRACE_DIR = ROOT / "figures" / "v14_centroid_contact"
FOLD_METRICS_PATH = TRACE_DIR / "v14_centroid_fold_metrics.csv"
RUN_MANIFEST_PATH = TRACE_DIR / "run_manifest.json"
DISTANCE_CSV = ROOT / "data" / "derived" / "litho_contact_centroid_distance_v14.csv"
DISTANCE_GPKG = ROOT / "data" / "derived" / "litho_contact_centroid_distance_v14.gpkg"
V13_DATA = ROOT / "data" / "copperbelt_training_v13_litho_contact.csv"
V13_DISTANCE_CSV = ROOT / "data" / "derived" / "litho_contact_distance_v13.csv"
DISTANCE_COLUMN = "distance_to_lithology_contact"
LITHOLOGY_COLUMN = "litho_contact_litho_class"
FEATURES = ["distance_to_fault", DISTANCE_COLUMN, "bouguer"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def map_domain(value: str) -> str:
    text = str(value).lower()
    for token, code in (
        ("3a", "NRB_3a"),
        ("3b", "NRB_3b"),
        ("crz", "CRZ"),
        ("srb", "SRB"),
        ("nkb", "NKB"),
        ("mmsb", "MMSB"),
    ):
        if token in text:
            return code
    return "Unknown"


def load_model_frame() -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    frame = pd.read_csv(V14_DATA_PATH)
    required = ["id", "centroid_x", "centroid_y", "domain", LITHOLOGY_COLUMN] + FEATURES
    frame = frame.dropna(subset=required).copy()
    if len(frame) != 1872:
        raise AssertionError(f"Expected 1,872 complete-case cells, found {len(frame)}")
    if frame["id"].duplicated().any():
        raise AssertionError("V14 modeling frame contains duplicated grid-cell ids")

    frame["daly_domain"] = frame["domain"].map(map_domain)
    frame = frame[frame["daly_domain"] != "Unknown"].copy()
    domains = sorted(frame["daly_domain"].unique())
    domain_to_idx = {domain: index for index, domain in enumerate(domains)}
    frame["domain_idx"] = frame["daly_domain"].map(domain_to_idx)
    frame["spatial_block"] = get_along_belt_folds(frame, n_folds=4)
    encoded = pd.get_dummies(frame, columns=[LITHOLOGY_COLUMN], drop_first=True, dtype=float)
    rocks = [column for column in encoded.columns if column.startswith(f"{LITHOLOGY_COLUMN}_")]
    return encoded, rocks, domain_to_idx


def validate_trace(fold: int, frame: pd.DataFrame, rocks: list[str]) -> dict:
    requested = TRACE_DIR / f"v14_centroid_fold_{fold + 1}_trace.nc"
    if not trace_exists(requested):
        raise FileNotFoundError(f"Missing V14 trace for fold {fold + 1}: {requested}")
    resolved = resolve_trace_path(requested)
    trace = load_trace(resolved)

    train = frame[frame["spatial_block"] != fold].copy()
    test = frame[frame["spatial_block"] == fold].copy()
    y_train = train["deposit_present"].values.astype(np.int32)
    observed = trace.observed_data["y_obs"].values.reshape(-1)
    if observed.shape != y_train.shape or not np.array_equal(observed, y_train):
        raise AssertionError(f"Fold {fold + 1} trace labels do not match V14 training labels")

    chain_count = int(trace.posterior.sizes.get("chain", -1))
    draw_count = int(trace.posterior.sizes.get("draw", -1))
    if chain_count != 2 or draw_count != 1500:
        raise AssertionError(
            f"Fold {fold + 1} trace has chain={chain_count}, draw={draw_count}; "
            "expected 2 chains and 1,500 retained draws"
        )

    y_train_labels = train["deposit_present"].values.astype(np.int32)
    valid_rocks = [
        column for column in rocks
        if ((train[column] == 1) & (y_train_labels == 1)).sum() > 0
        and ((train[column] == 1) & (y_train_labels == 0)).sum() > 0
    ]
    rock_shape = trace.posterior["beta_rocks"].shape[-1]
    if rock_shape != len(valid_rocks):
        raise AssertionError(
            f"Fold {fold + 1} beta_rocks dimension {rock_shape} != {len(valid_rocks)}"
        )

    parameter_vars = [
        "alpha_dom", "beta_f_lin", "beta_f_sq", "beta_l_lin", "beta_l_sq",
        "beta_grav", "beta_rocks", "mu_f_lin", "mu_f_sq", "mu_l_lin", "mu_l_sq",
    ]
    summary = az.summary(trace, var_names=parameter_vars, hdi_prob=0.95)
    divergences = int(trace.sample_stats["diverging"].values.sum())
    max_rhat = float(summary["r_hat"].max())
    min_ess_bulk = float(summary["ess_bulk"].min())
    min_ess_tail = float(summary["ess_tail"].min())

    return {
        "fold": fold + 1,
        "requested_path": str(requested.relative_to(ROOT)),
        "resolved_path": str(resolved.relative_to(ROOT)),
        "format": "pickle" if resolved.suffix.lower() == ".pkl" else "netCDF",
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
        "chains": chain_count,
        "draws_per_chain": draw_count,
        "training_cells": int(len(train)),
        "training_positives": int(y_train.sum()),
        "held_out_cells": int(len(test)),
        "held_out_positives": int(test["deposit_present"].sum()),
        "labels_match": True,
        "rock_terms": int(rock_shape),
        "divergences": divergences,
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess_bulk,
        "min_ess_tail": min_ess_tail,
    }


def oof_metrics(oof: pd.DataFrame) -> tuple[dict, dict]:
    folds = {}
    for fold, group in oof.groupby("fold", sort=True):
        y = group["deposit_present"].to_numpy()
        p = group["prob_v14"].to_numpy()
        folds[str(int(fold) + 1)] = {
            "cells": int(len(group)),
            "positives": int(y.sum()),
            "auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
        }
    y = oof["deposit_present"].to_numpy()
    p = oof["prob_v14"].to_numpy()
    pooled = {
        "cells": int(len(oof)),
        "positives": int(y.sum()),
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }
    return folds, pooled


def main(output_path: Path | None) -> None:
    frame, rocks, domains = load_model_frame()
    trace_records = [validate_trace(fold, frame, rocks) for fold in range(4)]

    v14_distance = pd.read_csv(DISTANCE_CSV)
    v13_distance = pd.read_csv(V13_DISTANCE_CSV)
    if not v14_distance["id"].is_unique or len(v14_distance) != 1899:
        raise AssertionError("V14 distance CSV does not contain 1,899 unique cell ids")
    if not v13_distance["id"].is_unique or len(v13_distance) != 1899:
        raise AssertionError("V13 historical distance CSV does not contain 1,899 unique ids")
    distance_delta = (
        v14_distance.set_index("id")[DISTANCE_COLUMN]
        - v13_distance.set_index("id")[DISTANCE_COLUMN]
    )

    oof_record = None
    if not V14_OOF_PATH.is_file():
        raise FileNotFoundError(
            f"Missing V14 OOF predictions: {V14_OOF_PATH}. Run src/extract_v14_oof.py."
        )
    oof = load_v14_oof()
    expected = frame[["id", "deposit_present", "spatial_block"]].copy()
    check = expected.merge(
        oof[["id", "deposit_present", "fold"]], on="id", how="outer",
        suffixes=("_model", "_oof"), validate="one_to_one", indicator=True,
    )
    if not (check["_merge"] == "both").all():
        raise AssertionError("V14 OOF ids do not cover the V14 modeling frame exactly")
    if not np.array_equal(check["deposit_present_model"], check["deposit_present_oof"]):
        raise AssertionError("V14 OOF labels differ from the modeling frame")
    if not np.array_equal(check["spatial_block"], check["fold"]):
        raise AssertionError("V14 OOF folds differ from the modeling frame")

    fold_metrics, pooled = oof_metrics(oof)
    if FOLD_METRICS_PATH.is_file():
        saved = pd.read_csv(FOLD_METRICS_PATH)
        for row in saved.itertuples(index=False):
            fold_key = str(int(row.fold))
            if not np.isclose(float(row.auc), fold_metrics[fold_key]["auc"], atol=1e-12):
                raise AssertionError(f"Fold {fold_key} AUC disagrees with saved metrics")

    converged = all(
        record["divergences"] == 0
        and record["max_rhat"] <= 1.01
        and record["min_ess_bulk"] >= 400
        and record["min_ess_tail"] >= 400
        for record in trace_records
    )
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "PASS" if converged else "FAIL_CONVERGENCE",
        "model_identity": "V14 centroid-contact refit of the frozen V11 specification",
        "dataset": {
            "path": str(V14_DATA_PATH.relative_to(ROOT)),
            "sha256": sha256(V14_DATA_PATH),
            "manifest": str(V14_MANIFEST_PATH.relative_to(ROOT)),
            "manifest_sha256": sha256(V14_MANIFEST_PATH),
            "modeling_rows": int(len(frame)),
            "positive_cells": int(frame["deposit_present"].sum()),
            "domains": domains,
            "fold_sizes": [int(value) for value in frame.groupby("spatial_block").size().tolist()],
            "fold_positives": [
                int(value) for value in frame.groupby("spatial_block")["deposit_present"].sum().tolist()
            ],
        },
        "traces": trace_records,
        "oof": {
            "path": str(V14_OOF_PATH.relative_to(ROOT)),
            "sha256": sha256(V14_OOF_PATH),
            "rows": int(len(oof)),
            "unique_ids": int(oof["id"].nunique()),
            "folds": fold_metrics,
            "pooled": pooled,
            "labels_and_folds_match": True,
        },
        "distance_definition": {
            "definition": "minimum Euclidean distance from each cell centroid to the nearest mapped internal lithological contact, excluding the outer study-area perimeter",
            "crs": "EPSG:32735",
            "units": "metres",
            "v14_csv": str(DISTANCE_CSV.relative_to(ROOT)),
            "v14_csv_sha256": sha256(DISTANCE_CSV),
            "v14_gpkg": str(DISTANCE_GPKG.relative_to(ROOT)),
            "v14_gpkg_sha256": sha256(DISTANCE_GPKG),
            "v13_csv": str(V13_DISTANCE_CSV.relative_to(ROOT)),
            "v13_csv_sha256": sha256(V13_DISTANCE_CSV),
            "rows_differ_from_v13": int((distance_delta != 0).sum()),
            "mean_v14_minus_v13_m": float(distance_delta.mean()),
            "max_abs_v14_minus_v13_m": float(distance_delta.abs().max()),
        },
        "run_manifest": json.loads(RUN_MANIFEST_PATH.read_text(encoding="utf-8"))
        if RUN_MANIFEST_PATH.is_file() else None,
    }

    if output_path is None:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        output_path = ROOT / "figures" / "audit" / f"v14_model_output_validation_{stamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not converged:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    main(parser.parse_args().output)
