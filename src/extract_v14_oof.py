"""Reconstruct V14 centroid-contact out-of-fold probabilities from V14 traces.

The V14 run uses the frozen V11 model specification and the validated
centroid-to-internal-contact training table. Predictions are linked to the
modeling frame only by stable grid-cell ``id`` and are checked against each
trace's observed-label vector before the OOF artifact is written.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.runtime_shim  # noqa: F401,E402  (must precede pandas/arviz)

import numpy as np
import pandas as pd

from src.numerics import StandardScaler
from src.oof_provenance import (
    V14_DATA_PATH,
    V14_MANIFEST_PATH,
    V14_OOF_PATH,
    sha256_file,
    write_manifest,
)
from src.trace_io import load_trace, resolve_trace_path, trace_exists
from src.validation_strategies import get_along_belt_folds

OUTPUT_DIR = ROOT / "figures" / "v14_centroid_contact"
OUTPUT_COLUMNS = [
    "id",
    "fold",
    "deposit_present",
    "daly_domain",
    "centroid_x",
    "centroid_y",
    "prob_v14",
]
CODE_FILES = [
    "src/extract_v14_oof.py",
    "src/v14_centroid_spatial_stability.py",
    "src/v11_spatial_stability_final.py",
    "src/validation_strategies.py",
    "src/trace_io.py",
    "src/oof_provenance.py",
    "src/numerics.py",
]


def map_daly_domain(value):
    value = str(value).lower()
    if "3a" in value:
        return "NRB_3a"
    if "3b" in value:
        return "NRB_3b"
    if "crz" in value:
        return "CRZ"
    if "srb" in value:
        return "SRB"
    if "nkb" in value:
        return "NKB"
    if "mmsb" in value:
        return "MMSB"
    return "Unknown"


def extract_v14_probabilities():
    print("--- 1. Loading V14 dataset and frozen folds ---")
    df = pd.read_csv(V14_DATA_PATH)
    lithology_col = "litho_contact_litho_class"
    features = ["distance_to_fault", "distance_to_lithology_contact", "bouguer"]
    df = df.dropna(
        subset=["id", "centroid_x", "centroid_y", "domain", lithology_col] + features
    ).copy()
    if len(df) != 1872:
        raise ValueError(f"Expected 1,872 complete-case cells, found {len(df)}")
    if df["id"].duplicated().any():
        raise ValueError("V14 training table contains duplicated grid-cell ids")

    df["daly_domain"] = df["domain"].apply(map_daly_domain)
    df = df[df["daly_domain"] != "Unknown"].copy()
    unique_domains = sorted(df["daly_domain"].unique())
    domain_to_idx = {domain: index for index, domain in enumerate(unique_domains)}
    df["domain_idx"] = df["daly_domain"].map(domain_to_idx)
    n_domains = len(unique_domains)

    df["spatial_block"] = get_along_belt_folds(df, n_folds=4)
    df = pd.get_dummies(df, columns=[lithology_col], drop_first=True, dtype=float)
    all_rock_features = [
        column for column in df.columns if column.startswith(f"{lithology_col}_")
    ]
    df["prob_v14"] = np.nan
    trace_records = {}

    print("--- 2. Reconstructing out-of-fold probabilities ---")
    for fold in range(4):
        requested_path = OUTPUT_DIR / f"v14_centroid_fold_{fold + 1}_trace.nc"
        if not trace_exists(requested_path):
            raise FileNotFoundError(
                f"Missing V14 trace for fold {fold + 1} at {requested_path}"
            )
        resolved_path = resolve_trace_path(requested_path)
        trace = load_trace(resolved_path)
        print(f"Loading {resolved_path.name}...")

        train_df = df[df["spatial_block"] != fold].copy()
        test_df = df[df["spatial_block"] == fold].copy()
        y_train = train_df["deposit_present"].values.astype(np.int32)

        observed = trace.observed_data["y_obs"].values.reshape(-1)
        if observed.shape != y_train.shape or not np.array_equal(observed, y_train):
            raise ValueError(
                f"Fold {fold + 1} trace observed labels do not match the V14 "
                "training-fold labels"
            )

        scaler_fault = StandardScaler().fit(train_df[["distance_to_fault"]])
        scaler_lith = StandardScaler().fit(
            train_df[["distance_to_lithology_contact"]]
        )
        scaler_grav = StandardScaler().fit(train_df[["bouguer"]])
        for frame in (train_df, test_df):
            frame["fault_z"] = scaler_fault.transform(frame[["distance_to_fault"]])
            frame["lith_z"] = scaler_lith.transform(
                frame[["distance_to_lithology_contact"]]
            )
            frame["grav_z"] = scaler_grav.transform(frame[["bouguer"]])
            frame["fault_z_sq"] = frame["fault_z"] ** 2
            frame["lith_z_sq"] = frame["lith_z"] ** 2

        valid_rocks = [
            column
            for column in all_rock_features
            if ((train_df[column] == 1) & (y_train == 1)).sum() > 0
            and ((train_df[column] == 1) & (y_train == 0)).sum() > 0
        ]
        if trace.posterior["beta_rocks"].shape[-1] != len(valid_rocks):
            raise ValueError(
                f"Fold {fold + 1} rock-term count differs from the saved trace"
            )

        alpha_samples = trace.posterior["alpha_dom"].values.reshape(-1, n_domains)
        b_f_lin_samples = trace.posterior["beta_f_lin"].values.reshape(-1, n_domains)
        b_f_sq_samples = trace.posterior["beta_f_sq"].values.reshape(-1, n_domains)
        b_l_lin_samples = trace.posterior["beta_l_lin"].values.reshape(-1, n_domains)
        b_l_sq_samples = trace.posterior["beta_l_sq"].values.reshape(-1, n_domains)
        b_grav_samples = trace.posterior["beta_grav"].values.flatten()
        b_rocks_samples = trace.posterior["beta_rocks"].values.reshape(
            -1, len(valid_rocks)
        )

        test_domain_idx = test_df["domain_idx"].values
        logit_test = (
            alpha_samples[:, test_domain_idx]
            + b_f_lin_samples[:, test_domain_idx] * test_df["fault_z"].values
            + b_f_sq_samples[:, test_domain_idx] * test_df["fault_z_sq"].values
            + b_l_lin_samples[:, test_domain_idx] * test_df["lith_z"].values
            + b_l_sq_samples[:, test_domain_idx] * test_df["lith_z_sq"].values
            + b_grav_samples[:, None] * test_df["grav_z"].values
            + np.dot(b_rocks_samples, test_df[valid_rocks].values.T)
        )
        probability = 1.0 / (1.0 + np.exp(-logit_test))
        df.loc[df["spatial_block"] == fold, "prob_v14"] = probability.mean(axis=0)

        trace_records[f"fold_{fold + 1}"] = {
            "requested_path": str(requested_path.relative_to(ROOT)),
            "resolved_path": str(resolved_path.relative_to(ROOT)),
            "format": (
                "pickle (ArviZ InferenceData fallback)"
                if resolved_path.suffix.lower() == ".pkl"
                else "netCDF (ArviZ InferenceData)"
            ),
            "bytes": int(resolved_path.stat().st_size),
            "sha256": sha256_file(resolved_path),
            "held_out_cells": int(len(test_df)),
            "held_out_positives": int(test_df["deposit_present"].sum()),
            "observed_labels_match": True,
        }

    if df["prob_v14"].isna().any():
        raise ValueError("V14 reconstruction produced missing probabilities")
    if df["id"].duplicated().any():
        raise ValueError("V14 reconstruction produced duplicated grid-cell ids")

    df["fold"] = df["spatial_block"].astype(int)
    fold_counts = {
        str(int(fold)): int(count)
        for fold, count in df["fold"].value_counts().sort_index().items()
    }
    fold_positives = {
        str(int(fold)): int(count)
        for fold, count in df.groupby("fold")["deposit_present"].sum().sort_index().items()
    }

    print("--- 3. Saving V14 OOF predictions and provenance manifest ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df[OUTPUT_COLUMNS].to_csv(V14_OOF_PATH, index=False)
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "generator": "src/extract_v14_oof.py",
        "model_identity": "V14 centroid-to-internal-contact refit of the frozen V11 specification",
        "status": "PASS",
        "canonical_dataset": {
            "path": str(V14_DATA_PATH.relative_to(ROOT)),
            "sha256": sha256_file(V14_DATA_PATH),
            "rows": int(len(df)),
            "positive_cells": int(df["deposit_present"].sum()),
            "background_cells": int(len(df) - df["deposit_present"].sum()),
        },
        "folds": {
            fold: {"cells": fold_counts[fold], "positives": fold_positives[fold]}
            for fold in fold_counts
        },
        "traces": trace_records,
        "prediction_artifact": {
            "path": str(V14_OOF_PATH.relative_to(ROOT)),
            "sha256": sha256_file(V14_OOF_PATH),
            "rows": int(len(df)),
            "columns": OUTPUT_COLUMNS,
            "unique_ids": int(df["id"].nunique()),
            "duplicated_ids": int(df["id"].duplicated().sum()),
            "prob_v14_min": float(df["prob_v14"].min()),
            "prob_v14_max": float(df["prob_v14"].max()),
        },
        "code": {relative: sha256_file(ROOT / relative) for relative in CODE_FILES},
        "method": (
            "Posterior-mean predicted probability for each held-out V14 fold cell, "
            "reconstructed from the V14 trace with fold-specific scalers and rock "
            "terms fitted on training folds only."
        ),
    }
    write_manifest(V14_MANIFEST_PATH, manifest)
    print(f"[+] OOF predictions: {V14_OOF_PATH}")
    print(f"[+] Manifest: {V14_MANIFEST_PATH}")
    return df


if __name__ == "__main__":
    extract_v14_probabilities()
