"""Fixed-domain sensitivity for Phase 2b/2c fault variants.

The latest analysis instruction makes the full 1,872-cell frame primary.  It
contains an outside-NRB fixed effect plus the fixed NRB_3b terms.  The
1,677-cell NRB-only frame is retained as a secondary, descriptive analysis.

All folds are copied from the canonical V15 frame.  No fold is recomputed.
Continuous predictors are standardized inside each training partition, with
quadratic terms formed after standardization.  Distance to fault is rebuilt
from the surviving 126-line CACB structure layer (the Phase 2 geometry variant)
by default; Phase 2c can select the stored fault distance instead.  The
recorded log clipped-cell area remains the fixed cloglog offset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PHASE_2B = ROOT / "paper_1" / "v15" / "phase2b_two_domain"
PHASE_2C = ROOT / "paper_1" / "v15" / "phase2c_stored_faults"
PHASE = PHASE_2B
MODEL_FRAME = ROOT / "paper_1" / "v15" / "data" / "v15_model_frame.csv"
STRUCTURE_VARIANT = Path(r"C:\Project\data for projects\CACB_Structures.gpkg")
FAULT_COLUMN_REBUILT = "distance_to_fault_km"
FAULT_COLUMN_STORED = "distance_to_fault_stored_km"
FAULT_COLUMNS = (FAULT_COLUMN_REBUILT, FAULT_COLUMN_STORED)
SELECTED_FAULT_COLUMN = FAULT_COLUMN_REBUILT

SOURCE_COLUMNS = [
    "id",
    "deposit_present",
    "fold",
    "daly_domain",
    "centroid_x",
    "centroid_y",
    "distance_to_fault",
    "distance_to_lithology_contact",
    "bouguer",
    "litho_contact_litho_class",
    "area_km2",
    "log_area_km2",
]
CONTINUOUS = {
    "fault": "fault_distance_model_km",
    "lith": "distance_to_lithology_contact_km",
    "grav": "bouguer",
}
Y_COL = "deposit_present"
LITHO_COL = "litho_contact_litho_class"
MODEL_ORDER = [
    "N0_global",
    "N1_fixed_3b",
    "N2_3b_distance_interactions",
    "N3_3b_gravity_interaction",
]
MODEL_LABEL = {
    "N0_global": "N0",
    "N1_fixed_3b": "N1",
    "N2_3b_distance_interactions": "N2",
    "N3_3b_gravity_interaction": "N3",
}
FRAME_ORDER = ["primary", "secondary"]
INTERACTION_TERMS = [
    "beta_3b_f_lin",
    "beta_3b_f_sq",
    "beta_3b_l_lin",
    "beta_3b_l_sq",
]

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime_cache" / "matplotlib_phase2b"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".runtime_cache" / "phase2b"))
os.environ.setdefault(
    "PYTENSOR_FLAGS",
    f"compiledir={ROOT / '.runtime_cache' / 'pytensor_phase2b'}",
)

# QGIS's CPython needs its adjacent DLL directory on the Windows search path.
# Keep the handle open for the process lifetime.
_DLL_HANDLE = None
if os.name == "nt":
    _qgis_dll_dir = Path(sys.base_prefix).resolve().parents[1] / "bin"
    if _qgis_dll_dir.is_dir():
        os.environ["PATH"] = str(_qgis_dll_dir) + os.pathsep + os.environ["PATH"]
        _DLL_HANDLE = os.add_dll_directory(str(_qgis_dll_dir))

try:
    import platformdirs

    _CACHE_DIR = ROOT / ".runtime_cache" / "phase2b"
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    platformdirs.user_cache_dir = lambda *args, **kwargs: str(_CACHE_DIR)
    platformdirs.user_data_dir = lambda *args, **kwargs: str(_CACHE_DIR)
except Exception:
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_phase_path(path: Path) -> Path:
    path = path.resolve()
    root = PHASE.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Refusing to write outside {PHASE.name}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, value: Any) -> None:
    path = ensure_phase_path(path)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def select_fault_column(column: str) -> None:
    global PHASE, SELECTED_FAULT_COLUMN

    if column not in FAULT_COLUMNS:
        raise ValueError(f"Unknown fault column: {column}")
    SELECTED_FAULT_COLUMN = column
    PHASE = PHASE_2C if column == FAULT_COLUMN_STORED else PHASE_2B


def source_frame() -> Any:
    import pandas as pd

    frame = pd.read_csv(MODEL_FRAME, usecols=SOURCE_COLUMNS)
    required = set(SOURCE_COLUMNS)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing model-frame columns: {missing}")
    if frame["id"].duplicated().any():
        raise ValueError("Duplicate cell IDs in the V15 frame")
    return frame


def rebuild_fault_distance(frame: Any) -> Any:
    """Rebuild centroid-to-line distance from the Phase 2 structure variant."""

    import geopandas as gpd
    import numpy as np
    import shapely

    if not STRUCTURE_VARIANT.is_file():
        raise FileNotFoundError(STRUCTURE_VARIANT)
    structures = gpd.read_file(STRUCTURE_VARIANT).to_crs(32735)
    if structures.empty:
        raise ValueError("The Phase 2 structure variant contains no geometries")
    if structures.geometry.isna().any() or structures.geometry.is_empty.any():
        raise ValueError("The Phase 2 structure variant contains null/empty geometry")
    linework = shapely.union_all(structures.geometry.to_numpy())
    points = shapely.points(
        frame["centroid_x"].to_numpy(float),
        frame["centroid_y"].to_numpy(float),
    )
    rebuilt_m = shapely.distance(linework, points)
    if not np.isfinite(rebuilt_m).all() or (rebuilt_m < 0).any():
        raise ValueError("Invalid rebuilt fault distances")

    out = frame.copy()
    out["distance_to_fault_stored_km"] = out["distance_to_fault"].astype(float) / 1000.0
    out["distance_to_fault_km"] = rebuilt_m / 1000.0
    out["fault_distance_rebuilt_minus_stored_km"] = (
        out["distance_to_fault_km"] - out["distance_to_fault_stored_km"]
    )
    out["fault_distance_reproduced_within_1m"] = (
        (rebuilt_m - out["distance_to_fault"].astype(float)).abs() < 1.0
    )
    out["distance_to_lithology_contact_km"] = (
        out["distance_to_lithology_contact"].astype(float) / 1000.0
    )
    out["outside_nrb"] = (~out["daly_domain"].isin(["NRB_3a", "NRB_3b"])).astype(int)
    out["nrb_3b"] = (out["daly_domain"] == "NRB_3b").astype(int)
    return out


def frame_counts(frame: Any) -> dict[str, Any]:
    by_fold_domain = []
    for (fold, domain), group in frame.groupby(["fold", "daly_domain"], sort=True):
        by_fold_domain.append(
            {
                "fold": int(fold) + 1,
                "domain": str(domain),
                "cells": int(len(group)),
                "positives": int(group[Y_COL].sum()),
            }
        )
    by_fold = []
    for fold, group in frame.groupby("fold", sort=True):
        by_fold.append(
            {
                "fold": int(fold) + 1,
                "cells": int(len(group)),
                "positives": int(group[Y_COL].sum()),
            }
        )
    return {
        "cells": int(len(frame)),
        "positives": int(frame[Y_COL].sum()),
        "by_fold": by_fold,
        "by_fold_domain": by_fold_domain,
    }


def make_frames() -> tuple[Any, Any, dict[str, Any]]:
    import geopandas as gpd
    import pandas as pd

    full = rebuild_fault_distance(source_frame())
    if SELECTED_FAULT_COLUMN not in full.columns:
        raise ValueError(f"Selected fault column is unavailable: {SELECTED_FAULT_COLUMN}")
    full["fault_distance_model_km"] = full[SELECTED_FAULT_COLUMN].astype(float)
    outside_domains = sorted(full.loc[full["outside_nrb"] == 1, "daly_domain"].unique())
    expected_outside = {"CRZ", "MMSB", "NKB", "SRB"}
    if set(outside_domains) != expected_outside:
        raise ValueError(f"Unexpected outside-NRB domains: {outside_domains}")

    primary = full.copy()
    secondary = full.loc[full["daly_domain"].isin(["NRB_3a", "NRB_3b"])].copy()
    if len(primary) != 1872 or int(primary[Y_COL].sum()) != 138:
        raise ValueError("Primary full frame does not match 1,872 cells / 138 positives")
    if len(secondary) != 1677 or int(secondary[Y_COL].sum()) != 138:
        raise ValueError("Secondary NRB frame does not match 1,677 cells / 138 positives")
    if sorted(primary["fold"].unique().tolist()) != [0, 1, 2, 3]:
        raise ValueError("Canonical V15 folds were not preserved")
    if sorted(secondary["fold"].unique().tolist()) != [0, 1, 2, 3]:
        raise ValueError("NRB-only frame no longer contains all four V15 folds")

    audit = {
        "created_utc": utc_now(),
        "model_frame": {
            "path": str(MODEL_FRAME.relative_to(ROOT)),
            "sha256": sha256(MODEL_FRAME),
            "cells": int(len(full)),
            "positives": int(full[Y_COL].sum()),
        },
        "fault_variant": {
            "selected_column": SELECTED_FAULT_COLUMN,
            "role": (
                "stored fault distances"
                if SELECTED_FAULT_COLUMN == FAULT_COLUMN_STORED
                else "rebuilt Phase 2 fault distances"
            ),
            "path": str(STRUCTURE_VARIANT),
            "sha256": sha256(STRUCTURE_VARIANT),
            "features": int(len(gpd.read_file(STRUCTURE_VARIANT))),
            "definition": (
                "Euclidean distance in EPSG:32735 from each canonical V15 cell "
                "centroid to the union of the surviving 126-line CACB structure layer"
            ),
            "reproduced_within_1m": int(
                full["fault_distance_reproduced_within_1m"].sum()
            ),
            "not_reproduced": int(
                (~full["fault_distance_reproduced_within_1m"]).sum()
            ),
            "not_reproduced_positives": int(
                full.loc[
                    ~full["fault_distance_reproduced_within_1m"], Y_COL
                ].sum()
            ),
            "max_abs_difference_km": float(
                full["fault_distance_rebuilt_minus_stored_km"].abs().max()
            ),
        },
        "primary": frame_counts(primary),
        "secondary": frame_counts(secondary),
        "outside_nrb_domains": outside_domains,
        "fold_domain_confounding": {
            "fold_1": "NRB_3b plus outside-NRB cells; no NRB_3a",
            "fold_2": "mixed NRB_3a, NRB_3b and outside-NRB cells",
            "fold_3": "NRB_3a only",
            "fold_4": "NRB_3a plus a small outside-NRB component",
        },
    }
    return primary, secondary, audit


def specification(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": (
            "2c.1.0"
            if SELECTED_FAULT_COLUMN == FAULT_COLUMN_STORED
            else "2b.1.0"
        ),
        "created_utc": utc_now(),
        "status": (
            "pre_specified_before_phase2c_fits"
            if SELECTED_FAULT_COLUMN == FAULT_COLUMN_STORED
            else "pre_specified_before_phase2b_fits"
        ),
        "fault_column": SELECTED_FAULT_COLUMN,
        "primary_frame": {
            "name": "full_1872",
            "role": "primary; source of all verdicts",
            "cells": audit["primary"]["cells"],
            "positives": audit["primary"]["positives"],
            "domain_terms": [
                "fixed outside-NRB indicator for CRZ, MMSB, NKB and SRB",
                "fixed NRB_3b indicator in N1-N3",
            ],
        },
        "secondary_frame": {
            "name": "nrb_3a_plus_3b_1677",
            "role": "descriptive; no outside-NRB indicator; no verdicts",
            "cells": audit["secondary"]["cells"],
            "positives": audit["secondary"]["positives"],
        },
        "folds": (
            "V15 fold IDs copied unchanged: 1 = NRB_3b only; 2 = mixed; "
            "3-4 = NRB_3a only"
        ),
        "fault_distance": (
            (
                "Stored V15 fault distance; training-fold standardization; "
                "quadratic term formed after standardization"
            )
            if SELECTED_FAULT_COLUMN == FAULT_COLUMN_STORED
            else (
                "Rebuilt from the Phase 2 CACB structure variant; training-fold "
                "standardization; quadratic term formed after standardization"
            )
        ),
        "likelihood": "Bernoulli complementary log-log",
        "area": "log_area_km2 fixed offset with coefficient one",
        "priors": {
            "all_coefficients": "Normal(0, 1), independent",
            "hierarchical_scales": "none",
        },
        "models": {
            "N0_global": "global terms; outside-NRB only in the primary frame",
            "N1_fixed_3b": "N0 + fixed NRB_3b indicator",
            "N2_3b_distance_interactions": (
                "N1 + 3b x (fault, fault^2, contact, contact^2)"
            ),
            "N3_3b_gravity_interaction": "N2 + 3b x gravity",
        },
        "decision_rule": {
            "pooled": (
                "upper model improves both mean held-out log score and ROC-AUC"
            ),
            "uncertainty": (
                "95% 25-km spatial-block bootstrap interval for mean log-score "
                "gain excludes zero"
            ),
            "fold_replication": "mean log-score gain is positive in >= 3 of 4 folds",
            "coefficient_n1_n2": (
                "a majority (>= 2 of 4) of the added 3b x distance coefficients "
                "has a 95% interval excluding zero in >= 3 of 4 folds"
            ),
            "coefficient_n2_n3": (
                "the 3b x gravity interaction interval excludes zero in >= 3 of 4 folds"
            ),
            "supported": "all applicable conditions must hold",
        },
        "descriptive_check": (
            "N1-to-N2 and N2-to-N3 comparisons within Fold 2 only, because Fold 2 "
            "is the only mixed fold"
        ),
    }


def build_outputs() -> dict[str, Any]:
    import pandas as pd

    primary, secondary, audit = make_frames()
    data_dir = PHASE / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    primary.to_csv(data_dir / "primary_frame.csv", index=False)
    secondary.to_csv(data_dir / "secondary_frame.csv", index=False)
    write_json(data_dir / "input_audit.json", audit)
    spec = specification(audit)
    write_json(PHASE / "specification.json", spec)
    print(
        f"Primary: {len(primary)} cells / {int(primary[Y_COL].sum())} positives; "
        f"secondary: {len(secondary)} / {int(secondary[Y_COL].sum())}"
    )
    return {"primary": primary, "secondary": secondary, "audit": audit, "spec": spec}


def load_frames() -> dict[str, Any]:
    import pandas as pd

    paths = {
        "primary": PHASE / "data" / "primary_frame.csv",
        "secondary": PHASE / "data" / "secondary_frame.csv",
    }
    if not all(path.is_file() for path in paths.values()):
        build_outputs()
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    for frame in frames.values():
        if "fault_distance_model_km" not in frame.columns:
            if SELECTED_FAULT_COLUMN not in frame.columns:
                raise ValueError(
                    f"Selected fault column is absent from frame: {SELECTED_FAULT_COLUMN}"
                )
            frame["fault_distance_model_km"] = frame[SELECTED_FAULT_COLUMN]
    return frames


def prepare_fold(frame: Any, fold: int) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    train = frame.loc[frame["fold"] != fold].copy()
    test = frame.loc[frame["fold"] == fold].copy()
    if train.empty or test.empty:
        raise ValueError(f"Fold {fold + 1} does not contain both partitions")
    if len(set(train["id"]).intersection(set(test["id"]))) != 0:
        raise ValueError("Training and test cell IDs overlap")

    rock_values = frame[LITHO_COL].astype("Int64").astype(str)
    rock_dummies = pd.get_dummies(rock_values, prefix="rock", dtype=float)
    rock_dummies.index = frame.index
    train_dummies = rock_dummies.loc[train.index]
    test_dummies = rock_dummies.loc[test.index]
    y_train = train[Y_COL].to_numpy(dtype=int)
    valid_rocks = [
        column
        for column in rock_dummies.columns
        if ((train_dummies[column] == 1) & (y_train == 1)).sum() > 0
        and ((train_dummies[column] == 1) & (y_train == 0)).sum() > 0
    ]

    arrays: dict[str, Any] = {
        "train": {"rocks": train_dummies[valid_rocks].to_numpy(float)},
        "test": {"rocks": test_dummies[valid_rocks].to_numpy(float)},
        "valid_rocks": valid_rocks,
        "n_rocks": len(valid_rocks),
        "y_train": y_train,
        "y_test": test[Y_COL].to_numpy(dtype=int),
        "train_ids": train["id"].to_numpy(dtype=int),
        "test_ids": test["id"].to_numpy(dtype=int),
        "train_size": int(len(train)),
        "test_size": int(len(test)),
        "p_ref_train": float((y_train.sum() + 0.5) / (len(y_train) + 1.0)),
    }
    arrays["train"]["rocks"] = (
        train_dummies[valid_rocks].to_numpy(float)
        if valid_rocks
        else np.empty((len(train), 0))
    )
    arrays["test"]["rocks"] = (
        test_dummies[valid_rocks].to_numpy(float)
        if valid_rocks
        else np.empty((len(test), 0))
    )

    for key, data in (("train", train), ("test", test)):
        for short, column in CONTINUOUS.items():
            values = data[column].to_numpy(dtype=float)
            if key == "train":
                arrays[f"{short}_mean"] = float(np.mean(values))
                arrays[f"{short}_scale"] = float(np.std(values, ddof=0))
                if arrays[f"{short}_scale"] <= 0:
                    raise ValueError(f"Non-positive training scale for {short}")
            z = (values - arrays[f"{short}_mean"]) / arrays[f"{short}_scale"]
            arrays[key][f"{short}_z"] = z
            arrays[key][f"{short}_z2"] = z**2
        arrays[key]["outside_nrb"] = data["outside_nrb"].to_numpy(dtype=float)
        arrays[key]["nrb_3b"] = data["nrb_3b"].to_numpy(dtype=float)
        arrays[key]["log_area"] = data["log_area_km2"].to_numpy(dtype=float)
    arrays["area_train_sum"] = float(np.exp(arrays["train"]["log_area"]).sum())
    return arrays


def build_model(
    model_name: str,
    arrays: dict[str, Any],
    include_outside: bool,
):
    import pymc as pm
    import pytensor.tensor as pt

    if model_name not in MODEL_ORDER:
        raise KeyError(f"Unknown model {model_name}")
    train = arrays["train"]

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
        beta_f_lin = pm.Normal("beta_f_lin", mu=0.0, sigma=1.0)
        beta_f_sq = pm.Normal("beta_f_sq", mu=0.0, sigma=1.0)
        beta_l_lin = pm.Normal("beta_l_lin", mu=0.0, sigma=1.0)
        beta_l_sq = pm.Normal("beta_l_sq", mu=0.0, sigma=1.0)
        beta_grav = pm.Normal("beta_grav", mu=0.0, sigma=1.0)

        eta = alpha
        if include_outside:
            beta_outside = pm.Normal("beta_outside_nrb", mu=0.0, sigma=1.0)
            eta = eta + beta_outside * train["outside_nrb"]

        if model_name != "N0_global":
            beta_3b = pm.Normal("beta_3b", mu=0.0, sigma=1.0)
            eta = eta + beta_3b * train["nrb_3b"]

        eta = (
            eta
            + beta_f_lin * train["fault_z"]
            + beta_f_sq * train["fault_z2"]
            + beta_l_lin * train["lith_z"]
            + beta_l_sq * train["lith_z2"]
            + beta_grav * train["grav_z"]
        )

        if model_name in {
            "N2_3b_distance_interactions",
            "N3_3b_gravity_interaction",
        }:
            for name in INTERACTION_TERMS:
                coefficient = pm.Normal(name, mu=0.0, sigma=1.0)
                source = {
                    "beta_3b_f_lin": "fault_z",
                    "beta_3b_f_sq": "fault_z2",
                    "beta_3b_l_lin": "lith_z",
                    "beta_3b_l_sq": "lith_z2",
                }[name]
                eta = eta + coefficient * train["nrb_3b"] * train[source]

        if model_name == "N3_3b_gravity_interaction":
            beta_3b_grav = pm.Normal("beta_3b_grav", mu=0.0, sigma=1.0)
            eta = eta + beta_3b_grav * train["nrb_3b"] * train["grav_z"]

        if arrays["n_rocks"]:
            beta_rocks = pm.Normal(
                "beta_rocks", mu=0.0, sigma=1.0, shape=arrays["n_rocks"]
            )
            eta = eta + pt.dot(train["rocks"], beta_rocks)

        eta = eta + train["log_area"]
        probability = -pt.expm1(-pt.exp(eta))
        pm.Bernoulli(
            "y_obs",
            p=pt.clip(probability, 1e-12, 1.0 - 1e-12),
            observed=arrays["y_train"],
        )
    return model


def predictor_samples(
    trace: Any,
    arrays: dict[str, Any],
    split: str,
    include_outside: bool,
) -> Any:
    import numpy as np

    data = arrays[split]

    def samples(name: str) -> Any:
        values = np.asarray(trace.posterior[name].values)
        return values.reshape(-1, *values.shape[2:])

    beta_names = {name for name in trace.posterior.data_vars if name.startswith("beta_")}
    model_name = "N0_global"
    if "beta_3b" in beta_names:
        model_name = "N1_fixed_3b"
    if "beta_3b_f_lin" in beta_names:
        model_name = "N2_3b_distance_interactions"
    if "beta_3b_grav" in beta_names:
        model_name = "N3_3b_gravity_interaction"

    eta = samples("alpha")[:, None] * np.ones((1, len(data["fault_z"])))
    eta = eta + samples("beta_f_lin")[:, None] * data["fault_z"][None, :]
    eta = eta + samples("beta_f_sq")[:, None] * data["fault_z2"][None, :]
    eta = eta + samples("beta_l_lin")[:, None] * data["lith_z"][None, :]
    eta = eta + samples("beta_l_sq")[:, None] * data["lith_z2"][None, :]
    eta = eta + samples("beta_grav")[:, None] * data["grav_z"][None, :]
    if include_outside:
        eta = eta + samples("beta_outside_nrb")[:, None] * data["outside_nrb"][None, :]
    if model_name != "N0_global":
        eta = eta + samples("beta_3b")[:, None] * data["nrb_3b"][None, :]
    if model_name in {
        "N2_3b_distance_interactions",
        "N3_3b_gravity_interaction",
    }:
        for name in INTERACTION_TERMS:
            source = {
                "beta_3b_f_lin": "fault_z",
                "beta_3b_f_sq": "fault_z2",
                "beta_3b_l_lin": "lith_z",
                "beta_3b_l_sq": "lith_z2",
            }[name]
            eta = eta + samples(name)[:, None] * data["nrb_3b"][None, :] * data[source][None, :]
    if model_name == "N3_3b_gravity_interaction":
        eta = eta + samples("beta_3b_grav")[:, None] * data["nrb_3b"][None, :] * data["grav_z"][None, :]
    if arrays["n_rocks"]:
        eta = eta + samples("beta_rocks") @ data["rocks"].T
    eta = eta + data["log_area"][None, :]
    probability = -np.expm1(-np.exp(eta))
    return np.clip(probability, 1e-12, 1.0 - 1e-12)


def summarize_trace(trace: Any, model_name: str, fold: int) -> tuple[Any, dict[str, Any]]:
    import arviz as az
    import numpy as np
    import pandas as pd

    variables = [name for name in trace.posterior.data_vars if name != "y_obs"]
    summary = az.summary(
        trace,
        var_names=variables,
        round_to=8,
        hdi_prob=0.95,
    )
    summary.insert(0, "fold", fold + 1)
    summary.insert(0, "model", model_name)
    diverging = (
        int(np.asarray(trace.sample_stats["diverging"]).sum())
        if "diverging" in trace.sample_stats
        else 0
    )
    max_rhat = float(np.nanmax(summary["r_hat"]))
    min_bulk = float(np.nanmin(summary["ess_bulk"]))
    min_tail = float(np.nanmin(summary["ess_tail"]))
    info = {
        "model": model_name,
        "fold": fold + 1,
        "diverging": diverging,
        "max_rhat": max_rhat,
        "min_ess_bulk": min_bulk,
        "min_ess_tail": min_tail,
        "converged": bool(
            diverging == 0
            and np.isfinite(max_rhat)
            and max_rhat < 1.05
            and min_bulk >= 200
            and min_tail >= 200
        ),
    }
    return summary.reset_index(names="parameter"), info


def fit_one(
    frame_name: str,
    frame: Any,
    model_name: str,
    fold: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import pymc as pm

    include_outside = frame_name == "primary"
    arrays = prepare_fold(frame, fold)
    model = build_model(model_name, arrays, include_outside=include_outside)
    stem = f"{frame_name}/{model_name}/fold_{fold + 1}"
    trace_path = ensure_phase_path(PHASE / "traces" / f"{stem}.nc")
    prediction_path = ensure_phase_path(
        PHASE / "predictions" / frame_name / f"{model_name}_fold_{fold + 1}.csv"
    )
    if trace_path.exists() and prediction_path.exists() and not args.overwrite:
        return {
            "frame": frame_name,
            "model": model_name,
            "fold": fold + 1,
            "status": "SKIPPED_EXISTING",
        }

    started = time.perf_counter()
    with model:
        trace = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            target_accept=args.target_accept,
            random_seed=args.seed,
            progressbar=True,
            idata_kwargs={"log_likelihood": False},
        )
    elapsed = time.perf_counter() - started

    from src.trace_io import save_trace

    saved_trace = save_trace(trace, trace_path)
    probability = predictor_samples(trace, arrays, "test", include_outside)
    prediction = pd.DataFrame(
        {
            "id": arrays["test_ids"],
            "fold": fold + 1,
            "y": arrays["y_test"],
            "prob": probability.mean(axis=0),
            "prob_sd": probability.std(axis=0),
            "prob_lower": np.percentile(probability, 2.5, axis=0),
            "prob_upper": np.percentile(probability, 97.5, axis=0),
            "p_ref_train": arrays["p_ref_train"],
        }
    )
    prediction.to_csv(prediction_path, index=False)
    coefficient_table, info = summarize_trace(trace, model_name, fold)
    coefficient_table.to_csv(
        ensure_phase_path(
            PHASE
            / "diagnostics"
            / frame_name
            / f"{model_name}_fold_{fold + 1}_summary.csv"
        ),
        index=False,
    )
    info.update(
        {
            "frame": frame_name,
            "status": "PASS" if info["converged"] else "FAIL_CONVERGENCE",
            "elapsed_seconds": elapsed,
            "trace": str(saved_trace.relative_to(PHASE)),
            "prediction": str(prediction_path.relative_to(PHASE)),
            "n_train": arrays["train_size"],
            "n_test": arrays["test_size"],
            "positive_train": int(arrays["y_train"].sum()),
            "positive_test": int(arrays["y_test"].sum()),
            "n_rocks": arrays["n_rocks"],
            "draws": args.draws,
            "tune": args.tune,
            "chains": args.chains,
            "cores": args.cores,
            "target_accept": args.target_accept,
        }
    )
    return info


def fit_outputs(args: argparse.Namespace) -> None:
    frames = load_frames()
    results_path = PHASE / "diagnostics" / "fit_status.json"
    results = []
    if results_path.is_file():
        results = json.loads(results_path.read_text(encoding="utf-8"))
    for frame_name in args.frames:
        if frame_name not in frames:
            raise KeyError(f"Unknown frame {frame_name}")
        for model_name in args.models:
            if model_name not in MODEL_ORDER:
                raise KeyError(f"Unknown model {model_name}")
            for fold in args.folds:
                print(
                    f"[Phase 2b] {frame_name} {model_name} fold {fold + 1}",
                    flush=True,
                )
                result = fit_one(frame_name, frames[frame_name], model_name, fold, args)
                print(f"[Phase 2b] {result}", flush=True)
                existing = next(
                    (
                        row
                        for row in results
                        if row.get("frame") == frame_name
                        and row.get("model") == model_name
                        and row.get("fold") == fold + 1
                    ),
                    None,
                )
                if result.get("status") == "SKIPPED_EXISTING" and existing:
                    result = {**existing, "status": "SKIPPED_EXISTING"}
                results = [
                    row
                    for row in results
                    if not (
                        row.get("frame") == frame_name
                        and row.get("model") == model_name
                        and row.get("fold") == fold + 1
                    )
                ]
                results.append(result)
                write_json(results_path, results)
    failed = [
        row
        for row in results
        if row.get("status") not in {"PASS", "SKIPPED_EXISTING"}
    ]
    if failed:
        raise SystemExit(f"Failed or incomplete fits: {failed}")


def read_oof(frame_name: str, model_name: str) -> Any:
    import pandas as pd

    paths = sorted(
        (PHASE / "predictions" / frame_name).glob(f"{model_name}_fold_*.csv")
    )
    if len(paths) != 4:
        raise FileNotFoundError(
            f"Expected four predictions for {frame_name}/{model_name}; found {len(paths)}"
        )
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    return frame.sort_values(["fold", "id"]).reset_index(drop=True)


def spatial_block_bootstrap(
    lower: Any,
    upper: Any,
    replicates: int = 2000,
    block_m: float = 25000.0,
) -> dict[str, Any]:
    import numpy as np
    from src.v15.metrics import log_score

    frame = load_frames()["primary"].loc[:, ["id", "centroid_x", "centroid_y"]]
    merged = (
        lower[["id", "y", "prob"]]
        .merge(
            upper[["id", "prob"]],
            on="id",
            suffixes=("_lower", "_upper"),
            validate="one_to_one",
        )
        .merge(frame, on="id", validate="one_to_one")
    )
    delta = log_score(merged["y"], merged["prob_upper"]) - log_score(
        merged["y"], merged["prob_lower"]
    )
    bx = np.floor(
        (merged["centroid_x"].to_numpy(float) - merged["centroid_x"].min()) / block_m
    ).astype(int)
    by = np.floor(
        (merged["centroid_y"].to_numpy(float) - merged["centroid_y"].min()) / block_m
    ).astype(int)
    merged = merged.assign(delta=delta, block=bx * 100000 + by)
    grouped = merged.groupby("block")["delta"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(42)
    draws = np.empty(replicates)
    for index in range(replicates):
        sample = rng.integers(0, len(grouped), size=len(grouped))
        draws[index] = sums[sample].sum() / counts[sample].sum()
    return {
        "ci_lower": float(np.percentile(draws, 2.5)),
        "ci_upper": float(np.percentile(draws, 97.5)),
        "blocks": int(len(grouped)),
        "replicates": replicates,
        "block_size_m": block_m,
    }


def comparison(lower: Any, upper: Any, bootstrap: bool = True) -> dict[str, Any]:
    from src.v15.metrics import log_score, roc_auc

    merged = lower[["id", "fold", "y", "prob"]].merge(
        upper[["id", "prob"]],
        on="id",
        suffixes=("_lower", "_upper"),
        validate="one_to_one",
    )
    lower_score = log_score(merged["y"], merged["prob_lower"])
    upper_score = log_score(merged["y"], merged["prob_upper"])
    row = {
        "n": int(len(merged)),
        "positives": int(merged["y"].sum()),
        "auc_lower": roc_auc(merged["y"], merged["prob_lower"]),
        "auc_upper": roc_auc(merged["y"], merged["prob_upper"]),
        "delta_auc": roc_auc(merged["y"], merged["prob_upper"])
        - roc_auc(merged["y"], merged["prob_lower"]),
        "mean_delta_log_score": float((upper_score - lower_score).mean()),
        "total_delta_log_score": float((upper_score - lower_score).sum()),
    }
    if bootstrap:
        row.update(spatial_block_bootstrap(lower, upper))
    return row


def coefficient_evidence(frame_name: str, models: Iterable[str]) -> Any:
    import pandas as pd

    rows = []
    for model_name in models:
        for fold in range(1, 5):
            path = (
                PHASE
                / "diagnostics"
                / frame_name
                / f"{model_name}_fold_{fold}_summary.csv"
            )
            table = pd.read_csv(path)
            table = table[
                table["parameter"].str.contains(
                    r"3b|outside_nrb", regex=True, na=False
                )
            ]
            for row in table.to_dict("records"):
                rows.append(
                    {
                        "frame": frame_name,
                        "model": model_name,
                        "fold": fold,
                        "parameter": row["parameter"],
                        "mean": row["mean"],
                        "sd": row["sd"],
                        "interval_lower": row["hdi_2.5%"],
                        "interval_upper": row["hdi_97.5%"],
                        "excludes_zero": bool(
                            row["hdi_2.5%"] > 0 or row["hdi_97.5%"] < 0
                        ),
                    }
                )
    return pd.DataFrame(rows)


def summarize_outputs() -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    from src.v15.metrics import metric_row

    all_metrics = []
    all_comparisons = []
    all_fold_comparisons = []
    decisions = {}
    frames = load_frames()
    evaluation_frames = (
        ["primary"]
        if SELECTED_FAULT_COLUMN == FAULT_COLUMN_STORED
        else FRAME_ORDER
    )
    for frame_name in evaluation_frames:
        predictions = {
            name: read_oof(frame_name, name)
            for name in MODEL_ORDER
            if len(
                list(
                    (PHASE / "predictions" / frame_name).glob(
                        f"{name}_fold_*.csv"
                    )
                )
            )
            == 4
        }
        if set(predictions) != set(MODEL_ORDER):
            raise ValueError(f"Incomplete prediction set for {frame_name}")
        for name, prediction in predictions.items():
            prediction.to_csv(
                ensure_phase_path(
                    PHASE / "predictions" / frame_name / f"oof_{name}.csv"
                ),
                index=False,
            )
            all_metrics.append(
                {"frame": frame_name, "model": name, **metric_row(prediction, "pooled")}
            )
            for fold, group in prediction.groupby("fold"):
                all_metrics.append(
                    {
                        "frame": frame_name,
                        "model": name,
                        **metric_row(group, f"fold_{int(fold)}"),
                    }
                )

        coefficients = coefficient_evidence(frame_name, MODEL_ORDER)
        coefficients.to_csv(
            ensure_phase_path(PHASE / "metrics" / f"{frame_name}_coefficients.csv"),
            index=False,
        )

        frame_comparisons = []
        for lower_name, upper_name in (
            ("N0_global", "N1_fixed_3b"),
            ("N1_fixed_3b", "N2_3b_distance_interactions"),
            ("N2_3b_distance_interactions", "N3_3b_gravity_interaction"),
        ):
            row = {
                "frame": frame_name,
                "lower": lower_name,
                "upper": upper_name,
                **comparison(
                    predictions[lower_name], predictions[upper_name], bootstrap=True
                ),
            }
            all_comparisons.append(row)
            frame_comparisons.append(row)
            for fold in range(1, 5):
                fold_lower = predictions[lower_name].loc[
                    predictions[lower_name]["fold"] == fold
                ]
                fold_upper = predictions[upper_name].loc[
                    predictions[upper_name]["fold"] == fold
                ]
                all_fold_comparisons.append(
                    {
                        "frame": frame_name,
                        "fold": fold,
                        "lower": lower_name,
                        "upper": upper_name,
                        **comparison(fold_lower, fold_upper, bootstrap=False),
                    }
                )

        if frame_name == "primary":
            decisions = phase4_decisions(
                frame_comparisons, all_fold_comparisons, coefficients
            )
            fold2_rows = []
            for lower_name, upper_name in (
                ("N1_fixed_3b", "N2_3b_distance_interactions"),
                ("N2_3b_distance_interactions", "N3_3b_gravity_interaction"),
            ):
                lower = predictions[lower_name].loc[
                    predictions[lower_name]["fold"] == 2
                ]
                upper = predictions[upper_name].loc[
                    predictions[upper_name]["fold"] == 2
                ]
                fold2_rows.append(
                    {
                        "frame": frame_name,
                        "fold": 2,
                        "lower": lower_name,
                        "upper": upper_name,
                        **comparison(lower, upper, bootstrap=False),
                    }
                )
            pd.DataFrame(fold2_rows).to_csv(
                ensure_phase_path(PHASE / "metrics" / "fold2_descriptive.csv"),
                index=False,
            )

    metrics = pd.DataFrame(all_metrics)
    comparisons = pd.DataFrame(all_comparisons)
    fold_comparisons = pd.DataFrame(all_fold_comparisons)
    metrics.to_csv(ensure_phase_path(PHASE / "metrics" / "model_metrics.csv"), index=False)
    comparisons.to_csv(
        ensure_phase_path(PHASE / "metrics" / "comparisons.csv"), index=False
    )
    fold_comparisons.to_csv(
        ensure_phase_path(PHASE / "metrics" / "fold_comparisons.csv"), index=False
    )
    write_json(PHASE / "metrics" / "decision.json", decisions)
    if SELECTED_FAULT_COLUMN == FAULT_COLUMN_STORED:
        fault_variant_comparison_table()
    print(
        metrics.loc[metrics["sample"] == "pooled"][
            ["frame", "model", "n", "positives", "auc", "mean_log_score"]
        ].to_string(index=False)
    )
    print("\nPrimary comparisons\n" + comparisons.loc[
        comparisons["frame"] == "primary"
    ].to_string(index=False))
    print("\nDecisions\n" + json.dumps(decisions, indent=2))
    return {
        "metrics": metrics,
        "comparisons": comparisons,
        "fold_comparisons": fold_comparisons,
        "decisions": decisions,
    }


def phase4_decisions(
    comparisons: list[dict[str, Any]],
    fold_comparisons: Any,
    coefficients: Any,
) -> dict[str, Any]:
    import pandas as pd

    if not isinstance(fold_comparisons, pd.DataFrame):
        fold_comparisons = pd.DataFrame(fold_comparisons)
    decisions: dict[str, Any] = {}
    for row in comparisons:
        if row["lower"] == "N0_global":
            continue
        subset = fold_comparisons.loc[
            (fold_comparisons["frame"] == "primary")
            & (fold_comparisons["lower"] == row["lower"])
            & (fold_comparisons["upper"] == row["upper"])
        ]
        positive_folds = int((subset["mean_delta_log_score"] > 0).sum())
        predictive = bool(
            row["mean_delta_log_score"] > 0
            and row["delta_auc"] > 0
            and row["ci_lower"] > 0
        )
        fold_rule = positive_folds >= 3

        if row["upper"] == "N2_3b_distance_interactions":
            relevant = coefficients.loc[
                (coefficients["frame"] == "primary")
                & (coefficients["model"] == "N2_3b_distance_interactions")
                & (coefficients["parameter"].isin(INTERACTION_TERMS))
            ]
            stable_terms = []
            for parameter, group in relevant.groupby("parameter"):
                if int(group["excludes_zero"].sum()) >= 3:
                    stable_terms.append(parameter)
            coefficient_rule = len(stable_terms) >= 2
            coefficient_detail = {
                "criterion": (
                    "majority of the four added 3b x distance terms have a "
                    "95% interval excluding zero in >= 3 of 4 folds"
                ),
                "stable_terms": stable_terms,
                "stable_term_count": len(stable_terms),
            }
        else:
            relevant = coefficients.loc[
                (coefficients["frame"] == "primary")
                & (coefficients["model"] == "N3_3b_gravity_interaction")
                & (coefficients["parameter"] == "beta_3b_grav")
            ]
            folds_excluding_zero = int(relevant["excludes_zero"].sum())
            coefficient_rule = folds_excluding_zero >= 3
            coefficient_detail = {
                "criterion": (
                    "3b x gravity interaction interval excludes zero in >= 3 of 4 folds"
                ),
                "folds_excluding_zero": folds_excluding_zero,
            }

        decisions[f"{row['lower']}->{row['upper']}"] = {
            "supported": bool(predictive and fold_rule and coefficient_rule),
            "conditions": {
                "pooled_log_score_and_auc_improve": bool(
                    row["mean_delta_log_score"] > 0 and row["delta_auc"] > 0
                ),
                "spatial_block_95_interval_excludes_zero": bool(
                    row["ci_lower"] > 0
                ),
                "positive_log_score_folds_ge_3_of_4": fold_rule,
                "coefficient_stability": coefficient_rule,
            },
            "positive_log_score_folds": positive_folds,
            "coefficient_detail": coefficient_detail,
        }
    return decisions


def fault_variant_comparison_table() -> Any:
    """Build the Phase 2c stored/rebuilt 2x2 table plus V15 references."""

    import pandas as pd

    rows: list[dict[str, Any]] = []
    for variant, root in (
        ("stored", PHASE_2C),
        ("rebuilt", PHASE_2B),
    ):
        comparisons = pd.read_csv(root / "metrics" / "comparisons.csv")
        fold_comparisons = pd.read_csv(root / "metrics" / "fold_comparisons.csv")
        decisions = json.loads(
            (root / "metrics" / "decision.json").read_text(encoding="utf-8")
        )
        for lower, upper, label in (
            ("N1_fixed_3b", "N2_3b_distance_interactions", "N1-N2"),
            (
                "N2_3b_distance_interactions",
                "N3_3b_gravity_interaction",
                "N2-N3",
            ),
        ):
            comparison = comparisons.loc[
                (comparisons["frame"] == "primary")
                & (comparisons["lower"] == lower)
                & (comparisons["upper"] == upper)
            ].iloc[0]
            fold = fold_comparisons.loc[
                (fold_comparisons["frame"] == "primary")
                & (fold_comparisons["lower"] == lower)
                & (fold_comparisons["upper"] == upper)
            ]
            decision = decisions[f"{lower}->{upper}"]
            rows.append(
                {
                    "row_type": "fixed-effect ladder",
                    "fault_variant": variant,
                    "comparison": label,
                    "mean_log_score_gain": comparison["mean_delta_log_score"],
                    "ci_lower": comparison["ci_lower"],
                    "ci_upper": comparison["ci_upper"],
                    "positive_folds": int(
                        (fold["mean_delta_log_score"] > 0).sum()
                    ),
                    "verdict": (
                        "SUPPORTED" if decision["supported"] else "NOT SUPPORTED"
                    ),
                }
            )

    v15 = ROOT / "paper_1" / "v15"
    central = pd.read_csv(v15 / "metrics" / "central_comparisons.csv")
    bootstrap = pd.read_csv(v15 / "metrics" / "paired_bootstrap.csv")
    by_fold = pd.read_csv(v15 / "metrics" / "paired_by_fold.csv")
    for lower, upper, label, verdict in (
        (
            "M1_domain_intercept",
            "M2_domain_distance",
            "M1-M2",
            "NOT SUPPORTED",
        ),
        (
            "M2_domain_distance",
            "M3_domain_distance_gravity",
            "M2-M3",
            "SUPPORTED",
        ),
    ):
        comparison = central.loc[
            (central["lower"] == lower) & (central["upper"] == upper)
        ].iloc[0]
        interval = bootstrap.loc[
            (bootstrap["left"] == lower) & (bootstrap["right"] == upper)
        ].iloc[0]
        folds = by_fold.loc[
            (by_fold["left"] == lower) & (by_fold["right"] == upper)
        ]
        rows.append(
            {
                "row_type": "V15 hierarchical reference",
                "fault_variant": "stored",
                "comparison": label,
                "mean_log_score_gain": comparison["mean_delta_log_score"],
                "ci_lower": -interval["ci_upper"],
                "ci_upper": -interval["ci_lower"],
                "positive_folds": int(
                    (folds["mean_delta_log_score"] < 0).sum()
                ),
                "verdict": verdict,
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(
        ensure_phase_path(PHASE / "metrics" / "fault_variant_2x2.csv"),
        index=False,
    )
    return table


def fault_reading(table: Any) -> str:
    stored = table.loc[
        (table["row_type"] == "fixed-effect ladder")
        & (table["fault_variant"] == "stored")
    ].set_index("comparison")
    rebuilt = table.loc[
        (table["row_type"] == "fixed-effect ladder")
        & (table["fault_variant"] == "rebuilt")
    ].set_index("comparison")
    comparisons_match = bool(
        (stored["verdict"] == rebuilt["verdict"]).all()
    )
    if (
        stored.loc["N2-N3", "verdict"] == "SUPPORTED"
        and rebuilt.loc["N2-N3", "verdict"] == "NOT SUPPORTED"
    ):
        return (
            "N2-to-N3 meets the rule with stored faults only. The V15 gravity "
            "gain depends on the fault data."
        )
    if (
        stored.loc["N2-N3", "verdict"] == "NOT SUPPORTED"
        and rebuilt.loc["N2-N3", "verdict"] == "NOT SUPPORTED"
    ):
        return (
            "The fixed-effect verdicts match under both fault versions, so the "
            "Phase 2b fixed-effect conclusions are robust to the fault-layer "
            "choice. Neither fixed-effect N2-to-N3 comparison meets the rule, "
            "whereas the V15 hierarchical M2-to-M3 comparison does; the V15 "
            "gravity gain therefore depended on the six-domain hierarchical "
            "structure."
        )
    if comparisons_match:
        return (
            "The fixed-effect verdicts match under both fault versions. The "
            "Phase 2b conclusions are robust to the fault-layer choice."
        )
    return (
        "The fault-version verdicts do not fit the pre-stated directional "
        "cases; report the contrast as unresolved without reframing it."
    )


def write_report() -> Path:
    import pandas as pd

    audit = json.loads((PHASE / "data" / "input_audit.json").read_text())
    metrics = pd.read_csv(PHASE / "metrics" / "model_metrics.csv")
    comparisons = pd.read_csv(PHASE / "metrics" / "comparisons.csv")
    fold_comparisons = pd.read_csv(PHASE / "metrics" / "fold_comparisons.csv")
    coefficients = pd.read_csv(PHASE / "metrics" / "primary_coefficients.csv")
    fold2 = pd.read_csv(PHASE / "metrics" / "fold2_descriptive.csv")
    decisions = json.loads((PHASE / "metrics" / "decision.json").read_text())
    fit_status = json.loads((PHASE / "diagnostics" / "fit_status.json").read_text())
    pooled = metrics.loc[metrics["sample"] == "pooled"]
    is_stored = SELECTED_FAULT_COLUMN == FAULT_COLUMN_STORED

    lines = [
        (
            "# Phase 2c - stored-fault robustness"
            if is_stored
            else "# Phase 2b - two-domain fixed-effect frame"
        ),
        "",
        f"Generated: {utc_now()}",
        "",
        "## Frame and provenance",
        "",
        (
            "The **full 1,872-cell frame is primary**. It includes fixed outside-NRB "
            "and fixed NRB_3b terms under the same complementary log-log likelihood "
            "and fixed log-area offset as V15. The **1,677-cell NRB-only frame is "
            "secondary and descriptive only**."
        ),
        "",
        "| Frame | Cells | Positives | Role |",
        "|---|---:|---:|---|",
        (
            f"| Full frame | {audit['primary']['cells']:,} | "
            f"{audit['primary']['positives']} | Primary; source of verdicts |"
        ),
        (
            f"| NRB_3a + NRB_3b | {audit['secondary']['cells']:,} | "
            f"{audit['secondary']['positives']} | Secondary; no verdicts |"
        ),
        "",
        (
            "Distance to fault is the **stored V15 value**. The rebuilt variant "
            f"differs for {audit['fault_variant']['not_reproduced']} cells, "
            f"including {audit['fault_variant']['not_reproduced_positives']} "
            "positives. The six added segments in the rebuilt layer are "
            "classified AMBIGUOUS and neither fault version is privileged. "
            "V15 fold IDs were copied unchanged."
            if is_stored
            else (
                f"Distance to fault was rebuilt from "
                f"{audit['fault_variant']['features']} surviving structure lines "
                f"(`SHA-256 {audit['fault_variant']['sha256']}`). It reproduces "
                f"the stored distance within 1 m for "
                f"{audit['fault_variant']['reproduced_within_1m']:,} cells and "
                f"differs for {audit['fault_variant']['not_reproduced']} cells, "
                f"including {audit['fault_variant']['not_reproduced_positives']} "
                "positives. V15 fold IDs were copied unchanged."
            )
        ),
        "",
        "## Fold composition",
        "",
        "| Fold | Cells | Positives | NRB_3a | NRB_3b | Outside NRB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    primary_frame = pd.read_csv(PHASE / "data" / "primary_frame.csv")
    for fold in range(1, 5):
        group = primary_frame.loc[primary_frame["fold"] == fold - 1]
        counts = group["daly_domain"].value_counts()
        lines.append(
            f"| {fold} | {len(group)} | {int(group[Y_COL].sum())} | "
            f"{int(counts.get('NRB_3a', 0))} | {int(counts.get('NRB_3b', 0))} | "
            f"{int((group['daly_domain'].isin(['CRZ', 'MMSB', 'NKB', 'SRB'])).sum())} |"
        )
    lines.extend(
        [
            "",
            "| Secondary fold | Cells | Positives |",
            "|---:|---:|---:|",
        ]
    )
    secondary_frame = pd.read_csv(PHASE / "data" / "secondary_frame.csv")
    for fold in range(1, 5):
        group = secondary_frame.loc[secondary_frame["fold"] == fold - 1]
        lines.append(f"| {fold} | {len(group)} | {int(group[Y_COL].sum())} |")
    lines.extend(
        [
            "",
            (
                "The 3a/3b contrast is nearly confounded with fold: **Fold 1 is "
                "NRB_3b only (with outside-NRB support), Folds 3-4 are NRB_3a only "
                "(Fold 4 includes 16 MMSB cells), and only Fold 2 is mixed.** "
                "Fold-specific estimates are therefore descriptive and cannot "
                "identify a fold-invariant 3b effect."
            ),
            "",
            "## Models",
            "",
            "- N0: global terms; fixed outside-NRB term in the primary frame.",
            "- N1: N0 plus the fixed 3b indicator.",
            "- N2: N1 plus 3b x fault, fault squared, contact and contact squared.",
            "- N3: N2 plus 3b x gravity.",
            "- All coefficients have independent Normal(0, 1) priors; no hierarchical scales are estimated.",
            "",
            "## Held-out performance",
            "",
            "| Frame | Model | Cells | Positives | AUC | PR-AUC | Mean log score | Brier |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pooled.itertuples(index=False):
        lines.append(
            f"| {row.frame} | {row.model} | {row.n} | {row.positives} | "
            f"{row.auc:.4f} | {row.pr_auc:.4f} | {row.mean_log_score:.5f} | "
            f"{row.brier:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Primary comparisons",
            "",
            (
                "Orientation is upper minus lower. Positive log-score and AUC "
                "differences favour the upper model. Intervals use 2,000 "
                "resamples of 25-km spatial blocks (117 blocks in the primary "
                "frame)."
            ),
            "",
            "| Comparison | Delta AUC | Mean Delta log score | Spatial-block 95% CI | Positive folds |",
            "|---|---:|---:|---|---:|",
        ]
    )
    for row in comparisons.loc[comparisons["frame"] == "primary"].itertuples(index=False):
        subset = fold_comparisons.loc[
            (fold_comparisons["frame"] == "primary")
            & (fold_comparisons["lower"] == row.lower)
            & (fold_comparisons["upper"] == row.upper)
        ]
        positive = int((subset["mean_delta_log_score"] > 0).sum())
        lines.append(
            f"| {row.lower} -> {row.upper} | {row.delta_auc:+.4f} | "
            f"{row.mean_delta_log_score:+.5f} | "
            f"[{row.ci_lower:+.5f}, {row.ci_upper:+.5f}] | {positive}/4 |"
        )
    if is_stored:
        robustness = pd.read_csv(PHASE / "metrics" / "fault_variant_2x2.csv")
        lines.extend(
            [
                "",
                "## Fault-layer robustness",
                "",
                (
                    "The fixed-effect rows use the same primary frame, V15 folds, "
                    "prior, sampler, seed, spatial-block bootstrap and decision "
                    "rule. V15 hierarchical rows are reference contrasts using "
                    "stored fault distances."
                ),
                "",
                "| Ladder / fault version | Comparison | Mean log-score gain | 95% spatial-block CI | Positive folds | Verdict |",
                "|---|---|---:|---|---:|---|",
            ]
        )
        for row in robustness.itertuples(index=False):
            lines.append(
                f"| {row.row_type}: {row.fault_variant} | {row.comparison} | "
                f"{row.mean_log_score_gain:+.5f} | "
                f"[{row.ci_lower:+.5f}, {row.ci_upper:+.5f}] | "
                f"{row.positive_folds}/4 | {row.verdict} |"
            )
        lines.extend(["", f"**Reading:** {fault_reading(robustness)}"])
    if not is_stored:
        lines.extend(
            [
                "",
                "## Secondary-frame comparisons",
                "",
                "These values are descriptive. They are not used for any verdict.",
                "",
                "| Comparison | Delta AUC | Mean Delta log score | Spatial-block 95% CI | Positive folds |",
                "|---|---:|---:|---|---:|",
            ]
        )
        for row in comparisons.loc[comparisons["frame"] == "secondary"].itertuples(index=False):
            subset = fold_comparisons.loc[
                (fold_comparisons["frame"] == "secondary")
                & (fold_comparisons["lower"] == row.lower)
                & (fold_comparisons["upper"] == row.upper)
            ]
            positive = int((subset["mean_delta_log_score"] > 0).sum())
            lines.append(
                f"| {row.lower} -> {row.upper} | {row.delta_auc:+.4f} | "
                f"{row.mean_delta_log_score:+.5f} | "
                f"[{row.ci_lower:+.5f}, {row.ci_upper:+.5f}] | {positive}/4 |"
            )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                "The operational Phase 4 adaptation required: pooled log-score "
                "and AUC improvement; a 25-km spatial-block 95% interval for "
                "log-score gain excluding zero; improvement in at least 3 of 4 "
                "folds; and the stated coefficient-stability condition. For N1 -> "
                "N2, at least two of the four added 3b-distance coefficients must "
                "exclude zero in at least 3 of 4 folds."
            ),
            "",
        ]
    )
    for name, decision in decisions.items():
        lines.append(
            f"- **{name}: {'SUPPORTED' if decision['supported'] else 'NOT SUPPORTED'}.** "
            f"Positive folds: {decision['positive_log_score_folds']}/4; "
            f"coefficient evidence: {decision['coefficient_detail']}."
        )
    gravity_coefficients = coefficients.loc[
        (coefficients["frame"] == "primary")
        & (coefficients["model"] == "N3_3b_gravity_interaction")
        & (coefficients["parameter"] == "beta_3b_grav")
    ]
    gravity_negative_folds = int((gravity_coefficients["mean"] < 0).sum())
    gravity_excluding_folds = int(gravity_coefficients["excludes_zero"].sum())
    n3_decision = decisions[
        "N2_3b_distance_interactions->N3_3b_gravity_interaction"
    ]
    lines.extend(
        [
            "",
            (
                "The focal 3b x gravity coefficient is negative in "
                f"{gravity_negative_folds}/4 folds and its 95% interval excludes "
                f"zero in {gravity_excluding_folds}/4 folds. The N2-to-N3 "
                f"verdict is "
                f"{'supported' if n3_decision['supported'] else 'not supported'}; "
                f"the log score improves in "
                f"{n3_decision['positive_log_score_folds']}/4 folds."
            ),
            "",
            "| Fold | 3b x gravity posterior mean | 95% HDI | Excludes zero |",
            "|---:|---:|---|---|",
        ]
    )
    for row in gravity_coefficients.itertuples(index=False):
        lines.append(
            f"| {row.fold} | {row.mean:+.4f} | "
            f"[{row.interval_lower:+.4f}, {row.interval_upper:+.4f}] | "
            f"{'yes' if row.excludes_zero else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Fold 2 descriptive check",
            "",
            "| Comparison | Delta AUC | Mean Delta log score |",
            "|---|---:|---:|",
        ]
    )
    for row in fold2.itertuples(index=False):
        lines.append(
            f"| {row.lower} -> {row.upper} | {row.delta_auc:+.4f} | "
            f"{row.mean_delta_log_score:+.5f} |"
        )
    lines.extend(
        [
            "",
            "## Coefficient stability",
            "",
            "| Model | Parameter | Folds with interval excluding zero |",
            "|---|---|---:|",
        ]
    )
    grouped = coefficients.groupby(["model", "parameter"])["excludes_zero"].sum()
    for (model, parameter), count in grouped.items():
        lines.append(f"| {model} | {parameter} | {int(count)}/4 |")
    failed = [
        row
        for row in fit_status
        if row.get("status") not in {"PASS", "SKIPPED_EXISTING"}
    ]
    lines.extend(
        [
            "",
            "## Convergence and interpretation",
            "",
            (
                f"Fits recorded: {len(fit_status)}; convergence failures: "
                f"{len(failed)}. The convergence screen requires zero divergences, "
                "R-hat < 1.05, and bulk and tail ESS >= 200. Each fit used two "
                "chains, 500 retained draws, 750 tuning draws and target "
                "acceptance 0.95, with chains run two at a time in one model job."
            ),
            "",
            (
                "Conclusions are drawn only from the full primary frame. "
                + (
                    "The NRB-only frame was not refitted in Phase 2c; its "
                    "composition is shown for transparency."
                    if is_stored
                    else (
                        "The NRB-only frame is reported separately because four "
                        "outside-NRB domains have zero positives and therefore "
                        "cannot inform domain performance."
                    )
                )
            ),
            "",
            (
                f"**Fault-layer reading:** {fault_reading(robustness)}"
                if is_stored
                else (
                    "N3 improves pooled ranking and log score relative to N2, "
                    "but it worsens Brier score and does not meet the predictive "
                    "stability part of the decision rule. It is therefore not "
                    "promoted for the primary scientific conclusion."
                )
            ),
            "",
        ]
    )
    report_name = (
        "phase2c_stored_faults_report.md"
        if is_stored
        else "phase2b_report.md"
    )
    report_path = ensure_phase_path(PHASE / "reports" / report_name)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def report_outputs() -> Path:
    path = write_report()
    print(path)
    return path


def write_checksums() -> Path:
    checksum_path = PHASE / "SHA256SUMS"
    lines = []
    for path in sorted(PHASE.rglob("*")):
        if not path.is_file() or path == checksum_path:
            continue
        lines.append(f"{sha256(path)}  {path.relative_to(PHASE).as_posix()}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(checksum_path)
    return checksum_path


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument(
        "--fault-column",
        choices=FAULT_COLUMNS,
        default=FAULT_COLUMN_REBUILT,
        help=(
            "Fault-distance column used by the model. The stored option writes "
            "to paper_1/v15/phase2c_stored_faults."
        ),
    )
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    fit = sub.add_parser("fit")
    fit.add_argument(
        "--frames", nargs="+", choices=FRAME_ORDER, default=FRAME_ORDER
    )
    fit.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=MODEL_ORDER)
    fit.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3])
    fit.add_argument("--draws", type=int, default=1000)
    fit.add_argument("--tune", type=int, default=1500)
    fit.add_argument("--chains", type=int, default=2)
    fit.add_argument("--cores", type=int, default=1)
    fit.add_argument("--target-accept", type=float, default=0.95)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--overwrite", action="store_true")
    sub.add_parser("summarize")
    sub.add_parser("report")
    sub.add_parser("checksums")
    sub.add_parser("all")
    return root


def main() -> None:
    args = parser().parse_args()
    select_fault_column(args.fault_column)
    if args.command in {"build", "all"}:
        build_outputs()
    if args.command in {"fit", "all"}:
        if args.command == "all":
            args = parser().parse_args(
                [
                    "--fault-column",
                    SELECTED_FAULT_COLUMN,
                    "fit",
                    "--frames",
                    *FRAME_ORDER,
                    "--models",
                    *MODEL_ORDER,
                ]
            )
        fit_outputs(args)
    if args.command in {"summarize", "all"}:
        summarize_outputs()
    if args.command in {"report", "all"}:
        report_outputs()
    if args.command in {"checksums", "all"}:
        write_checksums()


if __name__ == "__main__":
    main()
