"""Data loading and training-only model design for transfer models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.transfer import config


def ensure_output_path(path: Path) -> Path:
    path = Path(path).resolve()
    root = config.OUTPUT.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Refusing to write outside transfer output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_transfer_frame() -> pd.DataFrame:
    frame = pd.read_csv(config.MODEL_FRAME)
    rebuilt = pd.read_csv(
        config.PHASE2B_FRAME,
        usecols=["id", "distance_to_fault_km"],
    ).rename(columns={"distance_to_fault_km": "fault_rebuilt_km"})
    frame = frame.merge(rebuilt, on="id", how="left", validate="one_to_one")
    if frame["fault_rebuilt_km"].isna().any():
        raise ValueError("Rebuilt fault distances are missing for complete-case cells")
    frame["fault_stored_km"] = frame["distance_to_fault"].astype(float) / 1000.0
    frame["contact_km"] = frame["distance_to_lithology_contact"].astype(float) / 1000.0
    frame["gravity_mgal"] = frame["bouguer"].astype(float)
    frame[config.OUTSIDE_NRB] = (
        ~frame["daly_domain"].isin(["NRB_3a", "NRB_3b"])
    ).astype(int)
    if len(frame) != 1872 or int(frame[config.Y_COL].sum()) != 138:
        raise ValueError("Transfer frame is not the established 1,872-cell / 138-positive frame")
    if sorted(frame[config.FOLD_COL].unique().tolist()) != [0, 1, 2, 3]:
        raise ValueError("Canonical V15 fold IDs are incomplete")
    return frame.reset_index(drop=True)


def model_options(model_name: str) -> dict[str, Any]:
    if model_name == config.PRIMARY_MODEL:
        return {"fault_column": "fault_stored_km", "quadratic": True, "outside_nrb": False}
    if model_name not in config.SENSITIVITY_MODELS:
        raise KeyError(f"Unknown transfer model: {model_name}")
    options = {
        "fault_column": "fault_stored_km",
        "quadratic": True,
        "outside_nrb": False,
        **config.SENSITIVITY_MODELS[model_name],
    }
    return options


def feature_names(options: dict[str, Any]) -> list[str]:
    names = ["fault_z"]
    if options["quadratic"]:
        names.append("fault_z2")
    names.append("contact_z")
    if options["quadratic"]:
        names.append("contact_z2")
    names.append("gravity_z")
    if options["outside_nrb"]:
        names.append("outside_nrb")
    return names


def build_design(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    model_name: str = config.PRIMARY_MODEL,
) -> dict[str, Any]:
    """Build arrays using only training-partition transformations."""

    options = model_options(model_name)
    train = frame.loc[train_mask].copy()
    test = frame.loc[test_mask].copy()
    if train.empty or test.empty:
        raise ValueError("Training and test partitions must both be non-empty")
    if set(train["id"]).intersection(set(test["id"])):
        raise ValueError("A cell appears in both training and test data")

    raw_columns = {
        "fault": options["fault_column"],
        "contact": "contact_km",
        "gravity": "gravity_mgal",
    }
    parameters: dict[str, float] = {}
    transformed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, column in raw_columns.items():
        train_values = train[column].to_numpy(dtype=float)
        test_values = test[column].to_numpy(dtype=float)
        mean = float(train_values.mean())
        scale = float(train_values.std(ddof=0))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid training scale for {name}")
        parameters[f"{name}_mean"] = mean
        parameters[f"{name}_scale"] = scale
        transformed[name] = (
            (train_values - mean) / scale,
            (test_values - mean) / scale,
        )

    train_blocks = []
    test_blocks = []
    for name in ("fault", "contact", "gravity"):
        train_z, test_z = transformed[name]
        train_blocks.append(train_z)
        test_blocks.append(test_z)
        if options["quadratic"] and name in {"fault", "contact"}:
            train_blocks.append(train_z**2)
            test_blocks.append(test_z**2)
    if options["outside_nrb"]:
        train_blocks.append(train[config.OUTSIDE_NRB].to_numpy(dtype=float))
        test_blocks.append(test[config.OUTSIDE_NRB].to_numpy(dtype=float))

    X_train = np.column_stack(train_blocks)
    X_test = np.column_stack(test_blocks)
    names = feature_names(options)
    if X_train.shape[1] != len(names):
        raise ValueError("Design matrix and feature names disagree")
    y_train = train[config.Y_COL].to_numpy(dtype=int)
    y_test = test[config.Y_COL].to_numpy(dtype=int)
    prevalence_base = np.log(
        (float(y_train.sum()) + 0.5)
        / (float(np.exp(train[config.AREA_COL]).sum()) + 0.5)
    )
    return {
        "model": model_name,
        "options": options,
        "train": train.reset_index(drop=True),
        "test": test.reset_index(drop=True),
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "offset_train": train[config.AREA_COL].to_numpy(dtype=float),
        "offset_test": test[config.AREA_COL].to_numpy(dtype=float),
        "features": names,
        "parameters": parameters,
        "intercept_mean": prevalence_base,
        "train_ids": train["id"].to_numpy(dtype=int),
        "test_ids": test["id"].to_numpy(dtype=int),
    }


def subset_design(data: dict[str, Any], indices: np.ndarray) -> dict[str, Any]:
    """Return an internally consistent row subset of a prepared design."""

    return {
        **data,
        "train": data["train"].iloc[indices].reset_index(drop=True),
        "X_train": data["X_train"][indices],
        "y_train": data["y_train"][indices],
        "offset_train": data["offset_train"][indices],
        "train_ids": data["train_ids"][indices],
    }

