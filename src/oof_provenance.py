"""Validated, ID-based linkage for the frozen V11 out-of-fold predictions.

The original ``v11_oof_predictions.csv`` carried only ``centroid_x``/
``centroid_y``/``prob_v11`` and downstream scripts merged it back onto the
modeling frame on floating-point centroids.  That "first row per coordinate"
pattern is exactly the defect this module removes: predictions now carry a
stable grid-cell ``id``, the held-out ``fold`` and the observed label, are
validated against the canonical training table, and are joined only on ``id``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OOF_PATH = ROOT / "figures" / "v11_oof_predictions.csv"
DEFAULT_MANIFEST_PATH = ROOT / "figures" / "v11_oof_predictions_manifest.json"
CANONICAL_DATA_PATH = ROOT / "data" / "copperbelt_training_v13_litho_contact.csv"
V14_OOF_PATH = (
    ROOT / "figures" / "v14_centroid_contact" / "v14_centroid_oof_predictions.csv"
)
V14_MANIFEST_PATH = (
    ROOT / "figures" / "v14_centroid_contact" / "v14_centroid_oof_predictions_manifest.json"
)
V14_DATA_PATH = ROOT / "data" / "copperbelt_training_v14_centroid_contact.csv"

REQUIRED_PROVENANCE_COLUMNS = (
    "id",
    "fold",
    "deposit_present",
    "daly_domain",
    "centroid_x",
    "centroid_y",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_v11_oof(path: Path | str = DEFAULT_OOF_PATH) -> pd.DataFrame:
    """Load and validate the ID-bearing V11 OOF artifact."""
    return load_oof(path, probability_column="prob_v11", model_label="V11")


def load_v14_oof(path: Path | str = V14_OOF_PATH) -> pd.DataFrame:
    """Load and validate the ID-bearing V14 centroid-contact OOF artifact."""
    return load_oof(path, probability_column="prob_v14", model_label="V14")


def load_oof(
    path: Path | str,
    *,
    probability_column: str,
    model_label: str,
) -> pd.DataFrame:
    """Load an OOF artifact and enforce stable-ID and score invariants."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{model_label} OOF artifact not found: {path}")
    frame = pd.read_csv(path)

    required = tuple(REQUIRED_PROVENANCE_COLUMNS) + (probability_column,)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required provenance columns {missing}. "
            "Re-run src/extract_v11_oof.py; the legacy centroid-only artifact "
            "is archived and must not be used."
        )

    if frame["id"].isna().any():
        raise ValueError(f"{path} contains missing grid-cell ids.")
    if frame["id"].duplicated().any():
        duplicates = frame.loc[frame["id"].duplicated(), "id"].head(10).tolist()
        raise ValueError(f"{path} contains duplicated grid-cell ids: {duplicates}")

    folds = frame["fold"].to_numpy()
    if not set(folds).issubset({0, 1, 2, 3}):
        raise ValueError(f"{path} contains folds outside 0..3: {sorted(set(folds))}")

    probabilities = frame[probability_column]
    if probabilities.isna().any():
        raise ValueError(f"{path} contains missing {probability_column} values.")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError(f"{path} contains {probability_column} values outside [0, 1].")

    return frame


def merge_v11_oof(
    model_df: pd.DataFrame,
    oof_df: pd.DataFrame | None = None,
    *,
    fold_col: str = "spatial_block",
    label_col: str = "deposit_present",
) -> tuple[pd.DataFrame, dict]:
    """Attach ``prob_v11`` to ``model_df`` by grid-cell ``id``.

    Row order of ``model_df`` is preserved so that seeded bootstrap results
    remain comparable with the frozen outputs.  Uniqueness, fold assignment
    and observed-label agreement are all checked before anything is returned.
    """
    oof = load_v11_oof() if oof_df is None else oof_df
    return merge_oof(
        model_df,
        oof,
        probability_column="prob_v11",
        model_label="V11",
        fold_col=fold_col,
        label_col=label_col,
    )


def merge_v14_oof(
    model_df: pd.DataFrame,
    oof_df: pd.DataFrame | None = None,
    *,
    fold_col: str = "spatial_block",
    label_col: str = "deposit_present",
) -> tuple[pd.DataFrame, dict]:
    """Attach V14 predictions to the V14 modeling frame by stable grid-cell id."""
    oof = load_v14_oof() if oof_df is None else oof_df
    return merge_oof(
        model_df,
        oof,
        probability_column="prob_v14",
        model_label="V14",
        fold_col=fold_col,
        label_col=label_col,
    )


def merge_oof(
    model_df: pd.DataFrame,
    oof: pd.DataFrame,
    *,
    probability_column: str,
    model_label: str,
    fold_col: str = "spatial_block",
    label_col: str = "deposit_present",
) -> tuple[pd.DataFrame, dict]:
    """Join a validated OOF artifact on id and verify fold/label agreement."""

    if "id" not in model_df.columns:
        raise ValueError(
            "The modeling frame has no 'id' column; centroid-based linkage is "
            "not permitted. Use the canonical V13 training table."
        )
    if model_df["id"].duplicated().any():
        duplicates = model_df.loc[model_df["id"].duplicated(), "id"].head(10).tolist()
        raise ValueError(f"Modeling frame has duplicated ids: {duplicates}")
    if fold_col not in model_df.columns:
        raise ValueError(f"Modeling frame has no '{fold_col}' column.")

    model_ids = set(model_df["id"].tolist())
    oof_ids = set(oof["id"].tolist())
    missing_from_oof = model_ids - oof_ids
    extra_in_oof = oof_ids - model_ids
    if missing_from_oof or extra_in_oof:
        raise ValueError(
            f"{model_label} OOF artifact does not cover the modeling frame: "
            f"{len(missing_from_oof)} model ids missing, "
            f"{len(extra_in_oof)} OOF ids unmatched."
        )

    indexed = oof.set_index("id")
    merged = model_df.merge(
        indexed[["fold", probability_column]], left_on="id", right_index=True,
        how="left", validate="one_to_one",
    )

    fold_mismatch = int(
        (merged[fold_col].to_numpy() != merged["fold"].to_numpy()).sum()
    )
    if fold_mismatch:
        raise ValueError(
            f"{fold_mismatch} rows have an OOF fold that differs from the "
            f"modeling frame's {fold_col}; fold assignments must be unchanged."
        )

    label_mismatch = int(
        (merged[label_col].to_numpy() != indexed.loc[merged["id"], "deposit_present"].to_numpy()).sum()
    )
    if label_mismatch:
        raise ValueError(
            f"{label_mismatch} rows have an OOF label that differs from the "
            "modeling frame's deposit_present column."
        )

    if merged[probability_column].isna().any():
        raise ValueError(f"Join produced missing {probability_column} values.")

    report = {
        "rows": int(len(merged)),
        "unique_ids": int(merged["id"].nunique()),
        "fold_counts": {
            str(int(fold)): int(count)
            for fold, count in merged["fold"].value_counts().sort_index().items()
        },
        "label_counts": {
            str(int(label)): int(count)
            for label, count in merged[label_col].value_counts().sort_index().items()
        },
        "fold_mismatches": fold_mismatch,
        "label_mismatches": label_mismatch,
        "linkage": "grid-cell id (one-to-one)",
    }
    return merged, report


def write_manifest(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
