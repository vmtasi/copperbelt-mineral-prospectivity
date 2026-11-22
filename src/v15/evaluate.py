"""Evaluate V15 predictions and produce the primary comparison tables."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.v15 import config  # noqa: E402
from src.v15.common import ensure_v15_path, load_frozen_oof  # noqa: E402
from src.v15.metrics import log_score, metric_row, paired_table, reliability_table, roc_auc  # noqa: E402


def load_model_predictions(model_name: str) -> pd.DataFrame:
    paths = sorted((config.V15 / "predictions").glob(f"{model_name}_fold_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No predictions for {model_name}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    model_frame = pd.read_csv(config.V15 / "data" / "v15_model_frame.csv", usecols=["id", "centroid_x", "centroid_y", "daly_domain"])
    frame = frame.merge(model_frame, on="id", how="left", validate="one_to_one")
    return frame.sort_values(["fold", "id"]).reset_index(drop=True)


def frozen_v14() -> pd.DataFrame:
    frame = load_frozen_oof().rename(columns={"prob_v14": "prob", "deposit_present": "y"})
    frame["fold"] = frame["fold"].astype(int) + 1
    model_frame = pd.read_csv(
        config.V15 / "data" / "v15_model_frame.csv",
        usecols=["id", "fold", "deposit_present", "centroid_x", "centroid_y"],
    )
    reference = {
        fold + 1: float(group.loc[group["fold"] != fold, "deposit_present"].mean())
        for fold, group in model_frame.groupby("fold")
    }
    frame["p_ref_train"] = frame["fold"].map(reference).clip(1e-6, 1 - 1e-6)
    return frame.merge(
        model_frame[["id", "centroid_x", "centroid_y"]], on="id", how="left", validate="one_to_one"
    )


def main() -> None:
    names = [
        name
        for name in config.PRIMARY_MODELS + config.SENSITIVITY_MODELS + ["RF_comparator"]
        if len(list((config.V15 / "predictions").glob(f"{name}_fold_*.csv"))) == 4
    ]
    predictions = {name: load_model_predictions(name) for name in names}
    predictions["V14_frozen"] = frozen_v14()

    metric_rows = []
    for name, frame in predictions.items():
        metric_rows.append({**metric_row(frame, "pooled"), "model": name})
        for fold, group in frame.groupby("fold"):
            metric_rows.append({**metric_row(group, f"fold_{int(fold)}"), "model": name})
        reliability_table(frame).to_csv(
            ensure_v15_path(config.V15 / "diagnostics" / f"reliability_{name}.csv"), index=False
        )
        frame.to_csv(ensure_v15_path(config.V15 / "predictions" / f"oof_{name}.csv"), index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(ensure_v15_path(config.V15 / "metrics" / "model_metrics.csv"), index=False)

    summary = metrics[metrics["sample"] == "pooled"].copy()
    summary.to_csv(ensure_v15_path(config.V15 / "metrics" / "model_summary.csv"), index=False)

    paired, bootstrap = paired_table(predictions, bootstrap=True)
    paired.to_csv(ensure_v15_path(config.V15 / "metrics" / "paired_comparisons.csv"), index=False)
    bootstrap.to_csv(ensure_v15_path(config.V15 / "metrics" / "paired_bootstrap.csv"), index=False)

    fold_pairs = []
    for fold in sorted(next(iter(predictions.values()))["fold"].unique()):
        fold_frames = {name: frame[frame["fold"] == fold] for name, frame in predictions.items()}
        table, _ = paired_table(fold_frames, bootstrap=False)
        table.insert(0, "fold", int(fold))
        fold_pairs.append(table)
    pd.concat(fold_pairs, ignore_index=True).to_csv(
        ensure_v15_path(config.V15 / "metrics" / "paired_by_fold.csv"), index=False
    )

    wanted = [
        ("M0_global", "M1_domain_intercept"),
        ("M1_domain_intercept", "M2_domain_distance"),
        ("M2_domain_distance", "M3_domain_distance_gravity"),
    ]
    rows = []
    for lower, upper in wanted:
        if lower not in predictions or upper not in predictions:
            continue
        merged = predictions[lower][["id", "y", "prob"]].merge(
            predictions[upper][["id", "prob"]],
            on="id",
            suffixes=("_lower", "_upper"),
            validate="one_to_one",
        )
        lower_scores = log_score(merged["y"], merged["prob_lower"])
        upper_scores = log_score(merged["y"], merged["prob_upper"])
        rows.append(
            {
                "lower": lower,
                "upper": upper,
                "n": int(len(merged)),
                "auc_lower": roc_auc(merged["y"], merged["prob_lower"]),
                "auc_upper": roc_auc(merged["y"], merged["prob_upper"]),
                "delta_auc": float(
                    roc_auc(merged["y"], merged["prob_upper"])
                    - roc_auc(merged["y"], merged["prob_lower"])
                ),
                "mean_log_score_lower": float(lower_scores.mean()),
                "mean_log_score_upper": float(upper_scores.mean()),
                "mean_delta_log_score": float(upper_scores.mean() - lower_scores.mean()),
                "total_delta_log_score": float(upper_scores.sum() - lower_scores.sum()),
            }
        )
    central = pd.DataFrame(rows)
    central.to_csv(ensure_v15_path(config.V15 / "metrics" / "central_comparisons.csv"), index=False)

    print(summary[["model", "n", "positives", "auc", "pr_auc", "mean_log_score", "brier", "calibration_slope"]].to_string(index=False))
    print("\nCentral comparisons\n" + central.to_string(index=False))


if __name__ == "__main__":
    main()


