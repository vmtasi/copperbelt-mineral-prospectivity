"""Random-forest comparator under the same spatial folds and preprocessing."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.v15 import config  # noqa: E402
from src.v15.common import ensure_v15_path, prepare_fold  # noqa: E402
from src.v15.metrics import metric_row, reliability_table  # noqa: E402


def feature_matrix(arrays, split: str) -> np.ndarray:
    data = arrays[split]
    return np.column_stack(
        [
            data["fault_z"],
            data["fault_z2"],
            data["lith_z"],
            data["lith_z2"],
            data["grav_z"],
            data["rocks"],
            data["log_area_z"],
        ]
    )


def main() -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, GroupKFold

    frame = pd.read_csv(config.V15 / "data" / "v15_model_frame.csv")
    rows = []
    predictions = []
    for fold in range(4):
        arrays = prepare_fold(frame, fold, config.MODELS["M2_domain_distance"])
        x_train = feature_matrix(arrays, "train")
        x_test = feature_matrix(arrays, "test")
        y_train = arrays["y_train"]
        cx = arrays["train"]["centroid_x"]
        cy = arrays["train"]["centroid_y"]
        groups = (
            np.floor((cx - cx.min()) / 25000).astype(int) * 100000
            + np.floor((cy - cy.min()) / 25000).astype(int)
        )
        grid = {
            "max_depth": [None, 12],
            "min_samples_leaf": [5, 10],
            "max_features": ["sqrt", 0.5],
        }
        search = GridSearchCV(
            RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1),
            grid,
            scoring="neg_log_loss",
            cv=GroupKFold(n_splits=3),
            n_jobs=1,
            refit=True,
        )
        search.fit(x_train, y_train, groups=groups)
        probability = np.clip(search.predict_proba(x_test)[:, 1], config.PCELL, 1 - config.PCELL)
        prediction = pd.DataFrame(
            {
                "id": arrays["test_ids"],
                "fold": fold + 1,
                "y": arrays["y_test"],
                "prob": probability,
                "prob_sd": np.nan,
                "prob_lower": np.nan,
                "prob_upper": np.nan,
                "p_ref_train": arrays["p_ref_train"],
            }
        )
        prediction.to_csv(
            ensure_v15_path(config.V15 / "predictions" / f"RF_comparator_fold_{fold + 1}.csv"), index=False
        )
        rows.append({**metric_row(prediction, f"fold_{fold + 1}"), "model": "RF_comparator", "best_params": str(search.best_params_)})
        predictions.append(prediction)
        print(f"fold {fold + 1}: {search.best_params_} AUC={rows[-1]['auc']:.3f}", flush=True)
    pooled = pd.concat(predictions, ignore_index=True)
    pooled.to_csv(ensure_v15_path(config.V15 / "predictions" / "oof_RF_comparator.csv"), index=False)
    rows.append({**metric_row(pooled, "pooled"), "model": "RF_comparator", "best_params": ""})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(ensure_v15_path(config.V15 / "metrics" / "comparator_metrics.csv"), index=False)
    reliability_table(pooled).to_csv(ensure_v15_path(config.V15 / "diagnostics" / "reliability_RF_comparator.csv"), index=False)
    print(metrics[["sample", "n", "positives", "auc", "pr_auc", "mean_log_score", "brier"]].to_string(index=False))


if __name__ == "__main__":
    main()

