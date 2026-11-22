"""Gravity heterogeneity diagnostics for the M3 posterior."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.trace_io import load_trace, resolve_trace_path  # noqa: E402
from src.v15 import config  # noqa: E402
from src.v15.common import ensure_v15_path  # noqa: E402
from src.v15.metrics import roc_auc  # noqa: E402


def main() -> None:
    frame = pd.read_csv(config.V15 / "data" / "v15_model_frame.csv")
    rows = []
    for fold in range(4):
        path = resolve_trace_path(config.V15 / "traces" / "M3_domain_distance_gravity" / f"fold_{fold + 1}.nc")
        trace = load_trace(path)
        draws = np.asarray(trace.posterior["beta_grav"].values).reshape(-1, len(config.DOMAINS))
        for index, domain in enumerate(config.DOMAINS):
            values = draws[:, index]
            rows.append(
                {
                    "fold": fold + 1,
                    "domain": domain,
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "p_positive": float((values > 0).mean()),
                    "p_negative": float((values < 0).mean()),
                    "lower_95": float(np.percentile(values, 2.5)),
                    "upper_95": float(np.percentile(values, 97.5)),
                }
            )
    coefficient_table = pd.DataFrame(rows)
    coefficient_table.to_csv(
        ensure_v15_path(config.V15 / "diagnostics" / "gravity_coefficients.csv"), index=False
    )

    fold_rows = []
    for fold, group in frame.groupby("fold"):
        fold_rows.append(
            {
                "fold": int(fold) + 1,
                "n": int(len(group)),
                "positives": int(group[config.Y_COL].sum()),
                "positive_rate": float(group[config.Y_COL].mean()),
                "gravity_mean": float(group["bouguer"].mean()),
                "gravity_sd": float(group["bouguer"].std(ddof=1)),
                "gravity_min": float(group["bouguer"].min()),
                "gravity_max": float(group["bouguer"].max()),
                # AUC of raw Bouguer (high gravity scored as positive); values
                # below 0.5 mean occurrences favour LOW gravity.
                "gravity_auc": roc_auc(group[config.Y_COL], group["bouguer"]),
                "gravity_auc_low_gravity_positive": 1.0 - roc_auc(group[config.Y_COL], group["bouguer"]),
                "corr_fault": float(group["bouguer"].corr(group["distance_to_fault"], method="spearman")),
                "corr_lith": float(group["bouguer"].corr(group["distance_to_lithology_contact"], method="spearman")),
            }
        )
    fold_table = pd.DataFrame(fold_rows)
    fold_table.to_csv(ensure_v15_path(config.V15 / "diagnostics" / "gravity_fold_summary.csv"), index=False)
    print(coefficient_table.to_string(index=False))
    print("\n" + fold_table.to_string(index=False))


if __name__ == "__main__":
    main()
