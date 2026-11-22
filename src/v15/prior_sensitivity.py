"""Compare M2 predictions and posterior scales under changed prior widths."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.v15 import config  # noqa: E402
from src.v15.common import ensure_v15_path, write_json  # noqa: E402
from src.v15.metrics import metric_row  # noqa: E402

TAGS = {"primary": "", "prior_0_5": "prior0p5_", "prior_2_0": "prior2p0_"}
KEY_SCALES = ["alpha_sigma", "sigma_beta_f_lin", "sigma_beta_f_sq", "sigma_beta_l_lin", "sigma_beta_l_sq"]


def predictions(tag: str) -> pd.DataFrame:
    paths = sorted(
        path
        for pattern in (f"{tag}M2_domain_distance_fold_*.csv", f"{tag}_M2_domain_distance_fold_*.csv")
        for path in (config.V15 / "predictions").glob(pattern)
    )
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def main() -> None:
    frames = {label: predictions(tag) for label, tag in TAGS.items()}
    frames = {label: frame for label, frame in frames.items() if not frame.empty}
    if "primary" not in frames:
        raise FileNotFoundError("Primary M2 predictions are required")
    metric_rows = []
    reference = frames["primary"][["id", "fold", "y", "prob"]].rename(columns={"prob": "prob_primary"})
    comparison_rows = []
    for label, frame in frames.items():
        for sample, group in [("pooled", frame)] + list(frame.groupby("fold")):
            metric_rows.append({"prior_sensitivity": label, **metric_row(group, str(sample))})
        merged = frame[["id", "prob"]].merge(reference, on="id", validate="one_to_one")
        comparison_rows.append(
            {
                "prior_sensitivity": label,
                "cells": int(len(merged)),
                "mean_probability": float(merged["prob"].mean()),
                "max_abs_probability_difference_from_primary": float(
                    (merged["prob"] - merged["prob_primary"]).abs().max()
                ),
                "mean_abs_probability_difference_from_primary": float(
                    (merged["prob"] - merged["prob_primary"]).abs().mean()
                ),
            }
        )
    pd.DataFrame(metric_rows).to_csv(
        ensure_v15_path(config.V15 / "metrics" / "prior_sensitivity_metrics.csv"), index=False
    )
    pd.DataFrame(comparison_rows).to_csv(
        ensure_v15_path(config.V15 / "metrics" / "prior_sensitivity_predictions.csv"), index=False
    )

    parameter_rows = []
    for label, tag in TAGS.items():
        for path in sorted(
            path
            for pattern in (
                f"{tag}M2_domain_distance_fold_*_summary.csv",
                f"{tag}_M2_domain_distance_fold_*_summary.csv",
            )
            for path in (config.V15 / "diagnostics").glob(pattern)
        ):
            table = pd.read_csv(path)
            table = table[table["parameter"].isin(KEY_SCALES)]
            for row in table.itertuples():
                parameter_rows.append(
                    {
                        "prior_sensitivity": label,
                        "fold": int(row.fold),
                        "parameter": row.parameter,
                        "mean": float(row.mean),
                        "sd": float(row.sd),
                        "r_hat": float(row.r_hat),
                        "ess_bulk": float(row.ess_bulk),
                    }
                )
    parameter_table = pd.DataFrame(parameter_rows)
    parameter_table.to_csv(
        ensure_v15_path(config.V15 / "diagnostics" / "prior_sensitivity_scales.csv"), index=False
    )
    write_json(
        config.V15 / "diagnostics" / "prior_sensitivity_summary.json",
        {
            "tags": TAGS,
            "key_scales": KEY_SCALES,
            "prediction_comparison": comparison_rows,
            "interpretation": (
                "If predictions and scale posteriors remain close under the two prior-width "
                "variants, the informative domains are predominantly data-supported. Large "
                "movement in sparse domains indicates prior dependence."
            ),
        },
    )
    print(pd.DataFrame(comparison_rows).to_string(index=False))
    print(parameter_table.groupby(["prior_sensitivity", "parameter"])["mean"].mean().to_string())


if __name__ == "__main__":
    main()
