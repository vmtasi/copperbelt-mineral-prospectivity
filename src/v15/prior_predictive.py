"""Prior predictive implications for the M2 exposure model."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pymc as pm  # noqa: E402

from src.v15 import config  # noqa: E402
from src.v15.common import cloglog_probability, ensure_v15_path, prepare_fold  # noqa: E402
from src.v15.models import build_model  # noqa: E402


def flat(idata, name: str) -> np.ndarray:
    values = np.asarray(idata.prior[name].values)
    return values.reshape(-1, *values.shape[2:])


def main() -> None:
    frame = pd.read_csv(config.V15 / "data" / "v15_model_frame.csv")
    spec = config.MODELS["M2_domain_distance"]
    arrays = prepare_fold(frame, 0, spec)
    model = build_model("M2_domain_distance", arrays, spec)
    with model:
        prior = pm.sample_prior_predictive(draws=2000, random_seed=42)
    data = arrays["test"]
    alpha = flat(prior, "alpha_dom")[:, data["domain_idx"]]
    eta = alpha
    for name, column in (
        ("beta_f_lin", "fault_z"),
        ("beta_f_sq", "fault_z2"),
        ("beta_l_lin", "lith_z"),
        ("beta_l_sq", "lith_z2"),
    ):
        eta = eta + flat(prior, name)[:, data["domain_idx"]] * data[column][None, :]
    eta = eta + flat(prior, "beta_grav")[:, None] * data["grav_z"][None, :]
    if arrays["n_rocks"]:
        eta = eta + flat(prior, "beta_rocks") @ data["rocks"].T
    eta = eta + data["log_area"][None, :]
    probability = cloglog_probability(eta)
    intensity = probability / np.exp(data["log_area"])[None, :]
    rows = []
    for domain_index, domain in enumerate(config.DOMAINS):
        domain_probability = probability[:, data["domain_idx"] == domain_index]
        if domain_probability.size == 0:
            continue
        rows.append(
            {
                "domain": domain,
                "test_cells": int(domain_probability.shape[1]),
                "prior_mean_probability": float(domain_probability.mean()),
                "prior_p2_5_probability": float(np.percentile(domain_probability, 2.5)),
                "prior_median_probability": float(np.median(domain_probability)),
                "prior_p97_5_probability": float(np.percentile(domain_probability, 97.5)),
                "prior_p2_5_intensity_per_km2": float(np.percentile(intensity[:, data["domain_idx"] == domain_index], 2.5)),
                "prior_p97_5_intensity_per_km2": float(np.percentile(intensity[:, data["domain_idx"] == domain_index], 97.5)),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(ensure_v15_path(config.V15 / "diagnostics" / "prior_predictive_by_domain.csv"), index=False)
    summary = pd.DataFrame(
        [
            {
                "draws": int(probability.shape[0]),
                "cells": int(probability.shape[1]),
                "prior_mean_probability": float(probability.mean()),
                "prior_p2_5_probability": float(np.percentile(probability, 2.5)),
                "prior_median_probability": float(np.median(probability)),
                "prior_p97_5_probability": float(np.percentile(probability, 97.5)),
                "prior_p99_5_probability": float(np.percentile(probability, 99.5)),
                "fraction_probability_above_0_5": float((probability > 0.5).mean()),
                "fraction_probability_above_0_9": float((probability > 0.9).mean()),
            }
        ]
    )
    summary.to_csv(ensure_v15_path(config.V15 / "diagnostics" / "prior_predictive_summary.csv"), index=False)
    print(table.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
