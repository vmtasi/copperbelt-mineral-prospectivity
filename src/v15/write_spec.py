"""Write and hash the frozen V15 experiment specification."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.v15 import config  # noqa: E402
from src.v15.common import ensure_v15_path, sha256  # noqa: E402

EQUATIONS = {
    "M0_global": (
        "cloglog(p_i) = alpha + b_f1 z_f + b_f2 z_f^2 + b_l1 z_l + b_l2 z_l^2 "
        "+ b_g g + x_rock'b_rock + log(A_i)"
    ),
    "M1_domain_intercept": (
        "cloglog(p_i) = alpha_d[i] + b_f1 z_f + b_f2 z_f^2 + b_l1 z_l + b_l2 z_l^2 "
        "+ b_g g + x_rock'b_rock + log(A_i)"
    ),
    "M2_domain_distance": (
        "cloglog(p_i) = alpha_d[i] + b_f1[d(i)] z_f + b_f2[d(i)] z_f^2 "
        "+ b_l1[d(i)] z_l + b_l2[d(i)] z_l^2 + b_g g + x_rock'b_rock + log(A_i)"
    ),
    "M3_domain_distance_gravity": (
        "cloglog(p_i) = alpha_d[i] + b_f1[d(i)] z_f + b_f2[d(i)] z_f^2 "
        "+ b_l1[d(i)] z_l + b_l2[d(i)] z_l^2 + b_g[d(i)] g + x_rock'b_rock + log(A_i)"
    ),
    "V14_control": (
        "logit(p_i) = alpha_d[i] + b_f1[d(i)] z_f + b_f2[d(i)] z_f^2 "
        "+ b_l1[d(i)] z_l + b_l2[d(i)] z_l^2 + b_g g + x_rock'b_rock"
    ),
    "M2_logit_no_area": (
        "logit(p_i) = alpha_d[i] + b_f1[d(i)] z_f + b_f2[d(i)] z_f^2 "
        "+ b_l1[d(i)] z_l + b_l2[d(i)] z_l^2 + b_g g + x_rock'b_rock"
    ),
    "M2_logit_area_covariate": (
        "logit(p_i) = alpha_d[i] + b_f1[d(i)] z_f + b_f2[d(i)] z_f^2 "
        "+ b_l1[d(i)] z_l + b_l2[d(i)] z_l^2 + b_g g + x_rock'b_rock + b_A z_A"
    ),
    "M2_cloglog_no_litho": (
        "cloglog(p_i) = alpha_d[i] + b_f1[d(i)] z_f + b_f2[d(i)] z_f^2 "
        "+ b_g g + log(A_i)"
    ),
}


def main() -> None:
    frame = pd.read_csv(config.V15 / "data" / "v15_model_frame.csv")
    audit = json.loads((config.V15 / "diagnostics" / "data_build_audit.json").read_text())
    spec = {
        "schema_version": "1.0.0",
        "experiment": "Paper 1 V15",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "frozen": True,
        "estimand": (
            "Incremental held-out predictive value of domain-varying distance relationships after "
            "accounting for domain intercepts and clipped-cell exposure."
        ),
        "domains": config.DOMAINS,
        "outcome": {
            "column": config.Y_COL,
            "meaning": "at least one compiled USGS occurrence in the cell; binary, not a count",
            "cells": int(len(frame)),
            "positive_cells": int(frame[config.Y_COL].sum()),
            "negative_cells": int(len(frame) - frame[config.Y_COL].sum()),
            "fold_cells": {str(k): int(v) for k, v in frame.groupby("fold").size().items()},
            "fold_positives": {str(k): int(v) for k, v in frame.groupby("fold")[config.Y_COL].sum().items()},
        },
        "area": {
            "definition": audit["definition"],
            "crs": audit["crs"],
            "unit": audit["unit"],
            "grid_cells": audit["cells"],
            "clipped_area_sum_km2": audit["clipped_area_sum_km2"],
            "complete_case_cells": audit["complete_case_rows"],
            "cells_below_1_km2": int((frame["area_km2"] < 1.0).sum()),
            "positive_cells_below_1_km2": int(frame.loc[frame["area_km2"] < 1.0, config.Y_COL].sum()),
            "offset": "log(area_km2); fixed coefficient 1 in the cloglog models",
        },
        "predictors": {
            "fault": "distance_to_fault, training-fold standardized, then squared",
            "contact": "distance_to_lithology_contact, training-fold standardized, then squared",
            "gravity": "bouguer, training-fold standardized",
            "lithology": "litho_contact_litho_class indicators retained when the training partition has both labels",
            "area": "log_area_km2 exposure offset; standardized covariate only in the area-covariate sensitivity",
        },
        "models": {name: {"spec": config.MODELS[name], "equation": EQUATIONS.get(name, "sensitivity model")} for name in config.MODELS},
        "priors": config.PRIORS,
        "sampler": {
            "config_default": config.SAMPLER,
            "executed": {
                "M0_global": {"chains": 2, "draws": 1000, "tune": 1500, "target_accept": 0.95},
                "M1_domain_intercept": {"chains": 2, "draws": 1000, "tune": 1500, "target_accept": 0.95},
                "M2_domain_distance": {"chains": 4, "draws": 1500, "tune": 2500, "target_accept": 0.995},
                "M3_domain_distance_gravity": {"chains": 4, "draws": 1500, "tune": 2500, "target_accept": 0.995},
                "sensitivity": {"chains": 2, "draws": 1000, "tune": 2000, "target_accept": 0.995},
            },
        },
        "comparisons": [
            ["M0_global", "M1_domain_intercept"],
            ["M1_domain_intercept", "M2_domain_distance"],
            ["M2_domain_distance", "M3_domain_distance_gravity"],
        ],
        "sensitivity": config.SENSITIVITY_MODELS
        + ["buffered_5km", "buffered_10km", "prior_scale_0.5", "prior_scale_2.0"],
        "inputs": {
            "model_frame": {
                "path": "paper_1/v15/data/v15_model_frame.csv",
                "sha256": sha256(config.V15 / "data" / "v15_model_frame.csv"),
            },
            "v14_table": {"path": str(config.V14_TABLE.relative_to(ROOT)), "sha256": sha256(config.V14_TABLE)},
            "v14_benchmark_checksums": {
                "path": str(config.BENCHMARK_SUMS.relative_to(ROOT)),
                "sha256": sha256(config.BENCHMARK_SUMS),
            },
            "config_source": {"path": "src/v15/config.py", "sha256": sha256(ROOT / "src" / "v15" / "config.py")},
        },
        "deviations_from_v14": [
            "V15 primary models use a complementary log-log likelihood with log clipped-cell area as a fixed offset.",
            "M2 and M3 were refitted after an initial run produced 2-11 divergences per fold; the initial traces are archived under paper_1/v15/models/nonconverged_initial and are not used for inference.",
            "The retained M2 and M3 runs use four chains, 1,500 draws, 2,500 tuning steps and target_accept 0.995; M0 and M1 use the earlier convergent two-chain configuration.",
            "The V14_control is a new V15 re-specification with the V14 likelihood, predictors and priors, not a byte reproduction of the frozen V14 chains. V14 itself used 1,500 draws, 2,500 tuning steps, two chains and target_accept 0.99.",
            "The frozen V14 OOF predictions remain the exact benchmark; no V14 file is modified.",
        ],
    }
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    path = ensure_v15_path(config.V15 / "specification" / "v15_experiment_specification.json")
    path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    ensure_v15_path(path.parent / "v15_experiment_specification.sha256").write_text(
        digest + "\n", encoding="utf-8"
    )
    print(digest)


if __name__ == "__main__":
    main()
