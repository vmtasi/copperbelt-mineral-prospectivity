"""Immutable pre-fitting specification for the transfer analysis."""

from __future__ import annotations

import hashlib
import json
import sys
from importlib.metadata import version
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.transfer import config
from src.transfer.data import ensure_output_path, load_transfer_frame
from src.transfer.partitions import build_partitions


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def specification_payload() -> dict[str, Any]:
    frame = load_transfer_frame()
    _, partition_info = build_partitions(frame)
    details = {
        name: table.to_dict(orient="records")
        for name, table in partition_info["details"].items()
    }
    return {
        "schema_version": "1.0.0",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "frozen_before_any_transfer_model_fit",
        "research_question": (
            "How well does a global prospectivity model trained on along-belt "
            "sectors predict unseen sectors, and what contributes to failure?"
        ),
        "frame": {
            "cells": int(len(frame)),
            "positives": int(frame[config.Y_COL].sum()),
            "outcome": "at least one compiled USGS occurrence in the cell",
            "folds": {
                str(fold + 1): {
                    "cells": int((frame[config.FOLD_COL] == fold).sum()),
                    "positives": int(
                        frame.loc[frame[config.FOLD_COL] == fold, config.Y_COL].sum()
                    ),
                }
                for fold in range(4)
            },
        },
        "fault_versions": {
            "stored": {
                "source": "distance_to_fault in v15_model_frame.csv, converted metres to km",
                "role": "primary T0 and V15 reference rows",
            },
            "rebuilt": {
                "source": "distance_to_fault_km in Phase 2b primary_frame.csv",
                "role": "pre-specified sensitivity",
            },
            "provenance_note": (
                "The 126-line layer is attributed to Selley et al. (2005) Fig. 1b. "
                "Features 120-125 were added later in the Zambian Copperbelt core, "
                "are classified AMBIGUOUS, and cannot be verified in the source. "
                "A leakage test was negative. Neither version is privileged."
            ),
        },
        "models": {
            "T0": {
                "role": "primary",
                "fault": "stored",
                "predictors": ["fault", "fault_sq", "contact", "contact_sq", "gravity"],
                "lithology": "excluded",
                "link": "complementary log-log",
                "offset": "log_area_km2, coefficient fixed to 1",
            },
            "T0_rebuilt": {
                "role": "sensitivity",
                "change_from_T0": "rebuilt fault distance only",
            },
            "T0_linear": {
                "role": "sensitivity",
                "change_from_T0": "remove fault_sq and contact_sq",
            },
            "T0_nrb": {
                "role": "sensitivity",
                "change_from_T0": "add fixed outside-NRB indicator",
            },
        },
        "preprocessing": {
            "standardization": (
                "Fault, contact and gravity are standardized with means and "
                "population standard deviations from each training partition. "
                "Squared terms are formed after standardization."
            ),
            "fault_units": "km",
            "contact_units": "km",
            "gravity_units": "mGal",
        },
        "priors": {
            "intercept": (
                "Normal(log((sum training y + 0.5)/(sum training area km2 + 0.5)), 1)"
            ),
            "coefficients": "independent Normal(0, 1)",
            "hierarchical_terms": "none",
        },
        "estimation": {
            "mcmc": {
                "models": ["T0"],
                "partitions": ["primary"],
                "chains": config.SAMPLER["chains"],
                "draws": config.SAMPLER["draws"],
                "tune": config.SAMPLER["tune"],
                "target_accept": config.SAMPLER["target_accept"],
                "seed": config.SAMPLER["random_seed"],
                "convergence": config.MCMC_TOLERANCE,
            },
            "map": {
                "models": [
                    "T0",
                    "T0_rebuilt",
                    "T0_linear",
                    "T0_nrb",
                ],
                "uses": [
                    "MAP-MCMC validation",
                    "single-sector and sector-pair transfer matrices",
                    "inner spatial cross-validation for local decomposition",
                    "random cross-validation",
                    "P-GAP and P-POS sensitivities",
                ],
                "interval": "Laplace approximation from the numerical Hessian",
                "stop_rule": (
                    "MAP may support other analyses only if pooled held-out "
                    "mean log score differs from T0 MCMC by no more than 0.002"
                ),
            },
        },
        "metrics": {
            "main": [
                "mean log score",
                "ROC-AUC with class-stratified bootstrap interval",
                "average precision and prevalence baseline",
                "Brier skill against training prevalence",
                "calibration-in-the-large",
                "calibration slope",
            ],
            "bootstrap_replicates": config.BOOTSTRAP_REPLICATES,
            "spatial_block_size_km": config.INNER_BLOCK_SIZE_M / 1000.0,
            "bootstrap_seed": config.BOOTSTRAP_SEED,
        },
        "analyses": {
            "A1": (
                "Leave-one-sector-out T0 MCMC transfer; report sector and pooled "
                "metrics with V14 benchmark AUC references."
            ),
            "A2": (
                "MAP predicts each target sector from each single source sector "
                "and each source pair; compare with three-sector leave-one-out. "
                "Single-sector fits with fewer than 15 training positives are "
                "flagged and not interpreted alone."
            ),
            "A3": {
                "transfer": "T0 trained on the other sectors",
                "base": "transfer slopes fixed; intercept re-estimated on target labels",
                "recal": (
                    "intercept and one multiplier on the non-intercept linear "
                    "predictor re-estimated on target labels; multiplier prior "
                    "Normal(1,1)"
                ),
                "local": (
                    "T0 fitted within the target sector using four inner folds "
                    "formed from intact 25-km blocks and cross-fitted"
                ),
                "oracle_note": (
                    "base, recalibration and local use held-out labels and are "
                    "diagnostic oracles, not deployable models"
                ),
            },
            "A4": {
                "univariate": (
                    "share outside the training 1st-99th percentile for each of "
                    "the five T0 predictor variables"
                ),
                "aoa": (
                    "Meyer-Pebesma dissimilarity index with equal predictor "
                    "weights after training-derived z standardization; training "
                    "DI normalized by mean nearest-neighbour distance; threshold "
                    "is maximum 5-fold cross-validated training DI after removing "
                    "Tukey upper-fence outliers"
                ),
            },
            "A5": config.PREDICTOR_VERDICT_RULE,
            "A6": {
                "random_cv": "20 stratified random four-fold repeats using MAP",
                "comparison": "random CV versus leave-one-sector-out",
                "buffered_reference": (
                    "existing V15 M0 5-km and 10-km buffered results, labelled "
                    "as a different model containing lithology"
                ),
            },
            "A7": (
                "Assemble existing V15, Phase 2b and Phase 2c summaries; no new "
                "domain-model fits."
            ),
            "A8": {
                "rules": config.PARTITION_RULE,
                "computed": partition_info["metadata"],
                "details": details,
                "primary_partition": "canonical V15 equal-cell folds",
                "robustness_rule": (
                    "transfer conclusions are robust only if A5 verdicts and "
                    "the direction of main A1 findings agree under all three "
                    "partitions"
                ),
            },
        },
        "primary_results": {
            "model": "T0",
            "fault": "stored",
            "partition": "canonical V15 folds",
            "analysis": "A1-A7; sensitivities do not change conclusions",
        },
        "software": {
            "requested_python": r".venv\Scripts\python.exe",
            "executed_python": r".venv-pymc-full\Scripts\python.exe",
            "reason": (
                "The repository .venv points to a missing Python 3.10 base "
                "interpreter; the isolated 3.12 environment supplies PyMC, "
                "SciPy, pandas and geopandas."
            ),
        },
        "frozen_inputs": {
            "v14_manifest_sha256": _sha256(config.ROOT / "paper_1" / "benchmark" / "SHA256SUMS"),
            "v14_external_manifest_sha256": _sha256(
                config.ROOT / "paper_1" / "benchmark" / "SHA256SUMS_external"
            ),
            "v15_manifest_sha256": _sha256(
                config.V15 / "checksums" / "SHA256SUMS_v15"
            ),
            "phase2b_manifest_sha256": _sha256(
                config.PHASE2B / "SHA256SUMS"
            ),
        },
    }


def write_specification() -> Path:
    path = ensure_output_path(config.OUTPUT / "specification.json")
    sidecar = ensure_output_path(config.OUTPUT / "specification.sha256")
    if path.exists() and sidecar.exists():
        expected = sidecar.read_text(encoding="ascii").split()[0]
        actual = _sha256(path)
        if actual == expected:
            return path
        raise RuntimeError(
            "Existing transfer specification does not match its recorded hash"
        )
    if path.exists():
        raise FileExistsError(
            "Transfer specification already exists; do not overwrite it after fitting."
        )
    payload = specification_payload()
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            default=lambda value: value.item()
            if hasattr(value, "item")
            else str(value),
        )
        + "\n",
        encoding="utf-8",
    )
    digest = _sha256(path)
    sidecar.write_text(f"{digest}  specification.json\n", encoding="ascii")
    return path


def write_specification_v2() -> Path:
    """Version the plan after the failed MAP-MCMC gate, leaving v1 untouched."""

    original_path = config.OUTPUT / "specification.json"
    if not original_path.exists():
        raise FileNotFoundError("The original specification is missing")
    original_hash = _sha256(original_path)
    original = json.loads(original_path.read_text(encoding="utf-8"))
    validation = json.loads(
        (config.OUTPUT / "diagnostics" / "map_mcmc_validation.json").read_text(
            encoding="utf-8"
        )
    )
    environment = json.loads(
        (config.OUTPUT / "diagnostics" / "environment_reproduction.json").read_text(
            encoding="utf-8"
        )
    )
    packages = {
        name: version(name)
        for name in (
            "numpy",
            "scipy",
            "pandas",
            "pymc",
            "arviz",
            "pytensor",
            "geopandas",
        )
    }
    payload = {
        "schema_version": "2.0.0",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "frozen_before_amendment_1_fits",
        "supersedes": {
            "path": "specification.json",
            "sha256": original_hash,
            "immutable": True,
        },
        "reason_for_version": (
            "The pre-registered MAP-MCMC agreement gate failed. MAP-minus-MCMC "
            f"mean held-out log score was "
            f"{validation['mean_log_score_difference_map_minus_mcmc']:.6f} per "
            f"cell, exceeding the {validation['tolerance']:.3f} tolerance. "
            "Amendment 1 retires MAP and replaces every remaining fit with full "
            "MCMC."
        ),
        "gate_result": validation,
        "unchanged_analysis_plan": original,
        "amendment_changes": {
            "estimation": {
                "primary_leave_one_sector_out": (
                    "retain the existing four-chain T0 fits without refitting"
                ),
                "all_other_fits": (
                    "full MCMC with 2 chains, 1,000 retained draws, 1,500 "
                    "tuning steps, target acceptance 0.97, recorded seeds, "
                    "and the same convergence criteria"
                ),
                "failed_fit_rule": (
                    "refit with doubled tuning, up to three attempts; never drop a fit"
                ),
                "retired": "MAP point estimates and Laplace predictive approximation",
            },
            "A6": {
                "repeats": 10,
                "reason": "compute-budget-only reduction from 20 to 10 repeats",
            },
            "A1_reporting": (
                "add log-score skill against training-partition prevalence "
                "and mean predicted probability minus observed rate; rank "
                "sectors by skill rather than raw log score"
            ),
            "A5_interpretation": (
                "Sector 1 fault AUC 0.24 is treated primarily as a likely "
                "map-coverage effect because the regional fault digitization "
                "is dense in DRC and sparse in the Zambian Copperbelt"
            ),
        },
        "fit_plan": {
            "existing_primary_fits_reused": 4,
            "A2_single_source_fits": 4,
            "A2_source_pair_fits": 6,
            "A3_inner_local_fits": 16,
            "A3_oracle_correction_fits": 8,
            "A6_random_fits": 40,
            "A8_loo_and_single_source_fits": 16,
            "sensitivity_fits": 12,
            "total_new_fits": 102,
            "estimated_runtime_hours": {
                "lower": 1.2,
                "upper": 3.8,
                "basis": (
                    "Existing four-chain full-frame fits required 68-228 seconds; "
                    "two-chain Amendment 1 fits were bounded using slower and "
                    "previously uncompiled model startups."
                ),
            },
        },
        "environment": {
            "interpreter": environment["interpreter"],
            "python": sys.version,
            "packages": packages,
            "existing_mcmc_probability_reproduction": environment,
        },
    }
    path = ensure_output_path(config.OUTPUT / "specification_v2.json")
    if path.exists():
        sidecar = ensure_output_path(config.OUTPUT / "specification_v2.sha256")
        if not sidecar.exists():
            raise FileExistsError("specification_v2 exists without its hash")
        expected = sidecar.read_text(encoding="ascii").split()[0]
        if _sha256(path) != expected:
            raise RuntimeError("specification_v2 does not match its frozen hash")
        return path
    path.write_text(
        json.dumps(payload, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value))
        + "\n",
        encoding="utf-8",
    )
    digest = _sha256(path)
    ensure_output_path(config.OUTPUT / "specification_v2.sha256").write_text(
        f"{digest}  specification_v2.json\n", encoding="ascii"
    )
    return path


if __name__ == "__main__":
    print(write_specification_v2())

