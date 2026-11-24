"""Full-MCMC implementation of Amendment 1."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import version
from typing import Any

import numpy as np
import pandas as pd

from src.trace_io import save_trace
from src.transfer import config
from src.transfer.data import build_design, ensure_output_path, load_transfer_frame
from src.transfer.modeling import (
    cloglog_probability,
    posterior_probability,
    sample_pymc,
)
from src.transfer.partitions import assign_block_folds, sector_pairs
from src.transfer.training import _prediction_frame

AMENDMENT_SAMPLER = {
    "chains": 2,
    "draws": 1000,
    "tune": 1500,
    "cores": 2,
    "target_accept": 0.97,
    "random_seed": 142,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _environment_record() -> dict[str, Any]:
    return {
        "interpreter": sys.executable,
        "python": sys.version,
        "packages": {
            name: version(name)
            for name in (
                "numpy",
                "scipy",
                "pandas",
                "pymc",
                "arviz",
                "pytensor",
            )
        },
    }


def _result_paths(fit_id: str) -> dict[str, Path]:
    return {
        "trace": config.OUTPUT / "traces" / "amendment1" / f"{fit_id}.nc",
        "prediction": config.OUTPUT / "predictions" / "amendment1" / f"{fit_id}.csv",
        "coefficient": config.OUTPUT / "coefficients" / "amendment1" / f"{fit_id}.csv",
        "status": config.OUTPUT / "diagnostics" / "mcmc_fits" / f"{fit_id}.json",
    }


def _completed(fit_id: str) -> bool:
    paths = _result_paths(fit_id)
    if not all(path.exists() for path in paths.values()):
        return False
    try:
        status = json.loads(paths["status"].read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(status.get("converged", False))


def _fit_standard_mcmc(
    data: dict[str, Any],
    fit_id: str,
    batch: str,
    seed: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    paths = _result_paths(fit_id)
    if _completed(fit_id):
        return {"fit_id": fit_id, "status": "SKIPPED_COMPLETED"}
    tune = AMENDMENT_SAMPLER["tune"]
    attempts = []
    trace = None
    summary = None
    info = None
    started = time.perf_counter()
    for attempt in range(1, 4):
        trace, summary, info = sample_pymc(
            data,
            draws=AMENDMENT_SAMPLER["draws"],
            tune=tune,
            chains=AMENDMENT_SAMPLER["chains"],
            cores=AMENDMENT_SAMPLER["cores"],
            target_accept=AMENDMENT_SAMPLER["target_accept"],
            seed=seed + attempt - 1,
        )
        attempts.append({**info, "attempt": attempt, "tune": tune})
        if info["converged"]:
            break
        tune *= 2
    elapsed = time.perf_counter() - started
    if trace is None or summary is None or info is None:
        raise RuntimeError(f"No trace returned for {fit_id}")
    save_trace(trace, ensure_output_path(paths["trace"]))
    probability = posterior_probability(trace, data)
    prediction = _prediction_frame(data, probability, "amendment1", fit_id).assign(
        model=data["model"],
        batch=batch,
        fit_id=fit_id,
        seed=seed,
    )
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            prediction[key] = value
    prediction.to_csv(ensure_output_path(paths["prediction"]), index=False)

    coefficient = summary.loc[
        summary["parameter"].str.startswith(("alpha", "beta_")), :
    ].copy()
    coefficient["fit_id"] = fit_id
    coefficient["batch"] = batch
    coefficient["model"] = data["model"]
    coefficient["seed"] = seed
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            coefficient[key] = value
    coefficient.to_csv(ensure_output_path(paths["coefficient"]), index=False)
    status = {
        "fit_id": fit_id,
        "batch": batch,
        "model": data["model"],
        "created_utc": _utc_now(),
        "seed": seed,
        "sampler": AMENDMENT_SAMPLER,
        "attempts": attempts,
        "converged": bool(info["converged"]),
        **info,
        "final": info,
        "elapsed_seconds": elapsed,
        "n_train": len(data["train"]),
        "positive_train": int(data["y_train"].sum()),
        "n_test": len(data["test"]),
        "positive_test": int(data["y_test"].sum()),
        "environment": _environment_record(),
        **metadata,
    }
    ensure_output_path(paths["status"]).write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    if not info["converged"]:
        raise RuntimeError(f"Fit failed after three tuning increases: {fit_id}")
    return status


def run_a2(frame: pd.DataFrame) -> list[dict[str, Any]]:
    statuses = []
    folds = frame[config.FOLD_COL].to_numpy(dtype=int)
    for source in range(4):
        fit_id = f"a2_single_source_{source + 1}"
        data = build_design(frame, folds == source, folds != source, config.PRIMARY_MODEL)
        status = _fit_standard_mcmc(
            data,
            fit_id,
            "a2",
            AMENDMENT_SAMPLER["random_seed"] + 100 + source,
            {"source_sector": source + 1},
        )
        prediction = pd.read_csv(_result_paths(fit_id)["prediction"])
        prediction["source_sector"] = source + 1
        prediction.to_csv(_result_paths(fit_id)["prediction"], index=False)
        statuses.append(status)
    for pair_index, pair in enumerate(sector_pairs()):
        fit_id = f"a2_pair_{pair[0] + 1}_{pair[1] + 1}"
        source_mask = np.isin(folds, pair)
        data = build_design(frame, source_mask, ~source_mask, config.PRIMARY_MODEL)
        status = _fit_standard_mcmc(
            data,
            fit_id,
            "a2",
            AMENDMENT_SAMPLER["random_seed"] + 200 + pair_index,
            {"source_sector_a": pair[0] + 1, "source_sector_b": pair[1] + 1},
        )
        prediction = pd.read_csv(_result_paths(fit_id)["prediction"])
        prediction["source_sector_a"] = pair[0] + 1
        prediction["source_sector_b"] = pair[1] + 1
        prediction.to_csv(_result_paths(fit_id)["prediction"], index=False)
        statuses.append(status)
    return statuses


def run_sensitivities(frame: pd.DataFrame) -> list[dict[str, Any]]:
    statuses = []
    folds = frame[config.FOLD_COL].to_numpy(dtype=int)
    for model_offset, model_name in enumerate(config.SENSITIVITY_MODELS, start=1):
        for fold in range(4):
            fit_id = f"sensitivity_{model_name}_fold_{fold + 1}"
            data = build_design(frame, folds != fold, folds == fold, model_name)
            status = _fit_standard_mcmc(
                data,
                fit_id,
                "sensitivities",
                AMENDMENT_SAMPLER["random_seed"] + 300 + model_offset * 10 + fold,
                {"fold": fold + 1},
            )
            statuses.append(status)
    return statuses


def run_random_cv(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Amendment 1 reduces A6 to ten repeats, but all fits use MCMC."""

    statuses = []
    y = frame[config.Y_COL].to_numpy(dtype=int)
    for repeat in range(10):
        rng = np.random.default_rng(config.BOOTSTRAP_SEED + repeat)
        fold = np.empty(len(frame), dtype=int)
        for label in (0, 1):
            indices = np.flatnonzero(y == label)
            indices = rng.permutation(indices)
            fold[indices] = np.arange(len(indices)) % config.RANDOM_CV_FOLDS
        for inner_fold in range(config.RANDOM_CV_FOLDS):
            fit_id = f"a6_random_repeat_{repeat + 1}_fold_{inner_fold + 1}"
            data = build_design(
                frame,
                fold != inner_fold,
                fold == inner_fold,
                config.PRIMARY_MODEL,
            )
            status = _fit_standard_mcmc(
                data,
                fit_id,
                "a6",
                AMENDMENT_SAMPLER["random_seed"] + 500 + repeat * 4 + inner_fold,
                {"repeat": repeat + 1, "fold": inner_fold + 1},
            )
            prediction = pd.read_csv(_result_paths(fit_id)["prediction"])
            prediction["repeat"] = repeat + 1
            prediction.to_csv(_result_paths(fit_id)["prediction"], index=False)
            statuses.append(status)
    return statuses


def run_partitions(frame: pd.DataFrame, partitions: pd.DataFrame) -> list[dict[str, Any]]:
    statuses = []
    partition_by_id = partitions.set_index("id")
    for partition_name, column in (("P_GAP", "fold_p_gap"), ("P_POS", "fold_p_pos")):
        labels = frame["id"].map(partition_by_id[column]).to_numpy(dtype=int)
        partition_frame = frame.copy()
        partition_frame["target_sector"] = labels
        for fold in range(4):
            fit_id = f"a8_{partition_name}_loo_fold_{fold + 1}"
            data = build_design(
                partition_frame,
                labels != fold,
                labels == fold,
                config.PRIMARY_MODEL,
            )
            status = _fit_standard_mcmc(
                data,
                fit_id,
                "a8",
                AMENDMENT_SAMPLER["random_seed"] + 700 + fold,
                {
                    "partition": partition_name,
                    "fold": fold + 1,
                    "target_sector": fold + 1,
                },
            )
            statuses.append(status)
        for source in range(4):
            fit_id = f"a8_{partition_name}_single_source_{source + 1}"
            data = build_design(
                partition_frame,
                labels == source,
                labels != source,
                config.PRIMARY_MODEL,
            )
            status = _fit_standard_mcmc(
                data,
                fit_id,
                "a8",
                AMENDMENT_SAMPLER["random_seed"] + 750 + source,
                {"partition": partition_name, "source_sector": source + 1},
            )
            prediction = pd.read_csv(_result_paths(fit_id)["prediction"])
            prediction["source_sector"] = source + 1
            prediction["target_sector"] = prediction["target_sector"] + 1
            prediction.to_csv(_result_paths(fit_id)["prediction"], index=False)
            statuses.append(status)
    return statuses


def run_inner_local(frame: pd.DataFrame) -> list[dict[str, Any]]:
    statuses = []
    for sector in range(4):
        subset = frame.loc[frame[config.FOLD_COL] == sector].reset_index(drop=True)
        inner = assign_block_folds(
            subset,
            n_folds=4,
            seed=config.BOOTSTRAP_SEED + 900 + sector,
        )
        for fold in range(4):
            fit_id = f"a3_local_sector_{sector + 1}_inner_{fold + 1}"
            data = build_design(
                subset,
                inner != fold,
                inner == fold,
                config.PRIMARY_MODEL,
            )
            status = _fit_standard_mcmc(
                data,
                fit_id,
                "a3_local",
                AMENDMENT_SAMPLER["random_seed"] + 1000 + sector * 4 + fold,
                {"target_sector": sector + 1, "inner_fold": fold + 1},
            )
            prediction = pd.read_csv(_result_paths(fit_id)["prediction"])
            prediction["target_sector"] = sector + 1
            prediction["inner_fold"] = fold + 1
            prediction.to_csv(_result_paths(fit_id)["prediction"], index=False)
            statuses.append(status)
    return statuses


def run_oracle_corrections(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Run full MCMC for the one- and two-parameter A3 oracle corrections."""

    import arviz as az
    import pymc as pm
    import pytensor.tensor as pt

    statuses = []
    for sector in range(4):
        task_frame = frame.copy()
        task_frame["target_sector"] = task_frame[config.FOLD_COL]
        train_mask = task_frame[config.FOLD_COL].to_numpy(dtype=int) != sector
        test_mask = ~train_mask
        data = build_design(task_frame, train_mask, test_mask, config.PRIMARY_MODEL)
        transfer = pd.read_csv(
            config.OUTPUT
            / "predictions"
            / f"T0_mcmc_fold_{sector + 1}.csv"
        ).set_index("id").reindex(data["test_ids"])
        if transfer["prob"].isna().any():
            raise ValueError(f"Transfer prediction ID mismatch in sector {sector + 1}")
        transfer_probability = transfer["prob"].to_numpy(dtype=float)
        eta_transfer = np.log(-np.log1p(-transfer_probability))
        residual_linear = eta_transfer - data["offset_test"]

        for correction in ("base", "recal"):
            fit_id = f"a3_oracle_{correction}_sector_{sector + 1}"
            paths = _result_paths(fit_id)
            seed = (
                AMENDMENT_SAMPLER["random_seed"]
                + 1200
                + sector * 2
                + int(correction == "recal")
            )
            with pm.Model() as model:
                delta = pm.Normal("delta", mu=0.0, sigma=1.0)
                if correction == "base":
                    eta = eta_transfer + delta
                    names = ["delta"]
                else:
                    multiplier = pm.Normal("multiplier", mu=1.0, sigma=1.0)
                    eta = data["offset_test"] + delta + multiplier * residual_linear
                    names = ["delta", "multiplier"]
                pm.Bernoulli(
                    "y_obs",
                    p=pt.clip(-pt.expm1(-pt.exp(eta)), 1e-12, 1 - 1e-12),
                    observed=data["y_test"],
                )
            attempts = []
            tune = AMENDMENT_SAMPLER["tune"]
            for attempt in range(1, 4):
                with model:
                    trace = pm.sample(
                        draws=AMENDMENT_SAMPLER["draws"],
                        tune=tune,
                        chains=AMENDMENT_SAMPLER["chains"],
                        cores=AMENDMENT_SAMPLER["cores"],
                        target_accept=AMENDMENT_SAMPLER["target_accept"],
                        seed=seed + attempt - 1,
                        progressbar=True,
                        idata_kwargs={"log_likelihood": False},
                    )
                summary = az.summary(
                    trace, var_names=names, hdi_prob=0.95, round_to=8
                )
                diverging = int(np.asarray(trace.sample_stats["diverging"]).sum())
                info = {
                    "diverging": diverging,
                    "max_rhat": float(np.nanmax(summary["r_hat"])),
                    "min_ess_bulk": float(np.nanmin(summary["ess_bulk"])),
                    "min_ess_tail": float(np.nanmin(summary["ess_tail"])),
                }
                info["converged"] = bool(
                    diverging == 0
                    and info["max_rhat"] < config.MCMC_TOLERANCE["max_rhat"]
                    and info["min_ess_bulk"] >= config.MCMC_TOLERANCE["min_ess_bulk"]
                    and info["min_ess_tail"] >= config.MCMC_TOLERANCE["min_ess_tail"]
                )
                attempts.append({**info, "attempt": attempt, "tune": tune})
                if info["converged"]:
                    break
                tune *= 2
            save_trace(trace, ensure_output_path(paths["trace"]))
            delta_draws = np.asarray(trace.posterior["delta"].values).reshape(-1)
            if correction == "base":
                probability = cloglog_probability(
                    eta_transfer[None, :] + delta_draws[:, None]
                )
            else:
                multiplier_draws = np.asarray(trace.posterior["multiplier"].values).reshape(-1)
                probability = cloglog_probability(
                    data["offset_test"][None, :]
                    + delta_draws[:, None]
                    + multiplier_draws[:, None] * residual_linear[None, :]
                )
            prediction = _prediction_frame(data, probability, "amendment1", fit_id).assign(
                model="T0",
                batch="a3_oracle",
                fit_id=fit_id,
                correction=correction,
                target_sector=sector + 1,
            )
            prediction.to_csv(ensure_output_path(paths["prediction"]), index=False)
            summary.reset_index(names="parameter").to_csv(
                ensure_output_path(paths["coefficient"]), index=False
            )
            status = {
                "fit_id": fit_id,
                "batch": "a3_oracle",
                "model": "T0",
                "created_utc": _utc_now(),
                "seed": seed,
                "sampler": AMENDMENT_SAMPLER,
                "attempts": attempts,
                "converged": info["converged"],
                **info,
                "final": info,
                "n_train": 0,
                "positive_train": int(data["y_test"].sum()),
                "n_test": len(data["test"]),
                "positive_test": int(data["y_test"].sum()),
                "target_sector": sector + 1,
                "correction": correction,
                "environment": _environment_record(),
            }
            ensure_output_path(paths["status"]).write_text(
                json.dumps(status, indent=2) + "\n", encoding="utf-8"
            )
            if not info["converged"]:
                raise RuntimeError(f"Oracle correction failed convergence: {fit_id}")
            statuses.append(status)
    return statuses


def run_batch(frame: pd.DataFrame, batch: str) -> list[dict[str, Any]]:
    partitions = pd.read_csv(
        config.OUTPUT / "partitions" / "transfer_partitions.csv"
    )
    if batch == "a2":
        return run_a2(frame)
    if batch == "a3":
        return [*run_inner_local(frame), *run_oracle_corrections(frame)]
    if batch == "a6":
        return run_random_cv(frame)
    if batch == "a8":
        return run_partitions(frame, partitions)
    if batch == "sensitivities":
        return run_sensitivities(frame)
    raise KeyError(f"Unknown Amendment 1 batch: {batch}")


def aggregate_status() -> pd.DataFrame:
    rows = []
    for path in sorted((config.OUTPUT / "diagnostics" / "mcmc_fits").glob("*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    table = pd.DataFrame(rows)
    table.to_csv(
        ensure_output_path(config.OUTPUT / "diagnostics" / "amendment1_status.csv"),
        index=False,
    )
    failures = table.loc[~table["converged"].astype(bool)]
    if not failures.empty:
        raise RuntimeError(f"Convergence failures: {failures['fit_id'].tolist()}")
    return table

