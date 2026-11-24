"""Fit the pre-registered MCMC and MAP transfer models."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.trace_io import save_trace
from src.transfer import config
from src.transfer.data import build_design, ensure_output_path, load_transfer_frame
from src.transfer.modeling import (
    coefficient_frame,
    fit_map,
    laplace_probability,
    posterior_probability,
    sample_pymc,
)
from src.transfer.partitions import assign_block_folds, sector_pairs


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _prediction_frame(
    data: dict[str, Any],
    probabilities: np.ndarray,
    partition_name: str,
    source: str,
) -> pd.DataFrame:
    test = data["test"].reset_index(drop=True)
    point = probabilities.mean(axis=0) if probabilities.ndim == 2 else probabilities
    target_sector = (
        test["target_sector"].to_numpy(dtype=int)
        if "target_sector" in test.columns
        else test[config.FOLD_COL].to_numpy(dtype=int) + 1
    )
    return pd.DataFrame(
        {
            "id": test["id"].to_numpy(dtype=int),
            "fold": test[config.FOLD_COL].to_numpy(dtype=int),
            "target_sector": target_sector,
            "y": test[config.Y_COL].to_numpy(dtype=int),
            "prob": point,
            "prob_lower": (
                np.percentile(probabilities, 2.5, axis=0)
                if probabilities.ndim == 2
                else point
            ),
            "prob_upper": (
                np.percentile(probabilities, 97.5, axis=0)
                if probabilities.ndim == 2
                else point
            ),
            "p_ref_train": float((data["y_train"].sum() + 0.5) / (len(data["y_train"]) + 1.0)),
            "partition": partition_name,
            "source": source,
        }
    )


def _write_status(path: Path, rows: list[dict[str, Any]]) -> None:
    path = ensure_output_path(path)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def rebuild_mcmc_coefficient_table() -> None:
    """Refresh summaries from existing traces without refitting."""

    import arviz as az

    from src.trace_io import load_trace

    rows = []
    for fold in range(1, 5):
        trace = load_trace(config.OUTPUT / "traces" / "T0" / f"fold_{fold}.nc")
        variables = [name for name in trace.posterior.data_vars if name != "y_obs"]
        summary = az.summary(trace, var_names=variables, hdi_prob=0.95, round_to=8)
        summary["median"] = [
            float(np.median(np.asarray(trace.posterior[name].values)))
            for name in summary.index
        ]
        summary = summary.reset_index(names="parameter")
        summary = summary.loc[
            summary["parameter"].str.startswith(("alpha", "beta_")), :
        ].copy()
        summary["fold"] = fold
        summary["model"] = "T0"
        summary["estimation"] = "MCMC"
        rows.append(summary)
    pd.concat(rows, ignore_index=True).to_csv(
        ensure_output_path(config.OUTPUT / "coefficients" / "mcmc_T0.csv"),
        index=False,
    )


def run_mcmc(frame: pd.DataFrame, overwrite: bool = False) -> None:
    trace_dir = config.OUTPUT / "traces" / "T0"
    prediction_dir = config.OUTPUT / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[pd.DataFrame] = []
    coefficients: list[pd.DataFrame] = []
    status: list[dict[str, Any]] = []
    for fold in range(4):
        trace_path = trace_dir / f"fold_{fold + 1}.nc"
        prediction_path = prediction_dir / f"T0_mcmc_fold_{fold + 1}.csv"
        if trace_path.exists() and prediction_path.exists() and not overwrite:
            continue
        train_mask = frame[config.FOLD_COL].to_numpy(dtype=int) != fold
        test_mask = frame[config.FOLD_COL].to_numpy(dtype=int) == fold
        data = build_design(frame, train_mask, test_mask, config.PRIMARY_MODEL)
        started = time.perf_counter()
        trace, summary, info = sample_pymc(
            data,
            draws=config.SAMPLER["draws"],
            tune=config.SAMPLER["tune"],
            chains=config.SAMPLER["chains"],
            cores=config.SAMPLER["cores"],
            target_accept=config.SAMPLER["target_accept"],
            seed=config.SAMPLER["random_seed"] + fold,
        )
        elapsed = time.perf_counter() - started
        save_trace(trace, trace_path)
        probability = posterior_probability(trace, data)
        prediction = _prediction_frame(
            data,
            probability,
            partition_name="primary",
            source="other_three_sectors",
        ).assign(model="T0", fold=fold + 1)
        prediction.to_csv(prediction_path, index=False)
        predictions.append(prediction)
        coefficient = summary.loc[
            summary["parameter"].str.startswith(("alpha", "beta_")), :
        ].copy()
        coefficient["fold"] = fold + 1
        coefficient["model"] = "T0"
        coefficient["estimation"] = "MCMC"
        coefficients.append(coefficient)
        status.append(
            {
                "model": "T0",
                "fold": fold + 1,
                "status": "PASS" if info["converged"] else "FAIL_CONVERGENCE",
                **info,
                "elapsed_seconds": elapsed,
                "n_train": len(data["train"]),
                "positive_train": int(data["y_train"].sum()),
                "n_test": len(data["test"]),
                "positive_test": int(data["y_test"].sum()),
                "draws": config.SAMPLER["draws"],
                "tune": config.SAMPLER["tune"],
                "chains": config.SAMPLER["chains"],
                "cores": config.SAMPLER["cores"],
                "target_accept": config.SAMPLER["target_accept"],
                "seed": config.SAMPLER["random_seed"] + fold,
            }
        )
        _write_status(config.OUTPUT / "diagnostics" / "mcmc_status.json", status)
        print(
            f"[transfer MCMC] fold {fold + 1}: {status[-1]['status']} "
            f"diverging={info['diverging']} max_rhat={info['max_rhat']:.4f}",
            flush=True,
        )
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(
            ensure_output_path(prediction_dir / "T0_mcmc_loo.csv"), index=False
        )
    if coefficients:
        pd.concat(coefficients, ignore_index=True).to_csv(
            ensure_output_path(config.OUTPUT / "coefficients" / "mcmc_T0.csv"),
            index=False,
        )
    if not (config.OUTPUT / "diagnostics" / "mcmc_status.json").exists():
        _write_status(config.OUTPUT / "diagnostics" / "mcmc_status.json", status)


def run_map_validation(frame: pd.DataFrame) -> None:
    """Fit only the four T0 MAP folds needed for the MAP-MCMC gate."""

    (config.OUTPUT / "predictions").mkdir(parents=True, exist_ok=True)
    (config.OUTPUT / "coefficients").mkdir(parents=True, exist_ok=True)
    predictions = []
    coefficients = []
    status = []
    for fold in range(4):
        prediction, coef, info = _fit_map_prediction(
            frame,
            frame[config.FOLD_COL].to_numpy(dtype=int) != fold,
            frame[config.FOLD_COL].to_numpy(dtype=int) == fold,
            config.PRIMARY_MODEL,
            f"loo_fold_{fold + 1}",
            config.SAMPLER["random_seed"] + fold,
        )
        prediction["fold"] = fold + 1
        for row in coef:
            row["fold"] = fold + 1
        predictions.append(prediction)
        coefficients.extend(coef)
        status.append({**info, "fold": fold + 1})
    pd.concat(predictions, ignore_index=True).to_csv(
        ensure_output_path(config.OUTPUT / "predictions" / "map_loo.csv"),
        index=False,
    )
    pd.DataFrame(coefficients).to_csv(
        ensure_output_path(config.OUTPUT / "coefficients" / "map_loo.csv"),
        index=False,
    )
    pd.DataFrame(status).to_csv(
        ensure_output_path(config.OUTPUT / "diagnostics" / "map_validation_status.csv"),
        index=False,
    )


def _fit_map_prediction(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    model_name: str,
    scenario: str,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    data = build_design(frame, train_mask, test_mask, model_name)
    fit = fit_map(data)
    probability = laplace_probability(fit, data, seed=seed)
    prediction = _prediction_frame(data, probability, "primary", scenario).assign(
        model=model_name
    )
    coefficients = coefficient_frame(fit, "MAP-Laplace")
    for row in coefficients:
        row.update({"model": model_name, "scenario": scenario, "fold": ""})
    return prediction, coefficients, {
        "model": model_name,
        "scenario": scenario,
        "optimization_success": fit.optimization_success,
        "optimization_message": fit.optimization_message,
        "n_train": int(len(data["train"])),
        "positive_train": int(data["y_train"].sum()),
        "n_test": int(len(data["test"])),
        "positive_test": int(data["y_test"].sum()),
    }


def run_map_analyses(frame: pd.DataFrame) -> None:
    (config.OUTPUT / "predictions").mkdir(parents=True, exist_ok=True)
    (config.OUTPUT / "coefficients").mkdir(parents=True, exist_ok=True)
    fit_rows: list[dict[str, Any]] = []

    loo_predictions = []
    loo_coefficients = []
    for model_name in [config.PRIMARY_MODEL, *config.SENSITIVITY_MODELS]:
        for fold in range(4):
            prediction, coefficients, info = _fit_map_prediction(
                frame,
                frame[config.FOLD_COL].to_numpy(dtype=int) != fold,
                frame[config.FOLD_COL].to_numpy(dtype=int) == fold,
                model_name,
                f"loo_fold_{fold + 1}",
                config.SAMPLER["random_seed"] + fold,
            )
            prediction["fold"] = fold + 1
            for row in coefficients:
                row["fold"] = fold + 1
            loo_predictions.append(prediction)
            loo_coefficients.extend(coefficients)
            fit_rows.append({**info, "fold": fold + 1})
    pd.concat(loo_predictions, ignore_index=True).to_csv(
        ensure_output_path(config.OUTPUT / "predictions" / "map_loo.csv"), index=False
    )
    pd.DataFrame(loo_coefficients).to_csv(
        ensure_output_path(config.OUTPUT / "coefficients" / "map_loo.csv"), index=False
    )

    single_predictions = []
    single_coefficients = []
    folds = frame[config.FOLD_COL].to_numpy(dtype=int)
    for source_fold in range(4):
        prediction, coefficients, info = _fit_map_prediction(
            frame,
            folds == source_fold,
            folds != source_fold,
            config.PRIMARY_MODEL,
            f"single_source_{source_fold + 1}",
            config.SAMPLER["random_seed"] + 100 + source_fold,
        )
        prediction["source_sector"] = source_fold + 1
        for row in coefficients:
            row.update({"source_sector": source_fold + 1, "fold": ""})
        single_predictions.append(prediction)
        single_coefficients.extend(coefficients)
        fit_rows.append(
            {
                **info,
                "fold": "",
                "source_sector": source_fold + 1,
                "few_training_positives": bool(info["positive_train"] < 15),
            }
        )
    pd.concat(single_predictions, ignore_index=True).to_csv(
        ensure_output_path(config.OUTPUT / "predictions" / "single_sector_transfer.csv"),
        index=False,
    )
    pd.DataFrame(single_coefficients).to_csv(
        ensure_output_path(config.OUTPUT / "coefficients" / "single_sector.csv"),
        index=False,
    )

    pair_predictions = []
    pair_coefficients = []
    for pair_index, pair in enumerate(sector_pairs()):
        source_mask = np.isin(folds, pair)
        prediction, coefficients, info = _fit_map_prediction(
            frame,
            source_mask,
            ~source_mask,
            config.PRIMARY_MODEL,
            f"pair_{pair[0] + 1}_{pair[1] + 1}",
            config.SAMPLER["random_seed"] + 200 + pair_index,
        )
        prediction["source_sector_a"] = pair[0] + 1
        prediction["source_sector_b"] = pair[1] + 1
        for row in coefficients:
            row.update(
                {
                    "source_sector_a": pair[0] + 1,
                    "source_sector_b": pair[1] + 1,
                    "fold": "",
                }
            )
        pair_predictions.append(prediction)
        pair_coefficients.extend(coefficients)
        fit_rows.append(
            {
                **info,
                "fold": "",
                "source_sector_a": pair[0] + 1,
                "source_sector_b": pair[1] + 1,
            }
        )
    pd.concat(pair_predictions, ignore_index=True).to_csv(
        ensure_output_path(config.OUTPUT / "predictions" / "pair_sector_transfer.csv"),
        index=False,
    )
    pd.DataFrame(pair_coefficients).to_csv(
        ensure_output_path(config.OUTPUT / "coefficients" / "pair_sector.csv"),
        index=False,
    )
    pd.DataFrame(fit_rows).to_csv(
        ensure_output_path(config.OUTPUT / "diagnostics" / "map_status.csv"),
        index=False,
    )


def run_partition_sensitivities(frame: pd.DataFrame, partitions: pd.DataFrame) -> None:
    partitions = partitions.set_index("id")
    rows = []
    predictions = []
    coefficients = []
    single_predictions = []
    single_coefficients = []
    for partition_name, column in (("P_GAP", "fold_p_gap"), ("P_POS", "fold_p_pos")):
        labels = frame["id"].map(partitions[column]).to_numpy(dtype=int)
        partition_frame = frame.copy()
        partition_frame["target_sector"] = labels
        for sector in range(4):
            prediction, coef, info = _fit_map_prediction(
                partition_frame,
                labels != sector,
                labels == sector,
                config.PRIMARY_MODEL,
                f"{partition_name}_target_{sector + 1}",
                config.SAMPLER["random_seed"] + 300 + sector,
            )
            prediction["fold"] = sector + 1
            prediction["target_sector"] = sector + 1
            prediction["partition"] = partition_name
            for row in coef:
                row.update({"fold": sector + 1, "partition": partition_name})
            predictions.append(prediction)
            coefficients.extend(coef)
            rows.append({**info, "fold": sector + 1, "partition": partition_name})
        for source_sector in range(4):
            prediction, coef, info = _fit_map_prediction(
                partition_frame,
                labels == source_sector,
                labels != source_sector,
                config.PRIMARY_MODEL,
                f"{partition_name}_single_source_{source_sector + 1}",
                config.SAMPLER["random_seed"] + 350 + source_sector,
            )
            prediction["fold"] = 0
            prediction["target_sector"] = labels[labels != source_sector] + 1
            prediction["partition"] = partition_name
            prediction["source_sector"] = source_sector + 1
            for row in coef:
                row.update(
                    {
                        "fold": "",
                        "partition": partition_name,
                        "source_sector": source_sector + 1,
                    }
                )
            single_predictions.append(prediction)
            single_coefficients.extend(coef)
            rows.append(
                {
                    **info,
                    "fold": "",
                    "partition": partition_name,
                    "source_sector": source_sector + 1,
                    "few_training_positives": bool(info["positive_train"] < 15),
                }
            )
    pd.concat(predictions, ignore_index=True).to_csv(
        ensure_output_path(config.OUTPUT / "predictions" / "partition_loo_map.csv"),
        index=False,
    )
    pd.DataFrame(coefficients).to_csv(
        ensure_output_path(config.OUTPUT / "coefficients" / "partition_loo_map.csv"),
        index=False,
    )
    pd.concat(single_predictions, ignore_index=True).to_csv(
        ensure_output_path(
            config.OUTPUT / "predictions" / "partition_single_sector_map.csv"
        ),
        index=False,
    )
    pd.DataFrame(single_coefficients).to_csv(
        ensure_output_path(
            config.OUTPUT / "coefficients" / "partition_single_sector_map.csv"
        ),
        index=False,
    )
    pd.DataFrame(rows).to_csv(
        ensure_output_path(config.OUTPUT / "diagnostics" / "partition_map_status.csv"),
        index=False,
    )


def run_random_cv(frame: pd.DataFrame) -> None:
    y = frame[config.Y_COL].to_numpy(dtype=int)
    prediction_rows = []
    status_rows = []
    for repeat in range(config.RANDOM_CV_REPEATS):
        rng = np.random.default_rng(config.BOOTSTRAP_SEED + repeat)
        fold = np.empty(len(frame), dtype=int)
        for label in (0, 1):
            indices = np.flatnonzero(y == label)
            indices = rng.permutation(indices)
            fold[indices] = np.arange(len(indices)) % config.RANDOM_CV_FOLDS
        for inner_fold in range(config.RANDOM_CV_FOLDS):
            prediction, _, info = _fit_map_prediction(
                frame,
                fold != inner_fold,
                fold == inner_fold,
                config.PRIMARY_MODEL,
                f"random_cv_repeat_{repeat + 1}",
                config.SAMPLER["random_seed"] + 400 + repeat * 4 + inner_fold,
            )
            prediction["repeat"] = repeat + 1
            prediction["fold"] = inner_fold + 1
            prediction_rows.append(prediction)
            status_rows.append(
                {
                    **info,
                    "repeat": repeat + 1,
                    "fold": inner_fold + 1,
                    "few_training_positives": bool(info["positive_train"] < 15),
                }
            )
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        ensure_output_path(config.OUTPUT / "predictions" / "random_cv_map.csv"),
        index=False,
    )
    pd.DataFrame(status_rows).to_csv(
        ensure_output_path(config.OUTPUT / "diagnostics" / "random_cv_status.csv"),
        index=False,
    )


def run_inner_local(frame: pd.DataFrame) -> None:
    predictions = []
    status = []
    for sector in range(4):
        subset = frame.loc[frame[config.FOLD_COL] == sector].reset_index(drop=True)
        inner = assign_block_folds(
            subset,
            n_folds=4,
            seed=config.BOOTSTRAP_SEED + 500 + sector,
        )
        for fold in range(4):
            prediction, _, info = _fit_map_prediction(
                subset,
                inner != fold,
                inner == fold,
                config.PRIMARY_MODEL,
                f"local_sector_{sector + 1}_inner_{fold + 1}",
                config.SAMPLER["random_seed"] + 600 + sector * 4 + fold,
            )
            prediction["target_sector"] = sector + 1
            prediction["inner_fold"] = fold + 1
            predictions.append(prediction)
            status.append({**info, "target_sector": sector + 1, "inner_fold": fold + 1})
    pd.concat(predictions, ignore_index=True).to_csv(
        ensure_output_path(config.OUTPUT / "predictions" / "inner_local_crossfit.csv"),
        index=False,
    )
    pd.DataFrame(status).to_csv(
        ensure_output_path(config.OUTPUT / "diagnostics" / "inner_local_status.csv"),
        index=False,
    )


def run_all_fits(frame: pd.DataFrame | None = None) -> None:
    frame = frame if frame is not None else load_transfer_frame()
    run_map_analyses(frame)
    partitions = pd.read_csv(config.OUTPUT / "partitions" / "transfer_partitions.csv")
    run_partition_sensitivities(frame, partitions)
    run_random_cv(frame)
    run_inner_local(frame)

