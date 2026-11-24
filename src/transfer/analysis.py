"""Summaries, decomposition, applicability and figures for transfer analysis."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.transfer import config
from src.transfer.data import build_design, ensure_output_path, load_transfer_frame
from src.transfer.metrics import (
    class_stratified_bootstrap,
    metric_summary,
    spatial_block_bootstrap_delta,
)
from src.transfer.modeling import (
    cloglog_probability,
    fit_intercept_oracle,
    fit_scaled_oracle,
    log_probability,
)
from src.transfer.partitions import (
    build_partitions,
    stratified_random_folds,
)
from src.v15.metrics import log_score, roc_auc


def _json_write(path, value: Any) -> None:
    path = ensure_output_path(path)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def _load_predictions(name: str) -> pd.DataFrame:
    frame = pd.read_csv(config.OUTPUT / "predictions" / name)
    if "centroid_x" not in frame.columns or "centroid_y" not in frame.columns:
        coordinates = pd.read_csv(
            config.MODEL_FRAME,
            usecols=["id", "centroid_x", "centroid_y", "area_km2"],
        )
        frame = frame.merge(coordinates, on="id", how="left", validate="many_to_one")
    required = {"id", "y", "prob", "target_sector"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Prediction file lacks columns {required - set(frame.columns)}: {name}")
    return frame


def write_partition_outputs(frame: pd.DataFrame) -> pd.DataFrame:
    partitions, info = build_partitions(frame)
    partition_dir = config.OUTPUT / "partitions"
    partition_dir.mkdir(parents=True, exist_ok=True)
    partitions.to_csv(ensure_output_path(partition_dir / "transfer_partitions.csv"), index=False)
    for name, table in info["details"].items():
        table.to_csv(
            ensure_output_path(partition_dir / f"{name}_summary.csv"),
            index=False,
        )
    _json_write(partition_dir / "partition_metadata.json", info["metadata"])
    return partitions


def validate_map_mcmc() -> dict[str, Any]:
    mcmc = _load_predictions("T0_mcmc_loo.csv")
    maps = _load_predictions("map_loo.csv")
    maps = maps.loc[
        (maps["model"] == "T0") & maps["source"].str.startswith("loo_fold_")
    ].copy()
    merged = mcmc[["id", "fold", "y", "prob"]].merge(
        maps[["id", "prob"]],
        on="id",
        suffixes=("_mcmc", "_map"),
        validate="one_to_one",
    )
    mcmc_score = log_score(merged["y"], merged["prob_mcmc"])
    map_score = log_score(merged["y"], merged["prob_map"])
    probability_difference = (merged["prob_map"] - merged["prob_mcmc"]).abs()
    pooled_difference = float(map_score.mean() - mcmc_score.mean())
    fold_rows = []
    for fold, group in merged.groupby("fold"):
        fold_rows.append(
            {
                "fold": int(fold),
                "correlation": float(
                    np.corrcoef(group["prob_mcmc"], group["prob_map"])[0, 1]
                ),
                "max_abs_probability_difference": float(
                    probability_difference.loc[group.index].max()
                ),
                "mean_log_score_difference": float(
                    (
                        log_score(group["y"], group["prob_map"])
                        - log_score(group["y"], group["prob_mcmc"])
                    ).mean()
                ),
            }
        )
    mcmc_coeff = pd.read_csv(config.OUTPUT / "coefficients" / "mcmc_T0.csv")
    map_coeff = pd.read_csv(config.OUTPUT / "coefficients" / "map_loo.csv")
    map_coeff = map_coeff.loc[
        (map_coeff["model"] == "T0")
        & map_coeff["scenario"].str.startswith("loo_fold_")
    ].copy()
    term_map = {
        "alpha": "intercept",
        "beta_fault_z": "beta_fault_z",
        "beta_fault_z2": "beta_fault_z2",
        "beta_contact_z": "beta_contact_z",
        "beta_contact_z2": "beta_contact_z2",
        "beta_gravity_z": "beta_gravity_z",
    }
    coefficient_rows = []
    for fold in range(1, 5):
        left = mcmc_coeff.loc[mcmc_coeff["fold"] == fold]
        right = map_coeff.loc[map_coeff["fold"] == fold]
        for mcmc_term, map_term in term_map.items():
            lrow = left.loc[left["parameter"] == mcmc_term]
            rrow = right.loc[right["term"] == map_term]
            if lrow.empty or rrow.empty:
                continue
            coefficient_rows.append(
                {
                    "fold": fold,
                    "term": mcmc_term,
                    "mcmc_median": float(lrow.iloc[0]["median"]),
                    "mcmc_lower": float(lrow.iloc[0]["hdi_2.5%"]),
                    "mcmc_upper": float(lrow.iloc[0]["hdi_97.5%"]),
                    "map_estimate": float(rrow.iloc[0]["estimate"]),
                    "map_lower": float(rrow.iloc[0]["ci_lower"]),
                    "map_upper": float(rrow.iloc[0]["ci_upper"]),
                }
            )
    coefficient_table = pd.DataFrame(coefficient_rows)
    coefficient_table.to_csv(
        ensure_output_path(config.OUTPUT / "metrics" / "map_mcmc_coefficients.csv"),
        index=False,
    )
    result = {
        "n": int(len(merged)),
        "pooled_log_score_mcmc": float(mcmc_score.mean()),
        "pooled_log_score_map": float(map_score.mean()),
        "mean_log_score_difference_map_minus_mcmc": pooled_difference,
        "absolute_mean_log_score_difference": abs(pooled_difference),
        "tolerance": config.MAP_MCMC_MAX_LOG_SCORE_DIFFERENCE,
        "pass": abs(pooled_difference) <= config.MAP_MCMC_MAX_LOG_SCORE_DIFFERENCE,
        "pooled_probability_correlation": float(
            np.corrcoef(merged["prob_mcmc"], merged["prob_map"])[0, 1]
        ),
        "max_abs_probability_difference": float(probability_difference.max()),
        "mean_abs_probability_difference": float(probability_difference.mean()),
        "fold_comparison": fold_rows,
    }
    _json_write(config.OUTPUT / "diagnostics" / "map_mcmc_validation.json", result)
    return result


def map_gate_passed() -> dict[str, Any]:
    path = config.OUTPUT / "diagnostics" / "map_mcmc_validation.json"
    if not path.exists():
        raise FileNotFoundError("Run MAP-MCMC validation before using MAP analyses")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not result["pass"]:
        raise RuntimeError(
            "MAP-MCMC validation failed: pooled held-out log-score difference "
            f"{result['mean_log_score_difference_map_minus_mcmc']:.6f} exceeds "
            f"{config.MAP_MCMC_MAX_LOG_SCORE_DIFFERENCE:.3f}"
        )
    return result


def analyse_a1(frame: pd.DataFrame) -> pd.DataFrame:
    predictions = _load_predictions("T0_mcmc_loo.csv")
    benchmark = pd.read_csv(
        config.ROOT / "figures" / "v14_centroid_contact" / "v14_centroid_fold_metrics.csv"
    ).set_index("fold")
    rows = [
        {
            "sample": "pooled",
            **_with_prevalence_skill(predictions, metric_summary(predictions, "pooled")),
        }
    ]
    for sector, group in predictions.groupby("target_sector"):
        row = _with_prevalence_skill(
            group, metric_summary(group, f"sector_{int(sector)}")
        )
        row["v14_benchmark_auc"] = float(benchmark.loc[int(sector), "auc"])
        rows.append(row)
    table = pd.DataFrame(rows)
    sector_mask = table["sample"].str.startswith("sector_")
    table.loc[sector_mask, "skill_rank"] = (
        table.loc[sector_mask, "log_score_skill_vs_prevalence"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    table.to_csv(ensure_output_path(config.OUTPUT / "metrics" / "a1_transfer_metrics.csv"), index=False)
    return table


def _with_prevalence_skill(
    predictions: pd.DataFrame,
    row: dict[str, Any],
) -> dict[str, Any]:
    y = predictions["y"].to_numpy(dtype=int)
    probability = predictions["prob"].to_numpy(dtype=float)
    reference = predictions["p_ref_train"].to_numpy(dtype=float)
    model_loss = -float(log_score(y, probability).mean())
    reference_loss = -float(log_score(y, reference).mean())
    return {
        **row,
        "prevalence_baseline_log_score": -reference_loss,
        "log_score_skill_vs_prevalence": reference_loss - model_loss,
        "relative_log_score_skill": (
            1.0 - model_loss / reference_loss if reference_loss > 0 else float("nan")
        ),
        "mean_minus_observed_rate": float(probability.mean() - y.mean()),
    }


def _matrix_metric_rows(
    predictions: pd.DataFrame,
    source_columns: list[str],
    source_type: str,
    partition: str,
) -> pd.DataFrame:
    rows = []
    group_columns = source_columns + ["target_sector"]
    for keys, group in predictions.groupby(group_columns):
        if not isinstance(keys, tuple):
            keys = (keys,)
        source = dict(zip(source_columns, keys[:-1]))
        target = int(keys[-1])
        summary = metric_summary(group, f"target_{target}")
        rows.append(
            {
                **source,
                "source_type": source_type,
                "partition": partition,
                "target_sector": target,
                "n": summary["n"],
                "positives": summary["positives"],
                "mean_log_score": summary["mean_log_score"],
                "auc": summary["auc"],
                "pr_auc": summary["pr_auc"],
            }
        )
    return pd.DataFrame(rows)


def analyse_a2(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    single = _load_predictions("single_sector_transfer.csv")
    pair = _load_predictions("pair_sector_transfer.csv")
    loo = _load_predictions("map_loo.csv")
    loo = loo.loc[
        (loo["model"] == "T0") & loo["source"].str.startswith("loo_fold_")
    ].copy()
    single_rows = _matrix_metric_rows(
        single, ["source_sector"], "single_sector", "primary"
    )
    pair_rows = _matrix_metric_rows(
        pair, ["source_sector_a", "source_sector_b"], "sector_pair", "primary"
    )
    loo_rows = []
    for sector, group in loo.groupby("target_sector"):
        summary = metric_summary(group, f"target_{int(sector)}")
        loo_rows.append(
            {
                "source_type": "other_three_sectors",
                "partition": "primary",
                "target_sector": int(sector),
                "n": summary["n"],
                "positives": summary["positives"],
                "mean_log_score": summary["mean_log_score"],
                "auc": summary["auc"],
                "pr_auc": summary["pr_auc"],
            }
        )
    all_rows = pd.concat(
        [single_rows, pair_rows, pd.DataFrame(loo_rows)], ignore_index=True
    )
    pair_counts = frame.groupby(config.FOLD_COL).agg(
        training_cells=("id", "size"),
        training_positives=(config.Y_COL, "sum"),
    )
    pair_counts["source_sector"] = pair_counts.index + 1
    single_rows = single_rows.merge(
        pair_counts[["source_sector", "training_cells", "training_positives"]],
        on="source_sector",
        how="left",
    )
    single_rows["few_training_positives"] = single_rows["training_positives"] < 15
    partition_view = frame.assign(
        axis_km=build_partitions(frame)[0]["axis_km"]
    ).groupby(config.FOLD_COL)["axis_km"].mean()
    single_rows["source_axis_centroid_km"] = single_rows["source_sector"].map(
        partition_view
    )
    single_rows["target_axis_centroid_km"] = single_rows["target_sector"].map(
        partition_view
    )
    single_rows["sector_separation_km"] = (
        single_rows["source_axis_centroid_km"] - single_rows["target_axis_centroid_km"]
    ).abs()
    all_rows = pd.concat(
        [
            single_rows,
            pair_rows,
            all_rows.loc[all_rows["source_type"] == "other_three_sectors"],
        ],
        ignore_index=True,
    )
    all_rows.to_csv(
        ensure_output_path(config.OUTPUT / "metrics" / "a2_transfer_matrix_long.csv"),
        index=False,
    )
    wide_rows = []
    for value in ("mean_log_score", "auc"):
        wide = single_rows.pivot(
            index="source_sector", columns="target_sector", values=value
        )
        wide.index.name = "source_sector"
        wide.columns = [f"target_{int(column)}" for column in wide.columns]
        wide = wide.reset_index()
        wide["metric"] = value
        wide_rows.append(wide)
    wide_table = pd.concat(wide_rows, ignore_index=True)
    wide_table.to_csv(
        ensure_output_path(config.OUTPUT / "metrics" / "a2_single_sector_matrices.csv"),
        index=False,
    )
    return all_rows, wide_table


def _map_t0_coefficients(fold: int) -> np.ndarray:
    table = pd.read_csv(config.OUTPUT / "coefficients" / "map_loo.csv")
    rows = table.loc[
        (table["model"] == "T0")
        & (table["fold"] == fold)
        & (table["scenario"] == f"loo_fold_{fold}")
    ]
    terms = [
        "intercept",
        "beta_fault_z",
        "beta_fault_z2",
        "beta_contact_z",
        "beta_contact_z2",
        "beta_gravity_z",
    ]
    values = []
    for term in terms:
        row = rows.loc[rows["term"] == term]
        if row.empty:
            raise ValueError(f"Missing MAP coefficient {term} for fold {fold}")
        values.append(float(row.iloc[0]["estimate"]))
    return np.asarray(values, dtype=float)


def decomposition_gap_values(
    transfer: np.ndarray,
    base: np.ndarray,
    recal: np.ndarray,
    local: np.ndarray,
) -> tuple[float, float, float]:
    return (
        float(np.mean(base - transfer)),
        float(np.mean(recal - base)),
        float(np.mean(local - recal)),
    )


def analyse_a3(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    map_predictions = _load_predictions("map_loo.csv")
    map_predictions = map_predictions.loc[
        (map_predictions["model"] == "T0")
        & map_predictions["source"].str.startswith("loo_fold_")
    ].copy()
    local = _load_predictions("inner_local_crossfit.csv")
    local = local.groupby("id", as_index=False).agg(
        y=("y", "first"), prob_local=("prob", "mean"), target_sector=("target_sector", "first")
    )
    rows = []
    cell_rows = []
    for sector in range(4):
        fold = sector
        train_mask = frame[config.FOLD_COL].to_numpy(dtype=int) != fold
        test_mask = ~train_mask
        data = build_design(frame, train_mask, test_mask, config.PRIMARY_MODEL)
        beta = _map_t0_coefficients(fold + 1)
        linear_score = data["X_test"] @ beta[1:]
        alpha_base = fit_intercept_oracle(
            data["y_test"],
            data["offset_test"],
            linear_score,
            data["intercept_mean"],
        )
        alpha_recal, slope_multiplier = fit_scaled_oracle(
            data["y_test"],
            data["offset_test"],
            linear_score,
            data["intercept_mean"],
        )
        transfer = map_predictions.loc[
            map_predictions["target_sector"] == sector + 1
        ].sort_values("id")
        local_sector = local.loc[local["target_sector"] == sector + 1].sort_values("id")
        if not np.array_equal(transfer["id"].to_numpy(), data["test_ids"]):
            raise ValueError(f"Prediction ID mismatch in sector {sector + 1}")
        if not np.array_equal(local_sector["id"].to_numpy(), data["test_ids"]):
            raise ValueError(f"Local prediction ID mismatch in sector {sector + 1}")
        probability = {
            "transfer": transfer["prob"].to_numpy(dtype=float),
            "base": cloglog_probability(
                alpha_base + data["offset_test"] + linear_score
            ),
            "recal": cloglog_probability(
                alpha_recal
                + data["offset_test"]
                + slope_multiplier * linear_score
            ),
            "local": local_sector["prob_local"].to_numpy(dtype=float),
        }
        score = {
            name: log_score(data["y_test"], values)
            for name, values in probability.items()
        }
        gap_base = score["base"] - score["transfer"]
        gap_recal = score["recal"] - score["base"]
        gap_local = score["local"] - score["recal"]
        gap_means = decomposition_gap_values(
            score["transfer"], score["base"], score["recal"], score["local"]
        )
        frame_sector = data["test"][["id", "centroid_x", "centroid_y"]].copy()
        frame_sector["y"] = data["y_test"]
        intervals = {}
        for name, delta in (
            ("base_minus_transfer", gap_base),
            ("recal_minus_base", gap_recal),
            ("local_minus_recal", gap_local),
        ):
            intervals[name] = spatial_block_bootstrap_delta(
                frame_sector,
                delta + 0.0,
                np.zeros_like(delta),
                seed=config.BOOTSTRAP_SEED + sector,
            )
        rows.append(
            {
                "sector": sector + 1,
                "n": len(data["y_test"]),
                "positives": int(data["y_test"].sum()),
                "L_transfer": float(score["transfer"].mean()),
                "L_base": float(score["base"].mean()),
                "L_recal": float(score["recal"].mean()),
                "L_local": float(score["local"].mean()),
                "base_minus_transfer": gap_means[0],
                "base_minus_transfer_ci_lower": intervals["base_minus_transfer"][0],
                "base_minus_transfer_ci_upper": intervals["base_minus_transfer"][1],
                "recal_minus_base": gap_means[1],
                "recal_minus_base_ci_lower": intervals["recal_minus_base"][0],
                "recal_minus_base_ci_upper": intervals["recal_minus_base"][1],
                "local_minus_recal": gap_means[2],
                "local_minus_recal_ci_lower": intervals["local_minus_recal"][0],
                "local_minus_recal_ci_upper": intervals["local_minus_recal"][1],
                "oracle_alpha_base": float(alpha_base),
                "oracle_alpha_recal": float(alpha_recal),
                "oracle_slope_multiplier": float(slope_multiplier),
            }
        )
        cell_rows.append(
            frame_sector.assign(
                sector=sector + 1,
                L_transfer=score["transfer"],
                L_base=score["base"],
                L_recal=score["recal"],
                L_local=score["local"],
                base_minus_transfer=gap_base,
                recal_minus_base=gap_recal,
                local_minus_recal=gap_local,
            )
        )
    table = pd.DataFrame(rows)
    cells = pd.concat(cell_rows, ignore_index=True)
    table.to_csv(
        ensure_output_path(config.OUTPUT / "metrics" / "a3_decomposition.csv"),
        index=False,
    )
    cells.to_csv(
        ensure_output_path(config.OUTPUT / "metrics" / "a3_decomposition_cells.csv"),
        index=False,
    )
    return table, cells


def _nearest_distance(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    from scipy.spatial.distance import cdist

    output = np.empty(len(query), dtype=float)
    chunk = 256
    for start in range(0, len(query), chunk):
        stop = min(start + chunk, len(query))
        output[start:stop] = cdist(query[start:stop], reference).min(axis=1)
    return output


def _training_nearest_distance(reference: np.ndarray) -> np.ndarray:
    from scipy.spatial.distance import cdist

    distances = cdist(reference, reference)
    np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1)


def _tukey_max(values: np.ndarray) -> tuple[float, float]:
    clean_values = values[np.isfinite(values)]
    q1, q3 = np.percentile(clean_values, [25, 75])
    fence = q3 + 1.5 * (q3 - q1)
    retained = clean_values[clean_values <= fence]
    return float(retained.max()), float(fence)


def _aoa_for_design(data: dict[str, Any], seed: int) -> dict[str, Any]:
    X_train = data["X_train"]
    X_test = data["X_test"]
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    train = (X_train - mean) / scale
    test = (X_test - mean) / scale
    training_nearest = _training_nearest_distance(train)
    reference_scale = float(training_nearest.mean())
    if reference_scale <= 0:
        raise ValueError("Training predictor space has zero nearest-neighbour distance")
    cv_folds = stratified_random_folds(
        data["y_train"], config.AOA_CV_FOLDS, seed
    )
    training_di = np.empty(len(train), dtype=float)
    for fold in range(config.AOA_CV_FOLDS):
        test_index = np.flatnonzero(cv_folds == fold)
        train_index = np.flatnonzero(cv_folds != fold)
        training_di[test_index] = (
            _nearest_distance(train[train_index], train[test_index])
            / reference_scale
        )
    threshold, fence = _tukey_max(training_di)
    test_di = _nearest_distance(train, test) / reference_scale
    return {
        "di": test_di,
        "aoa": (test_di <= threshold).astype(int),
        "threshold": threshold,
        "tukey_upper_fence": fence,
        "training_mean_nearest_distance": reference_scale,
    }


def analyse_a4(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    univariate_rows = []
    cell_rows = []
    for sector in range(4):
        train_mask = frame[config.FOLD_COL].to_numpy(dtype=int) != sector
        test_mask = ~train_mask
        data = build_design(frame, train_mask, test_mask, config.PRIMARY_MODEL)
        for index, name in enumerate(data["features"]):
            train_values = data["X_train"][:, index]
            test_values = data["X_test"][:, index]
            lower, upper = np.percentile(train_values, [1, 99])
            outside = (test_values < lower) | (test_values > upper)
            univariate_rows.append(
                {
                    "sector": sector + 1,
                    "predictor": name,
                    "training_p1": float(lower),
                    "training_p99": float(upper),
                    "share_outside": float(outside.mean()),
                    "cells_outside": int(outside.sum()),
                }
            )
        aoa = _aoa_for_design(data, config.BOOTSTRAP_SEED + sector)
        prediction = pd.read_csv(
            config.OUTPUT / "predictions" / f"T0_mcmc_fold_{sector + 1}.csv"
        ).set_index("id").reindex(data["test_ids"])
        if prediction["prob"].isna().any():
            raise ValueError(f"AOA prediction ID mismatch in sector {sector + 1}")
        cell_score = log_score(data["y_test"], prediction["prob"].to_numpy())
        cell_rows.append(
            data["test"][["id", "centroid_x", "centroid_y", "area_km2"]].assign(
                sector=sector + 1,
                DI=aoa["di"],
                AOA=aoa["aoa"],
                threshold=aoa["threshold"],
                mean_log_score=cell_score,
                y=data["y_test"],
            )
        )
    univariate = pd.DataFrame(univariate_rows)
    cells = pd.concat(cell_rows, ignore_index=True)
    univariate.to_csv(
        ensure_output_path(config.OUTPUT / "metrics" / "a4_univariate_shift.csv"),
        index=False,
    )
    cells.to_csv(
        ensure_output_path(config.OUTPUT / "metrics" / "a4_aoa_cells.csv"),
        index=False,
    )
    summary_rows = []
    for (sector, inside), group in cells.groupby(["sector", "AOA"]):
        summary_rows.append(
            {
                "sector": int(sector),
                "aoa_status": "inside" if inside == 1 else "outside",
                "cells": int(len(group)),
                "area_km2": float(group["area_km2"].sum()),
                "positives": int(group["y"].sum()),
                "mean_log_score": float(group["mean_log_score"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        ensure_output_path(config.OUTPUT / "metrics" / "a4_aoa_summary.csv"),
        index=False,
    )
    return univariate, summary


def _marginal_auc_rows(frame: pd.DataFrame, partition_column: str) -> pd.DataFrame:
    rows = []
    orientations = {
        "fault": ("fault_stored_km", -1.0),
        "contact": ("contact_km", -1.0),
        "gravity": ("gravity_mgal", -1.0),
    }
    for sector, group in frame.groupby(partition_column):
        y = group[config.Y_COL].to_numpy(dtype=int)
        for predictor, (column, direction) in orientations.items():
            score = direction * group[column].to_numpy(dtype=float)
            auc = roc_auc(y, score)
            lower, upper = class_stratified_bootstrap(y, score, roc_auc)
            rows.append(
                {
                    "partition": "",
                    "sector": int(sector) + 1,
                    "predictor": predictor,
                    "orientation": (
                        "negative distance: closer scores higher"
                        if predictor in {"fault", "contact"}
                        else "negative gravity: lower gravity scores higher"
                    ),
                    "n": int(len(group)),
                    "positives": int(y.sum()),
                    "auc": auc,
                    "auc_ci_lower": lower,
                    "auc_ci_upper": upper,
                }
            )
    return pd.DataFrame(rows)


def _coefficient_sign_rows(include_single_map: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    mcmc = pd.read_csv(config.OUTPUT / "coefficients" / "mcmc_T0.csv")
    rows = []
    for fold, group in mcmc.groupby("fold"):
        for term, row in group.set_index("parameter").iterrows():
            if term.startswith("beta_"):
                rows.append(
                    {
                        "estimation": "MCMC leave-one-sector-out",
                        "source": f"fold_{int(fold)}",
                        "term": term,
                "estimate": float(row["median"]),
                        "ci_lower": float(row["hdi_2.5%"]),
                        "ci_upper": float(row["hdi_97.5%"]),
                    }
                )
    if include_single_map:
        single = pd.read_csv(config.OUTPUT / "coefficients" / "single_sector.csv")
        for row in single.itertuples(index=False):
            if str(row.term).startswith("beta_"):
                rows.append(
                    {
                        "estimation": "MAP single-sector",
                        "source": f"sector_{int(row.source_sector)}",
                        "term": row.term,
                        "estimate": float(row.estimate),
                        "ci_lower": float(row.ci_lower),
                        "ci_upper": float(row.ci_upper),
                    }
                )
    table = pd.DataFrame(rows)
    sign_table = (
        table.assign(sign=np.sign(table["estimate"]).astype(int))
        .groupby(["estimation", "term", "sign"])
        .size()
        .rename("fits")
        .reset_index()
    )
    table.to_csv(
        ensure_output_path(config.OUTPUT / "coefficients" / "a5_coefficient_stability.csv"),
        index=False,
    )
    sign_table.to_csv(
        ensure_output_path(config.OUTPUT / "coefficients" / "a5_sign_agreement.csv"),
        index=False,
    )
    return table, sign_table


def _verdict_from_signs_and_auc(
    sign_table: pd.DataFrame,
    marginal: pd.DataFrame,
    estimation: str,
) -> pd.DataFrame:
    term_groups = {
        "fault": ["beta_fault_z", "beta_fault_z2"],
        "contact": ["beta_contact_z", "beta_contact_z2"],
        "gravity": ["beta_gravity_z"],
    }
    rows = []
    for predictor, terms in term_groups.items():
        term_results = []
        for term in terms:
            counts = sign_table.loc[
                (sign_table["estimation"] == estimation)
                & (sign_table["term"] == term)
            ]
            same_sign = int(counts["fits"].max()) if not counts.empty else 0
            term_results.append(same_sign >= 3)
        auc_rows = marginal.loc[marginal["predictor"] == predictor]
        useful_sectors = int(
            (
                (auc_rows["auc"] > 0.5)
                & (auc_rows["auc_ci_lower"] > 0.5)
            ).sum()
        )
        sign_pass = bool(all(term_results))
        auc_pass = useful_sectors >= 3
        if sign_pass and auc_pass:
            verdict = "transfers"
        elif sign_pass or auc_pass:
            verdict = "partly transfers"
        else:
            verdict = "does not transfer"
        rows.append(
            {
                "predictor": predictor,
                "coefficient_sign_stable": sign_pass,
                "useful_auc_sectors": useful_sectors,
                "auc_condition": auc_pass,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def analyse_a5(
    frame: pd.DataFrame,
    include_single_map: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coefficients, signs = _coefficient_sign_rows(include_single_map=include_single_map)
    marginal = _marginal_auc_rows(frame, config.FOLD_COL)
    marginal["partition"] = "primary"
    verdicts = _verdict_from_signs_and_auc(
        signs, marginal, "MCMC leave-one-sector-out"
    )
    marginal.to_csv(
        ensure_output_path(config.OUTPUT / "metrics" / "a5_marginal_auc.csv"),
        index=False,
    )
    verdicts.to_csv(
        ensure_output_path(config.OUTPUT / "metrics" / "a5_predictor_verdicts.csv"),
        index=False,
    )
    return coefficients, marginal, verdicts


def analyse_a6(frame: pd.DataFrame, a1: pd.DataFrame) -> pd.DataFrame:
    predictions = _load_predictions("random_cv_map.csv")
    rows = []
    for repeat, group in predictions.groupby("repeat"):
        y = group["y"].to_numpy(dtype=int)
        probability = group["prob"].to_numpy(dtype=float)
        rows.append(
            {
                "repeat": int(repeat),
                "mean_log_score": float(log_score(y, probability).mean()),
                "auc": roc_auc(y, probability),
            }
        )
    repeats = pd.DataFrame(rows)
    loo = a1.loc[a1["sample"] == "pooled"].iloc[0]
    summary = pd.DataFrame(
        [
            {
                "validation": "repeated stratified random 4-fold",
                "mean_log_score": float(repeats["mean_log_score"].mean()),
                "log_score_sd": float(repeats["mean_log_score"].std(ddof=1)),
                "log_score_q2_5": float(repeats["mean_log_score"].quantile(0.025)),
                "log_score_q97_5": float(repeats["mean_log_score"].quantile(0.975)),
                "auc": float(repeats["auc"].mean()),
                "auc_sd": float(repeats["auc"].std(ddof=1)),
                "auc_q2_5": float(repeats["auc"].quantile(0.025)),
                "auc_q97_5": float(repeats["auc"].quantile(0.975)),
                "repeats": len(repeats),
            },
            {
                "validation": "leave-one-sector-out MCMC",
                "mean_log_score": float(loo["mean_log_score"]),
                "log_score_sd": float("nan"),
                "log_score_q2_5": float(loo["mean_log_score_block_ci_lower"]),
                "log_score_q97_5": float(loo["mean_log_score_block_ci_upper"]),
                "auc": float(loo["auc"]),
                "auc_sd": float("nan"),
                "auc_q2_5": float(loo["auc_ci_lower"]),
                "auc_q97_5": float(loo["auc_ci_upper"]),
                "repeats": 1,
            },
        ]
    )
    buffered = pd.read_csv(config.V15 / "metrics" / "buffered_validation_metrics.csv")
    for row in buffered.loc[buffered["model"] == "M0_global"].itertuples(index=False):
        summary.loc[len(summary)] = {
            "validation": f"V15 M0 buffered {row.buffer_km:g} km (different model)",
            "mean_log_score": row.mean_log_score,
            "log_score_sd": float("nan"),
            "log_score_q2_5": float("nan"),
            "log_score_q97_5": float("nan"),
            "auc": row.auc,
            "auc_sd": float("nan"),
            "auc_q2_5": float("nan"),
            "auc_q97_5": float("nan"),
            "repeats": 1,
        }
    repeats.to_csv(
        ensure_output_path(config.OUTPUT / "metrics" / "a6_random_cv_repeats.csv"),
        index=False,
    )
    summary.to_csv(
        ensure_output_path(config.OUTPUT / "metrics" / "a6_optimism_comparison.csv"),
        index=False,
    )
    return summary


def analyse_a7() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    v15_models = pd.read_csv(config.V15 / "metrics" / "model_metrics.csv")
    v15_models = v15_models.loc[v15_models["model"].isin(
        [
            "M0_global",
            "M1_domain_intercept",
            "M2_domain_distance",
            "M3_domain_distance_gravity",
        ]
    )].copy()
    v15_models.insert(0, "source", "V15 hierarchical")
    phase2b = pd.read_csv(config.PHASE2B / "metrics" / "model_metrics.csv")
    phase2b.insert(0, "source", "Phase2b fixed, rebuilt")
    phase2c = pd.read_csv(
        config.ROOT
        / "paper_1"
        / "v15"
        / "phase2c_stored_faults"
        / "metrics"
        / "model_metrics.csv"
    )
    phase2c.insert(0, "source", "Phase2c fixed, stored")
    columns = [
        "source",
        "frame",
        "model",
        "sample",
        "n",
        "positives",
        "auc",
        "pr_auc",
        "mean_log_score",
        "brier",
        "calibration_slope",
    ]
    domain_summary = pd.concat(
        [
            v15_models.assign(frame="primary"),
            phase2b[[column for column in columns if column in phase2b.columns]],
            phase2c[[column for column in columns if column in phase2c.columns]],
        ],
        ignore_index=True,
    )
    domain_summary = domain_summary[
        [column for column in columns if column in domain_summary.columns]
    ]
    domain_summary.to_csv(
        ensure_output_path(config.OUTPUT / "metrics" / "a7_domain_summary.csv"),
        index=False,
    )
    fixed_b = pd.read_csv(config.PHASE2B / "metrics" / "comparisons.csv")
    fixed_c = pd.read_csv(
        config.ROOT
        / "paper_1"
        / "v15"
        / "phase2c_stored_faults"
        / "metrics"
        / "comparisons.csv"
    )
    pairwise = []
    for label, table, comparison in (
        (
            "Phase2b fixed, rebuilt",
            fixed_b,
            ("N2_3b_distance_interactions", "N3_3b_gravity_interaction"),
        ),
        (
            "Phase2c fixed, stored",
            fixed_c,
            ("N2_3b_distance_interactions", "N3_3b_gravity_interaction"),
        ),
    ):
        row = table.loc[
            (table["frame"] == "primary")
            & (table["lower"] == comparison[0])
            & (table["upper"] == comparison[1])
        ].iloc[0]
        pairwise.append(
            {
                "source": label,
                "comparison": "gravity extension",
                "mean_delta_log_score": row["mean_delta_log_score"],
                "ci_lower": row["ci_lower"],
                "ci_upper": row["ci_upper"],
                "verdict": "not supported",
            }
        )
    v15_central = pd.read_csv(config.V15 / "metrics" / "central_comparisons.csv")
    v15_boot = pd.read_csv(config.V15 / "metrics" / "paired_bootstrap.csv")
    for lower, upper, label, verdict in (
        ("M1_domain_intercept", "M2_domain_distance", "distance heterogeneity", "not supported"),
        (
            "M2_domain_distance",
            "M3_domain_distance_gravity",
            "gravity extension",
            "supported but prior-dependent",
        ),
    ):
        row = v15_central.loc[
            (v15_central["lower"] == lower) & (v15_central["upper"] == upper)
        ].iloc[0]
        boot = v15_boot.loc[
            (v15_boot["left"] == lower) & (v15_boot["right"] == upper)
        ].iloc[0]
        pairwise.append(
            {
                "source": "V15 hierarchical, stored",
                "comparison": label,
                "mean_delta_log_score": row["mean_delta_log_score"],
                "ci_lower": -boot["ci_upper"],
                "ci_upper": -boot["ci_lower"],
                "verdict": verdict,
            }
        )
    gravity = pd.read_csv(
        config.ROOT
        / "paper_1"
        / "v15"
        / "phase2c_stored_faults"
        / "metrics"
        / "fault_variant_2x2.csv"
    )
    gravity_rows = gravity.loc[
        gravity["row_type"].isin(
            ["fixed-effect ladder", "V15 hierarchical reference"]
        )
    ].copy()
    gravity_rows.to_csv(
        ensure_output_path(config.OUTPUT / "metrics" / "a7_gravity_gradient.csv"),
        index=False,
    )
    paired = pd.DataFrame(pairwise)
    paired.to_csv(
        ensure_output_path(config.OUTPUT / "metrics" / "a7_paired_comparisons.csv"),
        index=False,
    )
    return domain_summary, paired, gravity_rows


def _partition_metric_table(
    predictions: pd.DataFrame,
    partition: str,
) -> pd.DataFrame:
    rows = [{"sample": "pooled", **metric_summary(predictions, "pooled")}]
    for sector, group in predictions.groupby("target_sector"):
        rows.append(
            {
                "sample": f"sector_{int(sector)}",
                **metric_summary(group, f"sector_{int(sector)}"),
            }
        )
    table = pd.DataFrame(rows)
    table.insert(0, "partition", partition)
    return table


def _partition_verdicts(
    frame: pd.DataFrame,
    partition_name: str,
    partition_column: str,
) -> pd.DataFrame:
    coefficient_path = config.OUTPUT / "coefficients" / "partition_loo_map.csv"
    coefficients = pd.read_csv(coefficient_path)
    coefficients = coefficients.loc[
        (coefficients["model"] == "T0")
        & (coefficients["partition"] == partition_name)
    ]
    sign_rows = []
    for fold, group in coefficients.groupby("fold"):
        for row in group.itertuples(index=False):
            if str(row.term).startswith("beta_"):
                sign_rows.append(
                    {
                        "estimation": "MAP partition leave-one-sector-out",
                        "term": row.term,
                        "sign": int(np.sign(row.estimate)),
                        "fits": 1,
                        "fold": int(fold),
                    }
                )
    sign = pd.DataFrame(sign_rows)
    if sign.empty:
        raise ValueError(f"No coefficients for partition {partition_name}")
    sign_counts = (
        sign.groupby(["estimation", "term", "sign"])
        .size()
        .rename("fits")
        .reset_index()
    )
    marginal = _marginal_auc_rows(frame, partition_column)
    marginal["partition"] = partition_name
    verdicts = _verdict_from_signs_and_auc(
        sign_counts,
        marginal,
        "MAP partition leave-one-sector-out",
    )
    verdicts.insert(0, "partition", partition_name)
    return verdicts


def analyse_a8(
    frame: pd.DataFrame,
    partitions: pd.DataFrame,
    primary_a1: pd.DataFrame,
    primary_verdicts: pd.DataFrame,
    a6: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    partition_predictions = _load_predictions("partition_loo_map.csv")
    metric_tables = []
    verdict_tables = [primary_verdicts.assign(partition="primary")]
    for partition_name in ("P_GAP", "P_POS"):
        group = partition_predictions.loc[
            partition_predictions["partition"] == partition_name
        ]
        metric_tables.append(_partition_metric_table(group, partition_name))
        verdict_tables.append(
            _partition_verdicts(
                frame,
                partition_name,
                f"fold_{partition_name.lower()}",
            )
        )
    metrics = pd.concat(metric_tables, ignore_index=True)
    metrics.to_csv(
        ensure_output_path(config.OUTPUT / "metrics" / "a8_partition_metrics.csv"),
        index=False,
    )
    verdicts = pd.concat(verdict_tables, ignore_index=True)
    verdicts.to_csv(
        ensure_output_path(config.OUTPUT / "metrics" / "a8_partition_verdicts.csv"),
        index=False,
    )

    p_pos_predictions = _load_predictions("partition_single_sector_map.csv")
    p_pos_predictions = p_pos_predictions.loc[
        p_pos_predictions["partition"] == "P_POS"
    ]
    pos_matrix = _matrix_metric_rows(
        p_pos_predictions,
        ["source_sector"],
        "single_sector",
        "P_POS",
    )
    pos_matrix.to_csv(
        ensure_output_path(config.OUTPUT / "metrics" / "a8_p_pos_transfer_matrix.csv"),
        index=False,
    )

    random_cv = a6.loc[
        a6["validation"] == "repeated stratified random 4-fold"
    ].iloc[0]
    optimism_rows = [
        {
            "partition": "primary",
            "random_cv_mean_log_score": float(random_cv["mean_log_score"]),
            "spatial_loo_mean_log_score": float(
                primary_a1.loc[primary_a1["sample"] == "pooled", "mean_log_score"].iloc[0]
            ),
        }
    ]
    for partition_name in ("P_GAP", "P_POS"):
        value = metrics.loc[
            (metrics["partition"] == partition_name)
            & (metrics["sample"] == "pooled"),
            "mean_log_score",
        ].iloc[0]
        optimism_rows.append(
            {
                "partition": partition_name,
                "random_cv_mean_log_score": float(random_cv["mean_log_score"]),
                "spatial_loo_mean_log_score": float(value),
            }
        )
    optimism = pd.DataFrame(optimism_rows)
    optimism["optimism_gap_random_minus_spatial"] = (
        optimism["random_cv_mean_log_score"]
        - optimism["spatial_loo_mean_log_score"]
    )
    optimism.to_csv(
        ensure_output_path(config.OUTPUT / "metrics" / "a8_optimism_gap.csv"),
        index=False,
    )

    primary_sector = primary_a1.loc[
        primary_a1["sample"].str.startswith("sector_"),
        ["sample", "mean_log_score"],
    ].set_index("sample")
    direction_rows = []
    for partition_name in ("P_GAP", "P_POS"):
        sector = metrics.loc[
            (metrics["partition"] == partition_name)
            & metrics["sample"].str.startswith("sector_"),
            ["sample", "mean_log_score"],
        ].set_index("sample")
        aligned = primary_sector.join(sector, lsuffix="_primary", rsuffix="_alternate")
        correlation = float(
            np.corrcoef(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
        )
        direction_rows.append(
            {
                "partition": partition_name,
                "sector_log_score_correlation_with_primary": correlation,
                "best_sector_primary": aligned.iloc[:, 0].idxmax(),
                "best_sector_alternate": aligned.iloc[:, 1].idxmax(),
            }
        )
    direction = pd.DataFrame(direction_rows)
    direction.to_csv(
        ensure_output_path(config.OUTPUT / "metrics" / "a8_partition_direction.csv"),
        index=False,
    )
    verdict_match = (
        verdicts.pivot(index="predictor", columns="partition", values="verdict")
        .nunique(axis=1)
        .eq(1)
    )
    robust = bool(verdict_match.all() and (direction["sector_log_score_correlation_with_primary"] > 0).all())
    _json_write(
        config.OUTPUT / "metrics" / "a8_conclusion.json",
        {
            "all_predictor_verdicts_match": bool(verdict_match.all()),
            "main_sector_direction_consistent": bool(
                (direction["sector_log_score_correlation_with_primary"] > 0).all()
            ),
            "partition_robust": robust,
            "oracle_decomposition": "primary partition only",
        },
    )
    return {
        "metrics": metrics,
        "verdicts": verdicts,
        "pos_matrix": pos_matrix,
        "optimism": optimism,
        "direction": direction,
    }


def make_figures(
    frame: pd.DataFrame,
    a2_wide: pd.DataFrame,
    decomposition: pd.DataFrame,
    aoa_cells: pd.DataFrame,
    coefficients: pd.DataFrame,
    marginal_auc: pd.DataFrame,
) -> list:
    import matplotlib.pyplot as plt

    figure_dir = config.OUTPUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for axis, metric, title in zip(
        axes, ("mean_log_score", "auc"), ("Mean log score", "ROC-AUC")
    ):
        table = a2_wide.loc[a2_wide["metric"] == metric].set_index("source_sector")
        matrix = table[[f"target_{index}" for index in range(1, 5)]].to_numpy(dtype=float)
        image = axis.imshow(matrix, cmap="viridis")
        axis.set_xticks(range(4), [f"Target {index}" for index in range(1, 5)])
        axis.set_yticks(range(4), [f"Source {index}" for index in range(1, 5)])
        axis.set_title(title)
        for row in range(4):
            for column in range(4):
                if row != column:
                    axis.text(
                        column,
                        row,
                        f"{matrix[row, column]:.3f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = figure_dir / "transfer_matrix.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    outputs.append(path)

    fig, axis = plt.subplots(figsize=(7, 4.5))
    width = 0.25
    indices = np.arange(4)
    for offset, column, label in (
        (-width, "base_minus_transfer", "base rate"),
        (0.0, "recal_minus_base", "slope calibration"),
        (width, "local_minus_recal", "local relationships"),
    ):
        axis.bar(indices + offset, decomposition[column], width=width, label=label)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(indices, [f"Sector {index}" for index in range(1, 5)])
    axis.set_ylabel("Log-score gap")
    axis.legend(frameon=False)
    axis.set_title("Oracle decomposition of transfer loss")
    fig.tight_layout()
    path = figure_dir / "decomposition_gaps.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    outputs.append(path)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for sector, axis in enumerate(axes.ravel(), start=1):
        group = aoa_cells.loc[aoa_cells["sector"] == sector]
        axis.scatter(
            group["centroid_x"],
            group["centroid_y"],
            c=group["AOA"].map({1: "#2a9d8f", 0: "#e76f51"}),
            s=8,
        )
        axis.set_title(
            f"Sector {sector}: outside={int((group['AOA'] == 0).sum())}/{len(group)}"
        )
        axis.set_aspect("equal")
    fig.suptitle("Area of applicability in held-out sectors")
    fig.tight_layout()
    path = figure_dir / "area_of_applicability.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    outputs.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    terms = [
        "beta_fault_z",
        "beta_fault_z2",
        "beta_contact_z",
        "beta_contact_z2",
        "beta_gravity_z",
    ]
    for term_index, term in enumerate(terms):
        subset = coefficients.loc[
            (coefficients["estimation"] == "MCMC leave-one-sector-out")
            & (coefficients["term"] == term)
        ]
        sources = subset["source"].tolist()
        x = term_index + np.linspace(-0.18, 0.18, len(subset))
        axes[0].errorbar(
            x,
            subset["estimate"],
            yerr=[
                subset["estimate"] - subset["ci_lower"],
                subset["ci_upper"] - subset["estimate"],
            ],
            fmt="o",
            alpha=0.75,
        )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(range(len(terms)), [term.replace("beta_", "") for term in terms], rotation=35, ha="right")
    axes[0].set_ylabel("Coefficient")
    axes[0].set_title("Leave-one-sector-out coefficient stability")
    for predictor, marker in (("fault", "o"), ("contact", "s"), ("gravity", "^")):
        subset = marginal_auc.loc[marginal_auc["predictor"] == predictor]
        axes[1].errorbar(
            subset["sector"],
            subset["auc"],
            yerr=[
                subset["auc"] - subset["auc_ci_lower"],
                subset["auc_ci_upper"] - subset["auc"],
            ],
            fmt=marker + "-",
            label=predictor,
        )
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xticks([1, 2, 3, 4])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Within-sector AUC")
    axes[1].legend(frameon=False)
    axes[1].set_title("Marginal predictor discrimination")
    fig.tight_layout()
    path = figure_dir / "coefficient_marginal_auc_stability.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    outputs.append(path)
    return outputs


def make_blocked_figures(
    aoa_cells: pd.DataFrame,
    coefficients: pd.DataFrame,
    marginal_auc: pd.DataFrame,
) -> list:
    """Figures that do not depend on the failed MAP approximation."""

    import matplotlib.pyplot as plt

    figure_dir = config.OUTPUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for sector, axis in enumerate(axes.ravel(), start=1):
        group = aoa_cells.loc[aoa_cells["sector"] == sector]
        axis.scatter(
            group["centroid_x"],
            group["centroid_y"],
            c=group["AOA"].map({1: "#2a9d8f", 0: "#e76f51"}),
            s=8,
        )
        axis.set_title(
            f"Sector {sector}: outside={int((group['AOA'] == 0).sum())}/{len(group)}"
        )
        axis.set_aspect("equal")
    fig.suptitle("Area of applicability in held-out sectors")
    fig.tight_layout()
    path = figure_dir / "area_of_applicability.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    outputs.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    terms = [
        "beta_fault_z",
        "beta_fault_z2",
        "beta_contact_z",
        "beta_contact_z2",
        "beta_gravity_z",
    ]
    for term_index, term in enumerate(terms):
        subset = coefficients.loc[
            (coefficients["estimation"] == "MCMC leave-one-sector-out")
            & (coefficients["term"] == term)
        ]
        x = term_index + np.linspace(-0.18, 0.18, len(subset))
        axes[0].errorbar(
            x,
            subset["estimate"],
            yerr=[
                subset["estimate"] - subset["ci_lower"],
                subset["ci_upper"] - subset["estimate"],
            ],
            fmt="o",
            alpha=0.75,
        )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(
        range(len(terms)),
        [term.replace("beta_", "") for term in terms],
        rotation=35,
        ha="right",
    )
    axes[0].set_ylabel("Coefficient")
    axes[0].set_title("MCMC coefficient stability")
    for predictor, marker in (("fault", "o"), ("contact", "s"), ("gravity", "^")):
        subset = marginal_auc.loc[marginal_auc["predictor"] == predictor]
        axes[1].errorbar(
            subset["sector"],
            subset["auc"],
            yerr=[
                subset["auc"] - subset["auc_ci_lower"],
                subset["auc_ci_upper"] - subset["auc"],
            ],
            fmt=marker + "-",
            label=predictor,
        )
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xticks([1, 2, 3, 4])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Within-sector AUC")
    axes[1].legend(frameon=False)
    axes[1].set_title("Marginal predictor discrimination")
    fig.tight_layout()
    path = figure_dir / "coefficient_marginal_auc_stability.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    outputs.append(path)
    return outputs

