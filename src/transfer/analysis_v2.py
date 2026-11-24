"""Amendment 1 analyses using full MCMC for every fit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.transfer import config
from src.transfer.analysis import (
    _marginal_auc_rows,
    _verdict_from_signs_and_auc,
    decomposition_gap_values,
)
from src.transfer.data import ensure_output_path, load_transfer_frame
from src.transfer.metrics import metric_summary, spatial_block_bootstrap_delta
from src.v15.metrics import log_score, roc_auc


def _load_amendment_predictions(prefix: str, batch: str | None = None) -> pd.DataFrame:
    paths = sorted(
        (config.OUTPUT / "predictions" / "amendment1").glob(f"{prefix}*.csv")
    )
    if prefix == "T0_mcmc_fold_":
        paths = sorted(
            (config.OUTPUT / "predictions").glob(f"{prefix}*.csv")
        )
    if not paths:
        raise FileNotFoundError(f"No Amendment 1 predictions for {prefix}")
    frames = []
    frame_info = pd.read_csv(
        config.MODEL_FRAME, usecols=["id", "centroid_x", "centroid_y", "area_km2"]
    )
    for path in paths:
        frame = pd.read_csv(path)
        if batch is not None and "batch" in frame.columns:
            frame = frame.loc[frame["batch"] == batch]
        if frame.empty:
            continue
        frames.append(frame.merge(frame_info, on="id", how="left", validate="many_to_one"))
    if not frames:
        raise FileNotFoundError(f"No predictions for prefix {prefix}, batch {batch}")
    return pd.concat(frames, ignore_index=True)


def _status_table() -> pd.DataFrame:
    rows = []
    for path in sorted((config.OUTPUT / "diagnostics" / "mcmc_fits").glob("*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    table = pd.DataFrame(rows)
    if not table.empty:
        for column in ("diverging", "max_rhat", "min_ess_bulk", "min_ess_tail"):
            if column not in table.columns:
                table[column] = np.nan
    return table


def _all_new_fits_converged() -> pd.DataFrame:
    status = _status_table()
    if status.empty:
        raise FileNotFoundError("No Amendment 1 fits found")
    failed = status.loc[~status["converged"].astype(bool)]
    if not failed.empty:
        raise RuntimeError(f"Amendment 1 convergence failures: {failed['fit_id'].tolist()}")
    return status


def a1_skill_table() -> pd.DataFrame:
    from src.transfer.analysis import analyse_a1

    return analyse_a1(load_transfer_frame())


def a2_mcmc_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    single = _load_amendment_predictions("a2_single_source_", batch="a2")
    pair = _load_amendment_predictions("a2_pair_", batch="a2")
    loo = _load_amendment_predictions("T0_mcmc_fold_")
    # Existing primary predictions use target_sector; standardize source keys.
    rows = []
    for (source, target), group in single.groupby(["source_sector", "target_sector"]):
        summary = metric_summary(group, f"target_{int(target)}")
        rows.append(
            {
                "source_type": "single_sector",
                "source_sector": int(source),
                "source_sector_a": np.nan,
                "source_sector_b": np.nan,
                "target_sector": int(target),
                "training_cells": int((frame[config.FOLD_COL] == int(source) - 1).sum()),
                "training_positives": int(
                    frame.loc[frame[config.FOLD_COL] == int(source) - 1, config.Y_COL].sum()
                ),
                **summary,
            }
        )
    for (source_a, source_b, target), group in pair.groupby(
        ["source_sector_a", "source_sector_b", "target_sector"]
    ):
        summary = metric_summary(group, f"target_{int(target)}")
        source_mask = frame[config.FOLD_COL].isin([int(source_a) - 1, int(source_b) - 1])
        rows.append(
            {
                "source_type": "sector_pair",
                "source_sector": np.nan,
                "source_sector_a": int(source_a),
                "source_sector_b": int(source_b),
                "target_sector": int(target),
                "training_cells": int(source_mask.sum()),
                "training_positives": int(frame.loc[source_mask, config.Y_COL].sum()),
                **summary,
            }
        )
    for target, group in loo.groupby("target_sector"):
        rows.append(
            {
                "source_type": "other_three_sectors",
                "source_sector": np.nan,
                "source_sector_a": np.nan,
                "source_sector_b": np.nan,
                "target_sector": int(target),
                "training_cells": 1404,
                "training_positives": int(frame.loc[frame[config.FOLD_COL] != int(target) - 1, config.Y_COL].sum()),
                **metric_summary(group, f"target_{int(target)}"),
            }
        )
    table = pd.DataFrame(rows)
    table["few_training_positives"] = table["training_positives"] < 15
    from src.transfer.partitions import build_partitions

    partition_view = (
        build_partitions(frame)[0].groupby("fold_primary")["axis_km"].mean()
    )
    table["source_axis_centroid_km"] = (table["source_sector"] - 1).map(partition_view)
    table["target_axis_centroid_km"] = (table["target_sector"] - 1).map(partition_view)
    table["sector_separation_km"] = (
        table["source_axis_centroid_km"] - table["target_axis_centroid_km"]
    ).abs()
    table.to_csv(
        ensure_output_path(config.OUTPUT / "metrics" / "a2_transfer_matrix_long.csv"),
        index=False,
    )
    wide_parts = []
    single_rows = table.loc[table["source_type"] == "single_sector"].copy()
    for metric in ("mean_log_score", "auc"):
        wide = single_rows.pivot(
            index="source_sector", columns="target_sector", values=metric
        )
        wide.columns = [f"target_{int(value)}" for value in wide.columns]
        wide = wide.reset_index()
        wide.insert(0, "metric", metric)
        wide_parts.append(wide)
    wide = pd.concat(wide_parts, ignore_index=True)
    wide.to_csv(
        ensure_output_path(config.OUTPUT / "metrics" / "a2_single_sector_matrices.csv"),
        index=False,
    )
    _write_a2_descriptors(table)
    return table, wide


def _write_a2_descriptors(table: pd.DataFrame) -> dict[str, Any]:
    singleton = table.loc[table["source_type"] == "single_sector"].copy()
    pair_scores = []
    for _, row in singleton.iterrows():
        reverse = singleton.loc[
            (singleton["source_sector"] == row["target_sector"])
            & (singleton["target_sector"] == row["source_sector"])
        ]
        if int(row["source_sector"]) < int(row["target_sector"]) and not reverse.empty:
            pair_scores.append(
                {
                    "source_a": int(row["source_sector"]),
                    "source_b": int(row["target_sector"]),
                    "a_to_b_log_score": float(row["mean_log_score"]),
                    "b_to_a_log_score": float(reverse.iloc[0]["mean_log_score"]),
                    "symmetry_abs_difference": float(
                        abs(row["mean_log_score"] - reverse.iloc[0]["mean_log_score"])
                    ),
                    "sector_separation_km": float(row["sector_separation_km"]),
                }
            )
    symmetry = pd.DataFrame(pair_scores)
    descriptors = {
        "mean_absolute_directional_log_score_difference": float(
            symmetry["symmetry_abs_difference"].mean()
        ),
        "separation_vs_mean_directional_score_correlation": float(
            np.corrcoef(
                symmetry["sector_separation_km"],
                symmetry[["a_to_b_log_score", "b_to_a_log_score"]].mean(axis=1),
            )[0, 1]
        ),
        "best_single_training_sector_by_target": {
            str(int(target)): int(
                group.sort_values("mean_log_score", ascending=False).iloc[0][
                    "source_sector"
                ]
            )
            for target, group in singleton.groupby("target_sector")
        },
        "mean_single_sector_log_score_by_training_sector": {
            str(int(source)): float(group["mean_log_score"].mean())
            for source, group in singleton.groupby("source_sector")
        },
    }
    (config.OUTPUT / "metrics" / "a2_descriptor.json").write_text(
        json.dumps(descriptors, indent=2) + "\n", encoding="utf-8"
    )
    return descriptors


def a3_mcmc_decomposition(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    loo = _load_amendment_predictions("T0_mcmc_fold_")
    local = _load_amendment_predictions("a3_local_", batch="a3_local")
    local = local.groupby("id", as_index=False).agg(
        y=("y", "first"),
        prob_local=("prob", "mean"),
        target_sector=("target_sector", "first"),
    )
    base = _load_amendment_predictions("a3_oracle_base_", batch="a3_oracle")
    recal = _load_amendment_predictions("a3_oracle_recal_", batch="a3_oracle")
    rows = []
    cells = []
    for sector in range(1, 5):
        transfer = loo.loc[loo["target_sector"] == sector].sort_values("id")
        local_sector = local.loc[local["target_sector"] == sector].sort_values("id")
        base_sector = base.loc[base["target_sector"] == sector].sort_values("id")
        recal_sector = recal.loc[recal["target_sector"] == sector].sort_values("id")
        ids = transfer["id"].tolist()
        for label, table in (
            ("local", local_sector),
            ("base", base_sector),
            ("recal", recal_sector),
        ):
            if table["id"].tolist() != ids:
                raise ValueError(f"A3 {label} ID mismatch in sector {sector}")
        scores = {
            "transfer": log_score(transfer["y"], transfer["prob"]),
            "base": log_score(base_sector["y"], base_sector["prob"]),
            "recal": log_score(recal_sector["y"], recal_sector["prob"]),
            "local": log_score(local_sector["y"], local_sector["prob_local"]),
        }
        gaps = decomposition_gap_values(
            scores["transfer"], scores["base"], scores["recal"], scores["local"]
        )
        source = transfer[["id", "centroid_x", "centroid_y"]].copy()
        intervals = {
            "base_minus_transfer": spatial_block_bootstrap_delta(
                source,
                scores["base"] - scores["transfer"],
                np.zeros(len(source)),
                seed=config.BOOTSTRAP_SEED + sector,
            ),
            "recal_minus_base": spatial_block_bootstrap_delta(
                source,
                scores["recal"] - scores["base"],
                np.zeros(len(source)),
                seed=config.BOOTSTRAP_SEED + 10 + sector,
            ),
            "local_minus_recal": spatial_block_bootstrap_delta(
                source,
                scores["local"] - scores["recal"],
                np.zeros(len(source)),
                seed=config.BOOTSTRAP_SEED + 20 + sector,
            ),
        }
        rows.append(
            {
                "sector": sector,
                "n": len(transfer),
                "positives": int(transfer["y"].sum()),
                "L_transfer": float(scores["transfer"].mean()),
                "L_base": float(scores["base"].mean()),
                "L_recal": float(scores["recal"].mean()),
                "L_local": float(scores["local"].mean()),
                "base_minus_transfer": gaps[0],
                "base_minus_transfer_ci_lower": intervals["base_minus_transfer"][0],
                "base_minus_transfer_ci_upper": intervals["base_minus_transfer"][1],
                "recal_minus_base": gaps[1],
                "recal_minus_base_ci_lower": intervals["recal_minus_base"][0],
                "recal_minus_base_ci_upper": intervals["recal_minus_base"][1],
                "local_minus_recal": gaps[2],
                "local_minus_recal_ci_lower": intervals["local_minus_recal"][0],
                "local_minus_recal_ci_upper": intervals["local_minus_recal"][1],
            }
        )
        cells.append(
            transfer.assign(
                sector=sector,
                L_transfer=scores["transfer"],
                L_base=scores["base"],
                L_recal=scores["recal"],
                L_local=scores["local"],
                base_minus_transfer=scores["base"] - scores["transfer"],
                recal_minus_base=scores["recal"] - scores["base"],
                local_minus_recal=scores["local"] - scores["recal"],
            )
        )
    result = pd.DataFrame(rows)
    cell_table = pd.concat(cells, ignore_index=True)
    result.to_csv(
        ensure_output_path(config.OUTPUT / "metrics" / "a3_decomposition.csv"),
        index=False,
    )
    cell_table.to_csv(
        ensure_output_path(config.OUTPUT / "metrics" / "a3_decomposition_cells.csv"),
        index=False,
    )
    return result, cell_table


def a5_mcmc(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mcmc = pd.read_csv(config.OUTPUT / "coefficients" / "mcmc_T0.csv")
    single_paths = sorted(
        (config.OUTPUT / "coefficients" / "amendment1").glob(
            "a2_single_source_*.csv"
        )
    )
    if not single_paths:
        raise FileNotFoundError("A2 single-sector MCMC coefficients are missing")
    singles = pd.concat([pd.read_csv(path) for path in single_paths], ignore_index=True)
    rows = []
    for fold, group in mcmc.groupby("fold"):
        for row in group.to_dict("records"):
            if str(row["parameter"]).startswith("beta_"):
                rows.append(
                    {
                        "estimation": "MCMC leave-one-sector-out",
                        "source": f"fold_{int(fold)}",
                        "term": row["parameter"],
                        "estimate": float(row["median"]),
                        "ci_lower": float(row["hdi_2.5%"]),
                        "ci_upper": float(row["hdi_97.5%"]),
                    }
                )
    for row in singles.to_dict("records"):
        if str(row["parameter"]).startswith("beta_"):
            rows.append(
                {
                    "estimation": "MCMC single-sector",
                    "source": f"sector_{int(row['source_sector'])}",
                    "term": row["parameter"],
                    "estimate": float(row["median"]),
                    "ci_lower": float(row["hdi_2.5%"]),
                    "ci_upper": float(row["hdi_97.5%"]),
                }
            )
    coefficients = pd.DataFrame(rows)
    sign = (
        coefficients.assign(sign=np.sign(coefficients["estimate"]).astype(int))
        .groupby(["estimation", "term", "sign"])
        .size()
        .rename("fits")
        .reset_index()
    )
    marginal = _marginal_auc_rows(frame, config.FOLD_COL)
    marginal["partition"] = "primary"
    verdicts = _verdict_from_signs_and_auc(
        sign, marginal, "MCMC leave-one-sector-out"
    )
    coefficients.to_csv(
        ensure_output_path(config.OUTPUT / "coefficients" / "a5_coefficient_stability.csv"),
        index=False,
    )
    sign.to_csv(
        ensure_output_path(config.OUTPUT / "coefficients" / "a5_sign_agreement.csv"),
        index=False,
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


def sensitivity_metrics() -> pd.DataFrame:
    predictions = _load_amendment_predictions(
        "sensitivity_", batch="sensitivities"
    )
    rows = []
    for model, group in predictions.groupby("model"):
        rows.append(
            {
                "model": model,
                "sample": "pooled",
                **metric_summary(group, "pooled"),
            }
        )
        for fold, fold_group in group.groupby("fold"):
            rows.append(
                {
                    "model": model,
                    "sample": f"fold_{int(fold)}",
                    **metric_summary(fold_group, f"fold_{int(fold)}"),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(
        ensure_output_path(config.OUTPUT / "metrics" / "sensitivity_metrics.csv"),
        index=False,
    )
    return table


def a6_mcmc(frame: pd.DataFrame, a1: pd.DataFrame) -> pd.DataFrame:
    predictions = _load_amendment_predictions("a6_", batch="a6")
    repeat_rows = []
    for repeat, group in predictions.groupby("repeat"):
        y = group["y"].to_numpy(dtype=int)
        probability = group["prob"].to_numpy(dtype=float)
        repeat_rows.append(
            {
                "repeat": int(repeat),
                "mean_log_score": float(log_score(y, probability).mean()),
                "auc": roc_auc(y, probability),
            }
        )
    repeats = pd.DataFrame(repeat_rows)
    loo = a1.loc[a1["sample"] == "pooled"].iloc[0]
    summary = pd.DataFrame(
        [
            {
                "validation": "10 repeats, stratified random 4-fold MCMC",
                "mean_log_score": repeats["mean_log_score"].mean(),
                "log_score_sd": repeats["mean_log_score"].std(ddof=1),
                "log_score_q2_5": repeats["mean_log_score"].quantile(0.025),
                "log_score_q97_5": repeats["mean_log_score"].quantile(0.975),
                "auc": repeats["auc"].mean(),
                "auc_sd": repeats["auc"].std(ddof=1),
                "auc_q2_5": repeats["auc"].quantile(0.025),
                "auc_q97_5": repeats["auc"].quantile(0.975),
            },
            {
                "validation": "leave-one-sector-out MCMC",
                "mean_log_score": loo["mean_log_score"],
                "log_score_sd": np.nan,
                "log_score_q2_5": loo["mean_log_score_block_ci_lower"],
                "log_score_q97_5": loo["mean_log_score_block_ci_upper"],
                "auc": loo["auc"],
                "auc_sd": np.nan,
                "auc_q2_5": loo["auc_ci_lower"],
                "auc_q97_5": loo["auc_ci_upper"],
            },
        ]
    )
    buffered = pd.read_csv(config.V15 / "metrics" / "buffered_validation_metrics.csv")
    for row in buffered.loc[
        (buffered["model"] == "M0_global") & (buffered["sample"] == "pooled")
    ].itertuples(index=False):
        summary.loc[len(summary)] = {
            "validation": f"V15 M0 buffered {row.buffer_km:g} km (different model)",
            "mean_log_score": row.mean_log_score,
            "log_score_sd": np.nan,
            "log_score_q2_5": np.nan,
            "log_score_q97_5": np.nan,
            "auc": row.auc,
            "auc_sd": np.nan,
            "auc_q2_5": np.nan,
            "auc_q97_5": np.nan,
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


def a8_mcmc(
    frame: pd.DataFrame,
    a1: pd.DataFrame,
    a6: pd.DataFrame,
    primary_verdicts: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    partitions = pd.read_csv(
        config.OUTPUT / "partitions" / "transfer_partitions.csv"
    ).set_index("id")
    frame = frame.copy()
    frame["fold_p_gap"] = frame["id"].map(partitions["fold_p_gap"])
    frame["fold_p_pos"] = frame["id"].map(partitions["fold_p_pos"])
    predictions = _load_amendment_predictions("a8_", batch="a8")
    metric_tables = []
    verdict_tables = [primary_verdicts.assign(partition="primary")]
    coefficient_rows = []
    for path in sorted(
        (config.OUTPUT / "coefficients" / "amendment1").glob("a8_*.csv")
    ):
        coefficient_rows.append(pd.read_csv(path))
    coefficients = pd.concat(coefficient_rows, ignore_index=True)
    coefficients.to_csv(
        ensure_output_path(config.OUTPUT / "coefficients" / "a8_all_coefficients.csv"),
        index=False,
    )
    for partition_name in ("P_GAP", "P_POS"):
        loo = predictions.loc[
            (predictions["partition"] == partition_name)
            & predictions["fit_id"].str.contains("_loo_"),
            :,
        ]
        rows = [{"sample": "pooled", **metric_summary(loo, "pooled")}]
        for sector, group in loo.groupby("target_sector"):
            rows.append(
                {"sample": f"sector_{int(sector)}", **metric_summary(group, f"sector_{int(sector)}")}
            )
        metric = pd.DataFrame(rows)
        metric.insert(0, "partition", partition_name)
        metric_tables.append(metric)
        sign_rows = []
        partition_coeff = coefficients.loc[
            (coefficients["partition"] == partition_name)
            & coefficients["fit_id"].str.contains("_loo_"),
            :,
        ]
        for row in partition_coeff.to_dict("records"):
            if str(row["parameter"]).startswith("beta_"):
                sign_rows.append(
                    {
                        "estimation": "MCMC partition leave-one-sector-out",
                        "term": row["parameter"],
                        "sign": int(np.sign(row["median"])),
                        "fits": 1,
                    }
                )
        sign = pd.DataFrame(sign_rows)
        sign = sign.groupby(["estimation", "term", "sign"]).size().rename("fits").reset_index()
        column = "fold_p_gap" if partition_name == "P_GAP" else "fold_p_pos"
        marginal = _marginal_auc_rows(frame, column)
        marginal["partition"] = partition_name
        verdict = _verdict_from_signs_and_auc(
            sign, marginal, "MCMC partition leave-one-sector-out"
        )
        verdict.insert(0, "partition", partition_name)
        verdict_tables.append(verdict)
    metrics = pd.concat(metric_tables, ignore_index=True)
    verdicts = pd.concat(verdict_tables, ignore_index=True)
    metrics.to_csv(
        ensure_output_path(config.OUTPUT / "metrics" / "a8_partition_metrics.csv"),
        index=False,
    )
    verdicts.to_csv(
        ensure_output_path(config.OUTPUT / "metrics" / "a8_partition_verdicts.csv"),
        index=False,
    )

    p_pos_single = predictions.loc[
        (predictions["partition"] == "P_POS")
        & predictions["fit_id"].str.contains("_single_source_"),
        :,
    ]
    matrix_rows = []
    for (source, target), group in p_pos_single.groupby(["source_sector", "target_sector"]):
        summary = metric_summary(group, f"target_{int(target)}")
        matrix_rows.append(
            {
                "source_sector": int(source),
                "target_sector": int(target),
                **summary,
            }
        )
    matrix = pd.DataFrame(matrix_rows)
    matrix.to_csv(
        ensure_output_path(config.OUTPUT / "metrics" / "a8_p_pos_transfer_matrix.csv"),
        index=False,
    )

    random_row = a6.loc[a6["validation"].str.startswith("10 repeats")].iloc[0]
    optimism_rows = []
    primary_value = a1.loc[a1["sample"] == "pooled", "mean_log_score"].iloc[0]
    for partition_name, spatial_value in [
        ("primary", primary_value),
        *[
            (
                name,
                metrics.loc[
                    (metrics["partition"] == name) & (metrics["sample"] == "pooled"),
                    "mean_log_score",
                ].iloc[0],
            )
            for name in ("P_GAP", "P_POS")
        ],
    ]:
        optimism_rows.append(
            {
                "partition": partition_name,
                "random_cv_mean_log_score": random_row["mean_log_score"],
                "spatial_loo_mean_log_score": spatial_value,
                "optimism_gap_random_minus_spatial": (
                    random_row["mean_log_score"] - spatial_value
                ),
            }
        )
    optimism = pd.DataFrame(optimism_rows)
    optimism.to_csv(
        ensure_output_path(config.OUTPUT / "metrics" / "a8_optimism_gap.csv"),
        index=False,
    )
    primary_sector = a1.loc[
        a1["sample"].str.startswith("sector_"), ["sample", "mean_log_score"]
    ].set_index("sample")
    direction_rows = []
    for partition_name in ("P_GAP", "P_POS"):
        sector = metrics.loc[
            (metrics["partition"] == partition_name)
            & metrics["sample"].str.startswith("sector_"),
            ["sample", "mean_log_score"],
        ].set_index("sample")
        aligned = primary_sector.join(sector, lsuffix="_primary", rsuffix="_alternate")
        direction_rows.append(
            {
                "partition": partition_name,
                "sector_log_score_correlation_with_primary": float(
                    np.corrcoef(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
                ),
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
    result = {
        "all_predictor_verdicts_match": bool(verdict_match.all()),
        "main_sector_direction_consistent": bool(
            (direction["sector_log_score_correlation_with_primary"] > 0).all()
        ),
        "partition_robust": bool(
            verdict_match.all()
            and (direction["sector_log_score_correlation_with_primary"] > 0).all()
        ),
        "oracle_decomposition": "primary partition only",
    }
    (config.OUTPUT / "metrics" / "a8_conclusion.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "metrics": metrics,
        "verdicts": verdicts,
        "pos_matrix": matrix,
        "optimism": optimism,
        "direction": direction,
    }


