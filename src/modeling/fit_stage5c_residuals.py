"""Stage 5C residual spatial diagnostics and buffered transfer experiments.

The accepted Stage 4 S0-without-gravity count model remains the reference. This
stage does not fit a detection surface and does not introduce domain-varying
coefficients. It characterises residual spatial dependence, tests whether it is
stable across spatial weights and partitions, and evaluates fixed-basis spatial
comparators under out-of-fold prediction.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modeling.fit_stage4_models import (  # noqa: E402
    Experiment,
    fit_experiment,
)
from src.modeling.stage4_common import (  # noqa: E402
    PROJECT_CRS,
    abs_path,
    build_id,
    file_record,
    relative_posix,
    sha256_file,
    software_versions,
    verify_protected,
    write_csv,
    write_gpkg,
    write_json,
)
from src.modeling.stage5_common import (  # noqa: E402
    markdown_table,
    software_stack,
    write_text,
)
from src.modeling.stage5c_common import (  # noqa: E402
    add_spatial_features,
    build_partition_frame,
    informative_fold_mean,
    load_config,
    load_verified_inputs,
    local_moran,
    moran_test,
    output_paths,
    partition_definition,
    randomized_quantile_residuals,
    residual_summaries,
    spatial_weights,
    weight_specs,
)


PALETTE = {
    "reference": "#1b7f79",
    "N0_AREA": "#9a9a9a",
    "S0_NO_GRAVITY": "#1b7f79",
    "S0_BELT_LINEAR": "#4778b5",
    "S0_BELT_CUBIC": "#6c63a6",
    "S0_2D_QUADRATIC": "#c67c2e",
    "S0_2D_RBF12": "#b84a4a",
}


def ordered_uniform_draws(frame: pd.DataFrame, seed: int) -> pd.Series:
    ordered = frame.sort_values("support_id").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    return pd.Series(rng.uniform(size=len(ordered)), index=ordered["support_id"], dtype=float)


def add_rqr(frame: pd.DataFrame, uniforms: pd.Series) -> pd.DataFrame:
    output = frame.copy()
    output["rqr_residual"] = randomized_quantile_residuals(
        output["observed_count"].to_numpy(dtype=float),
        output["predicted_count"].to_numpy(dtype=float),
        output["theta"].to_numpy(dtype=float),
        output["support_id"].map(uniforms).to_numpy(dtype=float),
    )
    return output


def fit_partitions(
    frame: pd.DataFrame,
    bundle: dict[str, Any],
    config: dict[str, Any],
    fit_config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]], dict[str, Any]]:
    reference_fold = bundle["model_table"].set_index("support_id")["new_fold"]
    reference_predictions = bundle["predictions"].loc[
        bundle["predictions"]["model_id"].eq(config["reference"]["model_id"])
    ].copy()
    reference_predictions = reference_predictions.set_index("support_id")
    informative_min = int(config["residuals"]["informative_fold_min_positive_cells"])
    outputs: dict[str, pd.DataFrame] = {}
    frames: dict[str, pd.DataFrame] = {}
    reproduction: dict[str, Any] = {}
    all_definitions: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []
    all_fold_metrics: list[pd.DataFrame] = []
    for partition in config["partitions"]:
        partition_frame = build_partition_frame(
            frame,
            partition,
            reference_fold if partition["family"] == "belt_axis" else None,
        )
        frames[partition["id"]] = partition_frame
        definition = partition_definition(partition_frame, partition, informative_min)
        all_definitions.append(definition)
        experiment = Experiment(
            id=f"S0_{partition['id']}",
            label=f"S0 {partition['label']}",
            predictors=tuple(config["reference"]["predictors"]),
            outcome=config["reference"]["outcome"],
            set_name="stage5c_partition",
        )
        predictions, metrics, coefficients, fold_metrics = fit_experiment(
            partition_frame, experiment, fit_config
        )
        for output in (predictions, metrics, fold_metrics, coefficients):
            output["partition_id"] = partition["id"]
            output["partition_label"] = partition["label"]
            output["partition_family"] = partition["family"]
            output["partition_role"] = partition["role"]
            output["partition_buffer_km"] = float(partition["buffer_km"])
        fold_metrics = fold_metrics.merge(
            definition[[
                "partition_id", "fold", "n_train", "n_buffer_excluded",
                "buffer_excluded_records", "train_positive_cells", "informative_fold",
            ]],
            on=["partition_id", "fold"],
            how="left",
            suffixes=("", "_partition"),
        )
        metrics = metrics.copy()
        informative = informative_fold_mean(fold_metrics, informative_min)
        metrics["informative_fold_count"] = int(fold_metrics["informative_fold"].sum())
        for key, value in informative.items():
            metrics[key] = value
        all_predictions.append(predictions)
        all_metrics.append(metrics)
        all_fold_metrics.append(fold_metrics)
        outputs[f"partition_{partition['id']}_fold_metrics"] = fold_metrics
        outputs[f"partition_{partition['id']}_coefficients"] = coefficients
        if partition["role"] == "reference":
            joined = predictions.set_index("support_id").join(
                reference_predictions, rsuffix="_stage4"
            )
            reproduction["reference_partition"] = {
                "n": len(joined),
                "max_abs_predicted_count_difference": float(
                    np.max(np.abs(joined["predicted_count"] - joined["predicted_count_stage4"]))
                ),
                "max_abs_probability_difference": float(
                    np.max(np.abs(
                        joined["predicted_any_probability"]
                        - joined["predicted_any_probability_stage4"]
                    ))
                ),
                "max_abs_pearson_residual_difference": float(
                    np.max(np.abs(joined["pearson_residual"] - joined["pearson_residual_stage4"]))
                ),
                "same_fold": bool(np.all(
                    joined["new_fold"].to_numpy() == joined["new_fold_stage4"].to_numpy()
                )),
            }
    outputs["partition_definitions"] = pd.concat(all_definitions, ignore_index=True)
    outputs["partition_predictions"] = pd.concat(all_predictions, ignore_index=True)
    outputs["partition_metrics"] = pd.concat(all_metrics, ignore_index=True)
    outputs["partition_fold_metrics"] = pd.concat(all_fold_metrics, ignore_index=True)
    return outputs, frames, reproduction


def fit_spatial_models(
    frame: pd.DataFrame,
    config: dict[str, Any],
    fit_config: dict[str, Any],
    stage4_reference: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    predictions: list[pd.DataFrame] = []
    metrics: list[pd.DataFrame] = []
    coefficients: list[pd.DataFrame] = []
    fold_metrics: list[pd.DataFrame] = []
    reproduction: dict[str, Any] = {}
    for index, specification in enumerate(config["spatial_models"]):
        predictors = tuple(config["reference"]["predictors"]) + tuple(specification["terms"])
        experiment = Experiment(
            id=specification["id"],
            label=specification["label"],
            predictors=predictors,
            outcome=config["reference"]["outcome"],
            set_name="stage5c_spatial_comparator",
        )
        pred, metric, coef, fold = fit_experiment(frame, experiment, fit_config)
        for output in (pred, metric, fold):
            output["spatial_terms"] = len(specification["terms"])
            output["total_terms"] = len(predictors)
            output["model_family"] = (
                "nonspatial_reference" if not specification["terms"]
                else "fixed_basis_spatial_comparator"
            )
        coef["spatial_terms"] = len(specification["terms"])
        coef["total_terms"] = len(predictors)
        informative = informative_fold_mean(
            fold, int(config["residuals"]["informative_fold_min_positive_cells"])
        )
        metric["informative_fold_count"] = int(
            fold["holdout_positive_cells"].ge(
                int(config["residuals"]["informative_fold_min_positive_cells"])
            ).sum()
        )
        for key, value in informative.items():
            metric[key] = value
        predictions.append(pred)
        metrics.append(metric)
        coefficients.append(coef)
        fold_metrics.append(fold)
        if specification["id"] == "S0_REFERENCE":
            joined = pred.set_index("support_id").join(
                stage4_reference.set_index("support_id"), rsuffix="_stage4"
            )
            reproduction = {
                "n": len(joined),
                "max_abs_predicted_count_difference": float(
                    np.max(np.abs(joined["predicted_count"] - joined["predicted_count_stage4"]))
                ),
                "max_abs_probability_difference": float(
                    np.max(np.abs(
                        joined["predicted_any_probability"]
                        - joined["predicted_any_probability_stage4"]
                    ))
                ),
                "max_abs_pearson_residual_difference": float(
                    np.max(np.abs(joined["pearson_residual"] - joined["pearson_residual_stage4"]))
                ),
            }
    return {
        "spatial_predictions": pd.concat(predictions, ignore_index=True),
        "spatial_metrics": pd.concat(metrics, ignore_index=True),
        "spatial_coefficients": pd.concat(coefficients, ignore_index=True),
        "spatial_fold_metrics": pd.concat(fold_metrics, ignore_index=True),
    }, reproduction


def build_moran_tables(
    baseline_predictions: pd.DataFrame,
    spatial_predictions: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    residual_config = config["residuals"]
    specs = weight_specs(config)
    baseline = baseline_predictions.loc[
        baseline_predictions["model_id"].isin(["N0_AREA", "S0_NO_GRAVITY"])
    ].copy()
    spatial_ids = [
        value for value in config["spatial_models"]
        if value["id"] != "S0_REFERENCE"
    ]
    spatial_ids = [value["id"] for value in spatial_ids]
    rows: list[dict[str, Any]] = []
    scopes = [("pooled_oof", baseline)] + [
        (f"fold_{fold}", baseline.loc[baseline["new_fold"].eq(fold)])
        for fold in (1, 2, 3, 4)
    ]
    primary_spec = next(spec for spec in specs if spec.id == residual_config["primary_weight"])
    for scope_index, (scope_name, scope_frame) in enumerate(scopes):
        coordinate_frame = scope_frame.loc[
            scope_frame["model_id"].eq("S0_NO_GRAVITY"), ["x_m", "y_m"]
        ]
        coordinates = coordinate_frame.to_numpy(dtype=float)
        for spec_index, spec in enumerate(specs):
            weights, metadata = spatial_weights(coordinates, spec)
            for model_index, model_id in enumerate(["N0_AREA", "S0_NO_GRAVITY"]):
                subset = scope_frame.loc[scope_frame["model_id"].eq(model_id)]
                seed = int(residual_config["seed"]) + scope_index * 10007 + spec_index * 101 + model_index
                test = moran_test(
                    subset["pearson_residual"].to_numpy(dtype=float),
                    weights,
                    int(residual_config["n_permutations"]),
                    seed,
                )
                test.update(metadata)
                test.update({
                    "model_id": model_id,
                    "model_class": "baseline_count_model",
                    "evaluation_scope": scope_name,
                    "residual_type": "Pearson count residual",
                    "n_permutations": int(residual_config["n_permutations"]),
                    "seed": seed,
                })
                rows.append(test)
            if scope_name == "pooled_oof":
                pooled_spatial = spatial_predictions.loc[
                    spatial_predictions["model_id"].isin(spatial_ids)
                ]
                for model_index, model_id in enumerate(spatial_ids):
                    subset = pooled_spatial.loc[pooled_spatial["model_id"].eq(model_id)]
                    seed = (
                        int(residual_config["seed"])
                        + 900001
                        + spec_index * 101
                        + model_index
                    )
                    test = moran_test(
                        subset["pearson_residual"].to_numpy(dtype=float),
                        weights,
                        int(residual_config["n_permutations"]),
                        seed,
                    )
                    test.update(metadata)
                    test.update({
                        "model_id": model_id,
                        "model_class": "fixed_basis_spatial_comparator",
                        "evaluation_scope": scope_name,
                        "residual_type": "Pearson count residual",
                        "n_permutations": int(residual_config["n_permutations"]),
                        "seed": seed,
                    })
                    rows.append(test)
        if scope_name in {"pooled_oof", "fold_1", "fold_2", "fold_3", "fold_4"}:
            primary_weights, primary_metadata = spatial_weights(coordinates, primary_spec)
            for model_id in ["N0_AREA", "S0_NO_GRAVITY"]:
                subset = scope_frame.loc[scope_frame["model_id"].eq(model_id)]
                test = moran_test(
                    subset["rqr_residual"].to_numpy(dtype=float),
                    primary_weights,
                    int(residual_config["n_permutations"]),
                    int(residual_config["seed"]) + scope_index * 10007 + 5001,
                )
                test.update(primary_metadata)
                test.update({
                    "model_id": model_id,
                    "model_class": "baseline_count_model",
                    "evaluation_scope": scope_name,
                    "residual_type": "randomized quantile residual",
                    "n_permutations": int(residual_config["n_permutations"]),
                    "seed": int(residual_config["seed"]) + scope_index * 10007 + 5001,
                })
                rows.append(test)
    pooled_spatial = spatial_predictions.loc[
        spatial_predictions["model_id"].isin(spatial_ids)
    ]
    coordinate_frame = pooled_spatial.drop_duplicates("support_id")[["x_m", "y_m"]]
    coordinates = coordinate_frame.to_numpy(dtype=float)
    primary_weights, primary_metadata = spatial_weights(coordinates, primary_spec)
    for model_index, model_id in enumerate(spatial_ids):
        subset = pooled_spatial.loc[pooled_spatial["model_id"].eq(model_id)]
        test = moran_test(
            subset["rqr_residual"].to_numpy(dtype=float),
            primary_weights,
            int(residual_config["n_permutations"]),
            int(residual_config["seed"]) + 900001 + model_index,
        )
        test.update(primary_metadata)
        test.update({
            "model_id": model_id,
            "model_class": "fixed_basis_spatial_comparator",
            "evaluation_scope": "pooled_oof",
            "residual_type": "randomized quantile residual",
            "n_permutations": int(residual_config["n_permutations"]),
            "seed": int(residual_config["seed"]) + 900001 + model_index,
        })
        rows.append(test)
    table = pd.DataFrame(rows)
    summary = (
        table.groupby(
            ["model_id", "model_class", "evaluation_scope", "residual_type"],
            as_index=False,
        )
        .agg(
            n_weight_specifications=("spatial_weight", "nunique"),
            moran_i_min=("moran_i", "min"),
            moran_i_median=("moran_i", "median"),
            moran_i_max=("moran_i", "max"),
            z_permutation_min=("z_permutation", "min"),
            z_permutation_max=("z_permutation", "max"),
            min_p_positive=("p_positive", "min"),
            max_p_positive=("p_positive", "max"),
        )
    )
    local_frame = spatial_predictions.loc[
        spatial_predictions["model_id"].eq("S0_REFERENCE")
    ].sort_values("support_id").reset_index(drop=True)
    local_weights, local_metadata = spatial_weights(
        local_frame[["x_m", "y_m"]].to_numpy(dtype=float), primary_spec
    )
    local = local_moran(
        local_frame,
        local_weights,
        int(residual_config["local_permutations"]),
        int(residual_config["seed"]) + 7777,
        float(residual_config["local_fdr"]),
    )
    local_metadata["macro_moran_i"] = float(local["local_moran_i"].mean())
    local_metadata["fdr"] = float(residual_config["local_fdr"])
    local_metadata["fdr_method"] = "normal-score Benjamini-Hochberg with permutation p<0.05 screen"
    local_metadata["permutation_screen"] = 0.05
    local_metadata["n_fdr_significant"] = int(local["significant_fdr"].sum())
    local_metadata["cluster_counts"] = {
        str(key): int(value)
        for key, value in local["cluster_class"].value_counts().sort_index().items()
    }
    return table, summary, local, local_metadata


def spatial_feature_diagnostics(
    frame: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    features = sorted({
        term for model in config["spatial_models"] for term in model["terms"]
    })
    base = config["reference"]["predictors"]
    rows: list[dict[str, Any]] = []
    correlation = frame[features].corr().abs()
    for feature in features:
        values = frame[feature].to_numpy(dtype=float)
        others = correlation.loc[feature].drop(labels=[feature])
        row: dict[str, Any] = {
            "feature": feature,
            "n": len(values),
            "min": float(np.min(values)),
            "q01": float(np.quantile(values, 0.01)),
            "median": float(np.median(values)),
            "q99": float(np.quantile(values, 0.99)),
            "max": float(np.max(values)),
            "sd": float(np.std(values, ddof=0)),
            "max_abs_correlation_with_other_spatial_feature": float(others.max())
            if len(others) else math.nan,
        }
        for predictor in base:
            if predictor == "BASE_CONTACT_CLASS":
                row[f"corr_with_{predictor}"] = math.nan
            else:
                row[f"corr_with_{predictor}"] = float(frame[[feature, predictor]].corr().iloc[0, 1])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_partition_maps(
    frames: dict[str, pd.DataFrame], geometries: dict[str, Any], config: dict[str, Any], path: str
) -> None:
    partitions = config["partitions"]
    n_columns = 2
    n_rows = int(math.ceil(len(partitions) / n_columns))
    figure, axes = plt.subplots(n_rows, n_columns, figsize=(13, 6.2 * n_rows))
    axes = np.atleast_1d(axes).reshape(-1)
    base = geometries["catchments"][["support_id", "geometry"]].copy()
    colors = {1: "#2166ac", 2: "#67a9cf", 3: "#ef8a62", 4: "#b2182b"}
    for axis, partition in zip(axes, partitions):
        frame = frames[partition["id"]][["support_id", "new_fold"]]
        layer = base.merge(frame, on="support_id", how="left")
        layer["color"] = layer["new_fold"].map(colors)
        layer.plot(color=layer["color"], linewidth=0, ax=axis)
        geometries["tracts"].boundary.plot(ax=axis, color="#222222", linewidth=0.6)
        axis.set_title(
            f"{partition['id']}\n{partition['label']}", fontsize=10, loc="left"
        )
        axis.set_aspect("equal")
        axis.set_axis_off()
        handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[fold],
                   markeredgecolor="none", markersize=9, label=f"Fold {fold}")
            for fold in sorted(colors)
        ]
        axis.legend(handles=handles, loc="lower left", frameon=False, fontsize=8)
    for axis in axes[len(partitions):]:
        axis.set_axis_off()
    figure.suptitle("Stage 5C spatial partitions and buffers", fontsize=14, y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_partition_transfer(
    fold_metrics: pd.DataFrame, config: dict[str, Any], path: str
) -> None:
    partition_order = [value["id"] for value in config["partitions"]]
    folds = [1, 2, 3, 4]
    auc = np.full((len(partition_order), len(folds)), np.nan, dtype=float)
    ap_ratio = np.full_like(auc, np.nan)
    annotations: list[list[str]] = [["" for _ in folds] for _ in partition_order]
    for row_index, partition_id in enumerate(partition_order):
        for column_index, fold in enumerate(folds):
            subset = fold_metrics.loc[
                fold_metrics["partition_id"].eq(partition_id)
                & fold_metrics["fold"].eq(fold)
            ]
            if subset.empty:
                continue
            row = subset.iloc[0]
            auc[row_index, column_index] = float(row["roc_auc"])
            ap_ratio[row_index, column_index] = float(row["ap_over_prevalence"])
            annotations[row_index][column_index] = (
                f"{row['holdout_positive_cells']:.0f} pos\n{row['holdout_records']:.0f} rec"
            )
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for axis, values, title, low, high in [
        (axes[0], auc, "Fold ROC-AUC", 0.0, 1.0),
        (axes[1], ap_ratio, "Fold AP / prevalence", 0.0, 4.0),
    ]:
        image = axis.imshow(values, cmap="YlGnBu", vmin=low, vmax=high, aspect="auto")
        axis.set_xticks(range(len(folds)), [f"Fold {fold}" for fold in folds])
        axis.set_yticks(range(len(partition_order)), partition_order)
        axis.set_title(title, loc="left")
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                text = "NA" if not np.isfinite(value) else f"{value:.3f}"
                if annotations[row_index][column_index]:
                    text += "\n" + annotations[row_index][column_index]
                axis.text(
                    column_index, row_index, text,
                    ha="center", va="center", fontsize=8,
                    color="white" if np.isfinite(value) and value > (low + high) / 2 else "#222222",
                )
        figure.colorbar(image, ax=axis, fraction=0.04, pad=0.03)
    figure.suptitle(
        "Spatial transfer is fold-specific; annotation gives holdout positive cells and records",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_moran_scales(moran: pd.DataFrame, path: str) -> None:
    subset = moran.loc[
        moran["model_id"].eq("S0_NO_GRAVITY")
        & moran["residual_type"].eq("Pearson count residual")
    ].copy()
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for axis, kind, title in [
        (axes[0], "knn", "Nearest-neighbour weights"),
        (axes[1], "distance", "Distance-band weights"),
    ]:
        block = subset.loc[subset["weight_kind"].eq(kind)]
        for scope, group in block.groupby("evaluation_scope"):
            group = group.sort_values("weight_parameter")
            if scope == "pooled_oof":
                axis.plot(
                    group["weight_parameter"], group["moran_i"],
                    marker="o", linewidth=2.5, color="#123c69", label="Pooled OOF",
                )
            elif int(scope.split("_")[1]) <= 3:
                axis.plot(
                    group["weight_parameter"], group["moran_i"],
                    marker="o", linewidth=1.0, alpha=0.65,
                    label=f"Fold {int(scope.split('_')[1])}",
                )
        axis.axhline(0.0, color="#777777", linestyle="--", linewidth=1.0)
        axis.set_title(title, loc="left")
        axis.set_xlabel("k neighbours" if kind == "knn" else "Distance band (km)")
        axis.set_ylabel("Moran's I of S0 Pearson residuals")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, frameon=False)
    figure.suptitle("Residual spatial dependence persists across weight definitions", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_local_clusters(
    local: pd.DataFrame, geometries: dict[str, Any], path: str
) -> None:
    layer = geometries["catchments"][["support_id", "geometry"]].merge(
        local[["support_id", "cluster_class", "local_moran_i", "q_two_sided_bh"]],
        on="support_id", how="left",
    )
    colors = {
        "not_significant": "#e5e5e5",
        "high_high": "#b2182b",
        "low_low": "#2166ac",
        "high_low": "#f4a582",
        "low_high": "#92c5de",
    }
    labels = {
        "high_high": "High-high cluster",
        "low_low": "Low-low cluster",
        "high_low": "High-low outlier",
        "low_high": "Low-high outlier",
        "not_significant": "Not significant",
    }
    figure, axis = plt.subplots(figsize=(10.5, 8.5))
    for cluster in ["not_significant", "low_low", "low_high", "high_low", "high_high"]:
        subset = layer.loc[layer["cluster_class"].eq(cluster)]
        if subset.empty:
            continue
        subset.plot(color=colors[cluster], linewidth=0, ax=axis, alpha=0.95)
    geometries["tracts"].boundary.plot(ax=axis, color="#222222", linewidth=0.7)
    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=colors[cluster],
               markeredgecolor="none", markersize=10,
               label=f"{labels[cluster]} (n={int((local['cluster_class'] == cluster).sum())})")
        for cluster in ["high_high", "low_low", "high_low", "low_high", "not_significant"]
    ]
    axis.legend(handles=handles, loc="lower left", frameon=True, fontsize=8)
    axis.set_title(
        "S0 out-of-fold Pearson residual clusters\n12-nearest-neighbour local Moran, normal-score FDR with permutation screen",
        fontsize=13, loc="left",
    )
    axis.set_aspect("equal")
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_residual_diagnostics(
    summaries: pd.DataFrame, path: str
) -> None:
    specifications = [
        ("domain", "Daly domain / mapped coverage"),
        ("tract_boundary_distance", "Distance to tract boundary"),
        ("support_area", "Voronoi support area"),
        ("fold", "Accepted spatial fold"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.reshape(-1)
    for axis, (grouping, title) in zip(axes, specifications):
        subset = summaries.loc[summaries["grouping"].eq(grouping)].copy()
        if grouping == "fold":
            subset = subset.sort_values("group_value")
        colors = np.where(subset["mean_pearson_residual"] >= 0, "#b2182b", "#2166ac")
        axis.bar(range(len(subset)), subset["mean_pearson_residual"], color=colors, alpha=0.85)
        axis.axhline(0.0, color="#333333", linewidth=0.8)
        axis.set_xticks(range(len(subset)), subset["group_value"], rotation=25, ha="right")
        axis.set_ylabel("Mean Pearson residual")
        axis.set_title(title, loc="left")
        axis.grid(axis="y", alpha=0.2)
        for index, row in enumerate(subset.itertuples(index=False)):
            if row.observed_records > 0:
                axis.text(
                    index, row.mean_pearson_residual,
                    f"O/P\n{row.observed_over_predicted:.2f}",
                    ha="center", va="bottom" if row.mean_pearson_residual >= 0 else "top",
                    fontsize=7,
                )
    figure.suptitle("Residual summaries by geological and spatial context", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_spatial_model_comparison(
    metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    moran: pd.DataFrame,
    config: dict[str, Any],
    path: str,
) -> None:
    model_order = [value["id"] for value in config["spatial_models"]]
    rows: list[dict[str, Any]] = []
    for model_id in model_order:
        folds = fold_metrics.loc[
            fold_metrics["model_id"].eq(model_id)
            & fold_metrics["holdout_positive_cells"].ge(
                int(config["residuals"]["informative_fold_min_positive_cells"])
            )
        ]
        primary_moran = moran.loc[
            moran["model_id"].eq(model_id)
            & moran["evaluation_scope"].eq("pooled_oof")
            & moran["spatial_weight"].eq(config["residuals"]["primary_weight"])
            & moran["residual_type"].eq("Pearson count residual")
        ]
        if primary_moran.empty and model_id == "S0_REFERENCE":
            primary_moran = moran.loc[
                moran["model_id"].eq("S0_NO_GRAVITY")
                & moran["evaluation_scope"].eq("pooled_oof")
                & moran["spatial_weight"].eq(config["residuals"]["primary_weight"])
                & moran["residual_type"].eq("Pearson count residual")
            ]
        rows.append({
            "model_id": model_id,
            "roc_auc": folds["roc_auc"].mean(),
            "ap_over_prevalence": folds["ap_over_prevalence"].mean(),
            "brier_skill": folds["brier_skill"].mean(),
            "moran_i": primary_moran["moran_i"].iloc[0] if len(primary_moran) else math.nan,
        })
    summary = pd.DataFrame(rows)
    figure, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.reshape(-1)
    for axis, column, title in [
        (axes[0], "roc_auc", "Mean informative-fold ROC-AUC"),
        (axes[1], "ap_over_prevalence", "Mean informative-fold AP / prevalence"),
        (axes[2], "brier_skill", "Mean informative-fold Brier skill"),
        (axes[3], "moran_i", "Pooled residual Moran's I (12 nearest neighbours)"),
    ]:
        values = summary[column].to_numpy(dtype=float)
        colors = [PALETTE.get(model, "#777777") for model in summary["model_id"]]
        axis.barh(range(len(summary)), values, color=colors, alpha=0.9)
        axis.set_yticks(range(len(summary)), summary["model_id"])
        axis.invert_yaxis()
        axis.set_title(title, loc="left")
        axis.grid(axis="x", alpha=0.2)
        if column == "brier_skill":
            axis.axvline(0.0, color="#333333", linewidth=0.8)
        if column != "moran_i":
            axis.set_xlim(min(0.0, np.nanmin(values)) - 0.03, max(0.32, np.nanmax(values) + 0.03))
    figure.suptitle("Fixed-basis spatial comparators: performance and remaining dependence", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_residual_maps(
    predictions: pd.DataFrame,
    geometries: dict[str, Any],
    config: dict[str, Any],
    path: str,
) -> None:
    model_order = [value["id"] for value in config["spatial_models"]]
    values = predictions.loc[predictions["model_id"].isin(model_order), "pearson_residual"]
    limit = float(np.quantile(np.abs(values), 0.985))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    figure, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.reshape(-1)
    base = geometries["catchments"][["support_id", "geometry"]]
    for axis, model_id in zip(axes, model_order):
        subset = predictions.loc[predictions["model_id"].eq(model_id), [
            "support_id", "pearson_residual"
        ]]
        layer = base.merge(subset, on="support_id", how="left")
        layer.plot(
            column="pearson_residual", cmap="RdBu_r", norm=norm,
            linewidth=0, ax=axis,
        )
        geometries["tracts"].boundary.plot(ax=axis, color="#222222", linewidth=0.6)
        axis.set_title(model_id, loc="left")
        axis.set_aspect("equal")
        axis.set_axis_off()
    color_axis = axes[len(model_order)]
    color_axis.set_axis_off()
    scalar = plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    scalar.set_array([])
    colorbar = figure.colorbar(scalar, ax=color_axis, fraction=0.55, pad=0.02)
    colorbar.set_label("Pearson count residual")
    color_axis.text(0.0, 0.3, "Common symmetric colour scale\n2nd-98th percentile", fontsize=9)
    figure.suptitle("Residual maps for the nonspatial reference and fixed-basis spatial comparators", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def build_report(
    config: dict[str, Any],
    bundle: dict[str, Any],
    stage5b_meta: dict[str, Any],
    outputs: dict[str, pd.DataFrame],
    reproduction: dict[str, Any],
    local_metadata: dict[str, Any],
) -> str:
    stage4_metrics = bundle["model_metrics"]
    s0_metric = stage4_metrics.loc[
        stage4_metrics["model_id"].eq("S0_NO_GRAVITY")
        & stage4_metrics["aggregation"].eq("pooled_oof")
    ].iloc[0]
    support_comparison = pd.read_csv(
        abs_path(stage5b_meta["manifest"]["outputs"]["support_comparison"]["path"])
    )
    support_metrics = pd.read_csv(
        abs_path(stage5b_meta["manifest"]["outputs"]["model_metrics"]["path"])
    )
    support_metrics = support_metrics.loc[support_metrics["aggregation"].eq("pooled_oof"), [
        "model_id", "support_scheme", "roc_auc", "average_precision", "ap_over_prevalence",
        "brier_skill", "calibration_slope",
    ]]
    support_table = support_comparison[[
        "support_scheme", "n_support", "n_retained_records", "n_positive_cells",
        "prevalence", "expected_records_retained_mc",
    ]].merge(support_metrics, on="support_scheme", how="left")
    support_table = support_table.loc[
        support_table["support_scheme"].isin(
            ["VORONOI_ALL", "HEX_ALL", "HEX_GE80", "HEX_GE99"]
        )
    ]

    moran = outputs["moran_tests"]
    primary_weight = config["residuals"]["primary_weight"]
    primary_s0 = moran.loc[
        moran["model_id"].eq("S0_NO_GRAVITY")
        & moran["evaluation_scope"].eq("pooled_oof")
        & moran["spatial_weight"].eq(primary_weight)
        & moran["residual_type"].eq("Pearson count residual")
    ].iloc[0]
    six_neighbour_s0 = moran.loc[
        moran["model_id"].eq("S0_NO_GRAVITY")
        & moran["evaluation_scope"].eq("pooled_oof")
        & moran["spatial_weight"].eq("knn_6")
        & moran["residual_type"].eq("Pearson count residual")
    ].iloc[0]
    pooled_s0 = moran.loc[
        moran["model_id"].eq("S0_NO_GRAVITY")
        & moran["evaluation_scope"].eq("pooled_oof")
        & moran["residual_type"].eq("Pearson count residual")
    ]
    fold_s0 = moran.loc[
        moran["model_id"].eq("S0_NO_GRAVITY")
        & moran["evaluation_scope"].ne("pooled_oof")
        & moran["residual_type"].eq("Pearson count residual")
    ]
    rqr_s0 = moran.loc[
        moran["model_id"].eq("S0_NO_GRAVITY")
        & moran["evaluation_scope"].eq("pooled_oof")
        & moran["residual_type"].eq("randomized quantile residual")
    ].iloc[0]
    rqr_fold_s0 = moran.loc[
        moran["model_id"].eq("S0_NO_GRAVITY")
        & moran["evaluation_scope"].ne("pooled_oof")
        & moran["residual_type"].eq("randomized quantile residual")
    ].copy()
    local_counts = (
        outputs["local_moran"]["cluster_class"].value_counts().rename_axis("cluster").reset_index(name="n")
    )
    local_summary = local_counts.copy()
    local_summary["share"] = local_summary["n"] / len(outputs["local_moran"])

    residual_summary = outputs["resid_summary"]
    domain_table = residual_summary.loc[residual_summary["grouping"].eq("domain")]
    boundary_table = residual_summary.loc[
        residual_summary["grouping"].eq("tract_boundary_distance")
    ]
    area_table = residual_summary.loc[residual_summary["grouping"].eq("support_area")]

    partition_fold = outputs["partition_fold_metrics"]
    partition_summary = outputs["partition_metrics"].loc[
        outputs["partition_metrics"]["aggregation"].eq("fold_mean")
    ].copy()
    partition_summary = partition_summary[[
        "partition_id", "partition_buffer_km", "informative_fold_count",
        "roc_auc", "ap_over_prevalence", "brier_skill", "calibration_slope",
        "informative_roc_auc_mean", "informative_ap_over_prevalence_mean",
        "informative_brier_skill_mean",
    ]]

    spatial_metrics = outputs["spatial_metrics"].loc[
        outputs["spatial_metrics"]["aggregation"].eq("fold_mean")
    ].copy()
    spatial_primary_moran = moran.loc[
        moran["evaluation_scope"].eq("pooled_oof")
        & moran["spatial_weight"].eq(primary_weight)
        & moran["residual_type"].eq("Pearson count residual")
        & moran["model_id"].isin([value["id"] for value in config["spatial_models"]])
    ][["model_id", "moran_i", "p_positive"]]
    spatial_table = spatial_metrics[[
        "model_id", "spatial_terms", "roc_auc", "ap_over_prevalence",
        "brier_skill", "calibration_slope", "informative_fold_count",
        "informative_roc_auc_mean", "informative_ap_over_prevalence_mean",
        "informative_brier_skill_mean",
    ]].merge(spatial_primary_moran, on="model_id", how="left")
    spatial_table = spatial_table.rename(columns={"moran_i": "moran_i_12nn"})
    spatial_table.loc[
        spatial_table["model_id"].eq("S0_REFERENCE"), "moran_i_12nn"
    ] = float(primary_s0["moran_i"])

    best_auc_row = spatial_table.sort_values("informative_roc_auc_mean", ascending=False).iloc[0]
    best_ap_row = spatial_table.sort_values("informative_ap_over_prevalence_mean", ascending=False).iloc[0]
    s0_spatial_row = spatial_table.loc[spatial_table["model_id"].eq("S0_REFERENCE")].iloc[0]
    quadratic_row = spatial_table.loc[spatial_table["model_id"].eq("S0_2D_QUADRATIC")].iloc[0]
    rbf_row = spatial_table.loc[spatial_table["model_id"].eq("S0_2D_RBF12")].iloc[0]
    min_moran = float(pooled_s0["moran_i"].min())
    max_moran = float(pooled_s0["moran_i"].max())
    min_fold_moran = float(fold_s0["moran_i"].min())
    max_fold_moran = float(fold_s0["moran_i"].max())
    min_rqr = float(moran.loc[
        moran["model_id"].eq("S0_NO_GRAVITY")
        & moran["evaluation_scope"].ne("pooled_oof")
        & moran["residual_type"].eq("randomized quantile residual"),
        "moran_i",
    ].min())
    max_rqr = float(moran.loc[
        moran["model_id"].eq("S0_NO_GRAVITY")
        & moran["evaluation_scope"].ne("pooled_oof")
        & moran["residual_type"].eq("randomized quantile residual"),
        "moran_i",
    ].max())
    domain_zero = int((domain_table["observed_records"] == 0).sum())
    boundary_max = boundary_table.sort_values("mean_pearson_residual", ascending=False).iloc[0]
    boundary_min = boundary_table.sort_values("mean_pearson_residual", ascending=True).iloc[0]
    area_max = area_table.sort_values("mean_pearson_residual", ascending=False).iloc[0]
    boundary_lookup = boundary_table.set_index("group_value")
    area_lookup = area_table.set_index("group_value")
    boundary_pattern_parts = []
    for boundary_class in ["0-1 km", "1-2 km", "2-5 km", "5-10 km", ">=10 km"]:
        ratio = boundary_lookup.loc[boundary_class, "observed_over_predicted"]
        boundary_pattern_parts.append(
            f"{boundary_class}: O/P "
            + ("not defined" if not np.isfinite(ratio) else f"{ratio:.2f}")
        )
    boundary_pattern = "; ".join(boundary_pattern_parts)
    area_15_25_ratio = float(area_lookup.loc["15-25 km2", "observed_over_predicted"])
    rqr_fold_summary = ", ".join(
        f"Fold {int(row.evaluation_scope.split('_')[1])} = {row.moran_i:.3f} "
        f"(p = {row.p_positive:.3f})"
        for row in rqr_fold_s0.sort_values("evaluation_scope").itertuples(index=False)
    )

    reproduction_text = (
        f"Reference partition reproduction: {reproduction['partition']['n']:,} cells; "
        f"maximum absolute prediction-count difference {reproduction['partition']['max_abs_predicted_count_difference']:.3e}, "
        f"probability difference {reproduction['partition']['max_abs_probability_difference']:.3e}, "
        f"and Pearson-residual difference {reproduction['partition']['max_abs_pearson_residual_difference']:.3e}. "
        f"The independent spatial-model refit agrees to {reproduction['spatial']['max_abs_predicted_count_difference']:.3e}."
    )
    report = f"""# Stage 5C residual spatial diagnostics

## Scientific scope

This stage asks whether the residual spatial dependence of the accepted Stage 4 `S0_NO_GRAVITY` count model is robust to the definition of spatial proximity, where it is concentrated, and whether constrained spatial trends or alternative validation partitions remove it. The response remains the intensity of historically compiled occurrence records per unit tract-supported area. It is not a detection-free mineralisation probability.

No domain-varying coefficient model, structural predictor, free detection surface or new occurrence record is introduced. The fixed-basis spatial comparators are diagnostics, not candidate production models.

## Inherited support decision

Stage 5B established that literal 80% and 99% hexagon populations are different estimands selected on tract-edge geography, not cleaner versions of the same sample. They are retained as sensitivity populations; `VORONOI_ALL` remains the reference support.

{markdown_table(support_table, ["support_scheme", "n_support", "n_retained_records", "n_positive_cells", "prevalence", "expected_records_retained_mc", "roc_auc", "ap_over_prevalence", "brier_skill", "calibration_slope"])}

The accepted reference is pooled ROC-AUC {s0_metric['roc_auc']:.3f}, AP {s0_metric['average_precision']:.3f} ({s0_metric['ap_over_prevalence']:.2f} times prevalence), Brier skill {s0_metric['brier_skill']:.3f}, and calibration slope {s0_metric['calibration_slope']:.3f}. Fold 4 contributes one record and is never treated as stable transfer evidence.

## Reference reproduction

{reproduction_text}

## Residual spatial dependence

The accepted six-neighbour Stage 4 diagnostic for `S0_NO_GRAVITY` is Moran's I = {six_neighbour_s0['moran_i']:.3f} (one-sided permutation p = {six_neighbour_s0['p_positive']:.3f}). The value 0.170 belongs to the exploratory `S0_BASE` comparator and is not the accepted reference. Stage 5C extends the diagnostic to 6, 12 and 24 nearest neighbours and 10, 25 and 50 km distance bands. Across the six definitions, pooled S0 Pearson-residual Moran's I ranges from {min_moran:.3f} to {max_moran:.3f}; the {primary_weight} primary weight gives {primary_s0['moran_i']:.3f} (one-sided permutation p = {primary_s0['p_positive']:.3f}). Every fold-level Pearson-residual estimate is positive, ranging from {min_fold_moran:.3f} to {max_fold_moran:.3f}.

The randomization-quantile residual is used to check how much of the Pearson field is driven by the imposed count-variance function. At the primary weight, pooled S0 Moran's I falls to {rqr_s0['moran_i']:.3f} for randomized quantile residuals (one-sided permutation p = {rqr_s0['p_positive']:.3f}), with fold estimates from {min_rqr:.3f} to {max_rqr:.3f}. The fold-specific results are {rqr_fold_summary}. Only Fold 1 reaches p < 0.05; Folds 2-4 do not. The Pearson residual therefore overstates standardized residual autocorrelation, and the normalized diagnostic, although positive in every fold, is not clearly established at fold level. The concern is attenuation and weak fold-level evidence, not sign reversal.

| Weight | Mean neighbours | Moran's I | Permutation z | One-sided p |
|---|---:|---:|---:|---:|
"""
    for row in pooled_s0.sort_values("weight_parameter").itertuples(index=False):
        report += (
            f"| {row.spatial_weight} | {row.mean_neighbours:.1f} | "
            f"{row.moran_i:.3f} | {row.z_permutation:.2f} | {row.p_positive:.3f} |\n"
        )
    report += f"""

The exact geographic scale changes the magnitude, but not the sign or substantive conclusion. For local clustering, `{local_metadata['spatial_weight']}` weights are used. Because 999 permutations cannot resolve a false-discovery threshold across 2,193 cells, local significance uses a normal-score approximation with 5% Benjamini-Hochberg control, screened by the permutation p-value (`p < 0.05`). This is an approximate exploratory screen, not an exact permutation FDR test.

## Local residual clusters

{markdown_table(local_summary, ["cluster", "n", "share"])}

The high-high clusters are above-baseline residual cells surrounded by above-baseline neighbours. The `low_high` class is a low-residual cell surrounded by higher-residual neighbours; it is not a low-low cluster. No low-low or high-low clusters were detected. These classes describe the joint geography of the fitted covariates and the observation inventory; they are not validation that mineralisation is present or absent.

## Residual context

### Daly domain and mapped coverage

{markdown_table(domain_table[["group_value", "n_support_cells", "area_km2", "observed_records", "predicted_records", "observed_over_predicted", "mean_pearson_residual", "mean_rqr"]], ["group_value", "n_support_cells", "area_km2", "observed_records", "predicted_records", "observed_over_predicted", "mean_pearson_residual", "mean_rqr"])}

Only NRB contains a meaningful positive population. {domain_zero} domain classes contain no recorded occurrences. These groups cannot identify separate geological relationships; their residual summaries are descriptive consequences of an observation inventory concentrated in one domain.

### Tract-boundary and support-area context

Boundary classes show a mixed, non-monotonic pattern: {boundary_pattern}. The 15-25 km2 support class has O/P = {area_15_25_ratio:.2f}. These patterns may reflect support geometry, catchment placement at tract edges and observation history, and they are potentially confounded with one another. They do not provide a clean boundary or area effect and are not evidence that such effects are negligible. They do not justify changing the accepted exposure model now; they remain sensitivity diagnostics.

{markdown_table(boundary_table[["group_value", "n_support_cells", "observed_records", "observed_over_predicted", "mean_pearson_residual", "mean_rqr"]], ["group_value", "n_support_cells", "observed_records", "observed_over_predicted", "mean_pearson_residual", "mean_rqr"])}

{markdown_table(area_table[["group_value", "n_support_cells", "observed_records", "observed_over_predicted", "mean_pearson_residual", "mean_rqr"]], ["group_value", "n_support_cells", "observed_records", "observed_over_predicted", "mean_pearson_residual", "mean_rqr"])}

## Buffered and alternative spatial partitions

The original four belt-axis folds are retained exactly. Increasing the buffer from 10 to 25 and 50 km tests whether the accepted transfer result depends on near-boundary training support. The 50 km checkerboard partition tests local spatial generalisation under interleaved blocks and does not replace the regional folds.

{markdown_table(partition_summary, ["partition_id", "partition_buffer_km", "informative_fold_count", "roc_auc", "ap_over_prevalence", "brier_skill", "informative_roc_auc_mean", "informative_ap_over_prevalence_mean", "informative_brier_skill_mean"])}

All partitions are dominated by folds with different positive counts and prevalence. The fold-level table is the primary comparison; pooled out-of-fold metrics still mix between-sector intercept shifts with within-sector ranking.

## Fixed-basis spatial comparators

Each comparator adds a fixed, training-standardised spatial basis to the accepted S0 predictors. The bases are ordered by flexibility: along-belt linear, along-belt cubic, two-dimensional quadratic, and twelve fixed radial-basis functions. The ridge penalty is unchanged from Stage 4. These models test whether broad smooth spatial variation can account for the residual field; they are not selection candidates.

{markdown_table(spatial_table, ["model_id", "spatial_terms", "informative_fold_count", "informative_roc_auc_mean", "informative_ap_over_prevalence_mean", "informative_brier_skill_mean", "calibration_slope", "moran_i_12nn"])}

The best informative-fold ROC-AUC is {best_auc_row['informative_roc_auc_mean']:.3f} for `{best_auc_row['model_id']}`; the best AP/prevalence is {best_ap_row['informative_ap_over_prevalence_mean']:.3f} for `{best_ap_row['model_id']}`. The nonspatial S0 reference has informative-fold ROC-AUC {s0_spatial_row['informative_roc_auc_mean']:.3f}, AP/prevalence {s0_spatial_row['informative_ap_over_prevalence_mean']:.3f}, and Brier skill {s0_spatial_row['informative_brier_skill_mean']:.3f}. The trade-off is explicit: `S0_2D_RBF12` has the best ranking and AP but Brier skill {rbf_row['informative_brier_skill_mean']:.3f} and residual Moran's I {rbf_row['moran_i_12nn']:.3f}; `S0_2D_QUADRATIC` reduces residual Moran's I to {quadratic_row['moran_i_12nn']:.3f} but has Brier skill {quadratic_row['informative_brier_skill_mean']:.3f}. No comparator improves ranking, probability accuracy and residual dependence together. None is accepted.

The primary-weight residual Moran's I for the spatial comparators is reported above. A flexible spatial basis that improved ranking while leaving substantial residual clustering would indicate that a smooth geological trend is not the main missing structure. A basis that removed all residual dependence by absorbing the observation geography would not be identifiable as a geological effect.

## Observation-process boundary

The inventory contains 184 in-tract records: 183 in NRB, one outside the reconstructed Daly envelope, and none in CRZ, MMSB or NKB. No external historical exploration, discovery or reporting-intensity dataset has been acquired. The decomposition

`lambda_observed(s) = lambda_geological(s) * p_observation(s)`

therefore remains unidentifiable from the present inventory alone. Stage 5C can quantify residual spatial structure but cannot attribute it to geology or observation effort. A free detection surface fitted jointly with a free geological intensity surface would be able to absorb the same spatial pattern and would not resolve that ambiguity.

The residual and partition results should therefore be read as conditional on the compiled inventory. In particular, the concentration of records in NRB cannot be used as evidence that the other domains are less prospective.

## Interpretation and limitations

- The Stage 4 S0 count model remains the reference. No spatial comparator is promoted merely because it adds terms.
- Pearson-residual dependence is positive across all six tested weight definitions and in every fold. It is not an artefact of using six nearest neighbours. Randomized-quantile residuals attenuate the dependence strongly; only Fold 1 is significant at the 5% level.
- The local clusters quantify where documented-record intensity departs from the model. They do not identify undiscovered mineralisation or verified absence.
- Alternative folds and buffers diagnose the sensitivity of transfer. They do not create independent geological observations; the same 184 records are reused.
- The checkerboard partition tests local interpolation, not regional geological transfer. The belt-axis folds remain the primary regional design.
- The spatial bases are fixed geometry. Their coefficients are descriptive associations under the fitted penalty and are not causal geological effects.
- The observation process remains unresolved. Any report of `p_observation` would require external effort, reporting-coverage or discovery data and explicit assumptions.
- Fold 4 and the outside-Daly population remain sparse; conclusions are driven by NRB and, within it, the western and central sectors.

## Recommendation

Retain `VORONOI_ALL` and `S0_NO_GRAVITY` as the reference for the next stage. Carry the six-weight Moran diagnostic, the local-cluster map, the 25/50 km buffer sensitivity, and the checkerboard partition into the research record. Do not promote a smooth spatial trend or a domain-varying model on the present evidence.

The next scientific step is Stage 5D: bounded observation-process scenarios using external or explicitly assumed observation proxies. No free detection surface should be estimated jointly with an unconstrained geological intensity surface. The first defensible question is how much of the residual pattern can be explained by a small set of declared observation proxies, not how much flexibility a model can absorb.

## Figures

![Partition maps](stage5c_partition_maps.png)

![Partition transfer](stage5c_partition_transfer.png)

![Moran scales](stage5c_moran_scales.png)

![Local clusters](stage5c_local_clusters.png)

![Residual diagnostics](stage5c_residual_diagnostics.png)

![Spatial model comparison](stage5c_spatial_model_comparison.png)

![Residual maps](stage5c_residual_maps.png)
"""
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage5c_v1.json")
    args = parser.parse_args()
    config_path = abs_path(args.config)
    config = load_config(config_path)
    paths = output_paths(config)
    bundle, geometries, metadata = load_verified_inputs(config)
    protected = verify_protected(bundle["stage4_config"])
    frame = bundle["model_table"].copy()
    frame, feature_diagnostics, spatial_metadata = add_spatial_features(frame, config)
    fit_config = copy.deepcopy(bundle["stage4_config"])
    spatial_features = sorted({
        term for model in config["spatial_models"] for term in model["terms"]
    })
    for feature in spatial_features:
        fit_config["models"]["continuous"][feature] = {"transform": "identity"}

    print("Fitting Stage 5C spatial partitions...")
    partition_outputs, partition_frames, partition_reproduction = fit_partitions(
        frame, bundle, config, fit_config
    )
    print("Fitting fixed-basis spatial comparators...")
    spatial_reference = bundle["predictions"].loc[
        bundle["predictions"]["model_id"].eq(config["reference"]["model_id"])
    ].copy()
    spatial_outputs, spatial_reproduction = fit_spatial_models(
        frame, config, fit_config, spatial_reference
    )

    uniforms = ordered_uniform_draws(frame, int(config["residuals"]["rqr_seed"]))
    baseline_predictions = bundle["predictions"].loc[
        bundle["predictions"]["model_id"].isin(["N0_AREA", "S0_NO_GRAVITY"])
    ].copy()
    baseline_predictions = add_rqr(baseline_predictions, uniforms)
    spatial_predictions = add_rqr(spatial_outputs["spatial_predictions"], uniforms)

    print("Computing multi-scale Moran and local cluster diagnostics...")
    moran, moran_summary, local, local_metadata = build_moran_tables(
        baseline_predictions, spatial_predictions, config
    )
    reference_predictions = spatial_predictions.loc[
        spatial_predictions["model_id"].eq("S0_REFERENCE")
    ].copy()
    residual_summary = residual_summaries(
        reference_predictions, geometries["tract_union"], config
    )
    observation_context = residual_summary.loc[
        residual_summary["grouping"].isin(
            ["tract_boundary_distance", "domain", "mapped_coverage"]
        )
    ].copy()
    feature_table = spatial_feature_diagnostics(frame, config)

    outputs = {
        **partition_outputs,
        **spatial_outputs,
        "moran_tests": moran,
        "moran_summary": moran_summary,
        "local_moran": local,
        "resid_summary": residual_summary,
        "observation_context": observation_context,
        "feature_diagnostics": feature_table,
    }
    write_csv(outputs["partition_definitions"], paths["partition_definitions"])
    write_csv(outputs["partition_fold_metrics"], paths["partition_fold_metrics"])
    write_csv(outputs["partition_metrics"], paths["partition_summary"])
    write_csv(outputs["partition_predictions"], paths["partition_predictions"])
    write_csv(outputs["spatial_metrics"], paths["spatial_model_metrics"])
    write_csv(outputs["spatial_fold_metrics"], paths["spatial_model_fold_metrics"])
    write_csv(outputs["spatial_predictions"], paths["spatial_model_predictions"])
    write_csv(outputs["spatial_coefficients"], paths["spatial_model_coefficients"])
    write_csv(feature_table, paths["spatial_feature_diagnostics"])
    write_csv(moran, paths["moran_tests"])
    write_csv(moran_summary, paths["moran_summary"])
    write_csv(local, paths["local_moran"])
    write_csv(residual_summary, paths["residual_summary"])
    write_csv(observation_context, paths["observation_context"])

    reproduction = {
        "partition": partition_reproduction["reference_partition"],
        "spatial": spatial_reproduction,
        "tolerances": {
            "max_abs_predicted_count_difference": 1e-10,
            "max_abs_probability_difference": 1e-10,
            "max_abs_pearson_residual_difference": 1e-10,
        },
        "spatial_basis_metadata": spatial_metadata,
        "local_moran_metadata": local_metadata,
    }
    write_json(reproduction, paths["reference_reproduction"])

    print("Rendering Stage 5C figures...")
    plot_partition_maps(partition_frames, geometries, config, paths["partition_figure"])
    plot_partition_transfer(
        outputs["partition_fold_metrics"], config, paths["partition_transfer_figure"]
    )
    plot_moran_scales(moran, paths["moran_figure"])
    plot_local_clusters(local, geometries, paths["local_cluster_figure"])
    plot_residual_diagnostics(residual_summary, paths["residual_diagnostic_figure"])
    plot_spatial_model_comparison(
        outputs["spatial_metrics"], outputs["spatial_fold_metrics"], moran,
        config, paths["spatial_model_figure"],
    )
    plot_residual_maps(
        spatial_predictions, geometries, config, paths["residual_maps_figure"]
    )

    report = build_report(
        config, bundle, metadata["stage5b"], outputs, reproduction, local_metadata
    )
    write_text(report, paths["report"])

    output_keys = [
        key for key in config["outputs"]
        if key not in {"processed_dir", "figure_dir", "manifest", "validation"}
    ]
    output_records = {
        key: file_record(paths[key], key) for key in output_keys
    }
    input_paths = {
        "stage4_config": bundle["stage4_config"]["protected_files"]["v14_training_csv"],
        "stage4_model_table": bundle["data_manifest"]["outputs"]["model_table"]["path"],
        "stage4_predictions": bundle["model_manifest"]["outputs"]["model_predictions"]["path"],
        "stage4_model_metrics": bundle["model_manifest"]["outputs"]["model_metrics"]["path"],
        "stage5b_manifest": config["stage5b"]["manifest"],
        "stage5b_validation": config["stage5b"]["validation"],
        "stage5a_manifest": config["stage5"]["inventory_manifest"],
        "stage5a_validation": config["stage5"]["inventory_validation"],
    }
    input_records = {
        key: file_record(value, key) for key, value in input_paths.items()
    }
    manifest = {
        "schema_version": "1.0.0",
        "build_id": build_id(),
        "status": "fitted",
        "purpose": config["purpose"],
        "config": file_record(config_path, "config"),
        "reference": config["reference"],
        "inputs": input_records,
        "protected_inputs_unchanged": protected,
        "stage5b_reference": metadata["stage5b"]["manifest"].get("build_id"),
        "outputs": output_records,
        "counts": {
            "n_support": int(len(frame)),
            "n_records": int(frame["n_records"].sum()),
            "n_positive_cells": int(frame["n_records"].gt(0).sum()),
            "n_partitions": len(config["partitions"]),
            "n_spatial_models": len(config["spatial_models"]),
            "n_moran_tests": int(len(moran)),
            "n_local_moran_cells": int(len(local)),
        },
        "reproduction": {
            "path": relative_posix(paths["reference_reproduction"]),
            "sha256": sha256_file(paths["reference_reproduction"]),
        },
        "validation": {
            "path": relative_posix(paths["validation"]),
            "sha256": None,
            "status": "pending",
        },
        "software": software_stack(),
        "notes": [
            "The response is documented-record intensity, not a geological probability.",
            "S0_NO_GRAVITY and VORONOI_ALL remain the accepted reference.",
            "The spatial comparators are fixed-basis diagnostics and are not promoted.",
            "No free observation-detection surface or domain-varying coefficient is fitted.",
            "Stage 5B support, Stage 4, Stage 3, V14 and the manuscript are not modified.",
        ],
    }
    write_json(manifest, paths["manifest"])
    print(json.dumps({
        "build_id": manifest["build_id"],
        "n_outputs": len(output_records),
        "n_moran_tests": int(len(moran)),
        "n_local_clusters": int((local["cluster_class"] != "not_significant").sum()),
        "report": paths["report"],
        "manifest": paths["manifest"],
    }, indent=2))


if __name__ == "__main__":
    main()
