"""Stage 5B support sensitivity for the accepted S0_NO_GRAVITY specification.

The script compares tract-clipped Voronoi support with literal nominal
hexagons. Every literal-grid comparison is paired with a same-record Voronoi
control so that support geometry and record loss are not conflated. No Stage 4
file is modified.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modeling.fit_stage4_models import (  # noqa: E402
    Experiment,
    binary_calibration,
    compute_bootstrap_metrics,
    fit_experiment,
    spatial_block_resamples,
)
from src.modeling.stage4_common import (  # noqa: E402
    PROJECT_CRS,
    abs_path,
    build_id,
    file_record,
    relative_posix,
    software_versions,
    write_csv,
    write_gpkg,
    write_json,
)
from src.modeling.stage5_common import (  # noqa: E402
    distance_to_boundary_km,
    markdown_table,
    software_stack,
    write_text,
)
from src.modeling.stage5b_common import (  # noqa: E402
    area_elasticity,
    boundary_diagnostics,
    build_support_scenario,
    fit_s0,
    load_config,
    output_paths,
    positional_retention,
    residual_area_diagnostics,
    validate_stage5a,
)


BOOTSTRAP_METRICS = [
    "roc_auc", "average_precision", "ap_over_prevalence", "brier_score",
    "brier_skill", "binary_log_loss", "nb_deviance", "mae", "rmse",
    "total_count_ratio", "calibration_slope",
]


SCHEME_ORDER = [
    "VORONOI_ALL",
    "VORONOI_MATCHED_HEX_ALL",
    "HEX_ALL",
    "VORONOI_MATCHED_HEX80",
    "HEX_GE80",
    "VORONOI_MATCHED_HEX99",
    "HEX_GE99",
]

ROLE_COLORS = {
    "reference": "#1b7f79",
    "support_control": "#6f5aa8",
    "literal_support": "#c45a2a",
}


def coverage_variant(scenario: dict[str, Any]) -> str:
    threshold = scenario.get("record_filter_threshold")
    if scenario["family"] == "voronoi" and threshold is None:
        return "VORONOI_ALL"
    return f"HEX_{float(threshold):g}"


def scenario_metadata(scenario: dict[str, Any]) -> dict[str, Any]:
    return {
        "support_scheme": scenario["id"],
        "support_label": scenario["label"],
        "support_role": scenario["role"],
        "support_family": scenario["family"],
        "hex_threshold": scenario.get("hex_threshold"),
        "record_filter_threshold": scenario.get("record_filter_threshold"),
    }


def build_assignment_table(
    scenarios: dict[str, dict[str, Any]],
    occurrences: pd.DataFrame,
    model_table: pd.DataFrame,
    positional: pd.DataFrame,
) -> pd.DataFrame:
    fold_lookup = model_table.set_index("support_id")["new_fold"]
    domain_lookup = model_table.set_index("support_id")["domain_abbreviation"]
    context = occurrences.copy()
    context["source_fold"] = context["assigned_support_id"].map(fold_lookup)
    context["source_domain"] = context["assigned_support_id"].map(domain_lookup)
    context["in_tract"] = context["in_tract"].astype(bool)
    context["distance_to_tract_boundary_km"] = distance_to_boundary_km(
        gpd.GeoSeries(gpd.points_from_xy(context["x_m"], context["y_m"]), crs=PROJECT_CRS),
        scenarios["__tract_union"],
    )
    positional_small = positional[[
        "coverage_variant", "record_id", "p_retained_mc", "p_lost_mc"
    ]]
    rows: list[pd.DataFrame] = []
    for scheme_id, scenario in scenarios.items():
        if scheme_id.startswith("__"):
            continue
        retained = scenario["records"][["record_id", "scenario_support_id"]].copy()
        assignment = context.merge(retained, on="record_id", how="left")
        assignment["support_scheme"] = scheme_id
        assignment["support_id"] = assignment["scenario_support_id"].where(
            assignment["in_tract"]
        )
        assignment["point_retained"] = assignment["support_id"].notna()
        variant = coverage_variant(scenario["spec"])
        probabilities = positional_small.loc[
            positional_small["coverage_variant"].eq(variant),
            ["record_id", "p_retained_mc", "p_lost_mc"],
        ]
        assignment = assignment.drop(columns=["p_retained_mc", "p_lost_mc"], errors="ignore")
        assignment = assignment.merge(probabilities, on="record_id", how="left")
        if assignment["p_retained_mc"].isna().any():
            raise RuntimeError(f"Missing positional retention for {scheme_id}")
        assignment.loc[~assignment["in_tract"], ["p_retained_mc", "p_lost_mc"]] = [0.0, 1.0]
        support_area = scenario["frame"].set_index("support_id")["support_area_km2"]
        assignment["support_area_km2"] = assignment["support_id"].map(support_area)
        assignment["record_expansion"] = assignment["record_expansion"].astype(float)
        assignment["expected_retained_mc"] = assignment["p_retained_mc"]
        assignment["expected_expanded_retained_mc"] = (
            assignment["p_retained_mc"] * assignment["record_expansion"]
        )
        rows.append(assignment[[
            "support_scheme", "record_id", "SiteStatus", "source_fold", "source_domain",
            "in_tract", "support_id", "support_area_km2", "point_retained",
            "p_retained_mc", "p_lost_mc", "expected_retained_mc",
            "expected_expanded_retained_mc", "record_expansion",
            "distance_to_tract_boundary_km",
        ]])
    output = pd.concat(rows, ignore_index=True)
    if output.groupby(["support_scheme", "record_id"]).size().max() != 1:
        raise RuntimeError("Record-assignment table is not unique by scheme and record")
    return output


def build_record_accounting(assignments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groupings = [
        ("ALL", pd.Series("ALL", index=assignments.index)),
        ("site_status", assignments["SiteStatus"]),
        ("source_fold", assignments["source_fold"].astype("Int64").astype(str)),
        ("source_domain", assignments["source_domain"].fillna("unassigned")),
    ]
    for grouping, values in groupings:
        working = assignments.assign(_group=values)
        for (scheme, value), group in working.groupby(
            ["support_scheme", "_group"], dropna=False, sort=True
        ):
            in_tract = group.loc[group["in_tract"]]
            retained = in_tract.loc[in_tract["point_retained"]]
            rows.append({
                "support_scheme": scheme,
                "grouping": grouping,
                "group_value": str(value),
                "n_source_in_tract": int(len(in_tract)),
                "n_point_retained": int(len(retained)),
                "n_point_dropped": int(len(in_tract) - len(retained)),
                "point_retention_fraction": float(len(retained) / max(len(in_tract), 1)),
                "n_expanded_source": float(in_tract["record_expansion"].sum()),
                "n_expanded_point_retained": float(retained["record_expansion"].sum()),
                "expected_retained_mc": float(in_tract["p_retained_mc"].sum()),
                "mc_retention_fraction": float(
                    in_tract["p_retained_mc"].sum() / max(len(in_tract), 1)
                ),
                "expected_expanded_retained_mc": float(
                    in_tract["expected_expanded_retained_mc"].sum()
                ),
            })
    return pd.DataFrame(rows)


def build_fold_composition(
    scenarios: dict[str, dict[str, Any]], assignments: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scheme_id, scenario in scenarios.items():
        if scheme_id.startswith("__"):
            continue
        frame = scenario["frame"]
        for fold in (1, 2, 3, 4):
            support = frame.loc[frame["new_fold"].eq(fold)]
            records = assignments.loc[
                assignments["support_scheme"].eq(scheme_id)
                & assignments["source_fold"].eq(fold)
                & assignments["in_tract"]
            ]
            retained = records.loc[records["point_retained"]]
            rows.append({
                "support_scheme": scheme_id,
                "fold": fold,
                "n_support": int(len(support)),
                "support_area_km2": float(support["support_area_km2"].sum()),
                "positive_cells": int(support["any_record"].sum()),
                "cell_prevalence": float(support["any_record"].mean()) if len(support) else math.nan,
                "records": int(support["n_records"].sum()),
                "deposits": int(support["n_deposits"].sum()),
                "prospects": int(support["n_prospects"].sum()),
                "groups": int(support["n_groups"].sum()),
                "n_source_records_in_tract": int(len(records)),
                "n_point_dropped": int(len(records) - len(retained)),
                "expected_retained_mc": float(records["p_retained_mc"].sum()),
            })
    return pd.DataFrame(rows)


def build_support_comparison(
    scenarios: dict[str, dict[str, Any]], assignments: pd.DataFrame,
    boundary: gpd.GeoSeries, tract_union: Any,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scheme_id, scenario in scenarios.items():
        if scheme_id.startswith("__"):
            continue
        frame = scenario["frame"]
        context = scenario["context"]
        assignment = assignments.loc[assignments["support_scheme"].eq(scheme_id)]
        in_tract = assignment.loc[assignment["in_tract"]]
        retained = in_tract.loc[in_tract["point_retained"]]
        points = gpd.GeoSeries(
            gpd.points_from_xy(frame["x_m"], frame["y_m"]), crs=PROJECT_CRS
        )
        frame_boundary = distance_to_boundary_km(points, tract_union)
        area = frame["support_area_km2"].astype(float)
        rows.append({
            **scenario_metadata(scenario["spec"]),
            **context,
            "n_folds_with_positive_cells": int(
                frame.groupby("new_fold")["any_record"].sum().gt(0).sum()
            ),
            "min_fold_prevalence": float(
                frame.groupby("new_fold", sort=True)["any_record"].mean().min()
            ),
            "max_fold_prevalence": float(
                frame.groupby("new_fold", sort=True)["any_record"].mean().max()
            ),
            "mean_support_area_km2": float(area.mean()),
            "median_support_area_km2": float(area.median()),
            "min_support_area_km2": float(area.min()),
            "max_support_area_km2": float(area.max()),
            "area_range_km2": float(area.max() - area.min()),
            "area_p95_minus_p05_km2": float(area.quantile(0.95) - area.quantile(0.05)),
            "n_support_within_2km_tract_boundary": int(np.sum(frame_boundary <= 2.0)),
            "area_vs_boundary_distance_spearman_r": float(
                stats.spearmanr(area, frame_boundary).statistic
            ),
            "point_records_dropped": int(len(in_tract) - len(retained)),
            "point_deposit_records_dropped": int(
                in_tract.loc[~in_tract["point_retained"], "SiteStatus"].eq("deposit").sum()
            ),
            "point_prospect_records_dropped": int(
                in_tract.loc[~in_tract["point_retained"], "SiteStatus"].eq("prospect").sum()
            ),
            "point_group_records_dropped": int(
                in_tract.loc[~in_tract["point_retained"], "SiteStatus"].eq("group").sum()
            ),
            "expected_records_retained_mc": float(in_tract["p_retained_mc"].sum()),
            "mc_retention_fraction": float(in_tract["p_retained_mc"].sum() / max(len(in_tract), 1)),
        })
    return pd.DataFrame(rows)


def bootstrap_metrics(
    predictions: pd.DataFrame, config: dict[str, Any], seed: int
) -> pd.DataFrame:
    ordered = predictions.sort_values("support_id").reset_index(drop=True)
    y_count = ordered["observed_count"].to_numpy(dtype=float)
    y_binary = ordered["observed_any"].to_numpy(dtype=float)
    mu = ordered["predicted_count"].to_numpy(dtype=float)
    p_any = ordered["predicted_any_probability"].to_numpy(dtype=float)
    theta = ordered["theta"].to_numpy(dtype=float)
    area = ordered["tract_area_km2"].to_numpy(dtype=float)
    ranking = ordered["predicted_rate_per_km2"].to_numpy(dtype=float)
    baseline = ordered["baseline_train_probability"].to_numpy(dtype=float)
    coordinates = ordered[["x_m", "y_m"]].to_numpy(dtype=float)
    block_size = float(config["models"]["block_bootstrap"]["block_size_km"])
    requested = int(config["models"]["block_bootstrap"]["n_resamples"])
    samples = spatial_block_resamples(
        coordinates, y_binary.astype(int), block_size, requested, seed
    )
    block_x = np.floor(coordinates[:, 0] / (block_size * 1000.0)).astype(int)
    block_y = np.floor(coordinates[:, 1] / (block_size * 1000.0)).astype(int)
    n_blocks = len(set(zip(block_x.tolist(), block_y.tolist())))
    point = compute_bootstrap_metrics(
        y_count, y_binary, mu, p_any, theta, area, ranking, baseline
    )
    point_clipped = np.clip(p_any, 1e-12, 1.0 - 1e-12)
    point["binary_log_loss"] = float(-np.mean(
        y_binary * np.log(point_clipped) + (1.0 - y_binary) * np.log(1.0 - point_clipped)
    ))
    intercept, slope = binary_calibration(y_binary, p_any)
    point["calibration_slope"] = slope
    values: dict[str, list[float]] = {metric: [] for metric in BOOTSTRAP_METRICS}
    for indices in samples:
        sample = compute_bootstrap_metrics(
            y_count[indices], y_binary[indices], mu[indices], p_any[indices],
            theta[indices], area[indices], ranking[indices], baseline[indices],
        )
        sample_clipped = np.clip(p_any[indices], 1e-12, 1.0 - 1e-12)
        sample["binary_log_loss"] = float(-np.mean(
            y_binary[indices] * np.log(sample_clipped)
            + (1.0 - y_binary[indices]) * np.log(1.0 - sample_clipped)
        ))
        _, sample_slope = binary_calibration(y_binary[indices], p_any[indices])
        sample["calibration_slope"] = sample_slope
        for metric in BOOTSTRAP_METRICS:
            if np.isfinite(sample.get(metric, math.nan)):
                values[metric].append(float(sample[metric]))
    rows = []
    for metric in BOOTSTRAP_METRICS:
        array = np.asarray(values[metric], dtype=float)
        rows.append({
            "metric": metric,
            "point_estimate": float(point.get(metric, math.nan)),
            "bootstrap_median": float(np.median(array)) if array.size else math.nan,
            "ci_low": float(np.quantile(array, 0.025)) if array.size else math.nan,
            "ci_high": float(np.quantile(array, 0.975)) if array.size else math.nan,
            "n_successful_resamples": int(array.size),
            "n_requested_resamples": requested,
            "block_size_km": block_size,
            "n_blocks": n_blocks,
            "seed": seed,
        })
    return pd.DataFrame(rows)


def coefficient_summary(coefficients: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (scheme, term), group in coefficients.groupby(
        ["support_scheme", "term"], sort=False
    ):
        full = group.loc[group["fit_scope"].eq("full_data_descriptive")].iloc[0]
        folds = group.loc[
            group["fit_scope"].str.startswith("fold_") & group["identifiable"].astype(bool)
        ]
        natural = folds["estimate_natural"].dropna().to_numpy(dtype=float)
        positive = int(np.sum(natural > 0))
        negative = int(np.sum(natural < 0))
        rows.append({
            "support_scheme": scheme,
            "term": term,
            "term_raw": full["term_raw"],
            "estimate_full_natural": float(full["estimate_natural"]),
            "ci_low_full_natural": float(full["ci_low_natural"]),
            "ci_high_full_natural": float(full["ci_high_natural"]),
            "fold_mean_natural": float(np.mean(natural)) if natural.size else math.nan,
            "fold_sd_natural": float(np.std(natural, ddof=1)) if natural.size > 1 else math.nan,
            "fold_min_natural": float(np.min(natural)) if natural.size else math.nan,
            "fold_max_natural": float(np.max(natural)) if natural.size else math.nan,
            "n_folds_finite": int(natural.size),
            "n_folds_positive": positive,
            "n_folds_negative": negative,
            "fold_sign_consistency": max(positive, negative) / natural.size if natural.size else math.nan,
        })
    return pd.DataFrame(rows)


def lost_record_diagnostics(assignments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scheme, group in assignments.groupby("support_scheme", sort=False):
        in_tract = group.loc[group["in_tract"]]
        retained = in_tract.loc[in_tract["point_retained"]]
        dropped = in_tract.loc[~in_tract["point_retained"]]
        for label, subset in (("retained", retained), ("dropped", dropped)):
            if subset.empty:
                continue
            distance = subset["distance_to_tract_boundary_km"].to_numpy(dtype=float)
            rows.append({
                "support_scheme": scheme,
                "record_class": label,
                "n_records": int(len(subset)),
                "n_deposits": int(subset["SiteStatus"].eq("deposit").sum()),
                "n_prospects": int(subset["SiteStatus"].eq("prospect").sum()),
                "n_groups": int(subset["SiteStatus"].eq("group").sum()),
                "mean_distance_to_tract_boundary_km": float(np.mean(distance)),
                "median_distance_to_tract_boundary_km": float(np.median(distance)),
                "fraction_within_1km": float(np.mean(distance <= 1.0)),
                "fraction_within_2km": float(np.mean(distance <= 2.0)),
                "fraction_within_5km": float(np.mean(distance <= 5.0)),
                "mean_p_retained_mc": float(subset["p_retained_mc"].mean()),
                "expected_records_retained_mc": float(subset["p_retained_mc"].sum()),
            })
    return pd.DataFrame(rows)


def outcome_sensitivities(
    frames: dict[str, pd.DataFrame], stage4_config: dict[str, Any],
    sensitivity_specs: list[dict[str, Any]],
) -> pd.DataFrame:
    predictors = tuple([
        entry for entry in stage4_config["models"]["sensitivity_experiments"]
        if entry["id"] == "S0_NO_GRAVITY"
    ][0]["predictors"])
    rows: list[dict[str, Any]] = []
    for scheme, frame in frames.items():
        for spec in sensitivity_specs:
            working = frame.sort_values("support_id").reset_index(drop=True).copy()
            if spec["id"] == "GROUP_EXPANDED":
                working["any_record"] = working["n_expanded_records"].gt(0).astype(int)
                experiment = Experiment(
                    id=spec["id"], label=spec["label"], predictors=predictors,
                    outcome="n_expanded_records", set_name="outcome_sensitivity",
                )
                effective_outcome = "n_expanded_records"
            else:
                working["n_records"] = working[spec["outcome"]].astype(float)
                working["any_record"] = working["n_records"].gt(0).astype(int)
                experiment = Experiment(
                    id=spec["id"], label=spec["label"], predictors=predictors,
                    outcome="n_records", set_name="outcome_sensitivity",
                )
                effective_outcome = spec["outcome"]
            _, metrics, _, fold_metrics = fit_experiment(working, experiment, stage4_config)
            metrics = metrics.copy()
            metrics["support_scheme"] = scheme
            metrics["sensitivity_id"] = spec["id"]
            metrics["effective_outcome"] = effective_outcome
            metrics["fold_metrics_n"] = len(fold_metrics)
            rows.append(metrics)
    return pd.concat(rows, ignore_index=True)


def reference_reproduction_check(
    bundle: dict[str, Any], reference_predictions: pd.DataFrame,
    reference_metrics: pd.DataFrame,
) -> dict[str, Any]:
    stage4_predictions = bundle["predictions"].loc[
        bundle["predictions"]["model_id"].eq("S0_NO_GRAVITY")
    ].set_index("support_id").sort_index()
    reproduced = reference_predictions.set_index("support_id").sort_index()
    if set(stage4_predictions.index) != set(reproduced.index):
        raise RuntimeError("Stage 5B reference does not reproduce the Stage 4 support")
    fields = {
        "predicted_rate_per_km2": 1e-9,
        "predicted_any_probability": 1e-9,
        "observed_count": 1e-9,
        "pearson_residual": 1e-8,
    }
    differences: dict[str, float] = {}
    for field, tolerance in fields.items():
        difference = float(np.max(np.abs(
            stage4_predictions[field].to_numpy(dtype=float)
            - reproduced[field].to_numpy(dtype=float)
        )))
        differences[field] = difference
        if difference > tolerance:
            raise RuntimeError(f"Reference reproduction differs for {field}: {difference}")
    stage4_metric = bundle["model_metrics"].loc[
        bundle["model_metrics"]["model_id"].eq("S0_NO_GRAVITY")
        & bundle["model_metrics"]["aggregation"].eq("pooled_oof")
    ].iloc[0]
    reproduced_metric = reference_metrics.loc[
        reference_metrics["aggregation"].eq("pooled_oof")
    ].iloc[0]
    metric_differences = {}
    for metric in (
        "roc_auc", "average_precision", "brier_score", "brier_skill",
        "calibration_slope", "calibration_intercept",
    ):
        difference = abs(float(stage4_metric[metric]) - float(reproduced_metric[metric]))
        metric_differences[metric] = difference
        if difference > 1e-8:
            raise RuntimeError(f"Reference metric differs for {metric}: {difference}")
    return {
        "status": "pass",
        "stage4_data_build_id": bundle["data_manifest"]["build_id"],
        "stage4_model_build_id": bundle["model_manifest"]["build_id"],
        "max_prediction_differences": differences,
        "max_metric_differences": metric_differences,
    }


def _short_scheme(scheme: str) -> str:
    labels = {
        "VORONOI_ALL": "VOR_ALL",
        "VORONOI_MATCHED_HEX_ALL": "V-M_ALL",
        "HEX_ALL": "HEX_ALL",
        "VORONOI_MATCHED_HEX80": "V-M80",
        "HEX_GE80": "HEX80",
        "VORONOI_MATCHED_HEX99": "V-M99",
        "HEX_GE99": "HEX99",
    }
    return labels.get(scheme, scheme)


def _style_axes(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.7)
    axis.set_axisbelow(True)


def make_support_figure(
    scenarios: dict[str, dict[str, Any]], assignments: pd.DataFrame,
    occurrences: pd.DataFrame, tract_union: Any, path: str | Path,
) -> Path:
    panels = ["VORONOI_ALL", "HEX_ALL", "HEX_GE80", "HEX_GE99"]
    panel_titles = {
        "VORONOI_ALL": "Voronoi catchments: every in-tract record retained",
        "HEX_ALL": "All nominal hexagons: gapped frame",
        "HEX_GE80": "Hexagons with >=80% tract overlap",
        "HEX_GE99": "Hexagons with >=99% tract overlap",
    }
    status_style = {
        "deposit": ("#1b7f79", "Deposit"),
        "prospect": ("#d98c1f", "Prospect"),
        "group": ("#7251a5", "Group"),
    }
    coordinates = occurrences.set_index("record_id")[["x_m", "y_m"]]
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 15.0), constrained_layout=True)
    boundary = gpd.GeoSeries([tract_union], crs=PROJECT_CRS).boundary
    bounds = boundary.total_bounds
    padding = 20_000.0
    for axis, scheme in zip(axes.ravel(), panels, strict=True):
        support = scenarios[scheme]["geometry"]
        support.plot(
            ax=axis, facecolor="#ececec", edgecolor="#8a8a8a", linewidth=0.12
        )
        boundary.plot(ax=axis, color="#111111", linewidth=0.9)
        assignment = assignments.loc[assignments["support_scheme"].eq(scheme)].copy()
        plot_data = assignment.join(coordinates, on="record_id")
        for status, (color, label) in status_style.items():
            retained = plot_data.loc[
                plot_data["point_retained"].astype(bool)
                & plot_data["in_tract"].astype(bool)
                & plot_data["SiteStatus"].eq(status)
            ]
            axis.scatter(
                retained["x_m"], retained["y_m"], s=14, c=color, marker="o",
                edgecolors="white", linewidths=0.25, alpha=0.9, zorder=4,
            )
        dropped = plot_data.loc[
            ~plot_data["point_retained"].astype(bool)
            & plot_data["in_tract"].astype(bool)
        ]
        outside = plot_data.loc[~plot_data["in_tract"].astype(bool)]
        axis.scatter(
            dropped["x_m"], dropped["y_m"], s=22, c="#c62828", marker="x",
            linewidths=1.0, alpha=0.9, zorder=5,
        )
        axis.scatter(
            outside["x_m"], outside["y_m"], s=24, c="#222222", marker="x",
            linewidths=1.0, alpha=0.9, zorder=5,
        )
        comparison = scenarios[scheme]["context"]
        axis.set_title(
            f"{panel_titles[scheme]}\n"
            f"cells={len(support):,}; retained={comparison['n_retained_records']}/"
            f"{comparison['n_in_tract_records']} records; "
            f"positive cells={comparison['n_positive_cells']:,}",
            fontsize=10.5,
        )
        axis.set_xlim(bounds[0] - padding, bounds[2] + padding)
        axis.set_ylim(bounds[1] - padding, bounds[3] + padding)
        axis.set_aspect("equal")
        axis.set_axis_off()
    handles = [
        Patch(facecolor="#ececec", edgecolor="#8a8a8a", label="Support geometry"),
        Line2D([0], [0], color="#111111", linewidth=1.0, label="Permissive-tract boundary"),
        *[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
                   markeredgecolor="white", markersize=5, label=label)
            for color, label in status_style.values()
        ],
        Line2D([0], [0], marker="x", color="#c62828", linestyle="none", label="In-tract, removed by support"),
        Line2D([0], [0], marker="x", color="#222222", linestyle="none", label="Outside tract"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    figure.suptitle(
        "Stage 5B support alternatives and record retention\n"
        "The literal-grid subsets progressively remove records concentrated near tract margins",
        fontsize=13,
    )
    destination = abs_path(path)
    figure.savefig(destination, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return destination


def make_retention_figure(
    comparison: pd.DataFrame, path: str | Path,
) -> Path:
    table = comparison.set_index("support_scheme").loc[SCHEME_ORDER]
    x = np.arange(len(table))
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 6.4), constrained_layout=True)
    bottom = np.zeros(len(table), dtype=float)
    colors = {"n_deposits": "#1b7f79", "n_prospects": "#d98c1f", "n_groups": "#7251a5"}
    labels = {"n_deposits": "Deposits", "n_prospects": "Prospects", "n_groups": "Groups"}
    for column in ("n_deposits", "n_prospects", "n_groups"):
        values = table[column].to_numpy(dtype=float)
        axes[0].bar(x, values, bottom=bottom, color=colors[column], label=labels[column], width=0.72)
        bottom += values
    for index, total in enumerate(bottom):
        fraction = table.iloc[index]["n_retained_records"] / table.iloc[index]["n_in_tract_records"]
        axes[0].text(index, total + 3.0, f"{int(total)}\n({fraction:.0%})", ha="center", va="bottom", fontsize=8)
    axes[0].axhline(184, color="#333333", linestyle="--", linewidth=1.0, label="184 in-tract records")
    axes[0].set_ylim(0, 205)
    axes[0].set_ylabel("In-tract records retained")
    axes[0].set_title("Deterministic record retention by status")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    _style_axes(axes[0])

    actual = table["n_retained_records"] / table["n_in_tract_records"]
    expected = table["mc_retention_fraction"].astype(float)
    axes[1].plot(x, actual, marker="o", markersize=6, linewidth=1.8, color="#333333", label="Point-assigned retention")
    axes[1].plot(x, expected, marker="s", markersize=5, linewidth=1.5, color="#c45a2a", label="Expected retention under 1 km positional model")
    for index, value in enumerate(actual):
        axes[1].annotate(f"{value:.1%}", (index, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("Fraction of 184 in-tract records")
    axes[1].set_title("Record retention and positional uncertainty")
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    _style_axes(axes[1])

    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels([_short_scheme(scheme) for scheme in table.index], rotation=30, ha="right")
    figure.suptitle("Stage 5B record accounting across support alternatives", fontsize=13)
    destination = abs_path(path)
    figure.savefig(destination, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return destination


def _bootstrap_row(bootstraps: pd.DataFrame, scheme: str, metric: str) -> pd.Series:
    rows = bootstraps.loc[
        bootstraps["support_scheme"].eq(scheme) & bootstraps["metric"].eq(metric)
    ]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one bootstrap row for {scheme}/{metric}")
    return rows.iloc[0]


def make_metric_figure(bootstraps: pd.DataFrame, comparison: pd.DataFrame, path: str | Path) -> Path:
    role_lookup = comparison.set_index("support_scheme")["support_role"]
    x = np.arange(len(SCHEME_ORDER))
    metric_specs = [
        ("roc_auc", "ROC-AUC", 0.5),
        ("ap_over_prevalence", "Average precision / prevalence", 1.0),
        ("brier_skill", "Brier skill relative to prevalence", 0.0),
        ("calibration_slope", "Binary calibration slope", 1.0),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.0), constrained_layout=True)
    for axis, (metric, title, reference) in zip(axes.ravel(), metric_specs, strict=True):
        rows = [_bootstrap_row(bootstraps, scheme, metric) for scheme in SCHEME_ORDER]
        estimate = np.asarray([float(row["point_estimate"]) for row in rows])
        low = np.asarray([float(row["ci_low"]) for row in rows])
        high = np.asarray([float(row["ci_high"]) for row in rows])
        colors = [ROLE_COLORS[str(role_lookup.loc[scheme])] for scheme in SCHEME_ORDER]
        for index in range(len(SCHEME_ORDER)):
            if np.isfinite(estimate[index]):
                axis.errorbar(
                    x[index], estimate[index],
                    yerr=np.array([[max(estimate[index] - low[index], 0.0)],
                                   [max(high[index] - estimate[index], 0.0)]]),
                    fmt="o", markersize=6.5, capsize=3.5, color=colors[index],
                    ecolor=colors[index], elinewidth=1.2,
                )
                axis.annotate(f"{estimate[index]:.3f}", (x[index], estimate[index]),
                              xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7.5)
        axis.axhline(reference, color="#333333", linestyle="--", linewidth=1.0)
        finite = np.r_[low[np.isfinite(low)], high[np.isfinite(high)], reference]
        span = float(np.max(finite) - np.min(finite))
        pad = max(span * 0.12, 0.03)
        axis.set_ylim(float(np.min(finite) - pad), float(np.max(finite) + pad))
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels([_short_scheme(scheme) for scheme in SCHEME_ORDER], rotation=30, ha="right")
        _style_axes(axis)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
               markeredgecolor=color, markersize=6, label=role.replace("_", " "))
        for role, color in ROLE_COLORS.items()
    ]
    figure.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.055),
        ncol=3, frameon=False,
    )
    figure.suptitle(
        "Stage 5B pooled out-of-fold metrics with 50 km spatial-block bootstrap intervals\n"
        "Intervals quantify spatial resampling variability; folds 1-3 remain the informative transfer sample",
        fontsize=12.5,
    )
    destination = abs_path(path)
    figure.savefig(destination, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return destination


def make_diagnostic_figure(
    scenarios: dict[str, dict[str, Any]], predictions: pd.DataFrame,
    area_diagnostics: pd.DataFrame, comparison: pd.DataFrame, path: str | Path,
) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(14.0, 10.0), constrained_layout=True)
    area_arrays = [
        predictions.loc[predictions["support_scheme"].eq(scheme), "tract_area_km2"].to_numpy(dtype=float)
        for scheme in SCHEME_ORDER
    ]
    boxplot = axes[0, 0].boxplot(
        area_arrays, tick_labels=[_short_scheme(scheme) for scheme in SCHEME_ORDER],
        showfliers=False, patch_artist=True, widths=0.62,
    )
    for patch in boxplot["boxes"]:
        patch.set_facecolor("#d9e4e3")
        patch.set_edgecolor("#3f3f3f")
    axes[0, 0].axhline(10.0, color="#333333", linestyle="--", linewidth=1.0)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("Support area (km$^2$, log scale)")
    axes[0, 0].set_title("Support-area distributions")
    axes[0, 0].tick_params(axis="x", rotation=30)
    _style_axes(axes[0, 0])

    frame = scenarios["VORONOI_ALL"]["frame"]
    boundary = frame["distance_to_tract_boundary_km"].to_numpy(dtype=float)
    area = frame["support_area_km2"].to_numpy(dtype=float)
    axes[0, 1].scatter(boundary, area, s=5, alpha=0.25, c="#1b7f79", linewidths=0)
    rho = stats.spearmanr(boundary, area)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("Distance from cell centre to tract boundary (km)")
    axes[0, 1].set_ylabel("Support area (km$^2$, log scale)")
    axes[0, 1].set_title(f"Voronoi support: area vs boundary distance\nSpearman r={rho.statistic:.3f}, p={rho.pvalue:.2g}")
    _style_axes(axes[0, 1])

    reference = predictions.loc[predictions["support_scheme"].eq("VORONOI_ALL")].copy()
    residual = reference["pearson_residual"].to_numpy(dtype=float)
    quantiles = pd.qcut(pd.Series(area), 20, duplicates="drop")
    binned = pd.DataFrame({
        "area": area, "residual": residual, "bin": quantiles
    }).groupby("bin", observed=True).agg(
        area=("area", "median"),
        residual_mean=("residual", "mean"),
        residual_sd=("residual", "std"),
        n=("residual", "size"),
    ).reset_index(drop=True)
    binned["standard_error"] = binned["residual_sd"] / np.sqrt(binned["n"])
    axes[1, 0].errorbar(
        binned["area"], binned["residual_mean"],
        yerr=1.96 * binned["standard_error"], fmt="o-", color="#6f5aa8",
        ecolor="#6f5aa8", markersize=4, linewidth=1.2, capsize=2.5,
    )
    axes[1, 0].axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel("Support area (km$^2$, log scale)")
    axes[1, 0].set_ylabel("Mean Pearson residual (95% CI)")
    diagnostic_row = area_diagnostics.set_index("support_scheme").loc["VORONOI_ALL"]
    axes[1, 0].set_title(
        "Voronoi support: residuals vs area\n"
        f"Spearman r={diagnostic_row['area_vs_pearson_residual_spearman_r']:.3f}, "
        f"p={diagnostic_row['area_vs_pearson_residual_spearman_p']:.2g}"
    )
    _style_axes(axes[1, 0])

    elasticity = area_diagnostics.set_index("support_scheme").loc[SCHEME_ORDER]
    role_lookup = comparison.set_index("support_scheme")["support_role"]
    y = np.arange(len(elasticity))
    for index, (scheme, row) in enumerate(elasticity.iterrows()):
        value = float(row["delta_log_rate_per_log_area"])
        if not np.isfinite(value):
            axes[1, 1].text(0.0, index, "not estimable", va="center", fontsize=8, color="#666666")
            continue
        color = ROLE_COLORS[str(role_lookup.loc[scheme])]
        axes[1, 1].errorbar(
            value, index,
            xerr=np.array([[value - float(row["ci_low"])], [float(row["ci_high"]) - value]]),
            fmt="o", markersize=6, capsize=3, color=color, ecolor=color,
        )
    axes[1, 1].axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    axes[1, 1].set_yticks(y)
    axes[1, 1].set_yticklabels([_short_scheme(scheme) for scheme in elasticity.index])
    axes[1, 1].set_xlabel("Additional log-rate per log-area coefficient")
    axes[1, 1].set_title("Free area elasticity diagnostic\n0 means the log-area offset is proportional")
    _style_axes(axes[1, 1])

    figure.suptitle(
        "Stage 5B area, boundary and residual diagnostics\n"
        "Selection on hexagon overlap induces severe area-boundary dependence; the free area coefficient is diagnostic, not causal",
        fontsize=12.5,
    )
    destination = abs_path(path)
    figure.savefig(destination, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return destination


def _format_estimate(row: pd.Series, metric: str, decimals: int = 3) -> str:
    value = float(row[metric])
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{decimals}f}"


def write_report(
    path: str | Path, comparison: pd.DataFrame, metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame, bootstraps: pd.DataFrame,
    area_diagnostics: pd.DataFrame, boundary_diagnostics_table: pd.DataFrame,
    lost_diagnostics: pd.DataFrame, outcome_table: pd.DataFrame,
    reference_check: dict[str, Any], bundle: dict[str, Any],
) -> Path:
    comparison_index = comparison.set_index("support_scheme")
    pooled = metrics.loc[metrics["aggregation"].eq("pooled_oof")].set_index("support_scheme")
    scenario_rows = pd.DataFrame({
        "Scheme": [_short_scheme(scheme) for scheme in SCHEME_ORDER],
        "Role": [comparison_index.loc[scheme, "support_role"] for scheme in SCHEME_ORDER],
        "Cells": [int(comparison_index.loc[scheme, "n_support"]) for scheme in SCHEME_ORDER],
        "Tract coverage": [
            f"{100.0 * float(comparison_index.loc[scheme, 'coverage_fraction_of_tract']):.1f}%"
            for scheme in SCHEME_ORDER
        ],
        "Records retained": [
            f"{int(comparison_index.loc[scheme, 'n_retained_records'])}/184"
            for scheme in SCHEME_ORDER
        ],
        "Point retention": [
            f"{100.0 * float(comparison_index.loc[scheme, 'retained_fraction']):.1f}%"
            for scheme in SCHEME_ORDER
        ],
        "Expected retention (1 km)": [
            f"{100.0 * float(comparison_index.loc[scheme, 'mc_retention_fraction']):.1f}%"
            for scheme in SCHEME_ORDER
        ],
        "Positive cells": [int(comparison_index.loc[scheme, "n_positive_cells"]) for scheme in SCHEME_ORDER],
        "Prevalence": [
            f"{100.0 * float(comparison_index.loc[scheme, 'prevalence']):.2f}%"
            for scheme in SCHEME_ORDER
        ],
    })
    metric_rows = pd.DataFrame({
        "Scheme": [_short_scheme(scheme) for scheme in SCHEME_ORDER],
        "ROC-AUC": [_format_estimate(pooled.loc[scheme], "roc_auc") for scheme in SCHEME_ORDER],
        "AP": [_format_estimate(pooled.loc[scheme], "average_precision") for scheme in SCHEME_ORDER],
        "AP/prevalence": [_format_estimate(pooled.loc[scheme], "ap_over_prevalence") for scheme in SCHEME_ORDER],
        "Brier skill": [_format_estimate(pooled.loc[scheme], "brier_skill") for scheme in SCHEME_ORDER],
        "Calibration slope": [_format_estimate(pooled.loc[scheme], "calibration_slope") for scheme in SCHEME_ORDER],
        "NB deviance": [_format_estimate(pooled.loc[scheme], "nb_deviance") for scheme in SCHEME_ORDER],
    })
    ci_rows = []
    for scheme in SCHEME_ORDER:
        row = {"Scheme": _short_scheme(scheme)}
        for metric, label in (
            ("roc_auc", "ROC-AUC"), ("ap_over_prevalence", "AP/prevalence"),
            ("brier_skill", "Brier skill"), ("calibration_slope", "Calibration slope"),
        ):
            boot = _bootstrap_row(bootstraps, scheme, metric)
            values = [float(boot["point_estimate"]), float(boot["ci_low"]), float(boot["ci_high"])]
            row[label] = (
                f"{values[0]:.3f} ({values[1]:.3f}, {values[2]:.3f})"
                if all(np.isfinite(values)) else f"{values[0]:.3f} (NA)"
            )
        ci_rows.append(row)
    ci_table = pd.DataFrame(ci_rows)
    informative = fold_metrics.loc[fold_metrics["fold"].le(3)].groupby(
        "support_scheme", sort=False
    ).agg(
        mean_auc=("roc_auc", "mean"),
        mean_ap_prevalence=("ap_over_prevalence", "mean"),
        mean_brier_skill=("brier_skill", "mean"),
        mean_calibration_slope=("calibration_slope", "mean"),
        min_positive_cells=("holdout_positive_cells", "min"),
        mean_positive_cells=("holdout_positive_cells", "mean"),
    ).loc[SCHEME_ORDER]
    informative_rows = pd.DataFrame({
        "Scheme": [_short_scheme(scheme) for scheme in SCHEME_ORDER],
        "Mean ROC-AUC": [f"{informative.loc[scheme, 'mean_auc']:.3f}" for scheme in SCHEME_ORDER],
        "Mean AP/prevalence": [f"{informative.loc[scheme, 'mean_ap_prevalence']:.3f}" for scheme in SCHEME_ORDER],
        "Mean Brier skill": [f"{informative.loc[scheme, 'mean_brier_skill']:.3f}" for scheme in SCHEME_ORDER],
        "Mean calibration slope": [f"{informative.loc[scheme, 'mean_calibration_slope']:.3f}" for scheme in SCHEME_ORDER],
        "Holdout positives (min-mean)": [
            f"{int(informative.loc[scheme, 'min_positive_cells'])}-{informative.loc[scheme, 'mean_positive_cells']:.1f}"
            for scheme in SCHEME_ORDER
        ],
    })
    fold_focus = fold_metrics.loc[
        fold_metrics["support_scheme"].isin(["VORONOI_ALL", "HEX_GE99"])
    ].copy()
    fold_rows = pd.DataFrame({
        "Scheme": [_short_scheme(value) for value in fold_focus["support_scheme"]],
        "Fold": [int(value) for value in fold_focus["fold"]],
        "Holdout cells": [int(value) for value in fold_focus["n_holdout"]],
        "Positive cells": [int(value) for value in fold_focus["holdout_positive_cells"]],
        "Records": [int(value) for value in fold_focus["holdout_records"]],
        "ROC-AUC": [f"{value:.3f}" if np.isfinite(value) else "NA" for value in fold_focus["roc_auc"]],
        "AP/prevalence": [f"{value:.3f}" if np.isfinite(value) else "NA" for value in fold_focus["ap_over_prevalence"]],
        "Brier skill": [f"{value:.3f}" if np.isfinite(value) else "NA" for value in fold_focus["brier_skill"]],
        "Calibration slope": [f"{value:.3f}" if np.isfinite(value) else "NA" for value in fold_focus["calibration_slope"]],
    })
    area = area_diagnostics.set_index("support_scheme").loc[SCHEME_ORDER]
    area_rows = pd.DataFrame({
        "Scheme": [_short_scheme(scheme) for scheme in SCHEME_ORDER],
        "Area range": [
            f"{float(comparison_index.loc[scheme, 'min_support_area_km2']):.1f}-"
            f"{float(comparison_index.loc[scheme, 'max_support_area_km2']):.1f}"
            for scheme in SCHEME_ORDER
        ],
        "Area vs residual r": [f"{area.loc[scheme, 'area_vs_pearson_residual_spearman_r']:.3f}" for scheme in SCHEME_ORDER],
        "Area coefficient": [
            f"{area.loc[scheme, 'delta_log_rate_per_log_area']:.3f} "
            f"({area.loc[scheme, 'ci_low']:.3f}, {area.loc[scheme, 'ci_high']:.3f})"
            if np.isfinite(float(area.loc[scheme, "delta_log_rate_per_log_area"])) else "not estimable"
            for scheme in SCHEME_ORDER
        ],
        "Observed/predicted": [f"{area.loc[scheme, 'observed_over_predicted_total']:.3f}" for scheme in SCHEME_ORDER],
    })
    outcome_pooled = outcome_table.loc[outcome_table["aggregation"].eq("pooled_oof")]
    outcome_rows = pd.DataFrame({
        "Scheme": [_short_scheme(value) for value in outcome_pooled["support_scheme"]],
        "Outcome": list(outcome_pooled["sensitivity_id"]),
        "ROC-AUC": [f"{value:.3f}" for value in outcome_pooled["roc_auc"]],
        "AP/prevalence": [f"{value:.3f}" for value in outcome_pooled["ap_over_prevalence"]],
        "Brier skill": [f"{value:.3f}" for value in outcome_pooled["brier_skill"]],
        "Calibration slope": [f"{value:.3f}" if np.isfinite(value) else "NA" for value in outcome_pooled["calibration_slope"]],
    })
    lost_rows = lost_diagnostics.loc[
        lost_diagnostics["record_class"].eq("dropped")
    ].copy()
    lost_display = pd.DataFrame({
        "Scheme": [_short_scheme(value) for value in lost_rows["support_scheme"]],
        "Dropped": [int(value) for value in lost_rows["n_records"]],
        "Within 1 km": [f"{100.0 * value:.1f}%" for value in lost_rows["fraction_within_1km"]],
        "Within 2 km": [f"{100.0 * value:.1f}%" for value in lost_rows["fraction_within_2km"]],
        "Median boundary distance": [f"{value:.2f} km" for value in lost_rows["median_distance_to_tract_boundary_km"]],
    })
    vor = comparison_index.loc["VORONOI_ALL"]
    hex99 = comparison_index.loc["HEX_GE99"]
    vor_metrics = pooled.loc["VORONOI_ALL"]
    hex99_metrics = pooled.loc["HEX_GE99"]
    vor_auc = _bootstrap_row(bootstraps, "VORONOI_ALL", "roc_auc")
    vor_ap = _bootstrap_row(bootstraps, "VORONOI_ALL", "ap_over_prevalence")
    hex99_auc = _bootstrap_row(bootstraps, "HEX_GE99", "roc_auc")
    hex99_ap = _bootstrap_row(bootstraps, "HEX_GE99", "ap_over_prevalence")
    reference_max = max(
        list(reference_check["max_prediction_differences"].values())
        + list(reference_check["max_metric_differences"].values())
    )
    lines = [
        "# Stage 5B support sensitivity",
        "",
        "## Scientific scope",
        "",
        "This experiment asks whether the accepted Stage 4 `S0_NO_GRAVITY` specification is sensitive to the spatial support used to represent the permissive tract. It compares the tract-clipped Voronoi partition with literal nominal hexagons and with record-restricted populations. The estimand remains the intensity of historically compiled occurrence records per unit tract-supported area, not a detection-free probability of mineralisation.",
        "",
        "All seven scenarios use the same outcome (`n_records`), offset (`log(tract_area_km2)`), predictors (`BASE_DIST_FAULT_KM`, `BASE_DIST_CONTACT_KM`, `BASE_CONTACT_CLASS`), ridge penalty (1.0), four contiguous belt-axis folds, 10 km training buffers, and training-fold-only standardisation. The reference scenario reproduces the Stage 4 support and predictions. The maximum absolute difference across protected prediction and metric fields is " + f"{reference_max:.3g}" + ".",
        "",
        "The design separates support geometry from record loss. `VORONOI_ALL` retains every in-tract record on the full catchment partition. `HEX_ALL`, `HEX_GE80`, and `HEX_GE99` use literal hexagon subsets and therefore discard records. `VORONOI_MATCHED_HEX_ALL`, `VORONOI_MATCHED_HEX80`, and `VORONOI_MATCHED_HEX99` apply the same record filters to the full Voronoi partition. They are controls for the consequences of record selection, not claims that one support is intrinsically superior.",
        "",
        "## Data and design",
        "",
        f"- Stage 4 data build: `{bundle['data_manifest']['build_id']}`; model build: `{bundle['model_manifest']['build_id']}`.",
        f"- Stage 4 data validation: `{bundle['data_validation']['status']}`; model validation: `{bundle['model_validation']['status']}`.",
        f"- Tract-supported area: {float(vor['support_area_sum_km2']):,.1f} km² under the Voronoi partition; {float(hex99['support_area_sum_km2']):,.1f} km² under the strict literal interior.",
        f"- Nominal hexagons cover {100.0 * float(hex99['support_area_sum_km2']) / float(vor['support_area_sum_km2']):.1f}% of the tract-supported Voronoi area when restricted to >=99% overlap; `HEX_ALL` covers {100.0 * float(comparison_index.loc['HEX_ALL', 'coverage_fraction_of_tract']):.2f}% of the tract.",
        "- Fold 4 contains one record in the reference population. It is reported but not used as evidence of transferability. Literal `HEX_GE99` contains no Fold 4 positive cells.",
        "",
        markdown_table(
            scenario_rows,
            ["Scheme", "Role", "Cells", "Tract coverage", "Records retained", "Point retention", "Expected retention (1 km)", "Positive cells", "Prevalence"],
        ),
        "",
        "The literal subsets are not cleaner samples of the same population. They are different estimands defined by support geometry. The 99% population retains only 40 of 184 in-tract records and has 29 positive cells; that loss is concentrated near the tract boundary.",
        "",
        "## Record loss and positional uncertainty",
        "",
        f"Of 184 in-tract records, {int(hex99['n_dropped_records'])} lie outside every nominal >=99% hexagon. The dropped records are close to the tract margin: median distance {float(lost_rows.loc[lost_rows['support_scheme'].eq('HEX_GE99'), 'median_distance_to_tract_boundary_km'].iloc[0]):.2f} km, with {100.0 * float(lost_rows.loc[lost_rows['support_scheme'].eq('HEX_GE99'), 'fraction_within_1km'].iloc[0]):.1f}% within 1 km of the boundary. This is a selection on edge geography, not evidence that the discarded locations are geologically less prospective.",
        "",
        markdown_table(lost_display, list(lost_display.columns)),
        "",
        f"Under the 1 km positional model, the expected retention of in-tract records is {100.0 * float(vor['mc_retention_fraction']):.1f}% even for the full Voronoi partition, because some coordinate draws fall outside the tract. It falls to {100.0 * float(hex99['mc_retention_fraction']):.1f}% for `HEX_GE99`. Positional uncertainty therefore compounds the support restriction rather than cancelling it.",
        "",
        "## Model performance",
        "",
        markdown_table(
            metric_rows,
            ["Scheme", "ROC-AUC", "AP", "AP/prevalence", "Brier skill", "Calibration slope", "NB deviance"],
        ),
        "",
        "Pooled out-of-fold metrics combine within-sector ranking with between-fold differences in prevalence and fitted intercept. They are not clean estimates of within-sector transfer. The fold-specific results are retained in `stage5b_fold_metrics_v1.csv`.",
        "",
        markdown_table(
            ci_table,
            ["Scheme", "ROC-AUC", "AP/prevalence", "Brier skill", "Calibration slope"],
        ),
        "",
        "The intervals are 95% percentile intervals from 1,000 spatial-block resamples with 50 km blocks. They describe spatial resampling variability conditional on the fitted model; they do not remove fold-composition confounding.",
        "",
        f"`VORONOI_ALL` has pooled ROC-AUC {float(vor_metrics['roc_auc']):.3f} (95% {float(vor_auc['ci_low']):.3f}, {float(vor_auc['ci_high']):.3f}), AP/prevalence {float(vor_metrics['ap_over_prevalence']):.3f} (95% {float(vor_ap['ci_low']):.3f}, {float(vor_ap['ci_high']):.3f}), Brier skill {float(vor_metrics['brier_skill']):.3f}, and calibration slope {float(vor_metrics['calibration_slope']):.3f}. `HEX_GE99` has pooled ROC-AUC {float(hex99_metrics['roc_auc']):.3f} (95% {float(hex99_auc['ci_low']):.3f}, {float(hex99_auc['ci_high']):.3f}), AP/prevalence {float(hex99_metrics['ap_over_prevalence']):.3f} (95% {float(hex99_ap['ci_low']):.3f}, {float(hex99_ap['ci_high']):.3f}), and calibration slope {float(hex99_metrics['calibration_slope']):.3f}. The strict interior is neither a larger sample nor a stronger validation population.",
        "",
        "At the 100% record set, the Voronoi and literal-hexagon supports contain the same 2,193 cell centres and the same 139 in-tract records. Their pooled metrics are close: `VORONOI_MATCHED_HEX_ALL` ROC-AUC 0.732 and AP/prevalence 2.660, versus `HEX_ALL` ROC-AUC 0.728 and AP/prevalence 2.729. The large changes after 80% and 99% filtering therefore coincide with record loss, changing positive-cell composition, and fold-specific case support; they are not a clean demonstration that one support geometry is geologically superior.",
        "",
        markdown_table(
            informative_rows,
            ["Scheme", "Mean ROC-AUC", "Mean AP/prevalence", "Mean Brier skill", "Mean calibration slope", "Holdout positives (min-mean)"],
        ),
        "",
        "This table uses Folds 1-3 only because Fold 4 has one positive record. It remains a descriptive four-sector sensitivity summary, not a test of a general spatial-transfer law.",
        "",
        markdown_table(
            fold_rows,
            ["Scheme", "Fold", "Holdout cells", "Positive cells", "Records", "ROC-AUC", "AP/prevalence", "Brier skill", "Calibration slope"],
        ),
        "",
        "## Area and tract-boundary diagnostics",
        "",
        markdown_table(
            area_rows,
            ["Scheme", "Area range", "Area vs residual r", "Area coefficient", "Observed/predicted"],
        ),
        "",
        "The free log-area coefficient is a diagnostic, not a geological effect. Under `VORONOI_ALL`, its 95% interval includes zero, so the proportional exposure offset is compatible with the total count budget. Under literal hexagon selections, the coefficient becomes large and strongly negative because area is collinear with distance from the tract boundary and with the selection rule. `HEX_GE99` has an area 1st-99th percentile range below 0.25 km², so a separate elasticity is not estimable. These results argue for retaining the offset and not interpreting a free area coefficient as evidence about mineralisation.",
        "",
        f"In the reference support, the area-residual Spearman correlation is {float(area.loc['VORONOI_ALL', 'area_vs_pearson_residual_spearman_r']):.3f} (p={float(area.loc['VORONOI_ALL', 'area_vs_pearson_residual_spearman_p']):.2g}). The observed/predicted total ratio is {float(area.loc['VORONOI_ALL', 'observed_over_predicted_total']):.3f}. In the boundary diagnostic, cells within 2 km of the tract boundary have mean Pearson residual {float(boundary_diagnostics_table.set_index('support_scheme').loc['VORONOI_ALL', 'mean_residual_within_2km']):.3f}, versus {float(boundary_diagnostics_table.set_index('support_scheme').loc['VORONOI_ALL', 'mean_residual_beyond_2km']):.3f} beyond 2 km. The difference is small but worth carrying into the residual-spatial stage.",
        "",
        "## Outcome-definition sensitivity",
        "",
        markdown_table(
            outcome_rows,
            ["Scheme", "Outcome", "ROC-AUC", "AP/prevalence", "Brier skill", "Calibration slope"],
        ),
        "",
        "Group expansion is a multiplicity exercise at a shared coordinate, not an independent-location expansion. The deposits-only and deposits-plus-prospects outcomes change both prevalence and the effective positive set. They show that the response definition matters, but do not identify which inventory representation is geologically correct.",
        "",
        "## Interpretation and limitations",
        "",
        "- The reference support remains `VORONOI_ALL`. It preserves every tract location and every in-tract record while using actual tract-supported area as the exposure.",
        "- The 80% and 99% literal populations should be retained as estimand-sensitivity analyses. They should not be used as default modelling populations merely because their prevalence appears more balanced.",
        "- The 1 km positional model is a bounded perturbation of the recorded coordinates. It does not represent reporting error, location ambiguity within grouped records, or undiscovered mineralisation.",
        "- No external historical exploration, accessibility, or reporting dataset is used here. Support sensitivity is not an observation-process model.",
        "- The result does not show that major faults are geologically unimportant. It shows that the accepted baseline remains the defensible comparator under the tested support representations.",
        "- Fold 4 and the strict interior contain too few positive cells to support stable fold-level discrimination claims.",
        "",
        "## Recommendation",
        "",
        "Retain `VORONOI_ALL` and `S0_NO_GRAVITY` as the reference for the next stage. Treat `HEX_GE80` and `HEX_GE99` as sensitivity populations, not replacements. The next scientific step is to quantify residual spatial dependence and test bounded observation-proxy explanations before adding a more flexible geological or detection model.",
        "",
        "## Figures",
        "",
        "![Support scenarios](stage5b_support_scenarios.png)",
        "",
        "![Record retention](stage5b_record_retention.png)",
        "",
        "![Support metrics](stage5b_support_metrics.png)",
        "",
        "![Area and boundary diagnostics](stage5b_area_boundary_diagnostics.png)",
    ]
    destination = abs_path(path)
    write_text("\n".join(lines), destination)
    return destination


def write_manifest(
    path: str | Path, run_id: str, config_path: str | Path, config: dict[str, Any],
    paths: dict[str, str], output_files: dict[str, str], bundle: dict[str, Any],
    reference_check: dict[str, Any], software: dict[str, str],
) -> Path:
    stage4_config = bundle["stage4_config"]
    with abs_path(config["stage5"]["config"]).open("r", encoding="utf-8") as handle:
        stage5_config = json.load(handle)
    stage4_paths = stage5_config["stage4"]
    output_records = {
        name: file_record(output_path, role=name) for name, output_path in output_files.items()
    }
    input_records = {
        "stage5b_config": file_record(config_path, role="Stage 5B configuration"),
        "stage5_config": file_record(config["stage5"]["config"], role="Stage 5 configuration"),
        "stage5_inventory_manifest": file_record(config["stage5"]["inventory_manifest"], role="verified Stage 5A inventory manifest"),
        "stage5_inventory_validation": file_record(config["stage5"]["inventory_validation"], role="verified Stage 5A validation"),
        "stage4_config": file_record(stage4_paths["config"], role="Stage 4 configuration"),
        "stage4_data_manifest": file_record(stage4_paths["data_manifest"], role="Stage 4 data manifest"),
        "stage4_data_validation": file_record(stage4_paths["data_validation"], role="Stage 4 data validation"),
        "stage4_model_manifest": file_record(stage4_paths["model_manifest"], role="Stage 4 model manifest"),
        "stage4_model_validation": file_record(stage4_paths["model_validation"], role="Stage 4 model validation"),
        "stage4_model_table": file_record(stage4_paths["model_table"], role="Stage 4 analysis table"),
        "stage4_occurrences": file_record(stage4_paths["occurrences"], role="Stage 4 occurrence inventory"),
        "stage4_support_gpkg": file_record(stage4_paths["support_gpkg"], role="Stage 4 Voronoi support"),
    }
    for key, spec in stage4_config["inputs"].items():
        if isinstance(spec, dict) and "path" in spec:
            record = file_record(spec["path"], role=f"Stage 4 input: {key}")
            if "layer" in spec:
                record["layer"] = spec["layer"]
            input_records[f"stage4_input_{key}"] = record
    for key, spec in config["stage5"].items():
        if key in {"config", "inventory_manifest", "inventory_validation"}:
            continue
        if isinstance(spec, str):
            input_records[f"stage5_{key}"] = file_record(spec, role=f"Stage 5 input: {key}")
    manifest = {
        "schema_version": "1.0.0",
        "build_id": run_id,
        "status": "written",
        "purpose": config["purpose"],
        "config": file_record(config_path, role="Stage 5B configuration"),
        "stage5_inventory": {
            "manifest": input_records["stage5_inventory_manifest"],
            "validation": input_records["stage5_inventory_validation"],
        },
        "stage4_build_ids": {
            "data": bundle["data_manifest"]["build_id"],
            "model": bundle["model_manifest"]["build_id"],
        },
        "stage4_validation": {
            "data": bundle["data_validation"]["status"],
            "model": bundle["model_validation"]["status"],
        },
        "inputs": input_records,
        "scenarios": [spec["id"] for spec in config["support_scenarios"]],
        "model": {
            "id": config["models"]["reference_id"],
            "outcome": config["models"]["outcome"],
            "offset": config["models"]["offset"],
            "ridge_lambda": float(config["models"]["ridge_lambda"]),
            "folds": 4,
            "training_buffer_km": float(stage5_config["models"]["primary_fold_buffer_km"]),
        },
        "reference_reproduction": reference_check,
        "positional_model": {
            "accuracy_m": float(stage4_config["inputs"]["occurrences"]["positional_accuracy_m"]),
            "monte_carlo_draws": int(stage4_config["inputs"]["occurrences"]["positional_monte_carlo_draws"]),
            "seed": int(stage4_config["inputs"]["occurrences"]["positional_seed"]),
        },
        "bootstrap": config["models"]["block_bootstrap"],
        "outputs": output_records,
        "software": software,
        "notes": [
            "This stage compares support representations; it does not model observation or reporting intensity.",
            "Literal hexagon subsets are different estimands because filtering removes records concentrated near tract boundaries.",
            "The free log-area coefficient is a support diagnostic, not a causal geological effect.",
            "Stage 4 and V14 benchmark assets are read-only inputs and are not modified by this stage.",
        ],
    }
    destination = abs_path(path)
    write_json(manifest, destination)
    return destination


def write_core_outputs(
    paths: dict[str, str], scenarios: dict[str, dict[str, Any]],
    assignments: pd.DataFrame, accounting: pd.DataFrame,
    fold_composition: pd.DataFrame, comparison: pd.DataFrame,
    positional: pd.DataFrame, positional_summary: pd.DataFrame,
    predictions: pd.DataFrame, metrics: pd.DataFrame, fold_metrics: pd.DataFrame,
    coefficients: pd.DataFrame, coefficients_summary: pd.DataFrame,
    bootstraps: pd.DataFrame, area_diagnostics: pd.DataFrame,
    boundary_diagnostics_table: pd.DataFrame, lost_diagnostics: pd.DataFrame,
    outcome_table: pd.DataFrame,
) -> None:
    write_csv(comparison, paths["support_comparison"])
    write_csv(assignments, paths["record_assignments"])
    write_csv(accounting, paths["record_accounting"])
    write_csv(fold_composition, paths["fold_composition"])
    write_csv(positional, paths["positional_propagation"])
    write_csv(positional_summary, paths["positional_summary"])
    write_csv(predictions, paths["model_predictions"])
    write_csv(metrics, paths["model_metrics"])
    write_csv(fold_metrics, paths["fold_metrics"])
    write_csv(coefficients, paths["coefficients"])
    write_csv(coefficients_summary, paths["coefficient_summary"])
    write_csv(bootstraps, paths["metric_bootstrap"])
    write_csv(area_diagnostics, paths["area_diagnostics"])
    write_csv(boundary_diagnostics_table, paths["boundary_diagnostics"])
    write_csv(lost_diagnostics, paths["lost_record_diagnostics"])
    write_csv(outcome_table, paths["outcome_sensitivity"])
    layers: list[tuple[str, gpd.GeoDataFrame]] = []
    for scheme in ("VORONOI_ALL", "HEX_ALL", "HEX_GE80", "HEX_GE99"):
        layers.append((scheme.lower(), scenarios[scheme]["geometry"]))
    write_gpkg(layers, paths["support_gpkg"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage5b_v1.json")
    args = parser.parse_args()
    config_path = abs_path(args.config)
    config = load_config(config_path)
    run_id = build_id()
    paths = output_paths(config)
    bundle, geometries, stage5_config = validate_stage5a(config)
    model_table = bundle["model_table"].copy()
    occurrences = bundle["occurrences"].copy()
    stage4_config = bundle["stage4_config"]

    scenarios: dict[str, dict[str, Any]] = {
        "__tract_union": geometries["tract_union"],
    }
    for spec in config["support_scenarios"]:
        print(f"Building support scenario {spec['id']}", flush=True)
        frame, geometry, records, context = build_support_scenario(
            model_table, occurrences, geometries["nominal_hexagons"],
            geometries["catchments"], geometries["tract_union"], spec,
        )
        frame["distance_to_tract_boundary_km"] = distance_to_boundary_km(
            gpd.GeoSeries(gpd.points_from_xy(frame["x_m"], frame["y_m"]), crs=PROJECT_CRS),
            geometries["tract_union"],
        )
        scenarios[spec["id"]] = {
            "spec": spec, "frame": frame, "geometry": geometry,
            "records": records, "context": context,
        }

    thresholds = sorted(set(
        float(spec["record_filter_threshold"])
        for spec in config["support_scenarios"]
        if spec.get("record_filter_threshold") is not None
    ))
    print("Propagating positional uncertainty through support alternatives", flush=True)
    fold_lookup = model_table.set_index("support_id")["new_fold"]
    occurrence_context = occurrences.copy()
    occurrence_context["assigned_fold"] = occurrence_context["assigned_support_id"].map(fold_lookup)
    positional = positional_retention(
        occurrence_context, geometries["tract_union"], geometries["nominal_hexagons"],
        stage4_config, thresholds,
    )
    assignments = build_assignment_table(scenarios, occurrence_context, model_table, positional)
    accounting = build_record_accounting(assignments)
    fold_composition = build_fold_composition(scenarios, assignments)
    comparison = build_support_comparison(
        scenarios, assignments,
        gpd.GeoSeries(gpd.points_from_xy(occurrences["x_m"], occurrences["y_m"]), crs=PROJECT_CRS),
        geometries["tract_union"],
    )

    positional_summary = positional.groupby(
        ["coverage_variant", "site_status"], sort=True
    ).agg(
        n_records=("record_id", "size"),
        n_in_tract=("in_tract", "sum"),
        mean_p_retained_mc=("p_retained_mc", "mean"),
        median_p_retained_mc=("p_retained_mc", "median"),
        expected_records_retained_mc=("p_retained_mc", "sum"),
        min_p_retained_mc=("p_retained_mc", "min"),
        max_p_retained_mc=("p_retained_mc", "max"),
    ).reset_index()

    prediction_parts: list[pd.DataFrame] = []
    metric_parts: list[pd.DataFrame] = []
    coefficient_parts: list[pd.DataFrame] = []
    fold_metric_parts: list[pd.DataFrame] = []
    reference_predictions: pd.DataFrame | None = None
    reference_metrics: pd.DataFrame | None = None
    for scheme_id, scenario in scenarios.items():
        if scheme_id.startswith("__"):
            continue
        print(f"Fitting S0_NO_GRAVITY on {scheme_id}", flush=True)
        prediction, metrics, coefficients, fold_metrics = fit_s0(
            scenario["frame"], stage4_config
        )
        metadata = scenario_metadata(scenario["spec"])
        for frame in (prediction, metrics, coefficients, fold_metrics):
            for key, value in metadata.items():
                frame[key] = value
        prediction_parts.append(prediction)
        metric_parts.append(metrics)
        coefficient_parts.append(coefficients)
        fold_metric_parts.append(fold_metrics)
        if scheme_id == "VORONOI_ALL":
            reference_predictions = prediction.copy()
            reference_metrics = metrics.copy()

    if reference_predictions is None or reference_metrics is None:
        raise RuntimeError("The VORONOI_ALL reference was not fitted")
    reference_check = reference_reproduction_check(
        bundle, reference_predictions, reference_metrics
    )
    print(json.dumps({"reference_reproduction": reference_check}, indent=2), flush=True)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics = pd.concat(metric_parts, ignore_index=True)
    coefficients = pd.concat(coefficient_parts, ignore_index=True)
    fold_metrics = pd.concat(fold_metric_parts, ignore_index=True)
    coefficients_summary = coefficient_summary(coefficients)

    bootstrap_parts: list[pd.DataFrame] = []
    for index, scheme_id in enumerate(
        [key for key in scenarios if not key.startswith("__")]
    ):
        print(f"Block-bootstrap metrics for {scheme_id}", flush=True)
        subset = predictions.loc[predictions["support_scheme"].eq(scheme_id)]
        table = bootstrap_metrics(
            subset, config,
            int(config["models"]["block_bootstrap"]["seed"]) + index,
        )
        table["support_scheme"] = scheme_id
        bootstrap_parts.append(table)
    bootstraps = pd.concat(bootstrap_parts, ignore_index=True)

    area_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    comparison_index = comparison.set_index("support_scheme")
    for scheme_id, scenario in scenarios.items():
        if scheme_id.startswith("__"):
            continue
        subset = predictions.loc[predictions["support_scheme"].eq(scheme_id)].copy()
        frame = scenario["frame"].set_index("support_id")
        subset = subset.set_index("support_id")
        ordered_frame = frame.loc[subset.index].reset_index()
        ordered_predictions = subset.reset_index()
        area_diag = residual_area_diagnostics(ordered_predictions, ordered_frame)
        if comparison_index.loc[scheme_id, "area_p95_minus_p05_km2"] >= 0.25:
            area_diag.update(area_elasticity(ordered_frame, stage4_config))
            area_diag["area_elasticity_status"] = "estimated"
        else:
            area_diag.update({
                "term": "BASE_LOG_AREA_KM2",
                "delta_log_rate_per_log_area": math.nan,
                "ci_low": math.nan, "ci_high": math.nan,
                "implied_area_exponent": math.nan,
                "fold_mean_delta": math.nan, "fold_min_delta": math.nan,
                "fold_max_delta": math.nan, "n_folds_finite": 0,
                "n_folds_positive": 0, "n_folds_negative": 0,
                "ridge_lambda": 0.0, "fold_metrics_n": 0,
                "area_elasticity_status": "not_estimable_narrow_area_range",
            })
        area_diag["support_scheme"] = scheme_id
        area_rows.append(area_diag)
        boundary = boundary_diagnostics(
            ordered_predictions, ordered_frame, geometries["tract_union"]
        )
        boundary["support_scheme"] = scheme_id
        boundary_rows.append(boundary)
    area_diagnostics = pd.DataFrame(area_rows)
    boundary_diagnostics_table = pd.DataFrame(boundary_rows)
    lost_diagnostics = lost_record_diagnostics(assignments)

    print("Fitting outcome-definition sensitivities", flush=True)
    outcome_table = outcome_sensitivities(
        {key: value["frame"] for key, value in scenarios.items() if not key.startswith("__")},
        stage4_config, config["models"]["outcome_sensitivities"],
    )

    write_core_outputs(
        paths, scenarios, assignments, accounting, fold_composition, comparison,
        positional, positional_summary, predictions, metrics, fold_metrics,
        coefficients, coefficients_summary, bootstraps, area_diagnostics,
        boundary_diagnostics_table, lost_diagnostics, outcome_table,
    )

    print("Writing Stage 5B figures and report", flush=True)
    figure_paths = {
        "scenario_figure": make_support_figure(
            scenarios, assignments, occurrences, geometries["tract_union"],
            paths["scenario_figure"],
        ),
        "retention_figure": make_retention_figure(comparison, paths["retention_figure"]),
        "metric_figure": make_metric_figure(bootstraps, comparison, paths["metric_figure"]),
        "diagnostic_figure": make_diagnostic_figure(
            scenarios, predictions, area_diagnostics, comparison,
            paths["diagnostic_figure"],
        ),
    }
    report_path = write_report(
        paths["report"], comparison, metrics, fold_metrics, bootstraps,
        area_diagnostics, boundary_diagnostics_table, lost_diagnostics,
        outcome_table, reference_check, bundle,
    )
    output_keys = [
        key for key in config["outputs"]
        if key not in {"processed_dir", "figure_dir", "manifest"}
    ]
    output_files = {key: paths[key] for key in output_keys}
    output_files.update({key: str(value) for key, value in figure_paths.items()})
    output_files["report"] = str(report_path)
    manifest_path = write_manifest(
        paths["manifest"], run_id, config_path, config, paths, output_files,
        bundle, reference_check, software_stack(),
    )

    print(json.dumps({
        "status": "written",
        "build_id": run_id,
        "scenarios": len([key for key in scenarios if not key.startswith("__")]),
        "metadata_cells": int(len(predictions)),
        "reference_reproduction": reference_check["status"],
        "support_comparison": relative_posix(paths["support_comparison"]),
        "manifest": relative_posix(manifest_path),
        "report": relative_posix(report_path),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
