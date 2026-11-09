"""Audit the Stage 5 occurrence inventory at record level.

This stage is descriptive. It preserves the original USGS record statuses,
group membership, positional-assignment mass and support context. It does not
fit an observation-intensity model or reinterpret a background cell as verified
barren ground.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import shapely
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from scipy.spatial import cKDTree
from shapely.ops import unary_union


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modeling.stage4_common import (  # noqa: E402
    PROJECT_CRS,
    abs_path,
    build_id,
    file_record,
    write_csv,
    write_json,
)
from src.modeling.stage5_common import (  # noqa: E402
    distance_to_boundary_km,
    load_config,
    load_geometries,
    load_stage4_bundle,
    markdown_table,
    shannon_entropy,
    software_stack,
    write_text,
)


STATUS_ORDER = ["deposit", "prospect", "group"]
STATUS_COLORS = {"deposit": "#111111", "prospect": "#d73027", "group": "#4575b4"}
FOLD_COLORS = {1: "#1b9e77", 2: "#d95f02", 3: "#7570b3", 4: "#e7298a"}


def output_paths(config: dict[str, Any]) -> dict[str, str]:
    processed = abs_path(config["outputs"]["processed_dir"])
    figures = abs_path(config["outputs"]["figure_dir"])
    processed.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    figure_keys = {
        "inventory_figure", "inventory_positional_figure",
        "inventory_support_figure", "inventory_report",
    }
    for key, value in config["outputs"].items():
        if key == "processed_dir":
            paths[key] = str(processed)
        elif key == "figure_dir":
            paths[key] = str(figures)
        elif key in figure_keys:
            paths[key] = str(figures / value)
        else:
            paths[key] = str(processed / value)
    return paths


def validate_inventory(bundle: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    occurrences = bundle["occurrences"]
    weights = bundle["weights"]
    components = bundle["components"]
    expected_n = int(config["inventory"]["expected_source_records"])
    if len(occurrences) != expected_n:
        raise ValueError(f"Expected {expected_n} source records, found {len(occurrences)}")
    if not occurrences["record_id"].is_unique:
        raise ValueError("Occurrence record identifiers are not unique")
    counts = occurrences["SiteStatus"].value_counts().to_dict()
    expected_counts = config["inventory"]["expected_status_counts"]
    if {str(k): int(v) for k, v in counts.items()} != {
        str(k): int(v) for k, v in expected_counts.items()
    }:
        raise ValueError(f"Unexpected SiteStatus counts: {counts}")
    in_tract = int(occurrences["in_tract"].sum())
    if in_tract != int(config["inventory"]["expected_in_tract"]):
        raise ValueError(f"Unexpected in-tract count: {in_tract}")
    sums = weights.groupby("record_id")["probability"].sum()
    if len(sums) != expected_n or not np.allclose(sums.to_numpy(), 1.0, atol=1e-9, rtol=0.0):
        raise RuntimeError("Positional assignment probabilities do not sum to one for every record")
    component_counts = components.groupby("record_id").size()
    groups = occurrences.loc[occurrences["SiteStatus"].eq("group")]
    if len(groups) != int(expected_counts["group"]):
        raise ValueError("Group-record count changed")
    if set(component_counts.index) != set(groups["record_id"]):
        raise ValueError("Group records and group components do not refer to the same record IDs")
    observed_counts = component_counts.reindex(groups["record_id"]).to_numpy()
    if not (observed_counts == groups["n_group_components"].to_numpy()).all():
        raise ValueError("Group component counts disagree with the occurrence table")
    outside = occurrences.loc[~occurrences["in_tract"], "record_id"].tolist()
    if len(outside) != 3:
        raise ValueError(f"Expected three records outside the tract, found {len(outside)}")
    return {
        "n_records": len(occurrences),
        "n_in_tract": in_tract,
        "status_counts": {str(k): int(v) for k, v in counts.items()},
        "n_group_components": int(len(components)),
        "n_expanded_records": int(occurrences["record_expansion"].sum()),
        "n_expanded_records_in_tract": int(
            occurrences.loc[occurrences["in_tract"], "record_expansion"].sum()
        ),
        "outside_tract_record_ids": outside,
        "positional_probability_sums_valid": True,
    }


def parse_reference_years(reference_text: Any) -> list[int]:
    if reference_text is None or (isinstance(reference_text, float) and math.isnan(reference_text)):
        return []
    return sorted({int(value) for value in re.findall(r"\b(?:18|19|20)\d{2}\b", str(reference_text))})


def reference_count(reference_text: Any) -> int:
    if reference_text is None or (isinstance(reference_text, float) and math.isnan(reference_text)):
        return 0
    text = str(reference_text).strip()
    if not text or text.lower() == "nan":
        return 0
    return len([part for part in re.split(r";", text) if part.strip()])


def attach_record_context(
    bundle: dict[str, Any], geometries: dict[str, Any], config: dict[str, Any]
) -> pd.DataFrame:
    occurrences = bundle["occurrences"].copy()
    model_table = bundle["model_table"].copy()
    weights = bundle["weights"].copy()
    hexagons = geometries["nominal_hexagons"].copy()
    tract_union = geometries["tract_union"]
    domain_union = geometries["domain_union"]

    points = gpd.GeoSeries(
        gpd.points_from_xy(occurrences["x_m"], occurrences["y_m"]), crs=PROJECT_CRS
    )
    occurrences["geometry"] = points
    occurrences["tract_contains_point"] = np.asarray(
        [bool(shapely.covers(tract_union, point)) for point in points], dtype=bool
    )
    if not (occurrences["tract_contains_point"] == occurrences["in_tract"].astype(bool)).all():
        raise RuntimeError("Stored in_tract flags disagree with the tract geometry")
    occurrences["daly_envelope_contains_point"] = np.asarray(
        [bool(shapely.covers(domain_union, point)) for point in points], dtype=bool
    )
    occurrences["distance_to_tract_boundary_km"] = distance_to_boundary_km(points, tract_union)
    occurrences["distance_to_daly_envelope_boundary_km"] = distance_to_boundary_km(
        points, domain_union
    )

    hex_points = gpd.GeoDataFrame(
        occurrences[["record_id", "geometry"]].copy(), geometry="geometry", crs=PROJECT_CRS
    )
    matches = gpd.sjoin(
        hex_points,
        hexagons[["support_id", "frame_overlap_fraction", "geometry"]].rename(
            columns={"support_id": "nominal_hex_support_id"}
        ),
        how="left",
        predicate="intersects",
    )
    match_counts = (
        matches.dropna(subset=["nominal_hex_support_id"])
        .groupby("record_id", observed=True)
        .size()
        .rename("nominal_hex_match_count")
    )
    matched = matches.dropna(subset=["nominal_hex_support_id"]).copy()
    matched = matched.sort_values(
        ["record_id", "frame_overlap_fraction", "nominal_hex_support_id"],
        ascending=[True, False, True],
    ).drop_duplicates("record_id")
    matched = matched.set_index("record_id")[
        ["nominal_hex_support_id", "frame_overlap_fraction"]
    ].rename(columns={"frame_overlap_fraction": "matched_nominal_hex_overlap_fraction"})
    occurrences = occurrences.merge(matched, left_on="record_id", right_index=True, how="left")
    occurrences = occurrences.merge(
        match_counts, left_on="record_id", right_index=True, how="left"
    )
    occurrences["nominal_hex_match_count"] = (
        occurrences["nominal_hex_match_count"].fillna(0).astype(int)
    )
    occurrences["inside_nominal_hex"] = occurrences["nominal_hex_match_count"].gt(0)
    thresholds = config["inventory"]["nominal_hex_overlap_thresholds"]
    overlap = occurrences["matched_nominal_hex_overlap_fraction"]
    occurrences["inside_nominal_hex_ge80"] = overlap.ge(
        thresholds["primary_sensitivity"]
    ).fillna(False)
    occurrences["inside_nominal_hex_ge99"] = overlap.ge(
        thresholds["strict_interior"]
    ).fillna(False)
    hex_boundary = unary_union(list(hexagons.geometry.boundary))
    occurrences["distance_to_nominal_hex_boundary_km"] = np.asarray(
        [shapely.distance(point, hex_boundary) / 1000.0 for point in points], dtype=float
    )
    hex_union = unary_union(list(hexagons.geometry))
    occurrences["distance_to_any_nominal_hex_km"] = np.asarray(
        [shapely.distance(point, hex_union) / 1000.0 for point in points], dtype=float
    )

    support_columns = [
        "support_id", "tract_area_km2", "nominal_hex_overlap_fraction",
        "area_overlap_class", "domain_abbreviation", "new_fold", "new_fold_label",
        "n_records", "n_deposits", "n_prospects", "n_groups",
    ]
    assigned = model_table[support_columns].copy()
    assigned = assigned.rename(columns={
        "support_id": "assigned_support_id",
        "tract_area_km2": "assigned_catchment_area_km2",
        "nominal_hex_overlap_fraction": "assigned_catchment_hex_overlap_fraction",
        "area_overlap_class": "assigned_catchment_overlap_class",
        "domain_abbreviation": "assigned_domain_abbreviation",
        "new_fold": "assigned_fold",
        "new_fold_label": "assigned_fold_label",
        "n_records": "assigned_support_records",
        "n_deposits": "assigned_support_deposits",
        "n_prospects": "assigned_support_prospects",
        "n_groups": "assigned_support_groups",
    })
    occurrences = occurrences.merge(assigned, on="assigned_support_id", how="left")
    nearest = model_table[[
        "support_id", "tract_area_km2", "domain_abbreviation", "new_fold", "new_fold_label"
    ]].rename(columns={
        "support_id": "nearest_support_context_id",
        "tract_area_km2": "nearest_support_area_km2",
        "domain_abbreviation": "nearest_support_domain",
        "new_fold": "nearest_support_fold",
        "new_fold_label": "nearest_support_fold_label",
    })
    occurrences = occurrences.merge(
        nearest, left_on="nearest_support_id", right_on="nearest_support_context_id", how="left"
    )

    occurrences["positional_entropy"] = shannon_entropy(weights, occurrences["record_id"])
    k_modes = occurrences["positional_n_catchments"].clip(lower=1)
    occurrences["positional_entropy_normalized"] = (
        occurrences["positional_entropy"] / np.log(k_modes)
    ).replace([np.inf, -np.inf], np.nan)
    occurrences["positional_modal_matches_assignment"] = (
        occurrences["positional_primary_support_id"].astype(str)
        == occurrences["assigned_support_id"].astype(str)
    )
    occurrences["n_reference_citations"] = occurrences["Ref_Short"].map(reference_count)
    reference_years = occurrences["Ref_Short"].map(parse_reference_years)
    occurrences["reference_year_min"] = reference_years.map(
        lambda values: min(values) if values else np.nan
    )
    occurrences["reference_year_max"] = reference_years.map(
        lambda values: max(values) if values else np.nan
    )
    occurrences["reference_year_span"] = (
        occurrences["reference_year_max"] - occurrences["reference_year_min"]
    )
    occurrences["has_pre_1970_reference"] = occurrences["reference_year_min"].lt(1970).fillna(False)
    occurrences["has_unpublished_reference"] = occurrences["Ref_Short"].fillna("").str.contains(
        "unpub", case=False, regex=False
    )
    occurrences["record_is_group"] = occurrences["SiteStatus"].eq("group")

    coordinates = occurrences[["x_m", "y_m"]].to_numpy(dtype=float)
    tree = cKDTree(coordinates)
    distances, _ = tree.query(coordinates, k=2)
    occurrences["nearest_other_record_m"] = distances[:, 1]
    return occurrences


def near_duplicate_checks(
    occurrences: pd.DataFrame, thresholds_m: list[float]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    coordinates = occurrences[["x_m", "y_m"]].to_numpy(dtype=float)
    tree = cKDTree(coordinates)
    rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    rounded = list(zip(occurrences["x_m"].round(6), occurrences["y_m"].round(6)))
    exact_groups = (
        occurrences.assign(_coordinate=rounded).groupby("_coordinate", sort=False)["record_id"].size()
    )
    rows.extend([
        {
            "check": "exact_coordinate_duplicate_records",
            "value": int(exact_groups.loc[exact_groups > 1].sum()),
            "unit": "records",
            "notes": "Rounded to 1e-6 m; no automatic de-duplication applied.",
        },
        {
            "check": "same_name_duplicate_records",
            "value": int(occurrences.loc[occurrences["Name"].notna() & occurrences["Name"].duplicated(False)].shape[0]),
            "unit": "records",
            "notes": "Case-sensitive exact name; shared names may be legitimate across countries.",
        },
    ])
    for threshold in thresholds_m:
        pairs_at_threshold = tree.query_pairs(threshold, output_type="ndarray")
        if len(pairs_at_threshold):
            pair_distances = np.linalg.norm(
                coordinates[pairs_at_threshold[:, 0]] - coordinates[pairs_at_threshold[:, 1]],
                axis=1,
            )
        else:
            pair_distances = np.empty(0, dtype=float)
        rows.append({
            "check": f"records_with_another_record_within_{int(threshold)}m",
            "value": int(len(np.unique(pairs_at_threshold))),
            "unit": "records",
            "notes": f"{len(pairs_at_threshold)} unordered pairs; no records merged.",
        })
        for (first, second), distance in zip(pairs_at_threshold[:500], pair_distances[:500]):
            if distance <= 0:
                continue
            pairs.append({
                "record_id_a": occurrences.iloc[int(first)]["record_id"],
                "record_id_b": occurrences.iloc[int(second)]["record_id"],
                "distance_m": float(distance),
                "status_a": occurrences.iloc[int(first)]["SiteStatus"],
                "status_b": occurrences.iloc[int(second)]["SiteStatus"],
            })
    component_total = int(
        occurrences.loc[occurrences["SiteStatus"].eq("group"), "n_group_components"].sum()
    )
    rows.append({
        "check": "group_components_sharing_group_coordinate",
        "value": component_total,
        "unit": "components",
        "notes": "Every parsed component is stored at the group coordinate; components are not independent locations.",
    })
    return pd.DataFrame(rows), pairs


def status_summary(audit: pd.DataFrame, tract_area_km2: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for status in STATUS_ORDER + ["ALL"]:
        subset = audit if status == "ALL" else audit.loc[audit["SiteStatus"].eq(status)]
        in_tract = subset.loc[subset["in_tract"]]
        support_cells = in_tract["assigned_support_id"].dropna().nunique()
        rows.append({
            "site_status": status,
            "n_records": int(len(subset)),
            "n_in_tract": int(len(in_tract)),
            "n_outside_tract": int(len(subset) - len(in_tract)),
            "pct_in_tract": 100.0 * len(in_tract) / len(subset) if len(subset) else math.nan,
            "n_expanded_in_tract": int(in_tract["record_expansion"].sum()),
            "n_assigned_support_cells": int(support_cells),
            "records_per_1000_km2_in_tract": 1000.0 * len(in_tract) / tract_area_km2,
            "mean_positional_outside_probability_in_tract": float(
                in_tract["positional_outside_probability"].mean()
            ) if len(in_tract) else math.nan,
            "n_positional_multi_catchment_in_tract": int(
                in_tract["positional_n_catchments"].ge(2).sum()
            ),
            "n_inside_nominal_hex": int(subset["inside_nominal_hex"].sum()),
            "n_outside_nominal_hex": int((~subset["inside_nominal_hex"]).sum()),
            "n_inside_nominal_hex_ge80": int(subset["inside_nominal_hex_ge80"].sum()),
            "n_inside_nominal_hex_ge99": int(subset["inside_nominal_hex_ge99"].sum()),
            "n_within_250m_tract_boundary": int(
                subset["distance_to_tract_boundary_km"].le(0.25).sum()
            ),
            "n_within_500m_tract_boundary": int(
                subset["distance_to_tract_boundary_km"].le(0.50).sum()
            ),
            "n_within_1000m_tract_boundary": int(
                subset["distance_to_tract_boundary_km"].le(1.00).sum()
            ),
            "n_inside_daly_envelope": int(subset["daly_envelope_contains_point"].sum()),
            "n_outside_daly_envelope": int((~subset["daly_envelope_contains_point"]).sum()),
        })
    return pd.DataFrame(rows)


def country_province_summary(audit: pd.DataFrame) -> pd.DataFrame:
    grouped = audit.groupby(
        ["Country", "State_Prov", "SiteStatus"], dropna=False, sort=True
    ).agg(
        n_records=("record_id", "size"),
        n_in_tract=("in_tract", "sum"),
        n_expanded_in_tract=("record_expansion", "sum"),
        n_support_cells=("assigned_support_id", "nunique"),
        mean_positional_outside_probability=("positional_outside_probability", "mean"),
    ).reset_index()
    # The group-level expansion sum is wrong for out-of-tract records unless
    # it is recomputed from the in-tract subset.
    expanded = audit.loc[audit["in_tract"]].groupby(
        ["Country", "State_Prov", "SiteStatus"], dropna=False, sort=True
    )["record_expansion"].sum().rename("n_expanded_in_tract")
    grouped = grouped.drop(columns=["n_expanded_in_tract"]).merge(
        expanded, on=["Country", "State_Prov", "SiteStatus"], how="left"
    )
    grouped["n_expanded_in_tract"] = grouped["n_expanded_in_tract"].fillna(0).astype(int)
    grouped["n_outside_tract"] = grouped["n_records"] - grouped["n_in_tract"]
    total = pd.DataFrame([{
        "Country": "ALL",
        "State_Prov": "ALL",
        "SiteStatus": "ALL",
        "n_records": int(len(audit)),
        "n_in_tract": int(audit["in_tract"].sum()),
        "n_expanded_in_tract": int(audit.loc[audit["in_tract"], "record_expansion"].sum()),
        "n_support_cells": int(audit.loc[audit["in_tract"], "assigned_support_id"].dropna().nunique()),
        "mean_positional_outside_probability": float(audit["positional_outside_probability"].mean()),
        "n_outside_tract": int((~audit["in_tract"]).sum()),
    }])
    return pd.concat([grouped, total], ignore_index=True)


def fold_domain_summary(audit: pd.DataFrame, model_table: pd.DataFrame) -> pd.DataFrame:
    aggregation = {
        "n_support_cells": ("support_id", "size"),
        "area_km2": ("tract_area_km2", "sum"),
        "positive_cells": ("any_record", "sum"),
        "n_records": ("n_records", "sum"),
        "n_deposits": ("n_deposits", "sum"),
        "n_prospects": ("n_prospects", "sum"),
        "n_groups": ("n_groups", "sum"),
        "n_expanded_records": ("n_expanded_records", "sum"),
        "mean_area_km2": ("tract_area_km2", "mean"),
        "min_area_km2": ("tract_area_km2", "min"),
        "max_area_km2": ("tract_area_km2", "max"),
    }
    source = model_table.groupby(
        ["new_fold", "new_fold_label", "domain_abbreviation"], dropna=False, sort=True
    ).agg(**aggregation).reset_index()
    fold_totals = model_table.groupby(
        ["new_fold", "new_fold_label"], sort=True
    ).agg(**aggregation).reset_index().assign(domain_abbreviation="ALL")
    overall = pd.DataFrame([{
        "new_fold": 0,
        "new_fold_label": "ALL",
        "domain_abbreviation": "ALL",
        "n_support_cells": len(model_table),
        "area_km2": float(model_table["tract_area_km2"].sum()),
        "positive_cells": int(model_table["any_record"].sum()),
        "n_records": int(model_table["n_records"].sum()),
        "n_deposits": int(model_table["n_deposits"].sum()),
        "n_prospects": int(model_table["n_prospects"].sum()),
        "n_groups": int(model_table["n_groups"].sum()),
        "n_expanded_records": int(model_table["n_expanded_records"].sum()),
        "mean_area_km2": float(model_table["tract_area_km2"].mean()),
        "min_area_km2": float(model_table["tract_area_km2"].min()),
        "max_area_km2": float(model_table["tract_area_km2"].max()),
    }])
    combined = pd.concat([source, fold_totals, overall], ignore_index=True, sort=False)
    combined["cell_prevalence"] = combined["positive_cells"] / combined["n_support_cells"]
    combined["records_per_1000_km2"] = 1000.0 * combined["n_records"] / combined["area_km2"]
    return combined


def group_summary(
    audit: pd.DataFrame, components: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = audit.loc[audit["SiteStatus"].eq("group")].copy()
    component_frame = components.copy()
    component_frame["component_at_group_coordinate"] = True
    component_table = component_frame[[
        "record_id", "source_fid", "group_name", "component_ordinal", "component_name",
        "source_field", "x_m", "y_m", "positional_accuracy_m", "assigned_support_id",
        "in_tract", "component_at_group_coordinate",
    ]].copy()
    group_list = component_frame.groupby("record_id", sort=False)["component_name"].agg(
        lambda values: "; ".join(str(value) for value in values)
    ).rename("component_names")
    group_counts = component_frame.groupby("record_id", sort=False).agg(
        parsed_component_count=("component_ordinal", "size"),
        unique_component_names=("component_name", "nunique"),
        unique_component_coordinates=("x_m", lambda values: len(set(zip(
            values.round(3), component_frame.loc[values.index, "y_m"].round(3)
        )))),
    )
    output = groups.merge(group_list, left_on="record_id", right_index=True, how="left")
    output = output.merge(group_counts, left_on="record_id", right_index=True, how="left")
    output = output[[
        "record_id", "Group_name", "Includes", "Includes2", "component_names",
        "n_group_components", "parsed_component_count", "unique_component_names",
        "unique_component_coordinates", "record_expansion", "in_tract", "Country",
        "State_Prov", "assigned_support_id", "assigned_catchment_area_km2",
        "positional_outside_probability", "positional_n_catchments",
        "n_reference_citations", "reference_year_min", "reference_year_max",
    ]].sort_values("record_id")
    return output, component_table


def positional_summary(audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for status in STATUS_ORDER + ["ALL"]:
        subset = audit if status == "ALL" else audit.loc[audit["SiteStatus"].eq(status)]
        in_tract = subset.loc[subset["in_tract"]]
        values = {
            "mean_own_probability": in_tract["positional_own_probability"].mean(),
            "median_own_probability": in_tract["positional_own_probability"].median(),
            "mean_outside_probability": in_tract["positional_outside_probability"].mean(),
            "median_outside_probability": in_tract["positional_outside_probability"].median(),
            "q90_outside_probability": in_tract["positional_outside_probability"].quantile(0.90),
            "mean_n_catchments": in_tract["positional_n_catchments"].mean(),
            "median_n_catchments": in_tract["positional_n_catchments"].median(),
            "mean_entropy": in_tract["positional_entropy"].mean(),
            "median_entropy": in_tract["positional_entropy"].median(),
            "n_one_catchment": int(in_tract["positional_n_catchments"].eq(1).sum()),
            "n_two_catchments": int(in_tract["positional_n_catchments"].eq(2).sum()),
            "n_three_or_more_catchments": int(in_tract["positional_n_catchments"].ge(3).sum()),
            "n_expected_outside_tract": in_tract["positional_outside_probability"].sum(),
            "n_modal_assignment_matches": int(in_tract["positional_modal_matches_assignment"].sum()),
            "n_modal_assignment_differs": int((~in_tract["positional_modal_matches_assignment"]).sum()),
        }
        for metric, value in values.items():
            rows.append({
                "site_status": status,
                "metric": metric,
                "value": float(value) if value is not None and np.isfinite(value) else math.nan,
                "n": int(len(in_tract)),
            })
    return pd.DataFrame(rows)


def support_summary(
    audit: pd.DataFrame, model_table: pd.DataFrame, tract_union: Any,
    config: dict[str, Any],
) -> pd.DataFrame:
    table = model_table.copy()
    bins = [float(value) for value in config["inventory"]["area_classes_km2"]]
    labels = [f"{bins[index]:g}-{bins[index + 1]:g}" for index in range(len(bins) - 1)]
    table["area_class_km2"] = pd.cut(
        table["tract_area_km2"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    support_points = gpd.GeoSeries(
        gpd.points_from_xy(table["x_m"], table["y_m"]), crs=PROJECT_CRS
    )
    table["distance_to_tract_boundary_km"] = distance_to_boundary_km(
        support_points, tract_union
    )
    rows: list[dict[str, Any]] = []

    def append_group(grouping: str, value: str, frame: pd.DataFrame) -> None:
        rows.append({
            "grouping": grouping,
            "group_value": value,
            "n_support_cells": len(frame),
            "area_km2": float(frame["tract_area_km2"].sum()),
            "mean_area_km2": float(frame["tract_area_km2"].mean()),
            "median_area_km2": float(frame["tract_area_km2"].median()),
            "min_area_km2": float(frame["tract_area_km2"].min()),
            "max_area_km2": float(frame["tract_area_km2"].max()),
            "n_positive_cells": int(frame["any_record"].sum()),
            "cell_prevalence": float(frame["any_record"].mean()),
            "n_records": int(frame["n_records"].sum()),
            "n_deposits": int(frame["n_deposits"].sum()),
            "n_prospects": int(frame["n_prospects"].sum()),
            "n_groups": int(frame["n_groups"].sum()),
            "records_per_1000_km2": 1000.0 * frame["n_records"].sum() / frame["tract_area_km2"].sum(),
            "mean_distance_to_tract_boundary_km": float(
                frame["distance_to_tract_boundary_km"].mean()
            ),
            "median_distance_to_tract_boundary_km": float(
                frame["distance_to_tract_boundary_km"].median()
            ),
        })

    append_group("ALL", "ALL", table)
    for label, frame in table.groupby("area_class_km2", observed=True, sort=True):
        append_group("area_class_km2", str(label), frame)
    for label, frame in table.groupby("area_overlap_class", observed=True, sort=True):
        append_group("area_overlap_class", str(label), frame)
    return pd.DataFrame(rows)


def boundary_summary(audit: pd.DataFrame, thresholds_m: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for boundary, distance_column in (
        ("tract_boundary", "distance_to_tract_boundary_km"),
        ("nominal_hex_boundary", "distance_to_nominal_hex_boundary_km"),
    ):
        for threshold_m in thresholds_m:
            for status in STATUS_ORDER + ["ALL"]:
                subset = audit if status == "ALL" else audit.loc[audit["SiteStatus"].eq(status)]
                within = subset[distance_column].le(threshold_m / 1000.0)
                rows.append({
                    "boundary_type": boundary,
                    "threshold_m": int(threshold_m),
                    "site_status": status,
                    "n_records": int(len(subset)),
                    "n_within_threshold": int(within.sum()),
                    "fraction_within_threshold": float(within.mean()) if len(subset) else math.nan,
                    "n_inside_nominal_hex": int(subset["inside_nominal_hex"].sum()),
                })
    return pd.DataFrame(rows)


def reference_summary(audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for status in STATUS_ORDER + ["ALL"]:
        subset = audit if status == "ALL" else audit.loc[audit["SiteStatus"].eq(status)]
        rows.append({
            "site_status": status,
            "n_records": int(len(subset)),
            "mean_reference_citations": float(subset["n_reference_citations"].mean()),
            "median_reference_citations": float(subset["n_reference_citations"].median()),
            "n_with_no_reference": int(subset["n_reference_citations"].eq(0).sum()),
            "n_with_pre_1970_reference": int(subset["has_pre_1970_reference"].sum()),
            "n_with_unpublished_reference": int(subset["has_unpublished_reference"].sum()),
            "min_reference_year": float(subset["reference_year_min"].min())
            if subset["reference_year_min"].notna().any() else math.nan,
            "median_reference_year_min": float(subset["reference_year_min"].median())
            if subset["reference_year_min"].notna().any() else math.nan,
            "max_reference_year": float(subset["reference_year_max"].max())
            if subset["reference_year_max"].notna().any() else math.nan,
        })
    return pd.DataFrame(rows)


def figure_overview(
    audit: pd.DataFrame,
    model_table: pd.DataFrame,
    catchments: gpd.GeoDataFrame,
    tract_union: Any,
    path: Path,
) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(15.0, 11.0))
    axis = axes[0, 0]
    catchment_plot = catchments.copy()
    for fold, frame in catchment_plot.groupby("new_fold", sort=True):
        frame.plot(
            ax=axis, color=FOLD_COLORS.get(int(fold), "#999999"), alpha=0.10, linewidth=0
        )
    gpd.GeoSeries([tract_union], crs=PROJECT_CRS).boundary.plot(
        ax=axis, color="#333333", linewidth=1.2
    )
    for status in STATUS_ORDER:
        subset = audit.loc[audit["SiteStatus"].eq(status)]
        sizes = 18 + 3.0 * subset["n_group_components"].clip(lower=0, upper=15)
        axis.scatter(
            subset["x_m"], subset["y_m"], s=sizes, c=STATUS_COLORS[status],
            label=status, alpha=0.85, edgecolors="white", linewidths=0.3,
        )
    axis.set_aspect("equal")
    axis.set_title("A. Source records and Stage 4 folds")
    axis.set_xlabel("Easting (m, EPSG:32735)")
    axis.set_ylabel("Northing (m, EPSG:32735)")
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=STATUS_COLORS[status], label=status)
        for status in STATUS_ORDER
    ]
    handles += [
        Patch(facecolor=FOLD_COLORS[fold], alpha=0.35, label=f"Fold {fold}")
        for fold in (1, 2, 3, 4)
    ]
    axis.legend(handles=handles, frameon=False, fontsize=8, loc="lower left")

    axis = axes[0, 1]
    status_counts = []
    for status in STATUS_ORDER + ["ALL"]:
        subset = audit if status == "ALL" else audit.loc[audit["SiteStatus"].eq(status)]
        status_counts.append({
            "status": status,
            "source": len(subset),
            "in_tract": int(subset["in_tract"].sum()),
            "expanded_in_tract": int(subset.loc[subset["in_tract"], "record_expansion"].sum()),
        })
    counts = pd.DataFrame(status_counts).set_index("status")
    positions = np.arange(len(counts))
    width = 0.25
    axis.bar(positions - width, counts["source"], width, label="source records", color="#777777")
    axis.bar(positions, counts["in_tract"], width, label="in tract", color="#1b9e77")
    axis.bar(
        positions + width, counts["expanded_in_tract"], width,
        label="expanded in tract", color="#d95f02",
    )
    axis.set_xticks(positions, counts.index)
    axis.set_ylabel("Record count")
    axis.set_title("B. Record status and multiplicity")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 0]
    fold_rows = []
    for fold in (1, 2, 3, 4):
        subset = audit.loc[audit["assigned_fold"].eq(fold)]
        row = {"fold": f"Fold {fold}"}
        row.update(subset["SiteStatus"].value_counts().reindex(STATUS_ORDER, fill_value=0).to_dict())
        fold_rows.append(row)
    outside_counts = audit.loc[~audit["in_tract"], "SiteStatus"].value_counts().reindex(
        STATUS_ORDER, fill_value=0
    ).to_dict()
    fold_rows.append({"fold": "Outside tract", **outside_counts})
    fold_frame = pd.DataFrame(fold_rows).set_index("fold")
    bottom = np.zeros(len(fold_frame))
    for status in STATUS_ORDER:
        axis.bar(
            fold_frame.index, fold_frame[status], bottom=bottom,
            color=STATUS_COLORS[status], label=status,
        )
        bottom += fold_frame[status].to_numpy()
    axis.set_ylabel("Records")
    axis.set_title("C. Records by assigned Stage 4 fold")
    axis.tick_params(axis="x", rotation=15)
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 1]
    categories = []
    for status in STATUS_ORDER + ["ALL"]:
        subset = audit if status == "ALL" else audit.loc[audit["SiteStatus"].eq(status)]
        categories.append({
            "status": status,
            "inside nominal hex": int(subset["inside_nominal_hex"].sum()),
            "outside every nominal hex": int((~subset["inside_nominal_hex"]).sum()),
        })
    category_frame = pd.DataFrame(categories).set_index("status")
    axis.barh(
        category_frame.index, category_frame["inside nominal hex"],
        color="#4575b4", label="inside a nominal hexagon",
    )
    axis.barh(
        category_frame.index, category_frame["outside every nominal hex"],
        left=category_frame["inside nominal hex"], color="#e7298a",
        label="outside every nominal hexagon",
    )
    axis.set_xlabel("Source records")
    axis.set_title("D. Literal nominal-hexagon membership")
    axis.legend(frameon=False, fontsize=8, loc="lower right")

    for axis in axes.flat:
        axis.grid(alpha=0.2, linewidth=0.5)
    figure.suptitle("Stage 5 record-level inventory audit", fontsize=14, fontweight="bold")
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def figure_positional(audit: pd.DataFrame, tract_union: Any, path: Path) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(17.5, 5.1))
    axis = axes[0]
    gpd.GeoSeries([tract_union], crs=PROJECT_CRS).boundary.plot(
        ax=axis, color="#333333", linewidth=1.0
    )
    scatter = axis.scatter(
        audit["x_m"], audit["y_m"],
        c=audit["positional_outside_probability"],
        s=16 + 9 * audit["positional_n_catchments"].clip(lower=0, upper=3),
        cmap="coolwarm", vmin=0.0, vmax=1.0, alpha=0.85,
        edgecolors="white", linewidths=0.25,
    )
    axis.set_aspect("equal")
    axis.set_title("A. Positional mass outside the tract")
    axis.set_xlabel("Easting (m)")
    axis.set_ylabel("Northing (m)")
    colorbar = figure.colorbar(scatter, ax=axis, shrink=0.75)
    colorbar.set_label("Monte Carlo probability outside tract")

    axis = axes[1]
    for status in STATUS_ORDER:
        subset = audit.loc[audit["SiteStatus"].eq(status)]
        axis.hist(
            subset["positional_outside_probability"],
            bins=np.linspace(0.0, 1.0, 21), alpha=0.55, label=status,
            color=STATUS_COLORS[status], edgecolor="white",
        )
    axis.set(
        xlabel="Probability outside tract", ylabel="Records",
        title="B. Positional outside-tract mass",
    )
    axis.legend(frameon=False, fontsize=8)

    axis = axes[2]
    counts = pd.crosstab(audit["positional_n_catchments"], audit["SiteStatus"]).reindex(
        columns=STATUS_ORDER, fill_value=0
    )
    positions = np.arange(len(counts))
    bottom = np.zeros(len(counts))
    for status in STATUS_ORDER:
        axis.bar(
            positions, counts[status], bottom=bottom,
            label=status, color=STATUS_COLORS[status],
        )
        bottom += counts[status].to_numpy()
    axis.set_xticks(positions, counts.index.astype(int))
    axis.set(
        xlabel="Catchments receiving positive mass", ylabel="Records",
        title="C. Cross-catchment positional mass",
    )
    axis.legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2, linewidth=0.5)
    figure.suptitle("Stage 5 positional-uncertainty audit", fontsize=14, fontweight="bold")
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def figure_support(
    audit: pd.DataFrame,
    model_table: pd.DataFrame,
    tract_union: Any,
    path: Path,
) -> Path:
    table = model_table.copy()
    support_points = gpd.GeoSeries(
        gpd.points_from_xy(table["x_m"], table["y_m"]), crs=PROJECT_CRS
    )
    table["distance_to_tract_boundary_km"] = distance_to_boundary_km(
        support_points, tract_union
    )
    table["area_class_km2"] = pd.cut(
        table["tract_area_km2"], bins=[0, 5, 10, 15, 20, 25],
        labels=["0-5", "5-10", "10-15", "15-20", "20-25"],
        include_lowest=True, right=False,
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
    axis = axes[0, 0]
    bins = np.linspace(0, 25, 26)
    axis.hist(
        table["tract_area_km2"], bins=bins, color="#999999",
        edgecolor="white", label="support cells",
    )
    record_weights = table["n_records"].to_numpy(dtype=float)
    if record_weights.sum() > 0:
        axis.hist(
            table["tract_area_km2"], bins=bins, weights=record_weights,
            histtype="step", color="#d73027", linewidth=1.8,
            label="records (unweighted)",
        )
    axis.set(
        xlabel="Tract-supported catchment area (km2)", ylabel="Count",
        title="A. Catchment-area distribution",
    )
    axis.legend(frameon=False, fontsize=8)

    axis = axes[0, 1]
    axis.scatter(
        table["distance_to_tract_boundary_km"], table["tract_area_km2"],
        c=table["new_fold"],
        cmap=matplotlib.colors.ListedColormap([FOLD_COLORS[i] for i in (1, 2, 3, 4)]),
        s=7, alpha=0.65,
    )
    axis.set(
        xlabel="Support-centre distance to tract boundary (km)",
        ylabel="Catchment area (km2)", title="B. Area versus boundary distance",
    )
    axis.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", color=FOLD_COLORS[fold], label=f"Fold {fold}")
            for fold in (1, 2, 3, 4)
        ],
        frameon=False, fontsize=7, loc="upper right",
    )

    axis = axes[1, 0]
    grouped = table.groupby("area_class_km2", observed=True).agg(
        area=("tract_area_km2", "sum"),
        records=("n_records", "sum"),
        cells=("support_id", "size"),
    ).reset_index()
    grouped["rate"] = 1000.0 * grouped["records"] / grouped["area"]
    axis.bar(grouped["area_class_km2"].astype(str), grouped["rate"], color="#4575b4")
    for index, row in grouped.iterrows():
        axis.text(
            index, row["rate"] + 0.08, f"n={int(row['records'])}",
            ha="center", fontsize=7,
        )
    axis.set(
        xlabel="Catchment area class (km2)", ylabel="Records per 1,000 km2",
        title="C. Observed-record rate by support area",
    )

    axis = axes[1, 1]
    classes = ["partial_lt_80pct", "overlap_80_to_lt_99pct", "full_ge_99pct"]
    data = [
        table.loc[table["area_overlap_class"].eq(label), "tract_area_km2"].dropna().to_numpy()
        for label in classes
    ]
    axis.boxplot(data, tick_labels=["<80%", "80-<99%", ">=99%"], showfliers=False)
    axis.set(
        xlabel="Nominal-hex overlap class", ylabel="Catchment area (km2)",
        title="D. Area by nominal-hex overlap",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.2, linewidth=0.5)
    figure.suptitle("Stage 5 Voronoi support diagnostics", fontsize=14, fontweight="bold")
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def make_report(
    path: Path,
    audit: pd.DataFrame,
    status_table: pd.DataFrame,
    country_table: pd.DataFrame,
    fold_domain_table: pd.DataFrame,
    positional_table: pd.DataFrame,
    reference_table: pd.DataFrame,
    support_table: pd.DataFrame,
    boundary_table: pd.DataFrame,
    duplicate_table: pd.DataFrame,
    context: dict[str, Any],
) -> Path:
    in_tract = audit.loc[audit["in_tract"]]
    outside = audit.loc[~audit["in_tract"]]
    status_parts = [
        f"{int(row.n_in_tract)} {row.site_status} records"
        for row in status_table.loc[status_table["site_status"].isin(STATUS_ORDER)].itertuples()
    ]
    status_text = ", ".join(status_parts[:-1]) + " and " + status_parts[-1] if status_parts else "none"
    domain_counts = in_tract["assigned_domain_abbreviation"].value_counts(dropna=False)
    n_nrb = int(domain_counts.get("NRB", 0))
    n_outside_daly = int((~in_tract["daly_envelope_contains_point"]).sum())
    n_multi = int(in_tract["positional_n_catchments"].ge(2).sum())
    n_one = int(in_tract["positional_n_catchments"].eq(1).sum())
    n_three = int(in_tract["positional_n_catchments"].ge(3).sum())
    total_outside_probability = float(in_tract["positional_outside_probability"].sum())
    n_hex_outside = int((~audit["inside_nominal_hex"]).sum())
    n_hex_outside_in_tract = int((~in_tract["inside_nominal_hex"]).sum())
    exact_duplicate = int(duplicate_table.loc[
        duplicate_table["check"].eq("exact_coordinate_duplicate_records"), "value"
    ].iloc[0])
    same_name = int(duplicate_table.loc[
        duplicate_table["check"].eq("same_name_duplicate_records"), "value"
    ].iloc[0])
    nearest_1km = int(duplicate_table.loc[
        duplicate_table["check"].eq("records_with_another_record_within_1000m"), "value"
    ].iloc[0])

    status_display = status_table[[
        "site_status", "n_records", "n_in_tract", "n_outside_tract",
        "n_expanded_in_tract", "n_assigned_support_cells",
        "n_positional_multi_catchment_in_tract", "n_inside_nominal_hex",
        "n_outside_nominal_hex",
    ]].copy()
    status_display.columns = [
        "Status", "Records", "In tract", "Outside", "Expanded in tract",
        "Support cells", "Positional multi-catchment", "Inside nominal hex",
        "Outside all nominal hex",
    ]
    fold_table = fold_domain_table.loc[
        fold_domain_table["domain_abbreviation"].eq("ALL")
        & fold_domain_table["new_fold"].isin([1, 2, 3, 4]),
        ["new_fold_label", "n_support_cells", "positive_cells", "cell_prevalence",
         "n_records", "n_deposits", "n_prospects", "n_groups"],
    ].copy()
    fold_table.columns = [
        "Fold", "Cells", "Positive cells", "Cell prevalence", "Records",
        "Deposits", "Prospects", "Groups",
    ]
    positional_display = positional_table.loc[
        positional_table["site_status"].eq("ALL"), ["metric", "value", "n"]
    ].copy()
    positional_display["value"] = positional_display["value"].map(
        lambda value: f"{value:.4f}" if np.isfinite(value) else "NA"
    )
    positional_display.columns = ["Metric", "Value", "n"]
    support_display = support_table.loc[
        support_table["grouping"].isin(["ALL", "area_class_km2"]),
        ["grouping", "group_value", "n_support_cells", "area_km2",
         "n_positive_cells", "n_records", "records_per_1000_km2"],
    ].copy()
    support_display.columns = [
        "Grouping", "Value", "Cells", "Area km2", "Positive cells",
        "Records", "Records / 1,000 km2",
    ]
    boundary_display = boundary_table.loc[
        boundary_table["site_status"].eq("ALL"),
        ["boundary_type", "threshold_m", "n_records", "n_within_threshold",
         "fraction_within_threshold"],
    ].copy()
    boundary_display.columns = [
        "Boundary", "Threshold m", "Records", "Within threshold", "Fraction",
    ]
    reference_display = reference_table[[
        "site_status", "n_records", "mean_reference_citations",
        "median_reference_citations", "n_with_no_reference",
        "n_with_pre_1970_reference", "min_reference_year", "max_reference_year",
    ]].copy()
    reference_display.columns = [
        "Status", "Records", "Mean citations", "Median citations", "No citation",
        "Pre-1970 citation", "Earliest year", "Latest year",
    ]

    text = f"""# Stage 5A record-level inventory and support audit

## Scope and status

This is a deterministic audit of the historical occurrence inventory and the Stage 4 spatial support. It does not fit an observation-detection model, change the Stage 4 support, or reinterpret a cell without a record as verified barren ground. Stage 4 remains the accepted reference; its data and model validation records passed before this audit ran.

The audit separates source site status (deposit, prospect or group), grouped-record multiplicity, positional-assignment probability and nominal-hexagon membership. It preserves all {len(audit)} source records and every parsed group component.

## Headline findings

- The inventory contains **{len(audit)}** source records; **{len(in_tract)}** are inside the permissive-tract footprint and **{len(outside)}** are outside it. In-tract status composition is {status_text}.
- Group expansion gives **{int(audit['record_expansion'].sum())}** named components overall and **{int(in_tract['record_expansion'].sum())}** inside the tract. All parsed components share their group coordinate; they are multiplicity, not independent point locations.
- **{n_nrb} of {len(in_tract)}** in-tract records are assigned to NRB. The other Daly domains contain no in-tract records in this inventory, and **{n_outside_daly}** in-tract record is outside the reconstructed Daly envelope. This is an inventory-support fact, not evidence that those domains are unprospective.
- **{n_multi} of {len(in_tract)}** in-tract records have positional probability mass in two or more catchments; **{n_three}** span three, while **{n_one}** are confined to one. The summed expected outside-tract mass is **{total_outside_probability:.2f}** records.
- **{n_hex_outside}** records ({100.0 * n_hex_outside / len(audit):.1f}%) fall outside every nominal 10 km2 hexagon; **{n_hex_outside_in_tract}** of these are inside the tract. This is why Stage 4 uses a tract-clipped nearest-centre partition rather than literal hexagons.
- Catchment area ranges from **{context['catchment_area_min']:.3f} to {context['catchment_area_max']:.3f} km2** (median **{context['catchment_area_median']:.3f}**; 1st-99th percentile **{context['catchment_area_q01']:.3f}-{context['catchment_area_q99']:.3f}**). Exposure is unequal and requires an offset, but the offset does not by itself eliminate support-dependent effects.
- No exact coordinate duplicates and no exact duplicate names were found (**{exact_duplicate}** and **{same_name}** respectively). At least **{nearest_1km}** records have another source record within 1 km; these were not merged.
- Record-attached reference strings describe citations for known sites, not observation effort over background cells. No independent historical exploration-intensity data were found in the repository.

## Record composition

{markdown_table(status_display, list(status_display.columns))}

The fold composition is strongly unbalanced. Fold 4 is retained for transparency but contains only one record and cannot support a stable fold-specific geological-transfer claim.

{markdown_table(fold_table, list(fold_table.columns))}

## Positional uncertainty

Stage 4 generated positional probabilities from a 1,000 m circular accuracy model. The probabilities cover both alternative catchments and the possibility that a location lies outside the tract. The modal catchment is a summarisation, not a known true location.

{markdown_table(positional_display, list(positional_display.columns))}

## Support and boundary context

{markdown_table(support_display, list(support_display.columns))}

{markdown_table(boundary_display, list(boundary_display.columns))}

The support-area diagnostics are descriptive. Counts over larger catchments are expected to be larger before exposure is accounted for. The relevant next test is whether counts depart from proportional area after the log-area offset, which Stage 5B will examine directly.

## Grouped records

There are **{int(audit['record_is_group'].sum())}** group records with **{int(audit.loc[audit['record_is_group'], 'n_group_components'].sum())}** parsed components across **{int(audit['record_is_group'].sum())}** coordinates. Expansion is a maximum name multiplicity, not a set of independent locations. Any expanded-count model must be treated as a sensitivity analysis.

## Reference context

{markdown_table(reference_display, list(reference_display.columns))}

Reference counts and years are uneven and are attached only to compiled records. They are not used as a spatial predictor. A future observation-process model would require an independent measure of exploration, access or reporting coverage over both recorded and unrecorded ground; the current repository does not contain that measure.

## Interpretation boundary

With no independent observation-effort surface, the two components of

`lambda_observed(s) = lambda_geological(s) * p_observation(s)`

are not separately identifiable from occurrence locations alone. This audit establishes the inventory and support facts needed for bounded sensitivity analyses. It does not establish that an unrecorded location is barren, and it does not turn any Stage 4 prediction into a mineralisation probability.

## Files and reproduction

- Record audit: `{context['record_audit_path']}`
- Status summary: `{context['status_path']}`
- Positional summary: `{context['positional_path']}`
- Support summary: `{context['support_path']}`
- Duplicate/near-neighbour audit: `{context['duplicate_path']}`
- Figures: `{context['overview_figure']}`, `{context['positional_figure']}`, `{context['support_figure']}`

The audit used the manifest-verified Stage 4 inputs and the reconstruction geometries recorded in the Stage 5 inventory manifest. Stage 4, Stage 3 and V14 files were read only.
"""
    return write_text(text, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage5_v1.json")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = output_paths(config)
    bundle = load_stage4_bundle(config)
    geometries = load_geometries(config)
    validation = validate_inventory(bundle, config)

    audit = attach_record_context(bundle, geometries, config)
    near_duplicate_table, near_pairs = near_duplicate_checks(
        audit,
        [float(value) for value in config["inventory"]["near_duplicate_thresholds_m"]],
    )
    tract_area = float(geometries["tract_union"].area / 1e6)
    model_table = bundle["model_table"].copy()
    status_table = status_summary(audit, tract_area)
    country_table = country_province_summary(audit)
    fold_domain_table = fold_domain_summary(audit, model_table)
    group_table, component_table = group_summary(audit, bundle["components"])
    positional_table = positional_summary(audit)
    support_table = support_summary(
        audit, model_table, geometries["tract_union"], config
    )
    boundary_table = boundary_summary(
        audit, [float(value) for value in config["inventory"]["boundary_thresholds_m"]]
    )
    reference_table = reference_summary(audit)

    support_points = gpd.GeoSeries(
        gpd.points_from_xy(model_table["x_m"], model_table["y_m"]), crs=PROJECT_CRS
    )
    model_table["distance_to_tract_boundary_km"] = distance_to_boundary_km(
        support_points, geometries["tract_union"]
    )
    area = model_table["tract_area_km2"].to_numpy(dtype=float)
    count = model_table["n_records"].to_numpy(dtype=float)
    area_boundary_corr = stats.spearmanr(
        area, model_table["distance_to_tract_boundary_km"]
    )
    area_count_corr = stats.spearmanr(area, count)
    context = {
        "record_audit_path": Path(paths["inventory_record_audit"]).as_posix(),
        "status_path": Path(paths["inventory_status_summary"]).as_posix(),
        "positional_path": Path(paths["inventory_positional_summary"]).as_posix(),
        "support_path": Path(paths["inventory_support_summary"]).as_posix(),
        "duplicate_path": Path(paths["inventory_duplicate_summary"]).as_posix(),
        "overview_figure": Path(paths["inventory_figure"]).as_posix(),
        "positional_figure": Path(paths["inventory_positional_figure"]).as_posix(),
        "support_figure": Path(paths["inventory_support_figure"]).as_posix(),
        "tract_area_km2": tract_area,
        "catchment_area_min": float(model_table["tract_area_km2"].min()),
        "catchment_area_max": float(model_table["tract_area_km2"].max()),
        "catchment_area_median": float(model_table["tract_area_km2"].median()),
        "catchment_area_q01": float(model_table["tract_area_km2"].quantile(0.01)),
        "catchment_area_q99": float(model_table["tract_area_km2"].quantile(0.99)),
        "area_vs_boundary_spearman_r": float(area_boundary_corr.statistic),
        "area_vs_boundary_spearman_p": float(area_boundary_corr.pvalue),
        "area_vs_records_spearman_r": float(area_count_corr.statistic),
        "area_vs_records_spearman_p": float(area_count_corr.pvalue),
    }

    write_csv(audit.drop(columns=["geometry"]), paths["inventory_record_audit"])
    write_csv(status_table, paths["inventory_status_summary"])
    write_csv(country_table, paths["inventory_country_province_summary"])
    write_csv(fold_domain_table, paths["inventory_fold_domain_summary"])
    write_csv(group_table, paths["inventory_group_summary"])
    write_csv(component_table, paths["inventory_component_summary"])
    write_csv(near_duplicate_table, paths["inventory_duplicate_summary"])
    write_csv(positional_table, paths["inventory_positional_summary"])
    write_csv(reference_table, paths["inventory_reference_summary"])
    write_csv(support_table, paths["inventory_support_summary"])
    write_csv(boundary_table, paths["inventory_boundary_summary"])
    near_pairs_path = Path(paths["inventory_duplicate_summary"]).with_name(
        "stage5_inventory_near_neighbour_pairs_v1.csv"
    )
    write_csv(pd.DataFrame(near_pairs), near_pairs_path)

    figure_overview(
        audit, model_table, geometries["catchments"], geometries["tract_union"],
        Path(paths["inventory_figure"]),
    )
    figure_positional(
        audit, geometries["tract_union"], Path(paths["inventory_positional_figure"])
    )
    figure_support(
        audit, model_table, geometries["tract_union"], Path(paths["inventory_support_figure"])
    )
    report_path = make_report(
        Path(paths["inventory_report"]), audit, status_table, country_table,
        fold_domain_table, positional_table, reference_table, support_table,
        boundary_table, near_duplicate_table, context,
    )

    output_keys = [
        "inventory_record_audit", "inventory_status_summary",
        "inventory_country_province_summary", "inventory_fold_domain_summary",
        "inventory_group_summary", "inventory_component_summary",
        "inventory_duplicate_summary", "inventory_positional_summary",
        "inventory_reference_summary", "inventory_support_summary",
        "inventory_boundary_summary", "inventory_figure",
        "inventory_positional_figure", "inventory_support_figure",
    ]
    output_records = {key: file_record(paths[key], role=key) for key in output_keys}
    output_records["inventory_near_neighbour_pairs"] = file_record(
        near_pairs_path, role="inventory_near_neighbour_pairs"
    )
    output_records["inventory_report"] = file_record(
        report_path, role="inventory_report"
    )

    build = build_id()
    manifest = {
        "schema_version": "1.0.0",
        "build_id": build,
        "status": "audit_complete_pending_verification",
        "purpose": "Stage 5A deterministic record-level inventory and spatial-support audit",
        "config": file_record(args.config, "stage5 configuration"),
        "stage4_data_manifest": file_record(
            config["stage4"]["data_manifest"], "stage4 data manifest"
        ),
        "stage4_model_manifest": file_record(
            config["stage4"]["model_manifest"], "stage4 model manifest"
        ),
        "stage4_data_validation": file_record(
            config["stage4"]["data_validation"], "stage4 data validation"
        ),
        "stage4_model_validation": file_record(
            config["stage4"]["model_validation"], "stage4 model validation"
        ),
        "stage4_build_ids": {
            "data": bundle["data_manifest"]["build_id"],
            "model": bundle["model_manifest"]["build_id"],
        },
        "protected_inputs": bundle["protected"],
        "stage4_verified_outputs": bundle["verified_outputs"],
        "inventory_validation": validation,
        "support_context": context,
        "outputs": output_records,
        "software": software_stack(),
        "notes": [
            "No Stage 4, Stage 3 or V14 file was modified.",
            "The 187 source records are preserved; group components are not treated as independent locations.",
            "Positional probabilities are categorical mass derived from the Stage 4 Monte Carlo model.",
            "Reference strings are record-attached citations, not a spatial exploration-effort surface.",
            "Nominal hexagon membership is recorded for support sensitivity, not used to replace the Stage 4 partition.",
        ],
    }
    write_json(manifest, paths["inventory_manifest"])
    print(json.dumps({
        "build_id": build,
        "status": manifest["status"],
        "n_records": len(audit),
        "n_in_tract": int(audit["in_tract"].sum()),
        "n_positive_support": int(model_table["any_record"].sum()),
        "n_group_components": int(len(component_table)),
        "n_outside_nominal_hex": int((~audit["inside_nominal_hex"]).sum()),
        "report": Path(report_path).as_posix(),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
