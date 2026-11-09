"""Build the Stage 4 spatial support, occurrence inventory and model table.

The script keeps the 2,193 accepted support centres, but replaces the
incomplete nominal hexagons with nearest-centre Voronoi catchments clipped to
the dissolved USGS permissive-tract footprint. Catchment area is the count-model
exposure. No Stage 3, V14 or manuscript file is modified.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, MultiPolygon, box
from shapely.ops import unary_union
from shapely.strtree import STRtree

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modeling.stage4_common import (  # noqa: E402
    PROJECT_CRS,
    abs_path,
    build_id,
    file_record,
    load_config,
    output_paths,
    read_gdf,
    relative_posix,
    sha256_file,
    software_versions,
    verify_expected_hash,
    verify_protected,
    write_csv,
    write_gpkg,
    write_json,
)


FOLD_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
STATUS_COLORS = {"deposit": "#111111", "prospect": "#d73027", "group": "#4575b4"}


def _polygonal_geometry(geometry: Any, label: str = "geometry") -> MultiPolygon:
    if geometry is None or geometry.is_empty:
        raise ValueError(f"Empty polygonal geometry for {label}")
    candidate = geometry
    if not shapely.is_valid(candidate):
        candidate = shapely.make_valid(candidate)
    if candidate.geom_type == "Polygon":
        return MultiPolygon([candidate])
    if candidate.geom_type == "MultiPolygon":
        return candidate
    parts: list[Any] = []
    for part in shapely.get_parts(candidate):
        if part.geom_type == "Polygon":
            parts.append(part)
        elif part.geom_type == "MultiPolygon":
            parts.extend(list(part.geoms))
    if not parts:
        raise ValueError(f"No polygonal component for {label}: {candidate.geom_type}")
    return MultiPolygon(parts)


def load_support_and_frame(config: dict[str, Any]) -> tuple[pd.DataFrame, gpd.GeoDataFrame, Any]:
    table_path = config["inputs"]["support_table"]["path"]
    expected = config["inputs"]["support_table"]["sha256"]
    support_hash = verify_expected_hash(table_path, expected, "support_table")
    support = pd.read_csv(abs_path(table_path))
    if len(support) != int(config["support"]["expected_n"]):
        raise ValueError(f"Unexpected support count: {len(support)}")
    if not support["support_id"].is_unique:
        raise ValueError("Support identifiers are not unique")
    support_points = read_gdf(
        config["inputs"]["support_gpkg"]["path"],
        layer=config["inputs"]["support_gpkg"]["layer"],
    )
    if len(support_points) != len(support):
        raise ValueError("Support point and table row counts differ")
    if support_points["support_id"].tolist() != support["support_id"].tolist():
        raise ValueError("Support point order differs from the feature table")
    point_xy = np.column_stack([support_points.geometry.x, support_points.geometry.y])
    table_xy = support[["x_m", "y_m"]].to_numpy()
    if not np.allclose(point_xy, table_xy, atol=1e-6, rtol=0.0):
        raise ValueError("Support point coordinates differ from the feature table")
    support.attrs["support_hash_record"] = support_hash

    tracts = read_gdf(config["inputs"]["tracts"]["path"])
    expected_tracts = int(config["inputs"]["tracts"]["expected_features"])
    if len(tracts) != expected_tracts:
        raise ValueError(f"Expected {expected_tracts} tract polygons, found {len(tracts)}")
    if not tracts.geometry.is_valid.all():
        tracts = tracts.copy()
        tracts["geometry"] = tracts.geometry.make_valid()
    tract_union = unary_union(list(tracts.geometry))
    if tract_union.is_empty:
        raise ValueError("Dissolved tract footprint is empty")
    if not shapely.is_valid(tract_union):
        tract_union = shapely.make_valid(tract_union)
    support.attrs["tract_union"] = tract_union
    support.attrs["tract_area_km2"] = float(tract_union.area / 1e6)
    support.attrs["tract_polygon_area_km2"] = float(tracts.geometry.area.sum() / 1e6)
    support.attrs["tracts_hash_record"] = verify_expected_hash(
        config["inputs"]["tracts"]["path"],
        config["inputs"]["tracts"]["sha256"],
        "tracts",
    )
    return support, support_points, tract_union


def build_catchments(support: pd.DataFrame, tract_union: Any) -> gpd.GeoDataFrame:
    points = shapely.points(support["x_m"].to_numpy(), support["y_m"].to_numpy())
    coordinates = np.column_stack([support["x_m"], support["y_m"]])
    duplicate_points = len(coordinates) - len(np.unique(coordinates, axis=0))
    if duplicate_points:
        raise ValueError(f"Support centre coordinates contain {duplicate_points} duplicates")

    minx, miny, maxx, maxy = tract_union.bounds
    padded_extent = box(minx - 5000.0, miny - 5000.0, maxx + 5000.0, maxy + 5000.0)
    voronoi = shapely.voronoi_polygons(MultiPoint(list(points)), extend_to=padded_extent)
    voronoi_cells = list(voronoi.geoms)
    if len(voronoi_cells) != len(points):
        raise ValueError("Voronoi construction did not return one cell per support centre")
    tree = STRtree(voronoi_cells)

    assigned_indices: list[int] = []
    for index, point in enumerate(points):
        candidates = np.atleast_1d(tree.query(point, predicate="intersects"))
        if len(candidates) == 0:
            candidates = np.array([int(tree.nearest(point))])
        if len(candidates) > 1:
            candidates = np.array(
                [int(candidate) for candidate in candidates if shapely.contains(voronoi_cells[int(candidate)], point)]
            )
        if len(candidates) != 1:
            raise RuntimeError(f"Could not resolve one Voronoi cell for support index {index}")
        assigned_indices.append(int(candidates[0]))
    if len(set(assigned_indices)) != len(points):
        raise RuntimeError("Voronoi cells were assigned to more than one support centre")

    clipped: list[MultiPolygon] = []
    areas_km2: list[float] = []
    for index, cell_index in enumerate(assigned_indices):
        geometry = shapely.intersection(voronoi_cells[cell_index], tract_union)
        polygonal = _polygonal_geometry(geometry, support.loc[index, "support_id"])
        if not (shapely.contains(polygonal, points[index]) or shapely.covers(polygonal, points[index])):
            raise RuntimeError(f"Support centre is outside its own catchment: {support.loc[index, 'support_id']}")
        clipped.append(polygonal)
        areas_km2.append(float(polygonal.area / 1e6))

    total_area = float(sum(areas_km2))
    tract_area = float(tract_union.area / 1e6)
    union_area = float(unary_union(clipped).area / 1e6)
    overlap_area = total_area - union_area
    uncovered_area_m2 = float(tract_union.difference(unary_union(clipped)).area)
    tolerance = float(1e-6)
    if abs(total_area - tract_area) > tolerance:
        raise RuntimeError(f"Catchment area sum {total_area} does not equal tract area {tract_area}")
    if abs(overlap_area) > tolerance:
        raise RuntimeError(f"Catchments overlap by {overlap_area} km2")
    if uncovered_area_m2 > float(1.0):
        raise RuntimeError(f"Catchments leave {uncovered_area_m2} m2 of tract uncovered")

    frame = support.copy()
    frame["tract_area_km2"] = areas_km2
    frame["tract_area_fraction"] = frame["tract_area_km2"] / tract_area
    frame["catchment_area_vs_nominal"] = frame["tract_area_km2"] / frame["nominal_area_km2"]
    catchment_gdf = gpd.GeoDataFrame(frame, geometry=clipped, crs=PROJECT_CRS)
    catchment_gdf.attrs["total_area_km2"] = total_area
    catchment_gdf.attrs["union_area_km2"] = union_area
    catchment_gdf.attrs["overlap_area_km2"] = overlap_area
    catchment_gdf.attrs["uncovered_area_m2"] = uncovered_area_m2
    catchment_gdf.attrs["tract_area_km2"] = tract_area
    catchment_gdf.attrs["duplicate_center_count"] = duplicate_points
    return catchment_gdf


def build_base_features(support: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    points = shapely.points(support["x_m"].to_numpy(), support["y_m"].to_numpy())
    contacts = read_gdf(
        config["inputs"]["contact_geometry"]["path"],
        layer=config["inputs"]["contact_geometry"]["layer"],
    )
    faults = read_gdf(
        config["inputs"]["fault_geometry"]["path"],
        layer=config["inputs"]["fault_geometry"]["layer"],
    )
    if len(contacts) != int(config["inputs"]["contact_geometry"]["expected_features"]):
        raise ValueError("Unexpected internal-contact count")
    if len(faults) != int(config["inputs"]["fault_geometry"]["expected_features"]):
        raise ValueError("Unexpected fault count")
    verify_expected_hash(
        config["inputs"]["contact_geometry"]["path"],
        config["inputs"]["contact_geometry"]["sha256"],
        "contact_geometry",
    )
    verify_expected_hash(
        config["inputs"]["fault_geometry"]["path"],
        config["inputs"]["fault_geometry"]["sha256"],
        "fault_geometry",
    )

    contact_distances = shapely.distance(points[:, None], contacts.geometry.to_numpy()[None, :])
    contact_ids = np.arange(len(contacts), dtype=int)
    contact_classes = contacts["litho_class"].astype(int).to_numpy()
    deterministic_order = np.lexsort((contact_ids, contact_classes))
    ordered_distances = contact_distances[:, deterministic_order]
    nearest_contact_ordered = np.argmin(ordered_distances, axis=1)
    nearest_contact_index = deterministic_order[nearest_contact_ordered]
    nearest_contact_distance = np.min(contact_distances, axis=1)
    nearest_contact_class = contact_classes[nearest_contact_index]
    tolerance_m = 1e-9
    tie_mask = np.abs(contact_distances - nearest_contact_distance[:, None]) <= tolerance_m
    tie_count = tie_mask.sum(axis=1).astype(int)
    tied_classes: list[str] = []
    for row_index in range(len(points)):
        classes = sorted(set(int(value) for value in contact_classes[tie_mask[row_index]]))
        tied_classes.append(";".join(str(value) for value in classes))

    fault_distances = shapely.distance(points[:, None], faults.geometry.to_numpy()[None, :])
    nearest_fault_index = np.argmin(fault_distances, axis=1)
    nearest_fault_distance = np.min(fault_distances, axis=1)

    v14 = pd.read_csv(
        abs_path(config["inputs"]["v14_table"]["path"]),
        usecols=["id", "centroid_x", "centroid_y", "bouguer", "litho_contact_litho_class"],
    )
    if not np.isfinite(v14[["centroid_x", "centroid_y", "bouguer"]].to_numpy()).all():
        raise ValueError("V14 gravity crosswalk contains non-finite values")
    gravity_tree = cKDTree(v14[["centroid_x", "centroid_y"]].to_numpy())
    gravity_distance, gravity_index = gravity_tree.query(support[["x_m", "y_m"]].to_numpy(), k=1)

    features = pd.DataFrame(
        {
            "support_id": support["support_id"].to_numpy(),
            "BASE_DIST_FAULT_KM": nearest_fault_distance / 1000.0,
            "BASE_DIST_CONTACT_KM": nearest_contact_distance / 1000.0,
            "BASE_CONTACT_CLASS": v14["litho_contact_litho_class"].to_numpy()[gravity_index].astype(int),
            "BASE_CONTACT_CLASS_SOURCE_ID": v14["id"].to_numpy()[gravity_index].astype(int),
            "BASE_CONTACT_CLASS_INHERIT_DIST_M": gravity_distance,
            "BASE_CONTACT_CLASS_REBUILT": nearest_contact_class.astype(int),
            "BASE_CONTACT_SOURCE_FID": nearest_contact_index.astype(int),
            "BASE_CONTACT_TIE_COUNT": tie_count,
            "BASE_CONTACT_TIED_CLASSES": tied_classes,
            "BASE_FAULT_SOURCE_FID": nearest_fault_index.astype(int),
            "BASE_BOUGUER_MGAL": v14["bouguer"].to_numpy()[gravity_index],
            "BASE_BOUGUER_SOURCE_ID": v14["id"].to_numpy()[gravity_index].astype(int),
            "BASE_BOUGUER_INHERIT_DIST_M": gravity_distance,
        }
    )
    summary = {
        "n_support": int(len(features)),
        "fault_km": features["BASE_DIST_FAULT_KM"].describe().to_dict(),
        "contact_km": features["BASE_DIST_CONTACT_KM"].describe().to_dict(),
        "contact_class_counts": {str(k): int(v) for k, v in features["BASE_CONTACT_CLASS"].value_counts().sort_index().items()},
        "rebuilt_contact_class_counts": {str(k): int(v) for k, v in features["BASE_CONTACT_CLASS_REBUILT"].value_counts().sort_index().items()},
        "contact_tie_count_distribution": {str(k): int(v) for k, v in features["BASE_CONTACT_TIE_COUNT"].value_counts().sort_index().items()},
        "bouguer_mgal": features["BASE_BOUGUER_MGAL"].describe().to_dict(),
        "gravity_inherit_distance_m": features["BASE_BOUGUER_INHERIT_DIST_M"].describe().to_dict(),
        "contact_class_provenance": "existing V14 legacy class inherited from the nearest V14 cell",
        "rebuilt_contact_tie_break": "minimum distance, then litho_class, then source fid; diagnostic only",
        "contact_tie_caveat": "Every support centre has coincident nearest contacts with multiple class labels; the rebuilt class is not used as the primary S0 covariate.",
        "gravity_provenance": "nearest V14-cell Bouguer value; not an independent raster intersection",
    }
    return features, summary


def parse_group_components(row: pd.Series, fields: list[str]) -> list[dict[str, str]]:
    components: list[dict[str, str]] = []
    seen: set[str] = set()
    for field in fields:
        value = row.get(field)
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text.lower() == "none":
            continue
        for part in re.split(r"[,;]", text):
            name = part.strip()
            if name and name not in seen:
                seen.add(name)
                components.append({"component_name": name, "source_field": field})
    return components


def build_occurrences(
    support: pd.DataFrame,
    tract_union: Any,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_config = config["inputs"]["occurrences"]
    occurrences = read_gdf(source_config["path"])
    verify_expected_hash(source_config["path"], source_config["sha256"], "occurrences")
    if len(occurrences) != int(source_config["expected_features"]):
        raise ValueError(f"Unexpected occurrence count: {len(occurrences)}")
    expected_counts = {str(k): int(v) for k, v in source_config["expected_status_counts"].items()}
    observed_counts = {str(k): int(v) for k, v in occurrences[source_config["site_status_field"]].value_counts().items()}
    if observed_counts != expected_counts:
        raise ValueError(f"SiteStatus counts changed: {observed_counts}")

    out = occurrences.copy()
    out["source_fid"] = np.arange(len(out), dtype=int)
    out["record_id"] = [f"USGS_{index + 1:04d}" for index in range(len(out))]
    out["display_name"] = out["Name"].where(out["Name"].notna(), out["Group_name"])
    out["x_m"] = out.geometry.x
    out["y_m"] = out.geometry.y
    out["in_tract"] = out.geometry.intersects(tract_union)

    support_tree = cKDTree(support[["x_m", "y_m"]].to_numpy())
    nearest_distance, nearest_index = support_tree.query(out[["x_m", "y_m"]].to_numpy(), k=1)
    out["nearest_support_id"] = support["support_id"].to_numpy()[nearest_index]
    out["nearest_support_distance_m"] = nearest_distance
    out["assigned_support_id"] = np.where(out["in_tract"], out["nearest_support_id"], None)

    expansion: list[int] = []
    component_records: list[dict[str, Any]] = []
    for _, row in out.iterrows():
        if row[source_config["site_status_field"]] == "group":
            components = parse_group_components(row, list(source_config["group_includes_fields"]))
            if not components:
                raise ValueError(f"Grouped record has no parseable components: {row['record_id']}")
            expansion.append(len(components))
            for ordinal, component in enumerate(components, start=1):
                component_records.append(
                    {
                        "record_id": row["record_id"],
                        "source_fid": int(row["source_fid"]),
                        "group_name": row.get("Group_name"),
                        "component_ordinal": ordinal,
                        "component_name": component["component_name"],
                        "source_field": component["source_field"],
                        "x_m": float(row["x_m"]),
                        "y_m": float(row["y_m"]),
                        "positional_accuracy_m": float(source_config["positional_accuracy_m"]),
                        "assigned_support_id": row["assigned_support_id"],
                        "in_tract": bool(row["in_tract"]),
                    }
                )
        else:
            expansion.append(1)
    out["n_group_components"] = 0
    out.loc[out[source_config["site_status_field"]] == "group", "n_group_components"] = [
        count for count, status in zip(expansion, out[source_config["site_status_field"]]) if status == "group"
    ]
    out["record_expansion"] = expansion
    component_frame = pd.DataFrame(component_records)
    return out, component_frame


def build_positional_weights(
    occurrences: pd.DataFrame,
    support: pd.DataFrame,
    tract_union: Any,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    accuracy_m = float(config["inputs"]["occurrences"]["positional_accuracy_m"])
    draws = int(config["inputs"]["occurrences"]["positional_monte_carlo_draws"])
    seed = int(config["inputs"]["occurrences"]["positional_seed"])
    support_tree = cKDTree(support[["x_m", "y_m"]].to_numpy())
    support_ids = support["support_id"].to_numpy()
    nearest_lookup = {sid: index for index, sid in enumerate(support_ids)}
    weight_rows: list[dict[str, Any]] = []

    own_probability = np.zeros(len(occurrences), dtype=float)
    outside_probability = np.zeros(len(occurrences), dtype=float)
    primary_support = np.array([None] * len(occurrences), dtype=object)
    primary_probability = np.zeros(len(occurrences), dtype=float)
    n_catchments = np.zeros(len(occurrences), dtype=int)

    for position, row in enumerate(occurrences.itertuples()):
        if not bool(row.in_tract):
            outside_probability[position] = 1.0
            weight_rows.append(
                {
                    "record_id": row.record_id,
                    "source_fid": int(row.source_fid),
                    "site_status": row.SiteStatus,
                    "assignment_type": "outside_tract",
                    "support_id": None,
                    "probability": 1.0,
                    "n_draws": draws,
                }
            )
            continue

        rng = np.random.default_rng([seed, int(row.source_fid)])
        radius = accuracy_m * np.sqrt(rng.random(draws))
        angle = 2.0 * np.pi * rng.random(draws)
        draw_x = float(row.x_m) + radius * np.cos(angle)
        draw_y = float(row.y_m) + radius * np.sin(angle)
        draw_points = shapely.points(draw_x, draw_y)
        inside = np.asarray(shapely.covers(tract_union, draw_points), dtype=bool)
        assignment = np.full(draws, -1, dtype=int)
        if inside.any():
            _, assigned = support_tree.query(np.column_stack([draw_x[inside], draw_y[inside]]), k=1)
            assignment[inside] = assigned

        own_index = nearest_lookup[str(row.assigned_support_id)]
        counts = np.bincount(assignment[inside], minlength=len(support)) if inside.any() else np.zeros(len(support), dtype=int)
        outside_count = draws - int(inside.sum())
        own_count = int(counts[own_index]) if own_index < len(counts) else 0
        own_probability[position] = own_count / draws
        outside_probability[position] = outside_count / draws
        n_catchments[position] = int(np.count_nonzero(counts))
        if counts.any():
            modal_index = int(np.argmax(counts))
            primary_support[position] = support_ids[modal_index]
            primary_probability[position] = float(counts[modal_index] / draws)

        if own_count:
            weight_rows.append(
                {
                    "record_id": row.record_id,
                    "source_fid": int(row.source_fid),
                    "site_status": row.SiteStatus,
                    "assignment_type": "own_catchment",
                    "support_id": support_ids[own_index],
                    "probability": own_count / draws,
                    "n_draws": draws,
                }
            )
        for support_index in np.flatnonzero(counts):
            if support_index == own_index:
                continue
            weight_rows.append(
                {
                    "record_id": row.record_id,
                    "source_fid": int(row.source_fid),
                    "site_status": row.SiteStatus,
                    "assignment_type": "adjacent_catchment",
                    "support_id": support_ids[int(support_index)],
                    "probability": float(counts[support_index] / draws),
                    "n_draws": draws,
                }
            )
        if outside_count:
            weight_rows.append(
                {
                    "record_id": row.record_id,
                    "source_fid": int(row.source_fid),
                    "site_status": row.SiteStatus,
                    "assignment_type": "outside_tract",
                    "support_id": None,
                    "probability": outside_count / draws,
                    "n_draws": draws,
                }
            )

    occurrence_out = occurrences.copy()
    occurrence_out["positional_own_probability"] = own_probability
    occurrence_out["positional_outside_probability"] = outside_probability
    occurrence_out["positional_primary_support_id"] = primary_support
    occurrence_out["positional_primary_probability"] = primary_probability
    occurrence_out["positional_n_catchments"] = n_catchments
    weight_frame = pd.DataFrame(weight_rows)
    sums = weight_frame.groupby("record_id")["probability"].sum()
    if not np.allclose(sums.to_numpy(), 1.0, atol=1e-9, rtol=0.0):
        raise RuntimeError("Positional assignment probabilities do not sum to one")
    return occurrence_out, weight_frame


def build_folds(support: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    n_folds = int(config["folds"]["n_folds"])
    buffer_km = float(config["folds"]["primary_buffer_km"])
    coordinates = support[["x_m", "y_m"]].to_numpy(dtype=float)
    covariance = np.cov(coordinates, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    reference = np.asarray(config["folds"]["belt_axis_reference_direction"], dtype=float)
    reference = reference / np.linalg.norm(reference)
    if float(np.dot(axis, reference)) < 0.0:
        axis = -axis
    belt_coordinate = coordinates @ axis
    order = np.lexsort((coordinates[:, 0], coordinates[:, 1], belt_coordinate))
    fold_sizes = np.full(n_folds, len(support) // n_folds, dtype=int)
    fold_sizes[: len(support) % n_folds] += 1
    fold = np.empty(len(support), dtype=int)
    start = 0
    for fold_number, size in enumerate(fold_sizes, start=1):
        fold[order[start : start + size]] = fold_number
        start += size

    output = support.copy()
    output["belt_axis_x"] = axis[0]
    output["belt_axis_y"] = axis[1]
    output["belt_coordinate_m"] = belt_coordinate
    output["new_fold"] = fold
    output["new_fold_label"] = [f"Fold {value}" for value in fold]

    for fold_number in range(1, n_folds + 1):
        heldout = fold == fold_number
        tree = cKDTree(coordinates[heldout])
        distance, _ = tree.query(coordinates, k=1)
        output[f"distance_to_fold_{fold_number}_m"] = distance
        output[f"train_included_fold_{fold_number}"] = (~heldout) & (distance > buffer_km * 1000.0)

    angle_degrees = math.degrees(math.atan2(axis[1], axis[0]))
    output["belt_axis_bearing_degrees"] = angle_degrees
    counts = pd.Series(fold).value_counts().sort_index()
    if int(counts.max() - counts.min()) > 1:
        raise ValueError(f"Primary folds are not equal in cell count: {counts.to_dict()}")
    return output


def add_occurrence_counts(support: pd.DataFrame, occurrences: pd.DataFrame) -> pd.DataFrame:
    output = support.copy()
    in_tract = occurrences[occurrences["in_tract"]].copy()
    support_index = pd.Series(np.arange(len(output)), index=output["support_id"].to_numpy())
    assigned = in_tract["assigned_support_id"].map(support_index)
    if assigned.isna().any():
        raise ValueError("An in-tract occurrence is not assigned to a support catchment")
    assigned_index = assigned.to_numpy(dtype=int)

    output["n_records"] = np.bincount(assigned_index, minlength=len(output)).astype(int)
    for status, column in [
        ("deposit", "n_deposits"),
        ("prospect", "n_prospects"),
        ("group", "n_groups"),
    ]:
        mask = in_tract["SiteStatus"].eq(status).to_numpy()
        output[column] = np.bincount(assigned_index[mask], minlength=len(output)).astype(int)
    expanded_weights = in_tract["record_expansion"].to_numpy(dtype=float)
    output["n_expanded_records"] = np.rint(
        np.bincount(assigned_index, weights=expanded_weights, minlength=len(output))
    ).astype(int)
    output["record_positional_outside_probability_sum"] = np.bincount(
        assigned_index,
        weights=in_tract["positional_outside_probability"].to_numpy(dtype=float),
        minlength=len(output),
    )
    output["any_record"] = (output["n_records"] > 0).astype(int)
    output["any_expanded_record"] = (output["n_expanded_records"] > 0).astype(int)
    if int(output["n_records"].sum()) != int(len(in_tract)):
        raise ValueError("Support counts do not reproduce the in-tract record total")
    if int(output["n_expanded_records"].sum()) != int(in_tract["record_expansion"].sum()):
        raise ValueError("Expanded support counts do not reproduce the expanded record total")
    return output


def overlap_class(values: pd.Series) -> pd.Series:
    result = pd.Series(index=values.index, dtype=object)
    result.loc[values < 0.8] = "partial_lt_80pct"
    result.loc[(values >= 0.8) & (values < 0.99)] = "overlap_80_to_lt_99pct"
    result.loc[values >= 0.99] = "full_ge_99pct"
    return result


def build_domain_overlap_crosstab(model_table: pd.DataFrame) -> pd.DataFrame:
    frame = model_table.copy()
    frame["area_overlap_class"] = overlap_class(frame["nominal_hex_overlap_fraction"])
    frame["outside_daly_envelope_flag"] = frame["domain_abbreviation"].eq("unassigned")
    grouped = (
        frame.groupby(["domain_abbreviation", "area_overlap_class"], dropna=False, sort=True)
        .agg(
            n_support=("support_id", "size"),
            catchment_area_km2=("tract_area_km2", "sum"),
            n_records=("n_records", "sum"),
            n_positive_cells=("any_record", "sum"),
            n_expanded_records=("n_expanded_records", "sum"),
            outside_daly_envelope=("outside_daly_envelope_flag", "max"),
        )
        .reset_index()
    )
    total = pd.DataFrame(
        [
            {
                "domain_abbreviation": "ALL",
                "area_overlap_class": "ALL",
                "n_support": int(len(frame)),
                "catchment_area_km2": float(frame["tract_area_km2"].sum()),
                "n_records": int(frame["n_records"].sum()),
                "n_positive_cells": int(frame["any_record"].sum()),
                "n_expanded_records": int(frame["n_expanded_records"].sum()),
                "outside_daly_envelope": bool(frame["outside_daly_envelope_flag"].any()),
            }
        ]
    )
    return pd.concat([grouped, total], ignore_index=True)


def build_fold_summary(model_table: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n_folds = int(config["folds"]["n_folds"])
    schemes = [
        ("along_belt_buffer_10km", "new_fold", "new_fold_label", True),
        ("along_belt_unbuffered", "new_fold", "new_fold_label", False),
        ("v14_crosswalk_unbuffered", "v14_fold", "v14_fold_label", False),
    ]
    for scheme, fold_field, label_field, use_buffer in schemes:
        for fold_value in sorted(model_table[fold_field].dropna().unique()):
            heldout = model_table[fold_field].eq(fold_value)
            if use_buffer:
                included = model_table[f"train_included_fold_{int(fold_value)}"].astype(bool)
            else:
                included = (~heldout) & model_table[fold_field].notna()
            holdout_frame = model_table.loc[heldout]
            domain_counts = {
                str(key): int(value)
                for key, value in holdout_frame["domain_abbreviation"].value_counts().sort_index().items()
            }
            rows.append(
                {
                    "scheme": scheme,
                    "fold": fold_value,
                    "fold_label": holdout_frame[label_field].iloc[0],
                    "n_holdout_support": int(heldout.sum()),
                    "n_train_included": int(included.sum()),
                    "n_train_excluded_by_buffer": int(((~heldout) & model_table[fold_field].notna() & (~included)).sum()),
                    "holdout_area_km2": float(holdout_frame["tract_area_km2"].sum()),
                    "holdout_records": int(holdout_frame["n_records"].sum()),
                    "holdout_positive_cells": int(holdout_frame["any_record"].sum()),
                    "holdout_cell_prevalence": float(holdout_frame["any_record"].mean()),
                    "holdout_expanded_records": int(holdout_frame["n_expanded_records"].sum()),
                    "holdout_outside_daly": int(holdout_frame["domain_abbreviation"].eq("unassigned").sum()),
                    "holdout_domain_counts": json.dumps(domain_counts, sort_keys=True),
                }
            )
    return pd.DataFrame(rows)


def plot_support_and_folds(
    model_table: pd.DataFrame,
    catchments: gpd.GeoDataFrame,
    tract_union: Any,
    occurrences: pd.DataFrame,
    path: Path,
) -> None:
    frame = catchments[["support_id", "geometry"]].merge(
        model_table[["support_id", "new_fold", "tract_area_km2", "domain_abbreviation"]],
        on="support_id",
        how="left",
    )
    frame = gpd.GeoDataFrame(frame, geometry="geometry", crs=PROJECT_CRS)
    fig, ax = plt.subplots(figsize=(11.0, 8.5))
    gpd.GeoSeries([tract_union], crs=PROJECT_CRS).plot(ax=ax, color="#f5f2eb", edgecolor="none")
    for fold_number, color in zip(range(1, 5), FOLD_COLORS):
        frame.loc[frame["new_fold"].eq(fold_number)].plot(
            ax=ax, color=color, edgecolor="white", linewidth=0.12, alpha=0.78
        )
    gpd.GeoSeries([tract_union], crs=PROJECT_CRS).boundary.plot(ax=ax, color="#222222", linewidth=0.8)
    for status, color in STATUS_COLORS.items():
        subset = occurrences[occurrences["SiteStatus"].eq(status) & occurrences["in_tract"]]
        ax.scatter(subset["x_m"], subset["y_m"], s=10 if status != "deposit" else 20, c=color,
                   edgecolors="white", linewidths=0.3, label=status.capitalize(), zorder=5)
    handlers = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=STATUS_COLORS[status],
               markeredgecolor="white", markersize=5, label=status.capitalize())
        for status in ["deposit", "prospect", "group"]
    ] + [
        Line2D([0], [0], color=color, linewidth=6, label=f"Fold {fold_number}")
        for fold_number, color in zip(range(1, 5), FOLD_COLORS)
    ]
    ax.legend(handles=handlers, loc="lower left", frameon=True, fontsize=8, ncol=2)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m, EPSG:32735)")
    ax.set_ylabel("Northing (m, EPSG:32735)")
    ax.set_title("Stage 4 nearest-centre catchments and along-belt validation folds")
    ax.text(0.01, 0.99, "Catchments partition the dissolved tract footprint\nCatchment area is the count-model exposure",
            transform=ax.transAxes, va="top", fontsize=8, bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"})
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_occurrence_counts(
    model_table: pd.DataFrame,
    catchments: gpd.GeoDataFrame,
    tract_union: Any,
    occurrences: pd.DataFrame,
    path: Path,
) -> None:
    from matplotlib.colors import BoundaryNorm, ListedColormap

    frame = catchments[["support_id", "geometry"]].merge(
        model_table[["support_id", "n_records"]], on="support_id", how="left"
    )
    frame = gpd.GeoDataFrame(frame, geometry="geometry", crs=PROJECT_CRS)
    colors = ["#f7f7f7", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    fig, ax = plt.subplots(figsize=(11.0, 8.5))
    gpd.GeoSeries([tract_union], crs=PROJECT_CRS).plot(ax=ax, color="#f5f2eb", edgecolor="none")
    frame.plot(ax=ax, column="n_records", cmap=cmap, norm=norm, edgecolor="white", linewidth=0.12)
    gpd.GeoSeries([tract_union], crs=PROJECT_CRS).boundary.plot(ax=ax, color="#222222", linewidth=0.8)
    for status, color in STATUS_COLORS.items():
        subset = occurrences[occurrences["SiteStatus"].eq(status) & occurrences["in_tract"]]
        ax.scatter(subset["x_m"], subset["y_m"], s=9 if status != "deposit" else 18, c=color,
                   edgecolors="white", linewidths=0.25, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m, EPSG:32735)")
    ax.set_ylabel("Northing (m, EPSG:32735)")
    ax.set_title("Observed USGS occurrence records per Stage 4 catchment")
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.7, ticks=[0, 1, 2, 3, 4])
    cbar.set_label("Original records (groups count once)")
    ax.text(0.01, 0.99, "Group multiplicity is retained separately in the expanded-record sensitivity analysis.",
            transform=ax.transAxes, va="top", fontsize=8, bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"})
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage4_v1.json")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = output_paths(config)
    built = build_id()

    protected = verify_protected(config)
    support, support_points, tract_union = load_support_and_frame(config)
    tract_polygon_area_km2 = float(support.attrs["tract_polygon_area_km2"])
    catchments = build_catchments(support, tract_union)
    support = support.merge(
        catchments[["support_id", "tract_area_km2", "tract_area_fraction", "catchment_area_vs_nominal"]],
        on="support_id",
        how="left",
        validate="one_to_one",
    )
    base_features, base_summary = build_base_features(support, config)
    occurrences, group_components = build_occurrences(support, tract_union, config)
    occurrences, positional_weights = build_positional_weights(occurrences, support, tract_union, config)
    folded = build_folds(support, config)
    model_table = add_occurrence_counts(folded, occurrences)
    model_table = model_table.merge(base_features, on="support_id", how="left", validate="one_to_one")
    model_table["log_tract_area_km2"] = np.log(model_table["tract_area_km2"].clip(lower=1e-12))
    model_table = model_table.rename(
        columns={
            "nominal_area_km2": "nominal_hex_area_km2",
            "frame_overlap_area_km2": "nominal_hex_overlap_area_km2",
            "frame_overlap_fraction": "nominal_hex_overlap_fraction",
        }
    )
    model_table["area_overlap_class"] = overlap_class(model_table["nominal_hex_overlap_fraction"])
    required = [
        "support_id", "x_m", "y_m", "tract_area_km2", "log_tract_area_km2", "new_fold",
        "BASE_DIST_FAULT_KM", "BASE_DIST_CONTACT_KM", "BASE_CONTACT_CLASS", "BASE_BOUGUER_MGAL",
        "STR_DIST_NAMED_HM_KM", "STR_NAMED_LINE_DENSITY_50KM", "n_records", "n_expanded_records",
    ]
    missing = [column for column in required if column not in model_table.columns]
    if missing:
        raise ValueError(f"Model table is missing required columns: {missing}")
    if model_table[required].isna().any().any():
        raise ValueError("Model table contains missing values in required columns")

    crosstab = build_domain_overlap_crosstab(model_table)
    fold_summary = build_fold_summary(model_table, config)
    support_audit = model_table[
        [
            "support_id", "x_m", "y_m", "tract_area_km2", "nominal_hex_area_km2",
            "nominal_hex_overlap_area_km2", "nominal_hex_overlap_fraction", "area_overlap_class",
            "full_support", "edge_hex", "domain_abbreviation", "new_fold", "new_fold_label",
            "v14_fold", "v14_fold_label", "n_records", "n_deposits", "n_prospects", "n_groups",
            "n_expanded_records", "any_record", "record_positional_outside_probability_sum",
        ]
    ].copy()

    write_csv(model_table.drop(columns=["geometry"], errors="ignore"), paths["model_table"])
    write_csv(occurrences.drop(columns=["geometry"], errors="ignore"), paths["occurrences_csv"])
    write_csv(group_components, paths["group_components_csv"])
    write_csv(positional_weights, paths["positional_weights_csv"])
    write_csv(support_audit, paths["support_audit_csv"])
    write_csv(fold_summary, paths["fold_summary_csv"])
    crosstab_path = abs_path(paths["processed_dir"]) / "stage4_domain_overlap_crosstab_v1.csv"
    write_csv(crosstab, crosstab_path)

    catchment_attributes = model_table.set_index("support_id").loc[catchments["support_id"]].reset_index()
    catchment_output = gpd.GeoDataFrame(catchment_attributes, geometry=catchments.geometry.to_numpy(), crs=PROJECT_CRS)
    point_output = gpd.GeoDataFrame(
        model_table[["support_id", "x_m", "y_m", "tract_area_km2", "domain_abbreviation", "new_fold",
                     "v14_fold_label", "n_records", "any_record"]].copy(),
        geometry=gpd.points_from_xy(model_table["x_m"], model_table["y_m"]),
        crs=PROJECT_CRS,
    )
    occurrence_output = gpd.GeoDataFrame(
        occurrences.copy(), geometry="geometry", crs=PROJECT_CRS
    )
    write_gpkg(
        [
            ("stage4_support_catchments", catchment_output),
            ("stage4_support_points", point_output),
            ("stage4_occurrences", occurrence_output),
        ],
        paths["support_gpkg"],
    )

    figure_paths = config["outputs"]["figures"]
    support_figure = Path(paths["figure_dir"]) / figure_paths["support_folds"]
    count_figure = Path(paths["figure_dir"]) / figure_paths["occurrence_counts"]
    plot_support_and_folds(model_table, catchments, tract_union, occurrences, support_figure)
    plot_occurrence_counts(model_table, catchments, tract_union, occurrences, count_figure)

    source_paths = [
        config["inputs"]["support_table"]["path"],
        config["inputs"]["support_gpkg"]["path"],
        config["inputs"]["tracts"]["path"],
        config["inputs"]["occurrences"]["path"],
        config["inputs"]["contact_geometry"]["path"],
        config["inputs"]["fault_geometry"]["path"],
        config["inputs"]["v14_table"]["path"],
    ]
    usgs_directory = abs_path(config["inputs"]["occurrences"]["path"]).parent
    for path in sorted(usgs_directory.glob("CACB_*")):
        if path.is_file():
            source_paths.append(str(path))
    unique_sources = sorted({relative_posix(path): path for path in source_paths}.values(), key=relative_posix)

    output_keys = [
        "model_table", "support_gpkg", "occurrences_csv", "group_components_csv",
        "positional_weights_csv", "support_audit_csv", "fold_summary_csv",
    ]
    output_records = {key: file_record(paths[key], key) for key in output_keys}
    output_records["domain_overlap_crosstab"] = file_record(crosstab_path, "domain_overlap_crosstab")
    output_records["support_folds_figure"] = file_record(support_figure, "support_and_folds_figure")
    output_records["occurrence_counts_figure"] = file_record(count_figure, "occurrence_counts_figure")

    outside = occurrences.loc[~occurrences["in_tract"], ["record_id", "Name", "SiteStatus", "Country",
                                                         "x_m", "y_m", "nearest_support_distance_m"]]
    manifest = {
        "schema_version": "1.0.0",
        "build_id": built,
        "status": "built_pending_independent_verification",
        "purpose": "Stage 4 support, occurrence inventory and analysis-ready model table; no model fitted",
        "config": file_record(args.config, "build configuration"),
        "builder": file_record(__file__, "builder source"),
        "protected_inputs": protected,
        "sources": [file_record(path) for path in unique_sources],
        "support": {
            "n_support": int(len(model_table)),
            "tract_area_km2": float(tract_union.area / 1e6),
            "tract_polygon_sum_area_km2": tract_polygon_area_km2,
            "catchment_area_sum_km2": float(model_table["tract_area_km2"].sum()),
            "catchment_union_area_km2": float(catchments.attrs["union_area_km2"]),
            "catchment_overlap_area_km2": float(catchments.attrs["overlap_area_km2"]),
            "uncovered_area_m2": float(catchments.attrs["uncovered_area_m2"]),
            "area_km2_distribution": model_table["tract_area_km2"].describe().to_dict(),
            "n_full_nominal_hex": int(model_table["full_support"].sum()),
            "n_overlap_ge_80pct_nominal_hex": int((model_table["nominal_hex_overlap_fraction"] >= 0.8).sum()),
            "n_outside_daly_envelope": int(model_table["domain_abbreviation"].eq("unassigned").sum()),
            "outside_daly_retained": True,
            "exposure": "log(tract_area_km2)",
        },
        "base_features": base_summary,
        "occurrences": {
            "source_records": int(len(occurrences)),
            "status_counts": {str(k): int(v) for k, v in occurrences["SiteStatus"].value_counts().items()},
            "in_tract": int(occurrences["in_tract"].sum()),
            "outside_tract": int((~occurrences["in_tract"]).sum()),
            "outside_tract_records": outside.to_dict(orient="records"),
            "group_records": int(occurrences["SiteStatus"].eq("group").sum()),
            "group_components": int(len(group_components)),
            "expanded_record_total_inside_tract": int(occurrences.loc[occurrences["in_tract"], "record_expansion"].sum()),
            "support_positive_cells": int((model_table["n_records"] > 0).sum()),
            "support_records": int(model_table["n_records"].sum()),
            "positional_accuracy_m": float(config["inputs"]["occurrences"]["positional_accuracy_m"]),
            "positional_monte_carlo_draws": int(config["inputs"]["occurrences"]["positional_monte_carlo_draws"]),
            "positional_probability_mass_inside_tract": float(1.0 - occurrences["positional_outside_probability"].sum() / max(len(occurrences), 1)),
        },
        "folds": {
            "belt_axis_bearing_degrees": float(model_table["belt_axis_bearing_degrees"].iloc[0]),
            "primary_scheme": "along_belt_buffer_10km",
            "fold_counts": {str(int(k)): int(v) for k, v in model_table["new_fold"].value_counts().sort_index().items()},
            "buffer_excluded_train_cells": {
                str(fold): int(((model_table["new_fold"] != fold) & (~model_table[f"train_included_fold_{fold}"])).sum())
                for fold in range(1, 5)
            },
        },
        "domain_overlap_crosstab": crosstab.to_dict(orient="records"),
        "outputs": output_records,
        "software": software_versions(),
        "notes": [
            "The 2,193 centres are retained; nominal hexagons are context only.",
            "Catchments are nearest-centre Voronoi cells clipped to the dissolved tract union.",
            "The 285 cells outside the Daly envelope are retained with domain_abbreviation=unassigned.",
            "n_records is the primary outcome and counts each original USGS record once.",
            "n_expanded_records replaces each group with its documented component count at the group location.",
            "BASE_BOUGUER_MGAL is inherited from the nearest V14 cell because the source raster is unavailable.",
            "Positional probabilities simulate uniform locations within the stated 1,000 m accuracy disk.",
        ],
    }
    write_json(manifest, paths["data_manifest"])
    print(json.dumps({
        "build_id": built,
        "n_support": len(model_table),
        "tract_area_km2": float(tract_union.area / 1e6),
        "catchment_area_sum_km2": float(model_table["tract_area_km2"].sum()),
        "n_in_tract_records": int(occurrences["in_tract"].sum()),
        "n_positive_cells": int((model_table["n_records"] > 0).sum()),
        "n_group_components": int(len(group_components)),
        "folds": {str(int(k)): int(v) for k, v in model_table["new_fold"].value_counts().sort_index().items()},
        "manifest": paths["data_manifest"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
