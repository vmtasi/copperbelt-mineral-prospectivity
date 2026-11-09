"""Independently verify the Stage 3 structural-predictor build.

The verifier is read-only apart from writing its validation JSON. It
recomputes support geometry, source-distance samples, density values,
named-system intersections and provenance hashes. It does not fit a model and
does not read any V14 occurrence field.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from PIL import Image
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from src.geology_foundation.common import relative_posix, resolve_path, sha256_file, write_json


DEFAULT_MANIFEST = "data/processed/structural_predictors/structural_predictor_manifest_v1.json"
DEFAULT_OUTPUT = "data/processed/structural_predictors/structural_predictor_validation_v1.json"
PROJECT_CRS = "EPSG:32735"


def _check(name: str, passed: bool, details: Any) -> dict[str, Any]:
    return {"name": name, "status": "pass" if passed else "fail", "details": details}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _file_mismatches(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for record in records:
        path = resolve_path(record["path"])
        if not path.exists():
            failures.append({"path": record["path"], "error": "missing"})
            continue
        if "bytes" in record and path.stat().st_size != int(record["bytes"]):
            failures.append({"path": record["path"], "error": "byte count mismatch"})
            continue
        if "sha256" in record:
            actual = sha256_file(path)
            if actual.lower() != str(record["sha256"]).lower():
                failures.append(
                    {
                        "path": record["path"],
                        "error": "sha256 mismatch",
                        "expected": record["sha256"],
                        "actual": actual,
                    }
                )
    return failures


def _configured_output_path(config: dict[str, Any], key: str) -> Path:
    value = Path(config["outputs"][key])
    if value.is_absolute() or len(value.parts) > 1:
        return resolve_path(value)
    return resolve_path(config["outputs"]["directory"]) / value


def _configured_figure_path(config: dict[str, Any], key: str) -> Path:
    return resolve_path(config["outputs"]["figure_directory"]) / config["outputs"]["figures"][key]


def _read_layer(path: Path, layer: str) -> gpd.GeoDataFrame:
    frame = gpd.read_file(path, layer=layer, engine="pyogrio")
    if frame.crs is None or frame.crs.to_string() != PROJECT_CRS:
        raise ValueError(f"Layer {layer} is not in {PROJECT_CRS}: {frame.crs}")
    return frame


def _nearest_distance_km(points: np.ndarray, geometry: Any) -> np.ndarray:
    return np.asarray(shapely.distance(points, geometry)) / 1000.0


def _line_length_in_radius(point: Point, radius_m: float, lines: list[Any]) -> float:
    circle = point.buffer(radius_m)
    return sum(line.intersection(circle).length for line in lines)


def _named_system_unions(segments: gpd.GeoDataFrame, statuses: set[str] | None = None) -> dict[str, Any]:
    selected = segments[segments["attributed_structure_id"].notna()].copy()
    if statuses is not None:
        selected = selected[selected["attributed_status"].isin(statuses)]
    return {
        str(structure_id): unary_union(list(group.geometry))
        for structure_id, group in selected.groupby("attributed_structure_id", sort=True)
    }


def _intersection_audit(segments: gpd.GeoDataFrame) -> list[dict[str, Any]]:
    systems = segments[segments["attributed_structure_id"].notna()].copy()
    unions: dict[str, dict[str, Any]] = {}
    for structure_id, group in systems.groupby("attributed_structure_id", sort=True):
        unions[str(structure_id)] = {
            "abbreviation": str(group["attributed_structure_abbreviation"].iloc[0]),
            "name": str(group["attributed_structure_name"].iloc[0]),
            "geometry": unary_union(list(group.geometry)),
        }
    records: list[dict[str, Any]] = []
    identifiers = sorted(unions)
    for index_a, structure_a in enumerate(identifiers):
        for structure_b in identifiers[index_a + 1 :]:
            intersection = unions[structure_a]["geometry"].intersection(unions[structure_b]["geometry"])
            if intersection.is_empty:
                continue
            if intersection.geom_type == "Point":
                points = [intersection]
            elif intersection.geom_type == "MultiPoint":
                points = list(intersection.geoms)
            else:
                raise RuntimeError(f"Unexpected intersection geometry: {intersection.geom_type}")
            for point in points:
                records.append(
                    {
                        "pair": tuple(sorted((unions[structure_a]["abbreviation"], unions[structure_b]["abbreviation"]))),
                        "a": unions[structure_a]["name"],
                        "b": unions[structure_b]["name"],
                        "x_m": float(point.x),
                        "y_m": float(point.y),
                    }
                )
    return records


def verify(manifest_path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest_file = resolve_path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    config_path = resolve_path(manifest["config"]["path"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    source_records = [*manifest["sources"], manifest["config"], manifest["builder"]]
    source_failures = _file_mismatches(source_records)
    checks.append(
        _check(
            "manifest_source_hashes_match_files",
            not source_failures,
            source_failures or {"checked_files": len(source_records)},
        )
    )

    protected_failures: list[dict[str, Any]] = []
    protected_details: dict[str, Any] = {}
    for key, record in manifest["protected"].items():
        expected = config["protected_hashes"][key]
        actual = sha256_file(resolve_path(record["path"]))
        matches = (
            actual.lower() == expected.lower()
            and str(record["expected_sha256"]).lower() == expected.lower()
            and str(record["observed_sha256"]).lower() == actual.lower()
        )
        protected_details[key] = {
            "path": record["path"],
            "expected_sha256": expected,
            "observed_sha256": actual,
            "matches": matches,
        }
        if not matches:
            protected_failures.append(protected_details[key])
    checks.append(
        _check(
            "v1_v14_protected_hashes_unchanged",
            not protected_failures,
            protected_failures or protected_details,
        )
    )

    output_failures = _file_mismatches(manifest["outputs"])
    checks.append(
        _check(
            "manifest_output_hashes_match_files",
            not output_failures,
            output_failures or {"checked_files": len(manifest["outputs"])},
        )
    )

    output_gpkg = _configured_output_path(config, "geopackage")
    expected_layers = list(manifest["layers"])
    observed_layers = [str(item[0]) for item in pyogrio.list_layers(output_gpkg)]
    checks.append(
        _check(
            "geopackage_layers_exact",
            observed_layers == expected_layers,
            {"expected": expected_layers, "observed": observed_layers},
        )
    )

    polygons = _read_layer(output_gpkg, config["outputs"]["layers"]["support_polygons"])
    points = _read_layer(output_gpkg, config["outputs"]["layers"]["support_points"])
    candidates = _read_layer(output_gpkg, config["outputs"]["layers"]["candidate_intersections"])
    crs_details = {
        "support_polygons": polygons.crs.to_string(),
        "support_points": points.crs.to_string(),
        "candidate_intersections": candidates.crs.to_string(),
    }
    checks.append(
        _check(
            "geopackage_layers_in_project_crs",
            all(value == PROJECT_CRS for value in crs_details.values()),
            crs_details,
        )
    )

    area_km2 = polygons.geometry.area / 1e6
    centroid_error = np.asarray(polygons.geometry.centroid.distance(polygons.geometry.centroid))
    point_centroid_errors = []
    polygon_lookup = polygons.set_index("support_id")
    for row in points[["support_id", "geometry"]].itertuples(index=False):
        if row.support_id not in polygon_lookup.index:
            point_centroid_errors.append(1e12)
            continue
        point_centroid_errors.append(row.geometry.distance(polygon_lookup.loc[row.support_id, "geometry"].centroid))
    geometry_details = {
        "n_polygons": int(len(polygons)),
        "n_points": int(len(points)),
        "polygon_area_min_km2": float(area_km2.min()),
        "polygon_area_max_km2": float(area_km2.max()),
        "valid_polygons": int(polygons.geometry.is_valid.sum()),
        "valid_points": int(points.geometry.is_valid.sum()),
        "max_point_to_polygon_centroid_error_m": float(max(point_centroid_errors)) if point_centroid_errors else None,
        "unused_centroid_probe_max_m": float(centroid_error.max()),
    }
    geometry_pass = (
        len(polygons) == len(points) == int(config["support"]["expected_n_support"])
        and np.allclose(area_km2.to_numpy(), float(config["support"]["nominal_cell_area_km2"]), atol=1e-6)
        and polygons.geometry.is_valid.all()
        and points.geometry.is_valid.all()
        and point_centroid_errors
        and max(point_centroid_errors) <= 1e-6
        and polygons["full_support"].sum() == int(config["support"]["expected_n_full_support"])
    )
    checks.append(_check("equal_area_support_geometry", bool(geometry_pass), geometry_details))

    feature_table_path = _configured_output_path(config, "feature_table")
    feature_table = pd.read_csv(feature_table_path)
    feature_support = pd.read_csv(_configured_output_path(config, "feature_support"))
    feature_dictionary = pd.read_csv(_configured_output_path(config, "feature_dictionary"))
    by_domain = pd.read_csv(_configured_output_path(config, "feature_by_domain"))
    by_fold = pd.read_csv(_configured_output_path(config, "feature_by_fold"))
    correlations = pd.read_csv(_configured_output_path(config, "correlations"))
    confidence = pd.read_csv(_configured_output_path(config, "confidence_sensitivity"))
    bridge = pd.read_csv(_configured_output_path(config, "bridge_sensitivity"))
    crosswalk = pd.read_csv(_configured_output_path(config, "v14_crosswalk"))
    shortlist = pd.read_csv(_configured_output_path(config, "shortlist"))
    intersections_csv = pd.read_csv(_configured_output_path(config, "candidate_intersections_csv"))

    feature_ids = [column for column in feature_table.columns if column.startswith("STR_")]
    feature_columns_ok = (
        len(feature_table) == int(config["support"]["expected_n_support"])
        and len(feature_ids) == int(manifest["features"]["n_features"])
        and not feature_table[feature_ids].isna().any().any()
        and set(feature_ids) == set(feature_dictionary["feature_id"])
        and (feature_table["occurrence_data_used"] == "no").all()
    )
    checks.append(
        _check(
            "feature_table_complete_and_occurrence_free",
            bool(feature_columns_ok),
            {
                "rows": int(len(feature_table)),
                "n_features": len(feature_ids),
                "n_missing": int(feature_table[feature_ids].isna().sum().sum()),
                "occurrence_flag_values": feature_table["occurrence_data_used"].value_counts().to_dict(),
                "dictionary_rows": int(len(feature_dictionary)),
            },
        )
    )

    forbidden_tokens = ("deposit", "prospect", "positive", "target")
    forbidden_columns = [
        column
        for column in feature_table.columns
        if column != "occurrence_data_used" and any(token in str(column).lower() for token in forbidden_tokens)
    ]
    occurrence_details = {
        "v14_columns_read": manifest["input_fields"]["v14_crosswalk"],
        "v14_occurrence_fields_read": manifest["input_fields"]["v14_occurrence_fields_read"],
        "feature_table_forbidden_columns": forbidden_columns,
        "crosswalk_columns": list(crosswalk.columns),
        "manifest_occurrence_data_used": manifest["occurrence_data_used"],
    }
    occurrence_pass = (
        manifest["input_fields"]["v14_crosswalk"] == ["id", "centroid_x", "centroid_y"]
        and manifest["input_fields"]["v14_occurrence_fields_read"] == []
        and manifest["occurrence_data_used"] is False
        and not forbidden_columns
        and "deposit_present" not in crosswalk.columns
    )
    checks.append(_check("no_occurrence_fields_read_or_propagated", occurrence_pass, occurrence_details))

    support_points = gpd.GeoDataFrame(
        feature_table.copy(),
        geometry=gpd.points_from_xy(feature_table["x_m"], feature_table["y_m"]),
        crs=PROJECT_CRS,
    )
    point_array = np.asarray(support_points.geometry.values)
    finite_hm = np.isfinite(feature_table["STR_DIST_NAMED_HM_KM"].to_numpy(dtype=float))
    checks.append(
        _check(
            "all_support_points_have_defined_hm_distance",
            bool(finite_hm.all() and (feature_table["STR_DIST_NAMED_HM_KM"] >= 0).all()),
            {
                "n": int(len(feature_table)),
                "n_finite": int(finite_hm.sum()),
                "min_km": float(feature_table["STR_DIST_NAMED_HM_KM"].min()),
                "max_km": float(feature_table["STR_DIST_NAMED_HM_KM"].max()),
            },
        )
    )

    support_summary = manifest["support"]
    overlap_fraction = polygons["frame_overlap_fraction"].astype(float)
    summary_pass = (
        len(polygons) == support_summary["n_support"]
        and int(polygons["full_support"].sum()) == support_summary["n_full_support"]
        and int((overlap_fraction >= float(config["support"]["edge_overlap_threshold"])).sum())
        == support_summary["n_overlap_ge_80pct"]
        and int((feature_table["domain_abbreviation"] == "unassigned").sum())
        == support_summary["n_unassigned_domains"]
        and all(float(overlap_fraction.min()) <= float(value) <= 1.000001 for value in overlap_fraction)
    )
    checks.append(
        _check(
            "support_summary_reproduces",
            bool(summary_pass),
            {
                "manifest": support_summary,
                "observed": {
                    "n_support": int(len(polygons)),
                    "n_full_support": int(polygons["full_support"].sum()),
                    "n_overlap_ge_80pct": int(
                        (overlap_fraction >= float(config["support"]["edge_overlap_threshold"])).sum()
                    ),
                    "n_unassigned_domains": int((feature_table["domain_abbreviation"] == "unassigned").sum()),
                },
            },
        )
    )

    domain_unique = by_domain.groupby("domain")["n_support"].nunique()
    fold_unique = by_fold.groupby("v14_fold")["n_support"].nunique()
    group_details = {
        "domains": {str(key): int(value) for key, value in feature_table["domain_abbreviation"].value_counts().items()},
        "folds": {str(key): int(value) for key, value in feature_table["v14_fold_label"].value_counts().items()},
        "domain_summary_groups": int(len(domain_unique)),
        "fold_summary_groups": int(len(fold_unique)),
        "domain_n_support_constant": bool((domain_unique == 1).all()),
        "fold_n_support_constant": bool((fold_unique == 1).all()),
    }
    checks.append(
        _check(
            "domain_and_fold_support_summaries_consistent",
            bool(
                (domain_unique == 1).all()
                and (fold_unique == 1).all()
                and int(by_domain.groupby("domain")["n_support"].first().sum()) == len(feature_table)
                and int(by_fold.groupby("v14_fold")["n_support"].first().sum()) == len(feature_table)
            ),
            group_details,
        )
    )

    source_structure_path = resolve_path(config["structures"]["path"])
    segments = gpd.read_file(
        source_structure_path,
        layer=config["structures"]["segments_layer"],
        engine="pyogrio",
    )
    hm_segments = segments[segments["attributed_status"].isin(["named_high", "named_medium"])]
    high_segments = segments[segments["attributed_status"] == "named_high"]
    all_named_segments = segments[segments["attributed_structure_id"].notna()]
    hm_union = unary_union(list(hm_segments.geometry))
    high_union = unary_union(list(high_segments.geometry))
    all_named_union = unary_union(list(all_named_segments.geometry))

    rng = np.random.default_rng(20261106)
    sample_size = min(50, len(feature_table))
    sample_indices = np.sort(rng.choice(len(feature_table), size=sample_size, replace=False))
    sample_points = point_array[sample_indices]
    sample_details: list[dict[str, Any]] = []
    sample_distances = {
        "STR_DIST_NAMED_HM_KM": hm_union,
        "STR_DIST_NAMED_HIGH_KM": high_union,
        "STR_DIST_NAMED_ALL_KM": all_named_union,
    }
    for feature, geometry in sample_distances.items():
        independent = _nearest_distance_km(sample_points, geometry)
        reported = feature_table.iloc[sample_indices][feature].to_numpy(dtype=float)
        sample_details.append(
            {
                "feature": feature,
                "n": int(sample_size),
                "max_abs_difference_km": float(np.max(np.abs(independent - reported))),
            }
        )
    nearest_audit_pass = all(item["max_abs_difference_km"] <= 1e-6 for item in sample_details)
    checks.append(_check("independent_nearest_distance_sample_audit", nearest_audit_pass, sample_details))

    zero_checks = []
    for line in list(hm_segments.geometry)[:10]:
        sample_point = line.interpolate(0.5, normalized=True)
        zero_checks.append(float(sample_point.distance(hm_union) / 1000.0))
    checks.append(
        _check(
            "hm_zero_distance_test",
            bool(max(zero_checks) <= 1e-9),
            {"n": len(zero_checks), "max_distance_km": max(zero_checks)},
        )
    )

    test_line = LineString([(0.0, 0.0), (10_000.0, 0.0)])
    symmetry = []
    for distance_m in [1000.0, 2000.0, 5000.0]:
        above = Point(5000.0, distance_m).distance(test_line)
        below = Point(5000.0, -distance_m).distance(test_line)
        symmetry.append(
            {
                "offset_m": distance_m,
                "above_m": float(above),
                "below_m": float(below),
                "abs_below_error_m": float(abs(below - distance_m)),
            }
        )
    monotonic = all(symmetry[index]["above_m"] < symmetry[index + 1]["above_m"] for index in range(len(symmetry) - 1))
    symmetry_pass = monotonic and all(item["abs_below_error_m"] <= 1e-6 for item in symmetry)
    checks.append(_check("synthetic_distance_monotonic_and_symmetric", symmetry_pass, symmetry))

    known_line = LineString([(-10_000.0, 0.0), (10_000.0, 0.0)])
    length_in_circle = _line_length_in_radius(Point(0.0, 0.0), 10_000.0, [known_line]) / 1000.0
    expected_length_km = 20.0
    expected_density = expected_length_km / (math.pi * 10.0 * 10.0 / 1000.0)
    observed_density = length_in_circle / (math.pi * 10.0 * 10.0 / 1000.0)
    checks.append(
        _check(
            "known_line_density_test",
            bool(abs(length_in_circle - expected_length_km) <= 1e-9 and abs(observed_density - expected_density) <= 1e-9),
            {
                "length_km": length_in_circle,
                "expected_length_km": expected_length_km,
                "density": observed_density,
                "expected_density": expected_density,
                "units": "km per 1,000 km2",
            },
        )
    )

    hm_lines = list(hm_segments.geometry)
    density_indices = np.sort(rng.choice(len(feature_table), size=min(10, len(feature_table)), replace=False))
    density_rows = []
    for index in density_indices:
        point = Point(float(feature_table.iloc[index]["x_m"]), float(feature_table.iloc[index]["y_m"]))
        length_km = _line_length_in_radius(point, 25_000.0, hm_lines) / 1000.0
        independent = length_km / (math.pi * 25.0 * 25.0 / 1000.0)
        reported = float(feature_table.iloc[index]["STR_NAMED_LINE_DENSITY_25KM"])
        density_rows.append({"support_id": str(feature_table.iloc[index]["support_id"]), "abs_difference": abs(independent - reported)})
    checks.append(
        _check(
            "independent_density_sample_audit",
            bool(max(row["abs_difference"] for row in density_rows) <= 1e-6),
            {"n": len(density_rows), "max_abs_difference": max(row["abs_difference"] for row in density_rows)},
        )
    )

    hm_system_unions = _named_system_unions(segments, {"named_high", "named_medium"})
    system_counts = np.zeros(len(feature_table), dtype=int)
    for system_geometry in hm_system_unions.values():
        system_counts += (_nearest_distance_km(point_array, system_geometry) <= 25.0).astype(int)
    count_difference = np.abs(system_counts - feature_table["STR_NAMED_SYSTEM_COUNT_25KM"].to_numpy(dtype=int))
    checks.append(
        _check(
            "named_system_count_reproduces",
            bool(count_difference.max() == 0),
            {
                "n_systems": len(hm_system_unions),
                "max_abs_difference": int(count_difference.max()),
                "values": feature_table["STR_NAMED_SYSTEM_COUNT_25KM"].value_counts().to_dict(),
            },
        )
    )

    candidate_distance_matrix = np.column_stack(
        [
            _nearest_distance_km(point_array, row.geometry)
            for row in candidates.itertuples()
        ]
    )
    candidate_nearest = candidate_distance_matrix.min(axis=1)
    candidate_distance_difference = np.abs(
        candidate_nearest - feature_table["STR_DIST_MAJOR_INTERSECTION_KM"].to_numpy(dtype=float)
    )
    candidate_counts_reported = feature_table["STR_MAJOR_INTERSECTION_COUNT_25KM"].to_numpy(dtype=int)
    candidate_counts_independent = (candidate_distance_matrix <= 25.0).sum(axis=1)
    candidate_count_difference = np.abs(candidate_counts_reported - candidate_counts_independent)
    checks.append(
        _check(
            "candidate_intersection_distance_and_count_reproduce",
            bool(candidate_distance_difference.max() <= 1e-6 and candidate_count_difference.max() == 0),
            {
                "n_candidates": int(len(candidates)),
                "max_distance_difference_km": float(candidate_distance_difference.max()),
                "max_count_difference": int(candidate_count_difference.max()),
            },
        )
    )

    intersections = _intersection_audit(segments)
    expected_pairs = {tuple(sorted(("Kf", "Mb"))), tuple(sorted(("Ky", "LB")))}
    observed_pairs = {tuple(sorted((str(row["pair"][0]), str(row["pair"][1])))) for row in intersections}
    coordinate_mismatches = []
    if len(intersections) == 2 and len(candidates) == 2:
        for row in intersections:
            pair_key = set(row["pair"])
            candidate_row = candidates[
                candidates.apply(
                    lambda candidate: {
                        candidate["system_a_abbreviation"],
                        candidate["system_b_abbreviation"],
                    }
                    == pair_key,
                    axis=1,
                )
            ]
            if candidate_row.empty:
                coordinate_mismatches.append({"pair": list(row["pair"]), "error": "candidate pair missing"})
                continue
            distance = math.hypot(
                float(candidate_row.iloc[0]["x_m"]) - row["x_m"],
                float(candidate_row.iloc[0]["y_m"]) - row["y_m"],
            )
            coordinate_mismatches.append({"pair": list(row["pair"]), "distance_m": distance})
    intersection_pass = (
        len(intersections) == 2
        and observed_pairs == expected_pairs
        and len(candidates) == 2
        and all(item.get("distance_m", 1e9) <= 0.01 for item in coordinate_mismatches)
        and candidates["candidate_status"].eq("candidate_contact_not_established_geological_node").all()
        and candidates["qualification"].str.contains("no independent evidence", case=False, regex=False).all()
    )
    checks.append(
        _check(
            "exact_two_candidate_intersections_preserved",
            bool(intersection_pass),
            {
                "independent_intersections": intersections,
                "coordinate_comparison": coordinate_mismatches,
                "candidate_status": candidates["candidate_status"].tolist(),
            },
        )
    )

    confidence_counts = confidence["scope_type"].value_counts().to_dict()
    bridge_counts = bridge["scope_type"].value_counts().to_dict()
    shortlist_roles = shortlist["shortlist_role"].value_counts().to_dict()
    diagnostic_pass = (
        set(confidence["comparison"].unique()) == {"HM_vs_HIGH", "HM_vs_ALL_NAMED", "HIGH_vs_ALL_NAMED"}
        and confidence_counts.get("overall", 0) == 3
        and set(bridge["structure_id"].unique()) == {"DALY2025_MSTR_MBFZ", "DALY2025_MSTR_NY"}
        and bridge_counts.get("overall", 0) == 2
        and "unclassified" not in shortlist_roles
        and len(correlations) == len(feature_ids) * (len(feature_ids) - 1) // 2
    )
    checks.append(
        _check(
            "confidence_bridge_correlation_and_shortlist_diagnostics",
            bool(diagnostic_pass),
            {
                "confidence_scope_counts": confidence_counts,
                "bridge_scope_counts": bridge_counts,
                "shortlist_roles": shortlist_roles,
                "correlation_pairs": int(len(correlations)),
            },
        )
    )

    map_records = [record for record in manifest["outputs"] if str(record["path"]).endswith(".png")]
    map_details = []
    map_failures = []
    for record in map_records:
        path = resolve_path(record["path"])
        try:
            with Image.open(path) as image:
                width, height = image.size
                image_format = image.format
            map_details.append({"path": record["path"], "width_px": width, "height_px": height, "format": image_format})
            if width < 2200 or height < 1600 or image_format != "PNG":
                map_failures.append({"path": record["path"], "error": "too small or wrong format"})
        except Exception as exc:
            map_failures.append({"path": record["path"], "error": str(exc)})
    checks.append(
        _check(
            "visual_products_exist_as_readable_png",
            len(map_records) == len(config["outputs"]["figures"]) and not map_failures,
            map_details if not map_failures else map_failures,
        )
    )

    crosswalk_pass = (
        len(crosswalk) == len(feature_table)
        and crosswalk["support_id"].is_unique
        and crosswalk["support_id"].equals(feature_table["support_id"])
    )
    checks.append(
        _check(
            "v14_crosswalk_preserves_support_order_and_ids",
            bool(crosswalk_pass),
            {
                "rows": int(len(crosswalk)),
                "unique_support_ids": int(crosswalk["support_id"].nunique()),
                "fold_counts": crosswalk["v14_fold_label"].value_counts().to_dict(),
            },
        )
    )

    feature_summary_details = {
        "rows": int(len(feature_support)),
        "n_missing_total": int(feature_support["n_missing"].sum()),
        "all_n_match": bool((feature_support["n"] == len(feature_table)).all()),
    }
    checks.append(
        _check(
            "feature_support_table_complete",
            bool(
                len(feature_support) == len(feature_ids)
                and feature_summary_details["n_missing_total"] == 0
                and feature_summary_details["all_n_match"]
            ),
            feature_summary_details,
        )
    )

    # The summary is written only after all checks have been evaluated.
    n_fail = sum(1 for item in checks if item["status"] == "fail")
    validation = {
        "schema_version": "1.0.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": relative_posix(manifest_file),
        "status": "pass" if n_fail == 0 else "fail",
        "n_checks": len(checks),
        "n_pass": len(checks) - n_fail,
        "n_fail": n_fail,
        "checks": checks,
        "notes": [
            "The verifier recalculates independent distances and densities from the v2 source geometry.",
            "No V14 occurrence field is read; the crosswalk uses only id, centroid_x and centroid_y.",
            "Candidate intersections are checked as cartographic contacts, not geological nodes.",
        ],
    }
    target = _configured_output_path(config, "validation")
    write_json(target, _json_safe(validation))
    print(f"Structural predictor verification: {validation['status']} ({validation['n_pass']}/{validation['n_checks']} checks)")
    print(f"Validation: {relative_posix(target)}")
    if n_fail:
        for item in checks:
            if item["status"] == "fail":
                print(f"FAILED: {item['name']}")
    return validation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    result = verify(args.manifest)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
