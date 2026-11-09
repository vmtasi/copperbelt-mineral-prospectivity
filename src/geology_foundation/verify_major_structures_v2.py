"""Verify the immutable v1 source and the reconstructed named-structure v2 outputs.

The verifier is read-only apart from writing its validation JSON. It does not
rebuild geometry, rerun attribution or fit any model.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from PIL import Image

from .common import relative_posix, resolve_path, sha256_file, write_json


DEFAULT_MANIFEST = "data/processed/geology_foundation/daly_structure_reconstruction_manifest_v2.json"
DEFAULT_OUTPUT = "data/processed/geology_foundation/daly_structure_reconstruction_validation_v2.json"
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


def _ids_from_structures(structures: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    accepted: list[str] = []
    candidates: list[str] = []
    for structure in structures:
        for part in structure.get("centerline_parts", []):
            accepted.extend(part.get("ordered_segment_ids", []))
        accepted.extend(item["segment_id"] for item in structure.get("supporting_segments", []))
        candidates.extend(
            item["segment_id"] for item in structure.get("candidate_segments_not_assigned", [])
        )
    return accepted, candidates


def _nearest_endpoint_distance(geometry: Any, x: float, y: float) -> float:
    coords = list(geometry.coords)
    return min(
        math.hypot(float(x) - float(px), float(y) - float(py))
        for px, py in (coords[0], coords[-1])
    )


def _configured_output_path(config: dict[str, Any], key: str) -> Path:
    value = Path(config["outputs"][key])
    if value.is_absolute() or len(value.parts) > 1:
        return resolve_path(value)
    return resolve_path(config["outputs"]["directory"]) / value


def verify(manifest_path: str | Path) -> dict[str, Any]:
    manifest_file = resolve_path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    config_path = resolve_path(manifest["config"]["path"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    attribution_path = resolve_path(config["sources"]["attribution_config"])
    attribution = json.loads(attribution_path.read_text(encoding="utf-8"))
    reference_path = resolve_path(config["sources"]["named_reference_csv"])
    reference = pd.read_csv(reference_path, dtype={"abbreviation_as_published": str})
    summary_path = _configured_output_path(config, "summary_json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    checks: list[dict[str, Any]] = []

    source_records = [*manifest["sources"], manifest["config"], manifest["attribution_config"]]
    source_failures = _file_mismatches(source_records)
    checks.append(
        _check(
            "manifest_source_hashes_match_files",
            not source_failures,
            source_failures or {"checked_files": len(source_records)},
        )
    )

    protected_failures: list[dict[str, Any]] = []
    protected_observed: dict[str, dict[str, Any]] = {}
    for key, expected in config["protected_source_hashes"].items():
        path = resolve_path(config["sources"][key])
        if not path.exists():
            protected_failures.append({"key": key, "path": relative_posix(path), "error": "missing"})
            continue
        actual = sha256_file(path)
        protected_observed[key] = {
            "path": relative_posix(path),
            "sha256": actual,
            "matches_config": actual.lower() == str(expected).lower(),
        }
        if actual.lower() != str(expected).lower():
            protected_failures.append(
                {
                    "key": key,
                    "path": relative_posix(path),
                    "expected": expected,
                    "actual": actual,
                }
            )
    checks.append(
        _check(
            "protected_v1_v14_and_source_hashes_unchanged",
            not protected_failures,
            protected_failures or protected_observed,
        )
    )

    v1_hash = sha256_file(resolve_path(config["sources"]["v1_gpkg"]))
    checks.append(
        _check(
            "v1_gpkg_sha256_unchanged",
            v1_hash.lower() == config["protected_source_hashes"]["v1_gpkg"].lower(),
            {"path": config["sources"]["v1_gpkg"], "sha256": v1_hash},
        )
    )
    v14_hash = sha256_file(resolve_path(config["sources"]["v14_training_csv"]))
    checks.append(
        _check(
            "v14_training_csv_sha256_unchanged",
            v14_hash.lower() == config["sources"]["v14_training_sha256"].lower(),
            {"path": config["sources"]["v14_training_csv"], "sha256": v14_hash},
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

    map_records = [record for record in manifest["outputs"] if str(record["path"]).endswith(".png")]
    map_details: list[dict[str, Any]] = []
    map_failures: list[dict[str, Any]] = []
    for record in map_records:
        path = resolve_path(record["path"])
        try:
            with Image.open(path) as image:
                width, height = image.size
                image_format = image.format
            map_details.append(
                {
                    "path": record["path"],
                    "width_px": int(width),
                    "height_px": int(height),
                    "format": image_format,
                }
            )
            if width < 2000 or height < 2000 or image_format != "PNG":
                map_failures.append({"path": record["path"], "error": "unexpected map dimensions or format"})
        except Exception as exc:  # pragma: no cover - defensive file check
            map_failures.append({"path": record["path"], "error": str(exc)})
    checks.append(
        _check(
            "visual_products_exist_as_readable_png",
            bool(map_records) and not map_failures,
            map_details if not map_failures else map_failures,
        )
    )

    output_gpkg = _configured_output_path(config, "named_gpkg")
    layer_names = [str(item[0]) for item in pyogrio.list_layers(output_gpkg)]
    expected_layers = list(manifest["layers"])
    checks.append(
        _check(
            "geopackage_layers_exact",
            layer_names == expected_layers,
            {"observed": layer_names, "expected": expected_layers},
        )
    )

    v1 = gpd.read_file(
        resolve_path(config["sources"]["v1_gpkg"]),
        layer=config["sources"]["v1_layer"],
        engine="pyogrio",
    )
    segments = gpd.read_file(output_gpkg, layer="v1_segments_attributed", engine="pyogrio")
    centerlines = gpd.read_file(output_gpkg, layer="named_structure_centrelines", engine="pyogrio")
    bridges = gpd.read_file(output_gpkg, layer="bridge_connectors", engine="pyogrio")
    catalogue = gpd.read_file(output_gpkg, layer="named_structure_catalogue", engine="pyogrio")

    crs_details = {
        "v1": v1.crs.to_string() if v1.crs else None,
        "v1_segments_attributed": segments.crs.to_string() if segments.crs else None,
        "named_structure_centrelines": centerlines.crs.to_string() if centerlines.crs else None,
        "bridge_connectors": bridges.crs.to_string() if bridges.crs else None,
        "named_structure_catalogue": catalogue.crs.to_string() if catalogue.crs else None,
    }
    checks.append(
        _check(
            "all_layers_in_project_crs",
            all(value == PROJECT_CRS for value in crs_details.values()),
            crs_details,
        )
    )

    validity_details = {
        "v1_segments_attributed": int(segments.geometry.is_valid.sum()),
        "named_structure_centrelines": int(centerlines.geometry.is_valid.sum()),
        "bridge_connectors": int(bridges.geometry.is_valid.sum()),
        "named_structure_catalogue": int(catalogue.geometry.is_valid.sum()),
    }
    validity_expected = {
        "v1_segments_attributed": len(segments),
        "named_structure_centrelines": len(centerlines),
        "bridge_connectors": len(bridges),
        "named_structure_catalogue": len(catalogue),
    }
    checks.append(
        _check(
            "all_output_geometries_valid",
            validity_details == validity_expected,
            {"valid_counts": validity_details, "feature_counts": validity_expected},
        )
    )
    checks.append(
        _check(
            "no_empty_output_geometries",
            all(bool((~frame.geometry.is_empty).all()) for frame in (segments, centerlines, bridges, catalogue)),
            {
                "v1_segments_attributed": int(segments.geometry.is_empty.sum()),
                "named_structure_centrelines": int(centerlines.geometry.is_empty.sum()),
                "bridge_connectors": int(bridges.geometry.is_empty.sum()),
                "named_structure_catalogue": int(catalogue.geometry.is_empty.sum()),
            },
        )
    )

    v1_ids = v1["structure_id"].astype(str).tolist()
    segment_ids = segments["structure_id"].astype(str).tolist()
    duplicate_ids = [key for key, count in Counter(segment_ids).items() if count != 1]
    missing_ids = sorted(set(v1_ids) - set(segment_ids))
    extra_ids = sorted(set(segment_ids) - set(v1_ids))
    checks.append(
        _check(
            "all_v1_segment_ids_appear_exactly_once",
            len(segments) == len(v1)
            and set(v1_ids) == set(segment_ids)
            and not duplicate_ids
            and not missing_ids
            and not extra_ids,
            {
                "v1_features": len(v1),
                "v2_features": len(segments),
                "duplicate_ids": duplicate_ids,
                "missing_ids": missing_ids,
                "extra_ids": extra_ids,
            },
        )
    )

    original_columns = [column for column in v1.columns if column != "geometry"]
    expected_appended_columns = [
        "attributed_structure_id",
        "attributed_structure_abbreviation",
        "attributed_structure_name",
        "attributed_status",
        "attribution_confidence",
        "mapping_role",
        "parent_part_id",
        "label_anchor_key",
        "label_anchor_px_x",
        "label_anchor_px_y",
        "label_distance_px",
        "attribution_evidence",
        "source_name_figure",
        "source_geometry_figure",
        "occurrence_evidence_used",
    ]
    observed_columns = [column for column in segments.columns if column != "geometry"]
    checks.append(
        _check(
            "v1_columns_preserved_and_v2_fields_appended",
            observed_columns == original_columns + expected_appended_columns,
            {
                "missing_original_columns": sorted(set(original_columns) - set(observed_columns)),
                "missing_appended_columns": sorted(set(expected_appended_columns) - set(observed_columns)),
                "column_count": len(observed_columns),
            },
        )
    )

    v1_sorted = v1.sort_values("structure_id").reset_index(drop=True)
    segment_sorted = segments.sort_values("structure_id").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            v1_sorted[original_columns],
            segment_sorted[original_columns],
            check_dtype=False,
            check_exact=True,
        )
        attributes_match = True
        attribute_error: str | None = None
    except AssertionError as exc:
        attributes_match = False
        attribute_error = str(exc)
    checks.append(
        _check(
            "original_v1_attributes_unchanged",
            attributes_match,
            attribute_error or f"all {len(original_columns)} original attribute columns match",
        )
    )

    geometry_mismatches = []
    for structure_id, source_geometry, output_geometry in zip(
        v1_sorted["structure_id"].astype(str),
        v1_sorted.geometry,
        segment_sorted.geometry,
        strict=True,
    ):
        if not source_geometry.equals_exact(output_geometry, 0.0):
            geometry_mismatches.append(structure_id)
    checks.append(
        _check(
            "original_v1_geometries_unchanged",
            not geometry_mismatches,
            {"mismatch_count": len(geometry_mismatches), "mismatches": geometry_mismatches[:20]},
        )
    )

    structures = attribution["structures"]
    expected_abbreviations = reference["abbreviation_as_published"].astype(str).tolist()
    catalogue_abbreviations = catalogue["abbreviation"].astype(str).tolist()
    checks.append(
        _check(
            "all_15_published_names_catalogued",
            len(reference) == 15
            and len(catalogue) == 15
            and Counter(expected_abbreviations) == Counter(catalogue_abbreviations),
            {
                "reference_count": len(reference),
                "catalogue_count": len(catalogue),
                "missing": sorted(set(expected_abbreviations) - set(catalogue_abbreviations)),
                "extra": sorted(set(catalogue_abbreviations) - set(expected_abbreviations)),
            },
        )
    )
    reference_pairs = set(zip(reference["abbreviation_as_published"], reference["structure_name"], strict=True))
    catalogue_pairs = set(zip(catalogue["abbreviation"], catalogue["structure_name"], strict=True))
    checks.append(
        _check(
            "catalogue_names_match_published_reference",
            reference_pairs == catalogue_pairs,
            {
                "missing_pairs": sorted(reference_pairs - catalogue_pairs),
                "extra_pairs": sorted(catalogue_pairs - reference_pairs),
            },
        )
    )

    unresolved = catalogue[catalogue["geometry_status"] == "unresolved_reference_only"]
    unresolved_source = [
        structure for structure in structures if structure["geometry_status"] == "unresolved_reference_only"
    ]
    unresolved_ids = set(unresolved["structure_id"].astype(str))
    centerline_ids = set(centerlines["structure_id"].astype(str))
    checks.append(
        _check(
            "unresolved_names_have_no_centerline_or_bridge",
            len(unresolved) == 3
            and len(unresolved_source) == 3
            and unresolved_ids.isdisjoint(centerline_ids)
            and unresolved_ids.isdisjoint(set(bridges["structure_id"].astype(str))),
            {
                "unresolved_catalogue": sorted(unresolved["structure_name"].tolist()),
                "centerline_intersection": sorted(unresolved_ids & centerline_ids),
                "bridge_intersection": sorted(unresolved_ids & set(bridges["structure_id"].astype(str))),
            },
        )
    )
    checks.append(
        _check(
            "unresolved_config_entries_have_empty_parts",
            all(not structure.get("centerline_parts") for structure in unresolved_source),
            {structure["abbreviation"]: len(structure.get("centerline_parts", [])) for structure in unresolved_source},
        )
    )

    accepted_ids, candidate_ids = _ids_from_structures(structures)
    duplicate_accepted = [key for key, count in Counter(accepted_ids).items() if count != 1]
    invalid_accepted = sorted(set(accepted_ids) - set(v1_ids))
    missing_candidate_refs = sorted(set(candidate_ids) - set(v1_ids))
    checks.append(
        _check(
            "configured_segments_exist_and_are_uniquely_assigned",
            not duplicate_accepted and not invalid_accepted and not missing_candidate_refs,
            {
                "accepted_segment_count": len(accepted_ids),
                "assigned_to_multiple_roles": duplicate_accepted,
                "accepted_not_in_v1": invalid_accepted,
                "candidate_not_in_v1": missing_candidate_refs,
            },
        )
    )

    attributed = segments[segments["attributed_structure_id"].notna()]
    attributed_ids = attributed["structure_id"].astype(str).tolist()
    checks.append(
        _check(
            "attributed_segment_table_matches_configuration",
            set(attributed_ids) == set(accepted_ids)
            and len(attributed_ids) == len(set(attributed_ids))
            and all(value == "no" for value in attributed["occurrence_evidence_used"].astype(str)),
            {
                "table_count": len(attributed_ids),
                "configured_count": len(accepted_ids),
                "table_only": sorted(set(attributed_ids) - set(accepted_ids)),
                "configuration_only": sorted(set(accepted_ids) - set(attributed_ids)),
            },
        )
    )

    checks.append(
        _check(
            "occurrence_evidence_not_used",
            bool((segments["occurrence_evidence_used"].astype(str) == "no").all())
            and bool((catalogue["occurrence_evidence_used"].astype(str) == "no").all()),
            {
                "segment_values": sorted(segments["occurrence_evidence_used"].astype(str).unique().tolist()),
                "catalogue_values": sorted(catalogue["occurrence_evidence_used"].astype(str).unique().tolist()),
            },
        )
    )

    centerline_parts = []
    for structure in structures:
        for part in structure.get("centerline_parts", []):
            centerline_parts.append(
                {
                    "structure_id": structure["structure_id"],
                    "part_id": part["part_id"],
                    "source_segment_ids": "|".join(part["ordered_segment_ids"]),
                }
            )
    expected_parts = pd.DataFrame(centerline_parts)
    observed_parts = centerlines[["structure_id", "part_id", "source_segment_ids"]].copy()
    parts_match = False
    if len(expected_parts) == len(observed_parts):
        expected_parts = expected_parts.sort_values(["structure_id", "part_id"]).reset_index(drop=True)
        observed_parts = observed_parts.sort_values(["structure_id", "part_id"]).reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(expected_parts, observed_parts, check_dtype=False)
            parts_match = True
        except AssertionError:
            parts_match = False
    checks.append(
        _check(
            "centerline_parts_match_ordered_configuration",
            parts_match,
            {"configured_parts": len(expected_parts), "geopackage_parts": len(observed_parts)},
        )
    )

    length_error_km = (
        centerlines.geometry.length / 1000.0 - centerlines["total_length_km"].astype(float)
    ).abs()
    checks.append(
        _check(
            "centerline_geometry_lengths_match_reported_lengths",
            bool((length_error_km <= 0.0015).all()),
            {"maximum_error_km": float(length_error_km.max()), "tolerance_km": 0.0015},
        )
    )

    declared_bridge_count = int(centerlines["n_bridges"].sum())
    checks.append(
        _check(
            "bridge_connector_count_matches_centerlines",
            declared_bridge_count == len(bridges) and len(bridges) > 0,
            {"centerline_declared": declared_bridge_count, "bridge_layer_rows": len(bridges)},
        )
    )

    center_bridge_counts = (
        centerlines.groupby("structure_id")["n_bridges"].sum().astype(int).rename("declared_count")
    )
    center_bridge_lengths = (
        centerlines.groupby("structure_id")["bridge_length_km"].sum().astype(float).rename("declared_length_km")
    )
    actual_bridge_counts = bridges.groupby("structure_id").size().rename("actual_count")
    actual_bridge_lengths = (
        (bridges.geometry.length / 1000.0).groupby(bridges["structure_id"]).sum().rename("actual_length_km")
    )
    bridge_audit = pd.concat(
        [center_bridge_counts, center_bridge_lengths, actual_bridge_counts, actual_bridge_lengths],
        axis=1,
    ).fillna(0.0)
    bridge_audit["count_delta"] = bridge_audit["actual_count"] - bridge_audit["declared_count"]
    bridge_audit["length_delta_km"] = bridge_audit["actual_length_km"] - bridge_audit["declared_length_km"]
    bridge_audit_ok = bool(
        (bridge_audit["count_delta"].abs() < 1e-9).all()
        and (bridge_audit["length_delta_km"].abs() <= 0.0015).all()
    )
    checks.append(
        _check(
            "bridge_counts_and_lengths_match_structure_metrics",
            bridge_audit_ok,
            bridge_audit.reset_index().to_dict(orient="records"),
        )
    )

    bridge_ids = bridges["bridge_id"].astype(str).tolist()
    bridge_gap_mismatches = []
    bridge_endpoint_mismatches = []
    segment_geometry_by_id = dict(zip(segments["structure_id"].astype(str), segments.geometry, strict=True))
    for row in bridges.itertuples(index=False):
        calculated_length = float(row.geometry.length)
        if abs(calculated_length - float(row.gap_m)) > 1e-3:
            bridge_gap_mismatches.append(
                {
                    "bridge_id": row.bridge_id,
                    "stored_gap_m": float(row.gap_m),
                    "geometry_length_m": calculated_length,
                }
            )
        from_distance = _nearest_endpoint_distance(
            segment_geometry_by_id[row.from_segment_id], float(row.from_x), float(row.from_y)
        )
        to_distance = _nearest_endpoint_distance(
            segment_geometry_by_id[row.to_segment_id], float(row.to_x), float(row.to_y)
        )
        if from_distance > 1e-3 or to_distance > 1e-3:
            bridge_endpoint_mismatches.append(
                {
                    "bridge_id": row.bridge_id,
                    "from_endpoint_error_m": from_distance,
                    "to_endpoint_error_m": to_distance,
                }
            )
    checks.append(
        _check(
            "bridge_ids_unique",
            len(bridge_ids) == len(set(bridge_ids)) and all(value.startswith("DALY2025_BRG_") for value in bridge_ids),
            {"count": len(bridge_ids), "example": bridge_ids[:3]},
        )
    )
    checks.append(
        _check(
            "bridge_geometries_match_stored_gap_lengths",
            not bridge_gap_mismatches,
            bridge_gap_mismatches or f"all {len(bridges)} bridge lengths match within 0.001 m",
        )
    )
    checks.append(
        _check(
            "bridge_endpoints_are_source_segment_endpoints",
            not bridge_endpoint_mismatches,
            bridge_endpoint_mismatches or f"all {len(bridges)} bridge endpoints match source endpoints",
        )
    )

    segment_csv = pd.read_csv(_configured_output_path(config, "segment_attribution_csv"))
    checks.append(
        _check(
            "segment_attribution_csv_matches_layer",
            len(segment_csv) == len(segments)
            and set(segment_csv["structure_id"].astype(str)) == set(segment_ids)
            and int(segment_csv["attributed_structure_id"].notna().sum()) == len(attributed),
            {
                "csv_rows": len(segment_csv),
                "layer_rows": len(segments),
                "csv_attributed": int(segment_csv["attributed_structure_id"].notna().sum()),
                "layer_attributed": len(attributed),
            },
        )
    )

    summary_csv = pd.read_csv(_configured_output_path(config, "structure_summary_csv"))
    summary_fields = [
        "structure_id",
        "abbreviation",
        "structure_name",
        "attribution_status",
        "attribution_confidence",
        "geometry_status",
        "n_attributed_segments",
        "n_centerline_parts",
        "n_bridges",
        "mapped_length_km",
        "bridge_length_km",
        "centerline_length_km",
    ]
    summary_sorted = summary_csv.sort_values("structure_id").reset_index(drop=True)
    catalogue_sorted = catalogue.sort_values("structure_id").reset_index(drop=True)
    summary_text_ok = True
    summary_numeric_ok = True
    numeric_fields = {
        "n_attributed_segments",
        "n_centerline_parts",
        "n_bridges",
        "mapped_length_km",
        "bridge_length_km",
        "centerline_length_km",
    }
    for field in summary_fields:
        if field in numeric_fields:
            if not np.allclose(
                summary_sorted[field].astype(float),
                catalogue_sorted[field].astype(float),
                atol=0.002,
                rtol=0.0,
            ):
                summary_numeric_ok = False
        elif not summary_sorted[field].astype(str).equals(catalogue_sorted[field].astype(str)):
            summary_text_ok = False
    checks.append(
        _check(
            "structure_summary_csv_matches_geopackage_catalogue",
            len(summary_csv) == len(catalogue) and summary_text_ok and summary_numeric_ok,
            {
                "csv_rows": len(summary_csv),
                "catalogue_rows": len(catalogue),
                "text_fields_match": summary_text_ok,
                "numeric_fields_match": summary_numeric_ok,
            },
        )
    )

    attribution_json = json.loads(
        _configured_output_path(config, "attribution_json").read_text(encoding="utf-8")
    )
    json_abbreviations = {structure["abbreviation"] for structure in attribution_json["structures"]}
    checks.append(
        _check(
            "attribution_json_contains_all_names_and_unresolved_register",
            len(attribution_json["structures"]) == 15
            and json_abbreviations == set(expected_abbreviations)
            and set(attribution_json["unresolved_reference_names"]) == set(unresolved["abbreviation"].astype(str)),
            {
                "structure_count": len(attribution_json["structures"]),
                "unresolved_register": attribution_json["unresolved_reference_names"],
            },
        )
    )

    summary_count_fields = {
        "v1_features": len(v1),
        "attributed_segments": len(attributed),
        "unnamed_segments": len(segments) - len(attributed),
        "centerline_parts": len(centerlines),
        "bridges": len(bridges),
        "published_names": len(catalogue),
        "unresolved_names": len(unresolved),
        "named_structures_with_geometry": len(centerline_ids),
    }
    summary_expected = {
        "v1_features": summary["source_v1"]["features"],
        "attributed_segments": summary["attributed_segments"],
        "unnamed_segments": summary["unnamed_segments"],
        "centerline_parts": summary["centerline_parts"],
        "bridges": summary["bridge_connectors"],
        "published_names": summary["published_names"],
        "unresolved_names": summary["unresolved_reference_only"],
        "named_structures_with_geometry": summary["named_structures_with_geometry"],
    }
    checks.append(
        _check(
            "summary_counts_match_layers",
            summary_count_fields == summary_expected,
            {"observed": summary_count_fields, "summary": summary_expected},
        )
    )

    bridge_summary = (
        bridges.groupby("structure_id")
        .agg(n_bridges=("bridge_id", "size"), bridge_length_km=("gap_km", "sum"), max_gap_km=("gap_km", "max"))
        .reset_index()
        .sort_values("structure_id")
    )
    valid = all(item["status"] == "pass" for item in checks)
    return _json_safe(
        {
            "schema_version": "1.0.0",
            "status": "pass" if valid else "fail",
            "verified_utc": datetime.now(timezone.utc).isoformat(),
            "manifest": relative_posix(manifest_file),
            "build_id": manifest["build_id"],
            "checks": checks,
            "source_files_and_fields": {
                "v1_geometry": {
                    "path": config["sources"]["v1_gpkg"],
                    "layer": config["sources"]["v1_layer"],
                    "key_column": "structure_id",
                    "length_column": "length_km",
                    "attributes_csv": config["sources"]["v1_attributes_csv"],
                },
                "names": {
                    "path": config["sources"]["attribution_config"],
                    "columns": ["abbreviation", "structure_id", "structure_name", "geometry_status", "label_anchor_key"],
                },
                "reference_table": {
                    "path": config["sources"]["named_reference_csv"],
                    "columns": ["abbreviation_as_published", "structure_name"],
                },
                "outputs": {
                    "geopackage": relative_posix(_configured_output_path(config, "named_gpkg")),
                    "segment_table": relative_posix(_configured_output_path(config, "segment_attribution_csv")),
                    "structure_table": relative_posix(_configured_output_path(config, "structure_summary_csv")),
                    "attribution_json": relative_posix(_configured_output_path(config, "attribution_json")),
                    "summary_json": relative_posix(_configured_output_path(config, "summary_json")),
                    "map": relative_posix(_configured_output_path(config, "named_map_png")),
                    "attribution_map": relative_posix(_configured_output_path(config, "attribution_map_png")),
                },
            },
            "counts": summary_count_fields,
            "bridge_summary": bridge_summary.to_dict(orient="records"),
            "protected_inputs": protected_observed,
            "v1_sha256": v1_hash,
            "v14_sha256": v14_hash,
            "outputs": manifest["outputs"],
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = verify(args.manifest)
    write_json(resolve_path(args.output), result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
