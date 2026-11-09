from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np

from .common import relative_posix, resolve_path, sha256_file


def _check(name: str, passed: bool, details: Any) -> dict[str, Any]:
    return {"name": name, "status": "pass" if passed else "fail", "details": details}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def verify(manifest_path: str | Path) -> dict[str, Any]:
    manifest_file = resolve_path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    config = json.loads(resolve_path(manifest["config"]["path"]).read_text(encoding="utf-8"))
    summary = json.loads(resolve_path(manifest["summary_file"]).read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    file_failures = []
    for record in [manifest["source"], manifest["config"], *manifest["outputs"]]:
        path = resolve_path(record["path"])
        if not path.exists():
            file_failures.append({"path": record["path"], "error": "missing"})
        elif path.stat().st_size != int(record["bytes"]):
            file_failures.append({"path": record["path"], "error": "byte count mismatch"})
        elif sha256_file(path) != record["sha256"]:
            file_failures.append({"path": record["path"], "error": "sha256 mismatch"})
    checks.append(_check("manifest_source_and_outputs_match_files", not file_failures, file_failures))

    output_directory = resolve_path(config["outputs"]["directory"])
    segments = gpd.read_file(
        output_directory / config["outputs"]["gpkg"],
        layer=config["outputs"]["layer"],
        engine="pyogrio",
    )
    checks.append(_check("feature_count_matches_summary", len(segments) == int(summary["features"]), {"layer": len(segments), "summary": int(summary["features"])}))
    checks.append(_check("all_geometries_are_linestrings", bool((segments.geom_type == "LineString").all()), dict(segments.geom_type.value_counts())))
    checks.append(_check("all_geometries_valid", bool(segments.geometry.is_valid.all()), int(segments.geometry.is_valid.sum())))
    checks.append(_check("all_geometries_nonempty", bool((~segments.geometry.is_empty).all()), int((~segments.geometry.is_empty).sum())))
    checks.append(_check("all_lengths_positive", bool((segments.geometry.length > 0).all()), float(segments.geometry.length.min())))
    checks.append(_check("crs_is_epsg_32735", segments.crs.to_string() == config["target_crs"], segments.crs.to_string()))
    checks.append(_check("structure_ids_unique", bool(segments["structure_id"].is_unique), int(segments["structure_id"].nunique())))
    checks.append(_check("names_remain_unassigned", bool(segments["structure_name"].isna().all()), int(segments["structure_name"].notna().sum())))
    checks.append(_check("types_remain_unassigned", bool(segments["structure_type"].isna().all()), int(segments["structure_type"].notna().sum())))
    checks.append(_check("hierarchy_remains_unassigned", bool((segments["hierarchy_status"] == "not assigned").all()), dict(segments["hierarchy_status"].value_counts())))
    checks.append(_check("source_doi_uniform", bool((segments["source_doi"] == config["source"]["doi"]).all()), sorted(segments["source_doi"].dropna().unique().tolist())))
    checks.append(_check("all_features_are_major_fault_zone_candidates", bool((segments["feature_class"] == "major_fault_zone_candidate").all()), dict(segments["feature_class"].value_counts())))

    summary_length = float(segments["length_km"].sum())
    checks.append(_check("total_length_matches_summary", abs(summary_length - float(summary["total_length_km"])) < 0.02, {"layer": round(summary_length, 3), "summary": float(summary["total_length_km"])}))

    expected_bounds = summary["source_extent_m"]
    xmin, ymin, xmax, ymax = segments.total_bounds
    tolerance_m = 2000.0
    bounds_ok = (
        xmin >= expected_bounds["xmin"] - tolerance_m
        and ymin >= expected_bounds["ymin"] - tolerance_m
        and xmax <= expected_bounds["xmax"] + tolerance_m
        and ymax <= expected_bounds["ymax"] + tolerance_m
    )
    checks.append(_check("features_stay_within_source_extent", bounds_ok, {"vector_bounds_m": [float(xmin), float(ymin), float(xmax), float(ymax)], "source_bounds_m": expected_bounds, "tolerance_m": tolerance_m}))

    v14_path = resolve_path(config["source"]["v14_training_csv"])
    v14_hash = sha256_file(v14_path)
    checks.append(_check("v14_training_table_unchanged", v14_hash == config["source"]["v14_training_sha256"], {"path": relative_posix(v14_path), "sha256": v14_hash}))

    quality = summary["quality_status"]
    checks.append(_check("summary_quality_status_all_pass", all(bool(value) for key, value in quality.items() if key != "crs") and quality["crs"] == config["target_crs"], quality))
    checks.append(_check("all_expected_overlays_present", all(resolve_path(record["path"]).exists() for record in manifest["outputs"] if str(record["path"]).endswith(".png")), [record["path"] for record in manifest["outputs"] if str(record["path"]).endswith(".png")]))

    valid = all(item["status"] == "pass" for item in checks)
    return _json_safe({
        "status": "pass" if valid else "fail",
        "manifest": relative_posix(manifest_file),
        "build_id": manifest["build_id"],
        "checks": checks,
        "feature_summary": {
            "features": int(len(segments)),
            "total_length_km": round(float(segments["length_km"].sum()), 3),
            "line_classes": {str(key): int(value) for key, value in segments["line_class"].value_counts().items()},
            "extraction_confidence": {str(key): int(value) for key, value in segments["extraction_confidence"].value_counts().items()},
        },
    })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="data/processed/geology_foundation/daly_major_structures_manifest_v1.json",
    )
    args = parser.parse_args(argv)
    result = verify(args.manifest)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
