from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from osgeo import gdal

from .common import resolve_path, relative_posix, sha256_file


def _check(name: str, passed: bool, details: Any) -> dict[str, Any]:
    return {"name": name, "status": "pass" if passed else "fail", "details": details}


def verify(manifest_path: str | Path) -> dict[str, Any]:
    gdal.UseExceptions()
    path = resolve_path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    config = json.loads(resolve_path(manifest["config"]["path"]).read_text(encoding="utf-8"))
    outdir = resolve_path(config["outputs"]["directory"])
    checks: list[dict[str, Any]] = []

    output_failures = []
    for record in manifest["outputs"]:
        output_path = resolve_path(record["path"])
        if not output_path.exists():
            output_failures.append({"path": record["path"], "error": "missing"})
        elif output_path.stat().st_size != int(record["bytes"]):
            output_failures.append({"path": record["path"], "error": "byte count mismatch"})
        elif sha256_file(output_path) != record["sha256"]:
            output_failures.append({"path": record["path"], "error": "sha256 mismatch"})
    checks.append(_check("manifest_outputs_match_files", not output_failures, output_failures))

    source_failures = []
    source_registry = pd.read_csv(outdir / config["outputs"]["source_registry_csv"])
    for row in source_registry.itertuples():
        source_path = resolve_path(row.local_path)
        if not source_path.exists():
            source_failures.append({"path": row.local_path, "error": "missing"})
        elif source_path.stat().st_size != int(row.bytes):
            source_failures.append({"path": row.local_path, "error": "byte count mismatch"})
        elif sha256_file(source_path) != row.sha256:
            source_failures.append({"path": row.local_path, "error": "sha256 mismatch"})
    checks.append(_check("registered_sources_match_files", not source_failures, source_failures))

    domains = gpd.read_file(
        outdir / config["outputs"]["domain_gpkg"],
        layer=config["outputs"]["domain_layer"],
        engine="pyogrio",
    )
    boundaries = gpd.read_file(
        outdir / config["outputs"]["boundary_gpkg"],
        layer=config["outputs"]["boundary_layer"],
        engine="pyogrio",
    )
    structural = gpd.read_file(
        outdir / config["outputs"]["structural_gpkg"],
        layer=config["outputs"]["structural_layer"],
        engine="pyogrio",
    )
    crosswalk = pd.read_csv(outdir / config["outputs"]["crosswalk_csv"])

    checks.append(_check("domain_layer_has_nine_classes", len(domains) == 9, {"features": len(domains)}))
    checks.append(_check("domain_ids_match_crosswalk", set(domains["domain_id"]) == set(crosswalk["domain_id"]), sorted(crosswalk["domain_id"])))
    checks.append(_check("domain_crs_is_epsg_32735", domains.crs.to_string() == "EPSG:32735", domains.crs.to_string()))
    checks.append(_check("domain_geometries_valid", bool(domains.geometry.is_valid.all()), int(domains.geometry.is_valid.sum())))
    overlaps = []
    for left in range(len(domains)):
        for right in range(left + 1, len(domains)):
            area_m2 = float(domains.geometry.iloc[left].intersection(domains.geometry.iloc[right]).area)
            if area_m2 > 1.0:
                overlaps.append({"domain_a": domains.domain_id.iloc[left], "domain_b": domains.domain_id.iloc[right], "area_m2": area_m2})
    checks.append(_check("domain_overlaps_absent", not overlaps, overlaps))

    checks.append(_check("boundary_layer_nonempty", not boundaries.empty, {"features": len(boundaries)}))
    checks.append(_check("boundary_crs_is_epsg_32735", boundaries.crs.to_string() == "EPSG:32735", boundaries.crs.to_string()))
    checks.append(_check("boundary_geometries_valid", bool(boundaries.geometry.is_valid.all()), int(boundaries.geometry.is_valid.sum())))

    expected_fids = list(range(1, 127))
    checks.append(_check("structural_feature_count", len(structural) == 126, {"features": len(structural)}))
    checks.append(_check("structural_crs_is_epsg_32735", structural.crs.to_string() == "EPSG:32735", structural.crs.to_string()))
    checks.append(_check("structural_geometries_valid", bool(structural.geometry.is_valid.all()), int(structural.geometry.is_valid.sum())))
    checks.append(_check("structural_source_fids_preserved", sorted(int(value) for value in structural["source_fid"]) == expected_fids, {"minimum": int(structural["source_fid"].min()), "maximum": int(structural["source_fid"].max()), "unique": int(structural["source_fid"].nunique())}))
    checks.append(_check("structural_names_absent", bool(structural["structure_name"].isna().all()), int(structural["structure_name"].isna().sum())))
    checks.append(_check("structural_types_absent", bool(structural["structure_type"].isna().all()), int(structural["structure_type"].isna().sum())))

    raster = gdal.Open(str(outdir / config["outputs"]["domain_raster"]))
    raster_values = sorted(int(value) for value in np.unique(raster.GetRasterBand(1).ReadAsArray()))
    raster_crs = raster.GetSpatialRef().GetAuthorityCode(None)
    checks.append(_check("domain_raster_classes_complete", raster_values == list(range(10)), raster_values))
    checks.append(_check("domain_raster_crs_is_epsg_32735", raster_crs == "32735", raster_crs))

    coverage = pd.read_csv(outdir / config["outputs"]["v14_coverage_csv"])
    validation = json.loads((outdir / config["outputs"]["validation_json"]).read_text(encoding="utf-8"))
    reference = config["v14_reference"]
    training = pd.read_csv(resolve_path(reference["training_csv"]))
    coverage_summary = validation["v14_reference_coverage"]
    complete_positive_missing = int(
        (
            coverage["complete_case"].astype(bool)
            & (coverage[reference["label_column"]] == 1)
            & (coverage["assignment_status"] != "assigned")
        ).sum()
    )
    checks.append(_check("v14_coverage_row_count", len(coverage) == int(reference["expected_rows"]), {"rows": len(coverage)}))
    checks.append(_check("v14_coverage_cell_ids_match_reference", set(coverage["cell_id"]) == set(training[reference["id_column"]]), {"coverage_ids": int(coverage["cell_id"].nunique()), "reference_ids": int(training[reference["id_column"]].nunique())}))
    checks.append(_check("v14_complete_case_positive_cells_all_assigned", complete_positive_missing == 0, {"missing": complete_positive_missing}))
    checks.append(_check("v14_coverage_counts_match_validation", coverage_summary["observed"]["rows"] == len(coverage) and coverage_summary["observed"]["complete_cases"] == int(coverage["complete_case"].sum()), coverage_summary["observed"]))

    preserved_v14 = resolve_path("archive/v14_benchmark_preservation_2026-11-05/contents/data/copperbelt_training_v14_centroid_contact.csv")
    current_v14 = resolve_path(reference["training_csv"])
    checks.append(_check("v14_training_matches_preservation_bundle", sha256_file(current_v14) == sha256_file(preserved_v14), {"current": relative_posix(current_v14), "preserved": relative_posix(preserved_v14), "sha256": sha256_file(current_v14)}))

    named = pd.read_csv(outdir / config["outputs"]["named_structures_csv"])
    checks.append(_check("named_structures_have_no_geometry_claim", len(named) == 15 and named["geometry_status"].str.contains("not linked", case=False).all(), {"rows": len(named)}))
    stratigraphy = json.loads((outdir / config["outputs"]["stratigraphy_status_json"]).read_text(encoding="utf-8"))
    checks.append(_check("bedrock_gap_is_explicit", stratigraphy["status"] == "missing_source", stratigraphy["status"]))

    valid = all(item["status"] == "pass" for item in checks)
    return {
        "status": "pass" if valid else "fail",
        "manifest": relative_posix(path),
        "build_id": manifest["build_id"],
        "checks": checks,
        "v14_coverage": coverage_summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/processed/geology_foundation/manifest.json")
    args = parser.parse_args(argv)
    result = verify(args.manifest)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
