from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from osgeo import gdal, osr
from pyproj import Geod

from .common import (
    ROOT,
    file_record,
    load_config,
    relative_posix,
    resolve_path,
    sha256_file,
    software_versions,
    write_csv,
    write_json,
)
from .domains import (
    classify_figure14,
    derive_boundaries,
    polygonize_domains,
    read_rgb_tiff,
    write_categorical_raster,
    write_diagnostic,
)
from .structural import (
    bedrock_stratigraphy_status,
    build_structural_baseline,
    named_structure_reference,
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _target_paths(config: dict[str, Any]) -> dict[str, Path]:
    outputs = config["outputs"]
    outdir = resolve_path(outputs["directory"])
    return {
        "domain_gpkg": outdir / outputs["domain_gpkg"],
        "boundary_gpkg": outdir / outputs["boundary_gpkg"],
        "domain_raster": outdir / outputs["domain_raster"],
        "structural_gpkg": outdir / outputs["structural_gpkg"],
        "crosswalk_csv": outdir / outputs["crosswalk_csv"],
        "source_registry_csv": outdir / outputs["source_registry_csv"],
        "named_structures_csv": outdir / outputs["named_structures_csv"],
        "v14_coverage_csv": outdir / outputs["v14_coverage_csv"],
        "stratigraphy_status_json": outdir / outputs["stratigraphy_status_json"],
        "validation_json": outdir / outputs["validation_json"],
        "diagnostic_png": resolve_path(outputs["diagnostic_png"]),
        "manifest_json": outdir / outputs["manifest_json"],
    }


def _crs_authid(wkt: str) -> str | None:
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    authority = srs.GetAuthorityName(None)
    code = srs.GetAuthorityCode(None)
    return f"{authority}:{code}" if authority and code else None


def _ground_pixel_size_m(geotransform: list[float], width: int, height: int) -> float:
    lon = geotransform[0] + geotransform[1] * width / 2
    lat = geotransform[3] + geotransform[5] * height / 2
    geod = Geod(ellps="WGS84")
    _, _, x_m = geod.inv(lon, lat, lon + geotransform[1], lat)
    _, _, y_m = geod.inv(lon, lat, lon, lat + abs(geotransform[5]))
    return float(round((x_m + y_m) / 2, 1))


def _source_rows(
    source_tiff: Path,
    source_structure: Path,
    raster_metadata: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    daly = config["daly"]
    structural = config["structure"]
    gt = raster_metadata["geotransform"]
    pixel_m = _ground_pixel_size_m(gt, raster_metadata["width"], raster_metadata["height"])
    return [
        {
            "source_id": "DALY2025_FIG14",
            "dataset_name": "Daly et al. (2025) tectonic-domain figure",
            "local_path": relative_posix(source_tiff),
            "upstream_locator": daly.get("upstream_locator", ""),
            "role": "domain classification source",
            "format": "GeoTIFF",
            "crs": _crs_authid(raster_metadata["crs_wkt"]),
            "feature_count": "",
            "ground_pixel_size_m": pixel_m,
            "source_date": "2025",
            "licence_status": daly["source_license"],
            "use_constraint": "Attribution required; retain DOI and Figure 14 citation.",
            "sha256": sha256_file(source_tiff),
            "bytes": source_tiff.stat().st_size,
            "verification_status": "local copy hashed; original georeferencing metadata unavailable",
            "notes": f"RGB {raster_metadata['width']} x {raster_metadata['height']} pixels; {raster_metadata['source_band_dtype']}; geotransform {gt}.",
        },
        {
            "source_id": "SELLEY2005_FAULTS",
            "dataset_name": "Selley et al. (2005) structural linework compilation",
            "local_path": relative_posix(source_structure),
            "upstream_locator": structural.get("upstream_locator", ""),
            "role": "anonymous structural baseline source",
            "format": "GeoPackage",
            "crs": "EPSG:32735",
            "feature_count": 126,
            "ground_pixel_size_m": "",
            "source_date": "2005",
            "licence_status": structural["source_license_status"],
            "use_constraint": "Local research use only until redistribution rights are verified.",
            "sha256": sha256_file(source_structure),
            "bytes": source_structure.stat().st_size,
            "verification_status": "local copy hashed; geometry not independently re-digitised in this build",
            "notes": "Only source, notes and geometry are present. No names, fault types, ages, displacements or mineralisation associations are available.",
        },
    ]


def _write_vector(
    frame: gpd.GeoDataFrame,
    path: Path,
    layer: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(path, layer=layer, driver="GPKG", engine="pyogrio", index=False)


def _check(name: str, passed: bool, details: Any) -> dict[str, Any]:
    return {"name": name, "status": "pass" if passed else "fail", "details": details}


def _v14_coverage(
    domains: gpd.GeoDataFrame,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference = config["v14_reference"]
    source_path = resolve_path(reference["training_csv"])
    frame = pd.read_csv(source_path)
    required_columns = [
        reference["id_column"],
        reference["x_column"],
        reference["y_column"],
        reference["label_column"],
        reference["legacy_domain_column"],
        *reference["complete_case_columns"],
    ]
    missing_columns = sorted(set(required_columns) - set(frame.columns))
    if missing_columns:
        raise KeyError(f"V14 reference table is missing columns: {missing_columns}")

    id_column = reference["id_column"]
    label_column = reference["label_column"]
    legacy_column = reference["legacy_domain_column"]
    complete_case = frame[reference["complete_case_columns"]].notna().all(axis=1)
    points = gpd.GeoDataFrame(
        frame[[id_column, reference["x_column"], reference["y_column"], label_column, legacy_column]].copy(),
        geometry=gpd.points_from_xy(frame[reference["x_column"]], frame[reference["y_column"]]),
        crs=config["target_crs"],
    )
    points["complete_case"] = complete_case.to_numpy()
    joined = gpd.sjoin(
        points,
        domains[["domain_id", "geometry"]],
        how="left",
        predicate="within",
    )
    if joined.index.duplicated().any():
        raise RuntimeError("V14 point-to-domain assignment produced ambiguous boundary matches")

    assigned = joined["domain_id"].notna()
    joined["assignment_status"] = np.where(assigned, "assigned", "outside_reconstructed_envelope")
    joined = joined.rename(columns={"domain_id": "reconstructed_domain_id"})
    rows = joined.drop(columns="geometry").reset_index(drop=True)
    rows = rows.rename(columns={id_column: "cell_id", reference["x_column"]: "centroid_x", reference["y_column"]: "centroid_y"})
    rows[legacy_column] = rows[legacy_column].fillna("")
    rows["reconstructed_domain_id"] = rows["reconstructed_domain_id"].fillna("")
    rows = rows[
        [
            "cell_id",
            "centroid_x",
            "centroid_y",
            label_column,
            legacy_column,
            "complete_case",
            "reconstructed_domain_id",
            "assignment_status",
        ]
    ]

    complete_mask = joined["complete_case"].astype(bool)
    positive_mask = joined[label_column].astype(bool)

    def count_summary(mask: pd.Series) -> dict[str, Any]:
        subset_assigned = assigned & mask
        subset_unassigned = (~assigned) & mask
        return {
            "cells": int(mask.sum()),
            "assigned_cells": int(subset_assigned.sum()),
            "unassigned_cells": int(subset_unassigned.sum()),
            "positive_cells": int((positive_mask & mask).sum()),
            "assigned_positive_cells": int((positive_mask & subset_assigned).sum()),
            "unassigned_positive_cells": int((positive_mask & subset_unassigned).sum()),
        }

    def domain_counts(mask: pd.Series) -> dict[str, int]:
        values = joined.loc[mask & assigned, "reconstructed_domain_id"].value_counts(dropna=True)
        return {str(key): int(value) for key, value in values.sort_index().items()}

    legacy_unassigned = joined.loc[~assigned, legacy_column].fillna("__missing__").value_counts()
    unassigned_ids = joined.loc[~assigned, id_column].astype(int).tolist()
    expected = {
        "rows": int(reference["expected_rows"]),
        "complete_cases": int(reference["expected_complete_cases"]),
        "positives": int(reference["expected_positives"]),
        "complete_case_positives": int(reference["expected_complete_case_positives"]),
    }
    observed = {
        "rows": int(len(joined)),
        "complete_cases": int(complete_mask.sum()),
        "positives": int(positive_mask.sum()),
        "complete_case_positives": int((positive_mask & complete_mask).sum()),
    }
    summary = {
        "reference_csv": relative_posix(source_path),
        "reference_sha256": sha256_file(source_path),
        "spatial_predicate": "cell centroid within reconstructed domain polygon; boundary-touching points are not forced into a domain",
        "expected": expected,
        "observed": observed,
        "matches_frozen_counts": {key: observed[key] == expected[key] for key in expected},
        "all_cells": count_summary(pd.Series(True, index=joined.index)),
        "complete_cases": count_summary(complete_mask),
        "all_cells_by_domain": domain_counts(pd.Series(True, index=joined.index)),
        "complete_cases_by_domain": domain_counts(complete_mask),
        "unassigned_cells_by_legacy_label": {
            str(key): int(value) for key, value in legacy_unassigned.sort_index().items()
        },
        "unassigned_cell_count": len(unassigned_ids),
        "coverage_csv": config["outputs"]["v14_coverage_csv"],
    }
    return rows.to_dict(orient="records"), summary


def _validate(
    assigned: np.ndarray,
    domains: gpd.GeoDataFrame,
    boundaries: gpd.GeoDataFrame,
    structural: gpd.GeoDataFrame,
    class_stats: dict[str, Any],
    named_structures: list[dict[str, Any]],
    v14_coverage: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    required = [item["class_value"] for item in config["daly"]["classes"]]
    present = sorted(int(value) for value in np.unique(assigned) if value != 0)
    checks: list[dict[str, Any]] = []
    checks.append(_check("all_required_domain_classes_present", present == required, {"expected": required, "observed": present}))
    checks.append(_check("domain_ids_unique", domains.domain_id.is_unique, list(domains.domain_id)))
    checks.append(_check("domain_geometries_valid", bool(domains.geometry.is_valid.all()), int(domains.geometry.is_valid.sum())))

    overlaps = []
    for left in range(len(domains)):
        for right in range(left + 1, len(domains)):
            area_m2 = domains.geometry.iloc[left].intersection(domains.geometry.iloc[right]).area
            if area_m2 > 1.0:
                overlaps.append(
                    {
                        "domain_a": domains.domain_id.iloc[left],
                        "domain_b": domains.domain_id.iloc[right],
                        "area_m2": float(area_m2),
                    }
                )
    checks.append(_check("domain_pairwise_overlaps_absent", not overlaps, overlaps))
    checks.append(_check("domain_boundaries_present", not boundaries.empty, int(len(boundaries))))
    checks.append(_check("boundary_ids_unique", boundaries.boundary_id.is_unique, int(len(boundaries))))
    checks.append(_check("boundary_domain_fields_present", bool(boundaries.domain_a.notna().all()), int(boundaries.domain_a.notna().sum())))
    checks.append(_check("structural_feature_count", len(structural) == 126, int(len(structural))))
    checks.append(_check("structural_geometries_valid", bool(structural.geometry.is_valid.all()), int(structural.geometry.is_valid.sum())))
    checks.append(_check("structural_crs_32735", structural.crs.to_string() == "EPSG:32735", structural.crs.to_string()))
    checks.append(_check("structural_names_absent", bool(structural.structure_name.isna().all()), int(structural.structure_name.isna().sum())))
    checks.append(_check("structural_types_absent", bool(structural.structure_type.isna().all()), int(structural.structure_type.isna().sum())))
    checks.append(_check("named_structure_reference_nonempty", len(named_structures) == 15, int(len(named_structures))))
    checks.append(_check("domain_pixel_total_matches_envelope", int(sum(class_stats["class_pixels"].values())) == int(class_stats["envelope_pixels"]), {"class_pixels": int(sum(class_stats["class_pixels"].values())), "envelope_pixels": int(class_stats["envelope_pixels"])}))
    checks.append(_check("v14_reference_counts_match_frozen_benchmark", all(v14_coverage["matches_frozen_counts"].values()), v14_coverage["matches_frozen_counts"]))
    checks.append(_check(
        "v14_reference_coverage_accounting",
        v14_coverage["all_cells"]["assigned_cells"] + v14_coverage["all_cells"]["unassigned_cells"] == v14_coverage["all_cells"]["cells"]
        and v14_coverage["complete_cases"]["assigned_cells"] + v14_coverage["complete_cases"]["unassigned_cells"] == v14_coverage["complete_cases"]["cells"],
        {
            "all_cells": v14_coverage["all_cells"],
            "complete_cases": v14_coverage["complete_cases"],
        },
    ))

    passed = all(item["status"] == "pass" for item in checks)
    return {
        "status": "pass" if passed else "fail",
        "checks": checks,
        "domain_area_km2": {row.domain_id: float(row.area_km2) for row in domains.itertuples()},
        "domain_pixel_counts": class_stats["class_pixels"],
        "domain_component_counts": class_stats["class_components"],
        "total_domain_area_km2": float(domains.geometry.area.sum() / 1_000_000),
        "boundary_length_km": {
            "internal": float(boundaries.loc[boundaries.boundary_type == "internal", "length_km"].sum()),
            "external": float(boundaries.loc[boundaries.boundary_type == "external", "length_km"].sum()),
        },
        "structural_total_length_km": float(structural.geometry.length.sum() / 1000.0),
        "v14_reference_coverage": v14_coverage,
        "valid": passed,
    }


def _verify_output_records(records: list[dict[str, Any]]) -> None:
    failures = []
    for record in records:
        path = resolve_path(record["path"])
        if not path.exists():
            failures.append({"path": record["path"], "error": "missing"})
            continue
        digest = sha256_file(path)
        if digest != record["sha256"]:
            failures.append({"path": record["path"], "error": "sha256 mismatch", "expected": record["sha256"], "observed": digest})
    if failures:
        raise RuntimeError(f"Post-build output verification failed: {failures}")


def _move_outputs(stage: Path, final_paths: dict[str, Path], overwrite: bool) -> None:
    expected_names = {path.name for path in final_paths.values()}
    staged_names = {path.name for path in stage.iterdir()}
    missing = sorted(expected_names - staged_names)
    if missing:
        raise FileNotFoundError(f"Staged outputs are missing: {missing}")

    sidecar_names = {
        name
        for name in staged_names - expected_names
        if name.endswith(".aux.xml") and name[: -len(".aux.xml")] in expected_names
    }
    unexpected = sorted(staged_names - expected_names - sidecar_names)
    if unexpected:
        raise RuntimeError(f"Undeclared staged files: {unexpected}")

    moves: list[tuple[Path, Path]] = []
    for target in final_paths.values():
        moves.append((stage / target.name, target))
        sidecar_name = f"{target.name}.aux.xml"
        if sidecar_name in sidecar_names:
            moves.append((stage / sidecar_name, target.with_name(sidecar_name)))

    for _, target in moves:
        if target.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {target}")

    for staged, target in moves:
        if target.exists():
            target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged, target)


def build(config_path: str | Path, overwrite: bool = False) -> dict[str, Any]:
    gdal.UseExceptions()
    config = load_config(config_path)
    build_id = _utc_stamp()
    targets = _target_paths(config)
    for key, path in targets.items():
        if key != "manifest_json" and path.exists() and not overwrite:
            raise FileExistsError(f"Output exists; use --overwrite only after review: {path}")

    source_tiff = resolve_path(config["daly"]["source_raster"])
    source_structure = resolve_path(config["structure"]["source_gpkg"])
    for source in (source_tiff, source_structure):
        if not source.exists():
            raise FileNotFoundError(source)

    outdir = resolve_path(config["outputs"]["directory"])
    outdir.mkdir(parents=True, exist_ok=True)
    stage = outdir / f"_staging_{build_id}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    final_paths = {path.name: path for path in targets.values()}

    try:
        rgb, geotransform, projection, raster_metadata = read_rgb_tiff(source_tiff)
        assigned, class_stats = classify_figure14(rgb, config["daly"])
        diagnostic_stage = stage / "daly_domain_reconstruction.png"
        write_diagnostic(rgb, assigned, config["daly"], diagnostic_stage)

        raster_stage = stage / config["outputs"]["domain_raster"]
        raster_info = write_categorical_raster(
            assigned,
            geotransform,
            projection,
            raster_stage,
            target_crs=config["target_crs"],
            resolution_m=int(config["daly"]["pixel_parameters"]["output_resolution_m"]),
        )
        domains = polygonize_domains(raster_stage, config["daly"])
        _write_vector(domains, stage / config["outputs"]["domain_gpkg"], config["outputs"]["domain_layer"])

        boundaries, boundary_stats = derive_boundaries(domains, config["daly"])
        _write_vector(boundaries, stage / config["outputs"]["boundary_gpkg"], config["outputs"]["boundary_layer"])

        structural, structural_stats = build_structural_baseline(source_structure, config["structure"])
        _write_vector(structural, stage / config["outputs"]["structural_gpkg"], config["outputs"]["structural_layer"])

        crosswalk_rows = []
        for row in domains.itertuples():
            crosswalk_rows.append(
                {
                    "domain_id": row.domain_id,
                    "domain_number": row.domain_number,
                    "is_subdomain": row.is_subdomain,
                    "abbreviation": row.abbreviation,
                    "domain_name": row.domain_name,
                    "parent_domain_id": row.parent_domain_id or "",
                    "figure_legend_text": row.figure_legend_text,
                    "class_value": row.class_value,
                    "class_pixels": class_stats["class_pixels"][row.domain_id],
                    "class_components": class_stats["class_components"][row.domain_id],
                    "area_km2": round(float(row.area_km2), 3),
                    "source_doi": row.source_doi,
                    "source_figure": row.source_figure,
                    "geometry_method": row.geometry_method,
                    "geometry_confidence": row.geometry_confidence,
                    "notes": row.notes,
                }
            )
        write_csv(stage / config["outputs"]["crosswalk_csv"], crosswalk_rows)

        named_structures = named_structure_reference()
        write_csv(stage / config["outputs"]["named_structures_csv"], named_structures)
        write_json(stage / config["outputs"]["stratigraphy_status_json"], bedrock_stratigraphy_status())
        sources = _source_rows(source_tiff, source_structure, raster_metadata, config)
        write_csv(stage / config["outputs"]["source_registry_csv"], sources)
        coverage_rows, v14_coverage = _v14_coverage(domains, config)
        write_csv(stage / config["outputs"]["v14_coverage_csv"], coverage_rows)

        validation = _validate(
            assigned,
            domains,
            boundaries,
            structural,
            class_stats,
            named_structures,
            v14_coverage,
            config,
        )
        write_json(stage / config["outputs"]["validation_json"], validation)
        if not validation["valid"]:
            raise RuntimeError(f"Geological-foundation validation failed: {validation['checks']}")

        output_records = []
        output_specs = [
            ("domain_gpkg", "tectonic-domain polygons", config["outputs"]["domain_layer"], len(domains), "EPSG:32735"),
            ("boundary_gpkg", "derived domain boundaries", config["outputs"]["boundary_layer"], len(boundaries), "EPSG:32735"),
            ("domain_raster", "categorical domain raster", None, None, "EPSG:32735"),
            ("structural_gpkg", "anonymous Tier 3 structural linework", config["outputs"]["structural_layer"], len(structural), "EPSG:32735"),
            ("crosswalk_csv", "human-readable domain crosswalk", None, None, None),
            ("source_registry_csv", "source provenance registry", None, None, None),
            ("named_structures_csv", "named structures without linked geometry", None, None, None),
            ("v14_coverage_csv", "V14 cell coverage against reconstructed domains", None, None, None),
            ("stratigraphy_status_json", "bedrock/stratigraphy acquisition gap", None, None, None),
            ("validation_json", "build validation", None, None, None),
            ("diagnostic_png", "reconstruction diagnostic", None, None, None),
        ]
        for key, role, layer, features, crs in output_specs:
            record = file_record(stage / targets[key].name, role=role)
            record["path"] = relative_posix(targets[key])
            record.update({"layer": layer, "features": features, "crs": crs})
            output_records.append(record)

        manifest = {
            "schema_version": config["schema_version"],
            "build_id": build_id,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "build_command": f"python -m src.geology_foundation.build --config {Path(config_path).as_posix()}",
            "target_crs": config["target_crs"],
            "software": software_versions(),
            "config": file_record(resolve_path(config_path), role="build configuration"),
            "sources": sources,
            "outputs": output_records,
            "domain_classes": crosswalk_rows,
            "domain_area_km2": validation["domain_area_km2"],
            "class_reconstruction": {
                "method": config["daly"]["classification_method"],
                "parameters": config["daly"]["pixel_parameters"],
                "legend_removal": class_stats["legend_removal"],
                "envelope_pixels": class_stats["envelope_pixels"],
                "envelope_components_retained": class_stats["envelope_components_retained"],
                "class_seed_pixels": class_stats["class_seed_pixels"],
                "source_raster": raster_metadata,
                "output_raster": raster_info,
            },
            "structural_baseline": {
                **structural_stats,
                "source": config["structure"]["source_reference"],
                "tier": config["structure"]["tier"],
                "attribution_status": config["structure"]["attribution_status"],
                "allowed_use": config["structure"]["allowed_use"],
                "prohibited_use": config["structure"]["prohibited_use"],
            },
            "boundary_derivation": boundary_stats,
            "v14_reference_coverage": v14_coverage,
            "data_gaps": {
                "bedrock_stratigraphy": bedrock_stratigraphy_status(),
                "named_structure_geometry": "Names are recorded separately. No name is linked to any of the 126 anonymous lines.",
            },
            "validation": {
                "status": validation["status"],
                "checks": validation["checks"],
                "validation_report": config["outputs"]["validation_json"],
            },
            "interpretation_limits": [
                "The domain polygons are a map-derived reconstruction of a published figure at about 600 m ground resolution, not a surveyed or author-supplied GIS layer.",
                "The map envelope is closed to remove cartographic linework and labels; small blank cartographic spaces are therefore not preserved as holes.",
                "Domain boundaries are changes of published figure colour class, not mapped faults or stratigraphic contacts.",
                "The 126 structural lines are anonymous Tier 3 linework. They cannot support named-fault, hierarchy, displacement or mineralisation claims.",
                "No bedrock or stratigraphic predictor is present because no validated cross-border source was available.",
            ],
        }
        write_json(stage / config["outputs"]["manifest_json"], manifest)
        _move_outputs(stage, final_paths, overwrite)
        _verify_output_records(output_records)
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)

    print(f"Build complete: {build_id}")
    print(f"Domain raster: {targets['domain_raster']}")
    print(f"Domain polygons: {targets['domain_gpkg']}")
    print(f"Structural baseline: {targets['structural_gpkg']}")
    print(f"Manifest: {targets['manifest_json']}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/geology_foundation.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    build(args.config, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
