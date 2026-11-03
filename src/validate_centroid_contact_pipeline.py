"""Independent validation of the V14 centroid-to-internal-contact feature.

The validator checks V14 against the preserved V13 base table and the actual
contact geometry. It does not reuse the distance-generation helper: all
centroid-to-line distances and tie diagnostics are recomputed from geometry.

Run:

    python src/validate_centroid_contact_pipeline.py \
        --report figures/audit/centroid_contact_pipeline_validation_2026-10-20.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.ops import nearest_points

CRS_EPSG = 32735
DISTANCE_COLUMN = "distance_to_lithology_contact"
TIE_TOLERANCE_M = 1e-9
V13_TRAINING = ROOT / "data" / "copperbelt_training_v13_litho_contact.csv"
V13_DISTANCE = ROOT / "data" / "derived" / "litho_contact_distance_v13.csv"
V14_TRAINING = ROOT / "data" / "copperbelt_training_v14_centroid_contact.csv"
V14_DISTANCE_CSV = ROOT / "data" / "derived" / "litho_contact_centroid_distance_v14.csv"
V14_DISTANCE_GPKG = ROOT / "data" / "derived" / "litho_contact_centroid_distance_v14.gpkg"
V14_MANIFEST = ROOT / "data" / "derived" / "canonical_training_manifest_v14_centroid_contact.json"
MODEL_COLUMNS = [
    "centroid_x",
    "centroid_y",
    "litho_contact_litho_class",
    "domain",
    DISTANCE_COLUMN,
    "distance_to_fault",
    "bouguer",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def domain_code(value: object) -> str:
    text = str(value).lower()
    if "3a" in text:
        return "NRB_3a"
    if "3b" in text:
        return "NRB_3b"
    if "crz" in text:
        return "CRZ"
    if "srb" in text:
        return "SRB"
    if "nkb" in text:
        return "NKB"
    if "mmsb" in text:
        return "MMSB"
    return "Unknown"


def model_frame(table: pd.DataFrame) -> pd.DataFrame:
    frame = table.dropna(
        subset=["id", "centroid_x", "centroid_y", "domain", "litho_contact_litho_class"]
        + ["distance_to_fault", DISTANCE_COLUMN, "bouguer"]
    ).copy()
    frame["daly_domain"] = frame["domain"].map(domain_code)
    frame = frame[frame["daly_domain"] != "Unknown"].copy().reset_index(drop=True)
    return frame


def verify_perimeter(source_path: Path) -> dict:
    contacts = gpd.read_file(source_path, fid_as_index=True).reset_index()
    perimeter_path = source_path.parent / "outer_boundary_line.gpkg"
    all_lines_path = source_path.parent / "all_lithology_lines.gpkg"
    require(perimeter_path.is_file(), f"Outer-boundary layer unavailable: {perimeter_path}")
    require(all_lines_path.is_file(), f"All-lines layer unavailable: {all_lines_path}")
    perimeter = gpd.read_file(perimeter_path)
    all_lines = gpd.read_file(all_lines_path)
    for label, frame in (("contacts", contacts), ("perimeter", perimeter), ("all_lines", all_lines)):
        require(frame.crs is not None and frame.crs.to_epsg() == CRS_EPSG,
                f"{label} CRS is not EPSG:{CRS_EPSG}")
        require(frame.geometry.notna().all(), f"{label} geometry contains nulls")
        require((~frame.geometry.is_empty).all(), f"{label} geometry contains empties")
        require(frame.geometry.is_valid.all(), f"{label} geometry contains invalids")
    perimeter_union = perimeter.geometry.union_all()
    contacts_length = float(contacts.geometry.length.sum())
    perimeter_length = float(perimeter_union.length)
    all_length = float(all_lines.geometry.length.sum())
    intersection_length = float(contacts.geometry.intersection(perimeter_union).length.sum())
    partition_difference = abs(contacts_length + perimeter_length - all_length)
    require(intersection_length <= 1e-6,
            f"Internal contacts overlap outer boundary by {intersection_length} m")
    require(partition_difference <= 1e-3,
            f"Internal/perimeter length partition differs by {partition_difference} m")
    return {
        "contacts_features": int(len(contacts)),
        "contacts_length_m": contacts_length,
        "perimeter_features": int(len(perimeter)),
        "perimeter_length_m": perimeter_length,
        "all_lines_features": int(len(all_lines)),
        "all_lines_length_m": all_length,
        "contacts_perimeter_intersection_length_m": intersection_length,
        "length_partition_difference_m": partition_difference,
    }


def main(report_path: Path) -> None:
    report_path = report_path.resolve()
    audit_dir = (ROOT / "figures" / "audit").resolve()
    require(report_path.parent == audit_dir, "Report must be directly under figures/audit")
    require(report_path.suffix.lower() == ".json", "Report must be JSON")
    require(not report_path.exists(), f"Refusing to overwrite report: {report_path}")
    for path in (V13_TRAINING, V13_DISTANCE, V14_TRAINING, V14_DISTANCE_CSV, V14_DISTANCE_GPKG, V14_MANIFEST):
        require(path.is_file(), f"Required artifact is missing: {path}")

    manifest = json.loads(V14_MANIFEST.read_text(encoding="utf-8"))
    require(manifest.get("dataset_version") == "V14_CENTROID_CONTACT",
            "Unexpected V14 manifest version")
    source_path = Path(manifest["source_geometry"]["path"])
    require(source_path.is_file(), f"Source geometry is missing: {source_path}")
    require(manifest["source_geometry"]["sha256"] == sha256(source_path),
            "Source-geometry hash differs from manifest")
    require(manifest["base_training"]["sha256"] == sha256(V13_TRAINING),
            "V13 base hash differs from manifest")
    require(manifest["outputs"]["distance_csv"]["sha256"] == sha256(V14_DISTANCE_CSV),
            "V14 distance CSV hash differs from manifest")
    require(manifest["outputs"]["distance_gpkg"]["sha256"] == sha256(V14_DISTANCE_GPKG),
            "V14 distance GeoPackage hash differs from manifest")
    require(manifest["outputs"]["training_table"]["sha256"] == sha256(V14_TRAINING),
            "V14 training-table hash differs from manifest")

    base = pd.read_csv(V13_TRAINING, encoding="utf-8-sig")
    training = pd.read_csv(V14_TRAINING)
    distance = pd.read_csv(V14_DISTANCE_CSV)
    gpkg = gpd.read_file(V14_DISTANCE_GPKG)
    contacts = gpd.read_file(source_path, fid_as_index=True).reset_index()

    require(len(base) == 1899, f"V13 base row count changed: {len(base)}")
    require(len(training) == 1899, f"V14 training row count is {len(training)}, expected 1899")
    require(len(distance) == 1899, f"V14 distance row count is {len(distance)}, expected 1899")
    require(training["id"].is_unique and training["id"].notna().all(),
            "V14 training IDs are missing or duplicated")
    require(distance["id"].is_unique and distance["id"].notna().all(),
            "V14 distance IDs are missing or duplicated")
    require(distance["id"].nunique() == 1899, "V14 distance unique-ID count is not 1899")
    require(set(distance["id"]) == set(base["id"]), "V14 distance ID set differs from V13")
    require(training["id"].equals(base["id"]), "V14 training row order/IDs differ from V13")
    distance_ordered = distance.set_index("id").loc[training["id"]].reset_index()
    require(distance_ordered["id"].equals(training["id"]),
            "V14 distance rows cannot be aligned one-to-one to training IDs")

    values = distance_ordered[DISTANCE_COLUMN].to_numpy(dtype=float)
    require(np.isfinite(values).all(), "V14 distances contain non-finite values")
    require((values >= 0.0).all(), "V14 distances contain negative values")
    require(np.isfinite(training[DISTANCE_COLUMN].to_numpy(dtype=float)).all(),
            "V14 training distances contain non-finite values")

    unchanged = [column for column in base.columns if column != DISTANCE_COLUMN]
    pd.testing.assert_frame_equal(
        base[unchanged], training[unchanged], check_dtype=False, check_names=True
    )
    require(training["litho_contact_litho_class"].equals(base["litho_contact_litho_class"]),
            "Host/contact lithology-class field changed")

    training_distance_diff = float(
        np.max(np.abs(training[DISTANCE_COLUMN].to_numpy() - values))
    )
    require(training_distance_diff <= 1e-9,
            f"Training distance differs from distance CSV by {training_distance_diff} m")

    require(gpkg.crs is not None and gpkg.crs.to_epsg() == CRS_EPSG,
            "V14 GeoPackage CRS is not EPSG:32735")
    require(len(gpkg) == 1899, f"V14 GeoPackage row count is {len(gpkg)}")
    require(gpkg.geometry.geom_type.eq("Point").all(), "V14 GeoPackage geometry is not Point")
    require(gpkg.geometry.notna().all() and not gpkg.geometry.is_empty.any(),
            "V14 GeoPackage contains null or empty geometries")
    gpkg_indexed = gpkg.set_index("id").sort_index()
    csv_indexed = distance.set_index("id").sort_index()
    require(gpkg_indexed.index.equals(csv_indexed.index), "CSV/GPKG ID sets differ")
    gpkg_distance_diff = float(
        np.max(np.abs(gpkg_indexed["distance_m"] - csv_indexed[DISTANCE_COLUMN]))
    )
    require(gpkg_distance_diff <= 1e-9,
            f"CSV/GPKG distances differ by {gpkg_distance_diff} m")
    gpkg_centroid_diff = float(
        np.max(
            np.hypot(
                gpkg_indexed.geometry.x - csv_indexed["centroid_x"],
                gpkg_indexed.geometry.y - csv_indexed["centroid_y"],
            )
        )
    )
    require(gpkg_centroid_diff <= 1e-8,
            f"GeoPackage points differ from stored centroids by {gpkg_centroid_diff} m")

    points = shapely.points(training[["centroid_x", "centroid_y"]].to_numpy(dtype=float))
    line_geometries = contacts.geometry.to_numpy()
    distance_matrix = shapely.distance(points[:, None], line_geometries[None, :])
    recomputed_minimum = distance_matrix.min(axis=1)
    maximum_geometry_difference = float(np.max(np.abs(recomputed_minimum - values)))
    require(maximum_geometry_difference <= 1e-8,
            f"Independent geometry distance differs by {maximum_geometry_difference} m")
    candidate_mask = np.isclose(
        distance_matrix, recomputed_minimum[:, None], rtol=0.0, atol=TIE_TOLERANCE_M
    )
    expected_tie_count = candidate_mask.sum(axis=1)
    require(np.array_equal(expected_tie_count, distance_ordered["tie_count"].to_numpy()),
            "Stored tie counts differ from independent geometry")

    class_values = contacts["litho_class"].to_numpy(dtype=int)
    fid_values = contacts["fid"].to_numpy(dtype=int)
    expected_fids: list[str] = []
    expected_classes: list[str] = []
    expected_rep_fids = np.empty(len(distance), dtype=int)
    for index in range(len(distance)):
        candidates = np.flatnonzero(candidate_mask[index])
        ordered = sorted(candidates, key=lambda position: (class_values[position], fid_values[position]))
        expected_fids.append(";".join(str(int(fid_values[position])) for position in ordered))
        expected_classes.append(
            ";".join(str(int(class_values[position])) for position in ordered)
        )
        expected_rep_fids[index] = fid_values[ordered[0]]
    require(expected_fids == distance_ordered["tied_contact_fids"].tolist(),
            "Stored tied-contact IDs differ from independent geometry")
    require(expected_classes == distance_ordered["tied_contact_classes"].tolist(),
            "Stored tied-contact classes differ from independent geometry")
    require(np.array_equal(
        expected_rep_fids, distance_ordered["representative_contact_fid"].to_numpy()
    ),
            "Stored representative contact differs from deterministic tie rule")

    representative_coords = np.empty((len(distance), 2), dtype=float)
    for index in range(len(distance)):
        candidate_position = int(np.flatnonzero(fid_values == expected_rep_fids[index])[0])
        representative = nearest_points(points[index], line_geometries[candidate_position])[1]
        representative_coords[index] = (representative.x, representative.y)
    representative_difference = float(
        np.max(
            np.hypot(
                representative_coords[:, 0] - distance_ordered["representative_x"].to_numpy(),
                representative_coords[:, 1] - distance_ordered["representative_y"].to_numpy(),
            )
        )
    )
    require(representative_difference <= 1e-8,
            f"Representative contact coordinates differ by {representative_difference} m")

    from src.validation_strategies import get_along_belt_folds

    v13_frame = model_frame(base)
    v14_frame = model_frame(training)
    require(len(v14_frame) == 1872, f"V14 complete-case frame has {len(v14_frame)} rows")
    require(int(v14_frame["deposit_present"].sum()) == 138,
            "V14 complete-case positive count is not 138")
    require(int((v14_frame["deposit_present"] == 0).sum()) == 1734,
            "V14 complete-case background count is not 1734")
    require(v13_frame["id"].equals(v14_frame["id"]), "Complete-case ID order changed")
    require(v13_frame["deposit_present"].equals(v14_frame["deposit_present"]),
            "Complete-case labels changed")
    require(v13_frame["daly_domain"].equals(v14_frame["daly_domain"]),
            "Complete-case domain assignments changed")
    v13_folds = get_along_belt_folds(v13_frame, n_folds=4)
    v14_folds = get_along_belt_folds(v14_frame, n_folds=4)
    require(np.array_equal(v13_folds, v14_folds), "Spatial fold membership changed")
    fold_counts = pd.Series(v14_folds).value_counts().sort_index().tolist()
    fold_positives = pd.DataFrame(
        {"fold": v14_folds, "deposit_present": v14_frame["deposit_present"].to_numpy()}
    ).groupby("fold")["deposit_present"].sum().tolist()
    require(fold_counts == [468, 468, 468, 468], f"Unexpected fold sizes: {fold_counts}")
    require(fold_positives == [9, 31, 37, 61],
            f"Unexpected fold positive counts: {fold_positives}")

    v13_distance = pd.read_csv(V13_DISTANCE, usecols=["id", DISTANCE_COLUMN])
    comparison = distance[["id", DISTANCE_COLUMN]].merge(
        v13_distance, on="id", suffixes=("_v14", "_v13"), validate="one_to_one"
    )
    delta = comparison[f"{DISTANCE_COLUMN}_v14"] - comparison[f"{DISTANCE_COLUMN}_v13"]
    require(delta.min() >= -1e-9, "A centroid distance is smaller than its V13 polygon distance")
    require(int((comparison[f"{DISTANCE_COLUMN}_v14"] == 0.0).sum()) == 0,
            "Centroid distance contains zero values")
    require(int((comparison[f"{DISTANCE_COLUMN}_v13"] == 0.0).sum()) == 270,
            "Unexpected V13 zero-distance count")
    comparison_summary = {
        "minimum_change_m": float(delta.min()),
        "mean_change_m": float(delta.mean()),
        "maximum_change_m": float(delta.max()),
        "v13_zero_distance_cells": 270,
        "v14_zero_distance_cells": 0,
        "changed_over_1e-9_m": int((delta.abs() > 1e-9).sum()),
    }
    require(comparison_summary["changed_over_1e-9_m"] == 1899,
            "Not all centroid distances differ from the polygon distances")

    perimeter = verify_perimeter(source_path)
    require(int(contacts.crs.to_epsg()) == CRS_EPSG and len(contacts) == 259,
            "Authoritative contact layer metadata changed")
    require(contacts.geometry.is_valid.all(), "Contact geometry is invalid")

    checks = [
        {"check": "one_row_per_grid_cell", "status": "PASS", "rows": 1899, "unique_ids": 1899},
        {"check": "no_missing_or_duplicate_ids", "status": "PASS"},
        {"check": "csv_gpkg_agree", "status": "PASS", "max_abs_diff_m": gpkg_distance_diff},
        {"check": "crs_and_grid_alignment", "status": "PASS", "crs_epsg": CRS_EPSG,
         "max_centroid_diff_m": gpkg_centroid_diff},
        {"check": "independent_geometry_distance", "status": "PASS",
         "max_abs_diff_m": maximum_geometry_difference},
        {"check": "tie_diagnostics", "status": "PASS",
         "tie_count_range": [int(expected_tie_count.min()), int(expected_tie_count.max())]},
        {"check": "non_distance_columns_unchanged", "status": "PASS"},
        {"check": "complete_case_frame", "status": "PASS", "rows": 1872,
         "positive": 138, "background": 1734},
        {"check": "spatial_folds_unchanged", "status": "PASS",
         "fold_sizes": fold_counts, "fold_positives": fold_positives},
        {"check": "perimeter_excluded", "status": "PASS",
         "intersection_length_m": perimeter["contacts_perimeter_intersection_length_m"]},
        {"check": "distance_positive_change_vs_v13", "status": "PASS",
         "minimum_change_m": comparison_summary["minimum_change_m"]},
    ]
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "validator": "src/validate_centroid_contact_pipeline.py",
        "status": "PASS",
        "dataset_version": "V14_CENTROID_CONTACT",
        "distance_definition": manifest["distance_definition"],
        "source_geometry": {
            "path": str(source_path),
            "sha256": sha256(source_path),
            "features": int(len(contacts)),
            "crs_epsg": int(contacts.crs.to_epsg()),
            "geometry_types": sorted(contacts.geometry.geom_type.unique().tolist()),
        },
        "hashes": {
            "v13_training_sha256": sha256(V13_TRAINING),
            "v13_distance_csv_sha256": sha256(V13_DISTANCE),
            "v14_training_sha256": sha256(V14_TRAINING),
            "v14_distance_csv_sha256": sha256(V14_DISTANCE_CSV),
            "v14_distance_gpkg_sha256": sha256(V14_DISTANCE_GPKG),
        },
        "distance_summary_m": {
            "minimum": float(values.min()),
            "median": float(np.median(values)),
            "mean": float(values.mean()),
            "maximum": float(values.max()),
            "zero_values": int((values == 0.0).sum()),
        },
        "comparison_with_v13": comparison_summary,
        "perimeter_evidence": perimeter,
        "checks": checks,
        "all_checks_passed": True,
    }

    manifest["validation"] = {
        "status": "PASS",
        "validator": "src/validate_centroid_contact_pipeline.py",
        "validated_utc": report["generated_utc"],
        "report_path": report_path.relative_to(ROOT).as_posix(),
        "independent_geometry_max_abs_diff_m": maximum_geometry_difference,
        "csv_gpkg_max_abs_diff_m": gpkg_distance_diff,
        "complete_case_rows": len(v14_frame),
        "complete_case_positives": int(v14_frame["deposit_present"].sum()),
        "fold_sizes": fold_counts,
        "fold_positives": fold_positives,
    }
    V14_MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    report["manifest_sha256_after_validation"] = sha256(V14_MANIFEST)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    main(parser.parse_args().report)
