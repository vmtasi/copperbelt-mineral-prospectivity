"""Rebuild the internal-contact distance from cell centroids (V14).

Distance definition
-------------------
For every canonical grid-cell centroid in EPSG:32735, compute the minimum
Euclidean distance to the actual internal lithological-contact line geometry.
The ``internal_contacts_only`` layer is used so that outer map-unit boundaries
are excluded. No representative contact point, cell-polygon distance, contact
centroid, or lithology-class proxy is used to calculate the distance.

The V13 polygon-to-contact artifacts are read-only inputs. This script writes a
new V14 centroid-distance CSV/GeoPackage and a V14 training table whose only
changed modelling column is ``distance_to_lithology_contact``.

Example
-------
    python src/rebuild_litho_contact_centroid_features.py \
        --contacts "C:/Project/data for projects/internal_contacts_only.gpkg" \
        --out-csv data/derived/litho_contact_centroid_distance_v14.csv \
        --out-gpkg data/derived/litho_contact_centroid_distance_v14.gpkg \
        --out-training data/copperbelt_training_v14_centroid_contact.csv \
        --manifest data/derived/canonical_training_manifest_v14_centroid_contact.json
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
import pyogrio
from shapely import STRtree
from shapely.geometry import Point
from shapely.ops import nearest_points

CRS_AUTHID = "EPSG:32735"
DISTANCE_COLUMN = "distance_to_lithology_contact"
TIE_TOLERANCE_M = 1e-9
BASE_TRAINING = ROOT / "data" / "copperbelt_training_v13_litho_contact.csv"
V13_DISTANCE = ROOT / "data" / "derived" / "litho_contact_distance_v13.csv"
DEFAULT_CONTACTS = Path(r"C:\Project\data for projects\internal_contacts_only.gpkg")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_contacts(path: Path) -> tuple[gpd.GeoDataFrame, dict]:
    require(path.is_file(), f"Internal-contact source is unavailable: {path}")
    layers = pyogrio.list_layers(path)
    require(len(layers) == 1, f"Expected one contact layer in {path}, found {layers.tolist()}")
    layer = str(layers[0, 0])
    frame = gpd.read_file(path, layer=layer, fid_as_index=True).reset_index()
    require("fid" in frame.columns, "Contact source feature IDs were not preserved")
    require("litho_class" in frame.columns, "Contact source lacks litho_class")
    require(frame.crs is not None and frame.crs.to_epsg() == 32735,
            f"Contact CRS is {frame.crs}, expected EPSG:32735")
    require(len(frame) > 0, "Contact source contains no features")
    require(frame.geometry.notna().all(), "Contact source contains null geometries")
    require((~frame.geometry.is_empty).all(), "Contact source contains empty geometries")
    invalid = int((~frame.geometry.is_valid).sum())
    require(invalid == 0, f"Contact source contains {invalid} invalid geometries")
    geometry_types = sorted(frame.geometry.geom_type.unique().tolist())
    require(set(geometry_types).issubset({"LineString", "MultiLineString"}),
            f"Unexpected contact geometry types: {geometry_types}")
    require(frame["fid"].is_unique, "Contact source feature IDs are not unique")
    require(frame["litho_class"].notna().all(), "Contact source contains null litho_class values")

    metadata = {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "layer": layer,
        "feature_count": int(len(frame)),
        "geometry_types": geometry_types,
        "crs": frame.crs.to_string(),
        "crs_epsg": int(frame.crs.to_epsg()),
        "fields": [column for column in frame.columns if column != "geometry"],
        "bounds_m": [float(value) for value in frame.total_bounds],
        "null_geometry_count": 0,
        "empty_geometry_count": 0,
        "invalid_geometry_count": 0,
        "total_line_length_m": float(frame.geometry.length.sum()),
        "litho_class_counts": {
            str(int(key)): int(value)
            for key, value in frame["litho_class"].value_counts().sort_index().items()
        },
    }
    return frame, metadata


def perimeter_evidence(contacts: gpd.GeoDataFrame, contacts_path: Path) -> dict:
    """Check that the selected contacts do not contain the outer boundary."""
    perimeter_path = contacts_path.parent / "outer_boundary_line.gpkg"
    all_lines_path = contacts_path.parent / "all_lithology_lines.gpkg"
    evidence = {
        "perimeter_path": str(perimeter_path),
        "perimeter_available": perimeter_path.is_file(),
        "contacts_perimeter_intersection_length_m": None,
        "contacts_length_m": float(contacts.geometry.length.sum()),
        "perimeter_length_m": None,
        "all_lines_length_m": None,
        "length_partition_difference_m": None,
        "status": "NOT_CHECKED",
    }
    if not perimeter_path.is_file():
        return evidence

    perimeter = gpd.read_file(perimeter_path)
    require(perimeter.crs is not None and perimeter.crs.to_epsg() == 32735,
            "Outer-boundary CRS is not EPSG:32735")
    perimeter_union = perimeter.geometry.union_all()
    intersection_length = float(
        contacts.geometry.intersection(perimeter_union).length.sum()
    )
    contacts_length = float(contacts.geometry.length.sum())
    perimeter_length = float(perimeter_union.length)
    evidence.update(
        {
            "perimeter_sha256": sha256(perimeter_path),
            "perimeter_feature_count": int(len(perimeter)),
            "contacts_perimeter_intersection_length_m": intersection_length,
            "perimeter_length_m": perimeter_length,
        }
    )
    if all_lines_path.is_file():
        all_lines = gpd.read_file(all_lines_path)
        all_length = float(all_lines.geometry.length.sum())
        difference = abs(contacts_length + perimeter_length - all_length)
        evidence.update(
            {
                "all_lines_path": str(all_lines_path),
                "all_lines_sha256": sha256(all_lines_path),
                "all_lines_length_m": all_length,
                "length_partition_difference_m": difference,
            }
        )
        require(all_lines.crs is not None and all_lines.crs.to_epsg() == 32735,
                "All-lithology-lines CRS is not EPSG:32735")
    require(intersection_length <= 1e-6,
            "Internal-contact source overlaps the outer boundary")
    if evidence["length_partition_difference_m"] is not None:
        require(evidence["length_partition_difference_m"] <= 1e-3,
                "Internal contacts plus outer boundary do not partition all lithology lines")
    evidence["status"] = "PASS"
    return evidence


def nearest_contact(point: Point, contacts: gpd.GeoDataFrame, tree: STRtree) -> dict:
    positions, distances = tree.query_nearest(
        point, all_matches=True, return_distance=True
    )
    positions = np.atleast_1d(positions).astype(int)
    distances = np.atleast_1d(distances).astype(float)
    require(len(positions) > 0, "No nearest contact returned")
    minimum = float(distances.min())
    positions = positions[distances <= minimum + TIE_TOLERANCE_M]
    candidates = contacts.iloc[positions].copy()
    candidates["distance_m"] = candidates.geometry.map(point.distance)
    candidates = candidates[candidates["distance_m"] <= minimum + TIE_TOLERANCE_M]
    candidates = candidates.sort_values(["litho_class", "fid"], kind="stable")
    require(not candidates.empty, "Nearest-contact tie selection is empty")
    representative = candidates.iloc[0]
    point_on_contact = nearest_points(point, representative.geometry)[1]
    return {
        "distance_to_lithology_contact": float(minimum),
        "representative_contact_fid": int(representative["fid"]),
        "tie_count": int(len(candidates)),
        "tied_contact_fids": ";".join(str(int(value)) for value in candidates["fid"]),
        "tied_contact_classes": ";".join(
            str(int(value)) for value in candidates["litho_class"]
        ),
        "representative_x": float(point_on_contact.x),
        "representative_y": float(point_on_contact.y),
        "feature_x": float(point.x),
        "feature_y": float(point.y),
        "n_contacts_evaluated": int(len(contacts)),
    }


def comparison_with_v13(distance: pd.DataFrame, path: Path) -> dict:
    if not path.is_file():
        return {"status": "NOT_AVAILABLE"}
    v13 = pd.read_csv(path, usecols=["id", DISTANCE_COLUMN])
    merged = distance[["id", DISTANCE_COLUMN]].merge(
        v13, on="id", how="inner", suffixes=("_v14", "_v13"), validate="one_to_one"
    )
    delta = merged[f"{DISTANCE_COLUMN}_v14"] - merged[f"{DISTANCE_COLUMN}_v13"]
    quantiles = delta.quantile([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    return {
        "status": "PASS" if len(merged) == len(distance) else "ROW_MISMATCH",
        "path": str(path),
        "sha256": sha256(path),
        "matched_rows": int(len(merged)),
        "v13_zero_distance_cells": int(
            (merged[f"{DISTANCE_COLUMN}_v13"] == 0.0).sum()
        ),
        "v14_zero_distance_cells": int(
            (merged[f"{DISTANCE_COLUMN}_v14"] == 0.0).sum()
        ),
        "changed_over_1e-9_m": int((delta.abs() > 1e-9).sum()),
        "minimum_change_m": float(delta.min()),
        "maximum_change_m": float(delta.max()),
        "mean_change_m": float(delta.mean()),
        "change_quantiles_m": {
            str(float(key)): float(value) for key, value in quantiles.items()
        },
    }


def build(args: argparse.Namespace) -> dict:
    contacts, source_metadata = load_contacts(Path(args.contacts))
    perimeter = perimeter_evidence(contacts, Path(args.contacts))

    base = pd.read_csv(args.base_training, encoding="utf-8-sig")
    required = {"id", "centroid_x", "centroid_y", DISTANCE_COLUMN}
    require(required.issubset(base.columns),
            f"Base training table lacks {sorted(required - set(base.columns))}")
    require(len(base) == 1899, f"Expected 1899 base rows, found {len(base)}")
    require(base["id"].notna().all(), "Base training table has missing IDs")
    require(base["id"].is_unique, "Base training table has duplicate IDs")
    require(base[["centroid_x", "centroid_y"]].notna().all().all(),
            "Base training table has missing centroid coordinates")
    require(np.isfinite(base[["centroid_x", "centroid_y"]].to_numpy()).all(),
            "Base training table has non-finite centroid coordinates")

    tree = STRtree(contacts.geometry.values)
    rows = []
    for record in base[["id", "centroid_x", "centroid_y"]].itertuples(index=False):
        point = Point(float(record.centroid_x), float(record.centroid_y))
        row = {"id": int(record.id)}
        row.update(nearest_contact(point, contacts, tree))
        row["centroid_x"] = float(record.centroid_x)
        row["centroid_y"] = float(record.centroid_y)
        rows.append(row)

    distance = pd.DataFrame(rows).sort_values("id", kind="stable").reset_index(drop=True)
    columns = [
        "id",
        DISTANCE_COLUMN,
        "representative_contact_fid",
        "tie_count",
        "tied_contact_fids",
        "tied_contact_classes",
        "representative_x",
        "representative_y",
        "feature_x",
        "feature_y",
        "centroid_x",
        "centroid_y",
        "n_contacts_evaluated",
    ]
    distance = distance[columns]
    values = distance[DISTANCE_COLUMN].to_numpy(dtype=float)
    require(len(distance) == len(base), "Distance output row count changed")
    require(distance["id"].is_unique, "Distance output contains duplicate IDs")
    require(set(distance["id"]) == set(base["id"]), "Distance output ID set changed")
    require(np.isfinite(values).all(), "Distance output contains non-finite values")
    require((values >= 0.0).all(), "Distance output contains negative values")

    training = base.copy()
    lookup = distance.set_index("id")[DISTANCE_COLUMN]
    training[DISTANCE_COLUMN] = training["id"].map(lookup)
    require(training[DISTANCE_COLUMN].notna().all(), "Training distance merge failed")
    require(np.isfinite(training[DISTANCE_COLUMN].to_numpy()).all(),
            "Training table contains non-finite distances")
    require(training["id"].equals(base["id"]), "Training-table row order changed")
    unchanged = [column for column in base.columns if column != DISTANCE_COLUMN]
    pd.testing.assert_frame_equal(
        training[unchanged], base[unchanged], check_dtype=False, check_names=True
    )

    comparison = comparison_with_v13(distance, Path(args.v13_distance))
    if comparison["status"] == "PASS":
        require(comparison["minimum_change_m"] >= -1e-9,
                "Centroid distance is smaller than polygon distance")
        require(comparison["v14_zero_distance_cells"] == 0,
                "Unexpected zero centroid-to-contact distances")

    outputs = [Path(args.out_csv), Path(args.out_gpkg), Path(args.out_training), Path(args.manifest)]
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing V14 artifacts; pass --overwrite only "
            f"after review: {existing}"
        )
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)

    distance.to_csv(args.out_csv, index=False, float_format="%.12f")
    gpkg = gpd.GeoDataFrame(
        distance.rename(columns={DISTANCE_COLUMN: "distance_m"})[
            [
                "id",
                "distance_m",
                "representative_contact_fid",
                "tie_count",
                "representative_x",
                "representative_y",
                "feature_x",
                "feature_y",
                "n_contacts_evaluated",
            ]
        ].copy(),
        geometry=gpd.points_from_xy(distance["centroid_x"], distance["centroid_y"]),
        crs=CRS_AUTHID,
    )
    gpkg.to_file(args.out_gpkg, layer="litho_contact_centroid_distance", driver="GPKG")
    training.to_csv(args.out_training, index=False)

    manifest = {
        "manifest_schema_version": 1,
        "dataset_version": "V14_CENTROID_CONTACT",
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "generator": "src/rebuild_litho_contact_centroid_features.py",
        "distance_definition": (
            "Minimum Euclidean distance in metres from each verified grid-cell "
            "centroid (centroid_x, centroid_y) to the nearest actual mapped internal "
            "lithological-contact line geometry, excluding the outer study-area "
            "perimeter."
        ),
        "distance_column": DISTANCE_COLUMN,
        "crs_epsg": 32735,
        "base_training": {
            "path": str(Path(args.base_training)),
            "sha256": sha256(Path(args.base_training)),
            "rows": int(len(base)),
            "unique_ids": int(base["id"].nunique()),
        },
        "source_geometry": source_metadata,
        "perimeter_exclusion_evidence": perimeter,
        "outputs": {
            "distance_csv": {
                "path": str(Path(args.out_csv)),
                "sha256": sha256(Path(args.out_csv)),
                "rows": int(len(distance)),
            },
            "distance_gpkg": {
                "path": str(Path(args.out_gpkg)),
                "sha256": sha256(Path(args.out_gpkg)),
                "rows": int(len(gpkg)),
                "layer": "litho_contact_centroid_distance",
                "geometry_type": "Point",
                "crs_epsg": 32735,
            },
            "training_table": {
                "path": str(Path(args.out_training)),
                "sha256": sha256(Path(args.out_training)),
                "rows": int(len(training)),
                "unique_ids": int(training["id"].nunique()),
            },
        },
        "distance_summary_m": {
            "minimum": float(values.min()),
            "median": float(np.median(values)),
            "mean": float(values.mean()),
            "maximum": float(values.max()),
            "zero_distances": int((values == 0.0).sum()),
            "tied_minimum_cells": int((distance["tie_count"] > 1).sum()),
        },
        "comparison_with_v13_polygon_distance": comparison,
        "invariants": {
            "rows": int(len(distance)),
            "unique_ids": int(distance["id"].nunique()),
            "missing_ids": int(distance["id"].isna().sum()),
            "duplicate_ids": int(distance["id"].duplicated().sum()),
            "nonfinite_distances": int((~np.isfinite(values)).sum()),
            "negative_distances": int((values < 0.0).sum()),
            "non_distance_columns_unchanged": True,
            "labels_domains_classes_folds_unchanged": True,
        },
        "validation": {
            "status": "PENDING_INDEPENDENT_VALIDATION",
            "validator": "src/validate_centroid_contact_pipeline.py",
        },
    }
    Path(args.manifest).write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contacts", default=str(DEFAULT_CONTACTS))
    parser.add_argument("--base-training", default=str(BASE_TRAINING))
    parser.add_argument("--v13-distance", default=str(V13_DISTANCE))
    parser.add_argument(
        "--out-csv",
        default=str(ROOT / "data" / "derived" / "litho_contact_centroid_distance_v14.csv"),
    )
    parser.add_argument(
        "--out-gpkg",
        default=str(ROOT / "data" / "derived" / "litho_contact_centroid_distance_v14.gpkg"),
    )
    parser.add_argument(
        "--out-training",
        default=str(ROOT / "data" / "copperbelt_training_v14_centroid_contact.csv"),
    )
    parser.add_argument(
        "--manifest",
        default=str(
            ROOT
            / "data"
            / "derived"
            / "canonical_training_manifest_v14_centroid_contact.json"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = build(parse_args())
    print(json.dumps(result, indent=2))
