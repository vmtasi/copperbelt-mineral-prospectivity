"""Independently verify the Stage 4 support, occurrence and model table outputs."""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from pathlib import Path
from typing import Any, Callable

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, MultiPolygon, box
from shapely.ops import unary_union
from shapely.strtree import STRtree


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modeling.stage4_common import (  # noqa: E402
    PROJECT_CRS,
    abs_path,
    load_config,
    read_gdf,
    relative_posix,
    sha256_file,
    verify_protected,
    write_json,
)


def record_check(checks: list[dict[str, Any]], name: str, passed: bool, details: Any) -> None:
    checks.append({"name": name, "status": "pass" if passed else "fail", "details": details})


def capture(checks: list[dict[str, Any]], name: str, function: Callable[[], Any]) -> Any:
    try:
        details = function()
        record_check(checks, name, True, details)
        return details
    except Exception as error:  # noqa: BLE001 - verifier must report all failures
        record_check(checks, name, False, {"error": f"{type(error).__name__}: {error}"})
        return None


def polygonal(geometry: Any) -> MultiPolygon:
    candidate = shapely.make_valid(geometry) if not shapely.is_valid(geometry) else geometry
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
        raise ValueError(f"Geometry has no polygonal part: {candidate.geom_type}")
    return MultiPolygon(parts)


def verify_file_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    checked = 0
    mismatches: list[dict[str, Any]] = []
    for record in records:
        path = abs_path(record["path"])
        observed = sha256_file(path)
        checked += 1
        if observed.lower() != str(record["sha256"]).lower():
            mismatches.append({"path": record["path"], "expected": record["sha256"], "observed": observed})
    return {"checked": checked, "mismatches": mismatches}


def recompute_catchments(
    source_support: pd.DataFrame,
    output_catchments: gpd.GeoDataFrame,
    tract_union: Any,
) -> dict[str, Any]:
    points = shapely.points(source_support["x_m"].to_numpy(), source_support["y_m"].to_numpy())
    minx, miny, maxx, maxy = tract_union.bounds
    extent = box(minx - 5000.0, miny - 5000.0, maxx + 5000.0, maxy + 5000.0)
    voronoi = list(shapely.voronoi_polygons(MultiPoint(list(points)), extend_to=extent).geoms)
    tree = STRtree(voronoi)
    assigned: list[int] = []
    for point in points:
        candidates = np.atleast_1d(tree.query(point, predicate="intersects"))
        candidates = [int(value) for value in candidates if shapely.contains(voronoi[int(value)], point)]
        if len(candidates) != 1:
            raise RuntimeError("Independent Voronoi assignment is not one-to-one")
        assigned.append(candidates[0])
    rebuilt = [polygonal(shapely.intersection(voronoi[index], tract_union)) for index in assigned]

    output_by_id = output_catchments.set_index("support_id").reindex(source_support["support_id"]).reset_index()
    if output_by_id.geometry.isna().any():
        raise RuntimeError("An output catchment is missing")
    geometry_area_difference = np.array(
        [
            shapely.symmetric_difference(output_by_id.geometry.iloc[index], rebuilt[index]).area
            for index in range(len(rebuilt))
        ]
    )
    area_rebuilt = np.array([geometry.area / 1e6 for geometry in rebuilt])
    area_output = output_by_id.geometry.area.to_numpy() / 1e6
    area_difference = np.abs(area_output - area_rebuilt)
    output_union = unary_union(list(output_by_id.geometry))
    rebuilt_union = unary_union(rebuilt)
    checks = {
        "n_support": int(len(output_by_id)),
        "n_centres_inside_own_catchment": int(sum(shapely.covers(output_by_id.geometry.iloc[i], points[i]) for i in range(len(points)))),
        "max_area_difference_km2": float(area_difference.max()),
        "total_symmetric_difference_m2": float(geometry_area_difference.sum()),
        "sum_area_km2": float(area_output.sum()),
        "tract_area_km2": float(tract_union.area / 1e6),
        "output_union_area_km2": float(output_union.area / 1e6),
        "rebuilt_union_area_km2": float(rebuilt_union.area / 1e6),
        "output_overlap_area_km2": float(output_by_id.geometry.area.sum() / 1e6 - output_union.area / 1e6),
        "uncovered_area_m2": float(tract_union.difference(output_union).area),
    }
    if checks["n_support"] != len(source_support):
        raise AssertionError("Support count changed")
    if checks["n_centres_inside_own_catchment"] != len(source_support):
        raise AssertionError("A support centre is not inside its catchment")
    if checks["max_area_difference_km2"] > 1e-6:
        raise AssertionError("Catchment areas differ from the independent construction")
    if checks["total_symmetric_difference_m2"] > 1.0:
        raise AssertionError("Catchment geometry differs from the independent construction")
    if abs(checks["sum_area_km2"] - checks["tract_area_km2"]) > 1e-6:
        raise AssertionError("Catchment areas do not sum to the tract area")
    if abs(checks["output_overlap_area_km2"]) > 1e-6:
        raise AssertionError("Catchments overlap")
    if checks["uncovered_area_m2"] > 1.0:
        raise AssertionError("Catchments do not cover the tract")
    return checks


def parse_components(row: pd.Series, fields: list[str]) -> list[str]:
    components: list[str] = []
    for field in fields:
        value = row.get(field)
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text.lower() == "none":
            continue
        components.extend([part.strip() for part in re.split(r"[,;]", text) if part.strip()])
    return list(dict.fromkeys(components))


def recompute_occurrence_assignments(
    model_table: pd.DataFrame,
    occurrences_source: gpd.GeoDataFrame,
    tract_union: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    status_field = config["inputs"]["occurrences"]["site_status_field"]
    tree = cKDTree(model_table[["x_m", "y_m"]].to_numpy())
    x = occurrences_source.geometry.x.to_numpy()
    y = occurrences_source.geometry.y.to_numpy()
    _, nearest = tree.query(np.column_stack([x, y]), k=1)
    in_tract = occurrences_source.geometry.intersects(tract_union).to_numpy()
    support_ids = model_table["support_id"].to_numpy()

    counts = {column: np.zeros(len(model_table), dtype=int) for column in ["n_records", "n_deposits", "n_prospects", "n_groups", "n_expanded_records"]}
    for index, row in occurrences_source.iterrows():
        if not in_tract[index]:
            continue
        support_index = int(nearest[index])
        counts["n_records"][support_index] += 1
        status = row[status_field]
        if status == "deposit":
            counts["n_deposits"][support_index] += 1
            counts["n_expanded_records"][support_index] += 1
        elif status == "prospect":
            counts["n_prospects"][support_index] += 1
            counts["n_expanded_records"][support_index] += 1
        elif status == "group":
            counts["n_groups"][support_index] += 1
            components = parse_components(row, list(config["inputs"]["occurrences"]["group_includes_fields"]))
            if not components:
                raise AssertionError(f"Group has no components: {index}")
            counts["n_expanded_records"][support_index] += len(components)
        else:
            raise AssertionError(f"Unexpected status: {status}")

    comparisons: dict[str, Any] = {}
    for column, values in counts.items():
        observed = model_table[column].to_numpy(dtype=int)
        differences = observed - values
        comparisons[column] = {
            "sum_output": int(observed.sum()),
            "sum_independent": int(values.sum()),
            "max_abs_difference": int(np.abs(differences).max()),
            "n_cell_differences": int(np.count_nonzero(differences)),
        }
        if np.any(differences):
            raise AssertionError(f"Independent occurrence counts differ for {column}")
    comparisons["n_in_tract"] = int(in_tract.sum())
    comparisons["n_outside_tract"] = int((~in_tract).sum())
    comparisons["status_counts_in_tract"] = {
        str(key): int(value) for key, value in occurrences_source.loc[in_tract, status_field].value_counts().items()
    }
    comparisons["n_support_with_any_record"] = int((counts["n_records"] > 0).sum())
    return comparisons


def verify_group_components(
    group_components: pd.DataFrame,
    occurrences_source: gpd.GeoDataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    status_field = config["inputs"]["occurrences"]["site_status_field"]
    components: list[dict[str, Any]] = []
    for index, row in occurrences_source.iterrows():
        if row[status_field] != "group":
            continue
        for component in parse_components(row, list(config["inputs"]["occurrences"]["group_includes_fields"])):
            components.append({"source_fid": int(index), "component_name": component})
    independent = pd.DataFrame(components)
    if len(independent) != len(group_components):
        raise AssertionError(f"Group component count differs: {len(group_components)} vs {len(independent)}")
    independent_counts = independent.groupby("source_fid").size().sort_index()
    output_counts = group_components.groupby("source_fid").size().sort_index()
    if not independent_counts.equals(output_counts):
        raise AssertionError("Group component counts differ by source record")
    component_sets = independent.groupby("source_fid")["component_name"].apply(lambda values: tuple(values))
    output_sets = group_components.groupby("source_fid")["component_name"].apply(lambda values: tuple(values))
    if not component_sets.equals(output_sets):
        raise AssertionError("Group component names differ from the source attributes")
    return {"n_group_components": int(len(group_components)), "n_group_records": int(independent_counts.size)}


def verify_positional_weights(
    weights: pd.DataFrame,
    occurrences: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    sums = weights.groupby("record_id")["probability"].sum()
    occurrence_ids = set(occurrences["record_id"])
    missing = occurrence_ids.difference(sums.index)
    if missing:
        raise AssertionError(f"Records missing positional weights: {sorted(missing)[:5]}")
    maximum_sum_error = float(np.abs(sums - 1.0).max())
    if maximum_sum_error > 1e-9:
        raise AssertionError("Positional weights do not sum to one")
    outside_records = occurrences.loc[~occurrences["in_tract"], "record_id"]
    outside_weights = weights[weights["assignment_type"].eq("outside_tract")]
    outside_sums = outside_weights.groupby("record_id")["probability"].sum().reindex(outside_records, fill_value=0.0)
    if not np.allclose(outside_sums.to_numpy(), 1.0, atol=1e-12, rtol=0.0):
        raise AssertionError("An outside-tract record does not have unit outside probability")
    own = weights[weights["assignment_type"].eq("own_catchment")].set_index("record_id")["probability"]
    occurrence_own = occurrences.set_index("record_id")["positional_own_probability"]
    own_compare = (occurrence_own - own.reindex(occurrence_own.index, fill_value=0.0)).abs()
    if float(own_compare.max()) > 1e-12:
        raise AssertionError("Occurrence own-catchment probabilities disagree with the positional table")
    return {
        "n_records": int(sums.size),
        "n_weight_rows": int(len(weights)),
        "max_probability_sum_error": maximum_sum_error,
        "max_own_probability_error": float(own_compare.max()),
        "n_outside_tract_records": int(len(outside_records)),
        "mean_outside_probability": float(occurrences["positional_outside_probability"].mean()),
    }


def verify_base_features(
    model_table: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    points = shapely.points(model_table["x_m"].to_numpy(), model_table["y_m"].to_numpy())
    contacts = read_gdf(
        config["inputs"]["contact_geometry"]["path"], layer=config["inputs"]["contact_geometry"]["layer"]
    )
    faults = read_gdf(
        config["inputs"]["fault_geometry"]["path"], layer=config["inputs"]["fault_geometry"]["layer"]
    )
    contact_distance = shapely.distance(points[:, None], contacts.geometry.to_numpy()[None, :])
    contact_fids = np.arange(len(contacts), dtype=int)
    contact_classes = contacts["litho_class"].astype(int).to_numpy()
    deterministic_order = np.lexsort((contact_fids, contact_classes))
    nearest_ordered = np.argmin(contact_distance[:, deterministic_order], axis=1)
    nearest_contact = deterministic_order[nearest_ordered]
    fault_distance = shapely.distance(points[:, None], faults.geometry.to_numpy()[None, :])
    nearest_fault = np.argmin(fault_distance, axis=1)

    v14 = pd.read_csv(
        abs_path(config["inputs"]["v14_table"]["path"]),
        usecols=["id", "centroid_x", "centroid_y", "bouguer", "litho_contact_litho_class"],
    )
    gravity_tree = cKDTree(v14[["centroid_x", "centroid_y"]].to_numpy())
    gravity_distance, gravity_index = gravity_tree.query(model_table[["x_m", "y_m"]].to_numpy(), k=1)
    differences = {
        "BASE_DIST_FAULT_KM": np.abs(model_table["BASE_DIST_FAULT_KM"].to_numpy() - fault_distance.min(axis=1) / 1000.0).max(),
        "BASE_DIST_CONTACT_KM": np.abs(model_table["BASE_DIST_CONTACT_KM"].to_numpy() - contact_distance.min(axis=1) / 1000.0).max(),
        "BASE_BOUGUER_INHERIT_DIST_M": np.abs(model_table["BASE_BOUGUER_INHERIT_DIST_M"].to_numpy() - gravity_distance).max(),
        "BASE_BOUGUER_MGAL": np.abs(model_table["BASE_BOUGUER_MGAL"].to_numpy() - v14["bouguer"].to_numpy()[gravity_index]).max(),
        "BASE_CONTACT_CLASS_INHERIT_DIST_M": np.abs(model_table["BASE_CONTACT_CLASS_INHERIT_DIST_M"].to_numpy() - gravity_distance).max(),
    }
    if any(float(value) > 1e-9 for value in differences.values()):
        raise AssertionError(f"Base features do not reproduce: {differences}")
    if not np.array_equal(
        model_table["BASE_CONTACT_CLASS"].to_numpy(dtype=int),
        v14["litho_contact_litho_class"].to_numpy()[gravity_index].astype(int),
    ):
        raise AssertionError("Inherited V14 contact class does not reproduce")
    if not np.array_equal(
        model_table["BASE_CONTACT_CLASS_SOURCE_ID"].to_numpy(dtype=int),
        v14["id"].to_numpy()[gravity_index].astype(int),
    ):
        raise AssertionError("Inherited V14 contact-class source identifiers do not reproduce")
    if not np.array_equal(model_table["BASE_CONTACT_CLASS_REBUILT"].to_numpy(dtype=int), contact_classes[nearest_contact]):
        raise AssertionError("Rebuilt nearest-contact class does not reproduce")
    if not np.array_equal(model_table["BASE_CONTACT_SOURCE_FID"].to_numpy(dtype=int), nearest_contact):
        raise AssertionError("Contact source identifiers do not reproduce")
    if not np.array_equal(model_table["BASE_FAULT_SOURCE_FID"].to_numpy(dtype=int), nearest_fault):
        raise AssertionError("Fault source identifiers do not reproduce")
    if not np.array_equal(model_table["BASE_BOUGUER_SOURCE_ID"].to_numpy(dtype=int), v14["id"].to_numpy()[gravity_index]):
        raise AssertionError("Gravity source identifiers do not reproduce")
    tie_mask = np.abs(contact_distance - contact_distance.min(axis=1)[:, None]) <= 1e-9
    tie_count = tie_mask.sum(axis=1)
    if not np.array_equal(model_table["BASE_CONTACT_TIE_COUNT"].to_numpy(dtype=int), tie_count):
        raise AssertionError("Nearest-contact tie counts do not reproduce")
    tied_classes = [
        ";".join(str(value) for value in sorted(set(int(item) for item in contact_classes[tie_mask[row]])))
        for row in range(len(model_table))
    ]
    if model_table["BASE_CONTACT_TIED_CLASSES"].astype(str).tolist() != tied_classes:
        raise AssertionError("Nearest-contact tied classes do not reproduce")
    return {
        "max_differences": {key: float(value) for key, value in differences.items()},
        "contact_class_counts": {str(k): int(v) for k, v in model_table["BASE_CONTACT_CLASS"].value_counts().sort_index().items()},
        "rebuilt_contact_class_counts": {str(k): int(v) for k, v in pd.Series(contact_classes[nearest_contact]).value_counts().sort_index().items()},
        "contact_tie_count_distribution": {str(k): int(v) for k, v in pd.Series(tie_count).value_counts().sort_index().items()},
        "n_contacts": int(len(contacts)),
        "n_faults": int(len(faults)),
        "gravity_inherit_distance_m_quantiles": np.quantile(gravity_distance, [0.0, 0.5, 0.95, 1.0]).tolist(),
    }


def verify_folds(model_table: pd.DataFrame, fold_summary: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    coordinates = model_table[["x_m", "y_m"]].to_numpy(dtype=float)
    covariance = np.cov(coordinates, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    reference = np.asarray(config["folds"]["belt_axis_reference_direction"], dtype=float)
    reference = reference / np.linalg.norm(reference)
    if float(np.dot(axis, reference)) < 0.0:
        axis = -axis
    coordinate = coordinates @ axis
    order = np.lexsort((coordinates[:, 0], coordinates[:, 1], coordinate))
    n_folds = int(config["folds"]["n_folds"])
    sizes = np.full(n_folds, len(model_table) // n_folds, dtype=int)
    sizes[: len(model_table) % n_folds] += 1
    expected = np.empty(len(model_table), dtype=int)
    start = 0
    for fold, size in enumerate(sizes, start=1):
        expected[order[start : start + size]] = fold
        start += size
    if not np.array_equal(expected, model_table["new_fold"].to_numpy(dtype=int)):
        raise AssertionError("Primary fold membership does not reproduce")
    buffer_m = float(config["folds"]["primary_buffer_km"]) * 1000.0
    buffer_excluded: dict[str, int] = {}
    for fold in range(1, n_folds + 1):
        heldout = expected == fold
        tree = cKDTree(coordinates[heldout])
        distance, _ = tree.query(coordinates, k=1)
        expected_included = (~heldout) & (distance > buffer_m)
        observed_included = model_table[f"train_included_fold_{fold}"].to_numpy(dtype=bool)
        if not np.array_equal(expected_included, observed_included):
            raise AssertionError(f"Training-buffer inclusion differs for fold {fold}")
        buffer_excluded[str(fold)] = int(((~heldout) & (~expected_included)).sum())
    primary_summary = fold_summary[fold_summary["scheme"].eq("along_belt_buffer_10km")]
    if len(primary_summary) != n_folds:
        raise AssertionError("Primary fold summary does not contain four folds")
    for row in primary_summary.itertuples():
        holdout = model_table["new_fold"].eq(int(row.fold))
        if int(row.n_holdout_support) != int(holdout.sum()):
            raise AssertionError("Fold summary support count differs")
        if int(row.holdout_records) != int(model_table.loc[holdout, "n_records"].sum()):
            raise AssertionError("Fold summary record count differs")
        domain_counts = json.loads(row.holdout_domain_counts)
        if sum(int(value) for value in domain_counts.values()) != int(holdout.sum()):
            raise AssertionError("Fold summary domain counts do not sum to holdout cells")
    if not model_table["v14_fold"].notna().all():
        raise AssertionError("Historical V14 fold crosswalk contains missing values")
    return {
        "fold_sizes": {str(k): int(v) for k, v in pd.Series(expected).value_counts().sort_index().items()},
        "buffer_excluded": buffer_excluded,
        "n_summary_rows": int(len(fold_summary)),
        "historical_fold_counts": {str(int(k)): int(v) for k, v in model_table["v14_fold"].value_counts().sort_index().items()},
    }


def verify_png(path: str | Path) -> dict[str, Any]:
    data = abs_path(path).read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise AssertionError(f"Not a PNG: {path}")
    width, height = struct.unpack(">II", data[16:24])
    if width < 500 or height < 500 or len(data) < 50_000:
        raise AssertionError(f"PNG is unexpectedly small: {path}")
    return {"path": relative_posix(path), "width": int(width), "height": int(height), "bytes": len(data)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage4_v1.json")
    args = parser.parse_args()
    config = load_config(args.config)
    processed = abs_path(config["outputs"]["processed_dir"])
    manifest_path = processed / config["outputs"]["data_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    def protected_check() -> dict[str, Any]:
        return verify_protected(config)

    capture(checks, "protected_inputs_unchanged", protected_check)

    def source_hashes() -> dict[str, Any]:
        return verify_file_records(manifest["sources"])

    def output_hashes() -> dict[str, Any]:
        return verify_file_records(list(manifest["outputs"].values()))

    capture(checks, "source_files_match_manifest", source_hashes)
    capture(checks, "stage4_outputs_match_manifest", output_hashes)

    source_support = pd.read_csv(abs_path(config["inputs"]["support_table"]["path"]))
    model_table = pd.read_csv(processed / config["outputs"]["model_table"])
    occurrences = pd.read_csv(processed / config["outputs"]["occurrences_csv"], keep_default_na=False, na_values=[""])
    group_components = pd.read_csv(processed / config["outputs"]["group_components_csv"])
    positional_weights = pd.read_csv(processed / config["outputs"]["positional_weights_csv"], keep_default_na=False, na_values=[""])
    fold_summary = pd.read_csv(processed / config["outputs"]["fold_summary_csv"])
    support_audit = pd.read_csv(processed / config["outputs"]["support_audit_csv"])
    crosstab = pd.read_csv(processed / "stage4_domain_overlap_crosstab_v1.csv")
    support_gpkg = processed / config["outputs"]["support_gpkg"]
    catchments = gpd.read_file(support_gpkg, layer="stage4_support_catchments", engine="pyogrio")
    support_points = gpd.read_file(support_gpkg, layer="stage4_support_points", engine="pyogrio")

    tracts = read_gdf(config["inputs"]["tracts"]["path"])
    tract_union = unary_union(list(tracts.geometry))
    occurrences_source = read_gdf(config["inputs"]["occurrences"]["path"])

    def table_check() -> dict[str, Any]:
        required = [
            "support_id", "x_m", "y_m", "tract_area_km2", "log_tract_area_km2", "new_fold",
            "BASE_DIST_FAULT_KM", "BASE_DIST_CONTACT_KM", "BASE_CONTACT_CLASS", "BASE_BOUGUER_MGAL",
            "STR_DIST_NAMED_HM_KM", "STR_NAMED_LINE_DENSITY_50KM", "n_records", "n_expanded_records",
        ]
        missing = [column for column in required if column not in model_table.columns]
        if missing:
            raise AssertionError(f"Missing required columns: {missing}")
        if len(model_table) != int(config["support"]["expected_n"]):
            raise AssertionError("Model table has the wrong number of rows")
        if not model_table["support_id"].is_unique:
            raise AssertionError("Support identifiers are not unique")
        if model_table["support_id"].tolist() != source_support["support_id"].tolist():
            raise AssertionError("Support order or identifiers changed from Stage 3")
        if model_table[required].isna().any().any():
            raise AssertionError("Required model-table fields contain missing values")
        if not np.allclose(model_table["log_tract_area_km2"], np.log(model_table["tract_area_km2"]), atol=1e-10):
            raise AssertionError("Exposure offset is not log(tract_area_km2)")
        if int((model_table["domain_abbreviation"] == "unassigned").sum()) != int(config["support"]["expected_n_outside_daly"]):
            raise AssertionError("Outside-Daly cell count changed")
        if int((model_table["nominal_hex_overlap_fraction"] >= 0.8).sum()) != int(config["support"]["expected_n_overlap_ge_80pct"]):
            raise AssertionError("Nominal 80% overlap count changed")
        if int(model_table["full_support"].sum()) != int(config["support"]["expected_n_full_support"]):
            raise AssertionError("Nominal 99% overlap count changed")
        return {
            "n_rows": int(len(model_table)),
            "n_columns": int(len(model_table.columns)),
            "outside_daly": int((model_table["domain_abbreviation"] == "unassigned").sum()),
            "nominal_overlap_ge_80pct": int((model_table["nominal_hex_overlap_fraction"] >= 0.8).sum()),
            "full_support": int(model_table["full_support"].sum()),
        }

    capture(checks, "model_table_schema_and_support", table_check)

    def point_check() -> dict[str, Any]:
        if len(support_points) != len(model_table):
            raise AssertionError("Support point layer has the wrong count")
        point_xy = np.column_stack([support_points.geometry.x, support_points.geometry.y])
        table_xy = model_table[["x_m", "y_m"]].to_numpy()
        maximum = float(np.abs(point_xy - table_xy).max())
        if maximum > 1e-6:
            raise AssertionError("Support point coordinates differ from the table")
        return {"n_points": int(len(support_points)), "max_coordinate_difference_m": maximum}

    capture(checks, "support_point_coordinates", point_check)
    capture(checks, "catchment_partition_reproduces", lambda: recompute_catchments(source_support, catchments, tract_union))
    capture(checks, "occurrence_counts_reproduce", lambda: recompute_occurrence_assignments(model_table, occurrences_source, tract_union, config))

    def occurrence_inventory_check() -> dict[str, Any]:
        expected = config["inputs"]["occurrences"]["expected_status_counts"]
        observed = occurrences_source["SiteStatus"].value_counts().to_dict()
        if {str(k): int(v) for k, v in observed.items()} != {str(k): int(v) for k, v in expected.items()}:
            raise AssertionError("Source occurrence status counts changed")
        if len(occurrences) != int(config["inputs"]["occurrences"]["expected_features"]):
            raise AssertionError("Occurrence output row count changed")
        if int(occurrences["in_tract"].sum()) != 184:
            raise AssertionError("Unexpected in-tract occurrence count")
        outside_names = set(occurrences.loc[~occurrences["in_tract"], "Name"])
        if outside_names != {"Kanaronga", "Luishia", "Nchanga Nose"}:
            raise AssertionError(f"Outside-tract records changed: {outside_names}")
        if not occurrences["record_id"].is_unique:
            raise AssertionError("Record identifiers are not unique")
        return {
            "n_records": int(len(occurrences)),
            "in_tract": int(occurrences["in_tract"].sum()),
            "outside_tract": int((~occurrences["in_tract"]).sum()),
            "outside_names": sorted(outside_names),
        }

    capture(checks, "occurrence_inventory_preserved", occurrence_inventory_check)
    capture(checks, "group_components_reproduce", lambda: verify_group_components(group_components, occurrences_source, config))
    capture(checks, "positional_probabilities_consistent", lambda: verify_positional_weights(positional_weights, occurrences, config))
    capture(checks, "base_covariates_reproduce", lambda: verify_base_features(model_table, config))
    capture(checks, "folds_and_buffer_reproduce", lambda: verify_folds(model_table, fold_summary, config))

    def audit_check() -> dict[str, Any]:
        if len(support_audit) != len(model_table):
            raise AssertionError("Support audit row count differs")
        if support_audit["support_id"].tolist() != model_table["support_id"].tolist():
            raise AssertionError("Support audit order differs")
        all_row = crosstab[crosstab["domain_abbreviation"].eq("ALL") & crosstab["area_overlap_class"].eq("ALL")]
        if len(all_row) != 1:
            raise AssertionError("Cross-tabulation lacks an ALL row")
        row = all_row.iloc[0]
        if int(row["n_support"]) != len(model_table):
            raise AssertionError("Cross-tabulation support total differs")
        if int(row["n_records"]) != int(model_table["n_records"].sum()):
            raise AssertionError("Cross-tabulation record total differs")
        if abs(float(row["catchment_area_km2"]) - float(model_table["tract_area_km2"].sum())) > 1e-6:
            raise AssertionError("Cross-tabulation area total differs")
        return {
            "n_audit_rows": int(len(support_audit)),
            "crosstab_all_n_support": int(row["n_support"]),
            "crosstab_all_n_records": int(row["n_records"]),
        }

    capture(checks, "audit_and_crosstab_totals", audit_check)

    def figure_check() -> list[dict[str, Any]]:
        figure_dir = abs_path(config["outputs"]["figure_dir"])
        return [
            verify_png(figure_dir / config["outputs"]["figures"]["support_folds"]),
            verify_png(figure_dir / config["outputs"]["figures"]["occurrence_counts"]),
        ]

    capture(checks, "data_figures_readable", figure_check)

    passed = all(check["status"] == "pass" for check in checks)
    validation = {
        "schema_version": "1.0.0",
        "build_id": manifest["build_id"],
        "status": "pass" if passed else "fail",
        "checks": checks,
        "n_checks": len(checks),
        "n_failed": int(sum(check["status"] == "fail" for check in checks)),
        "verified_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "notes": [
            "Catchments, counts, group expansion, covariates and folds were recomputed from source files.",
            "Positional probabilities were checked for record-level probability conservation.",
            "Model fitting is outside the scope of this data verifier.",
        ],
    }
    validation_path = processed / config["outputs"]["data_validation"]
    write_json(validation, validation_path)
    if passed:
        manifest["status"] = "verified"
        manifest["validation"] = {
            "status": "pass",
            "path": relative_posix(validation_path),
            "sha256": sha256_file(validation_path),
        }
        write_json(manifest, manifest_path)

    print(json.dumps({
        "status": validation["status"],
        "n_checks": validation["n_checks"],
        "n_failed": validation["n_failed"],
        "failed": [check for check in checks if check["status"] == "fail"],
        "validation": relative_posix(validation_path),
    }, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
