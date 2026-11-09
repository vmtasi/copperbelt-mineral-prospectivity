"""Independently verify the Stage 5A inventory and support audit.

The verifier recomputes the audit's principal counts from the manifest-verified
Stage 4 inputs and source geometries. It deliberately does not import the
Stage 5A audit functions. Only after every check passes does it mark the Stage
5A manifest as ``verified`` and write a separate validation record.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Callable

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy import stats
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modeling.stage4_common import (  # noqa: E402
    PROJECT_CRS,
    abs_path,
    file_record,
    relative_posix,
    sha256_file,
    write_json,
)
from src.modeling.stage5_common import (  # noqa: E402
    distance_to_boundary_km,
    load_config,
    load_geometries,
    load_stage4_bundle,
    software_stack,
)


def record_check(checks: list[dict[str, Any]], name: str, passed: bool, details: Any) -> None:
    checks.append({"name": name, "status": "pass" if passed else "fail", "details": details})


def capture(checks: list[dict[str, Any]], name: str, function: Callable[[], Any]) -> Any:
    try:
        details = function()
        record_check(checks, name, True, details)
        return details
    except Exception as error:  # noqa: BLE001 - verify every check independently
        record_check(checks, name, False, {"error": f"{type(error).__name__}: {error}"})
        return None


def verify_records(records: dict[str, Any]) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    checked = 0
    for name, record in records.items():
        path = abs_path(record["path"])
        checked += 1
        if not path.exists():
            mismatches.append({"name": name, "path": record["path"], "error": "missing"})
            continue
        observed_hash = sha256_file(path)
        if observed_hash.lower() != str(record["sha256"]).lower():
            mismatches.append({
                "name": name, "path": record["path"], "error": "sha256 mismatch",
                "expected": record["sha256"], "observed": observed_hash,
            })
        if int(record.get("bytes", -1)) != path.stat().st_size:
            mismatches.append({
                "name": name, "path": record["path"], "error": "byte-size mismatch",
                "expected": int(record.get("bytes", -1)), "observed": path.stat().st_size,
            })
    if mismatches:
        raise RuntimeError(f"{len(mismatches)} Stage 5A output records failed verification")
    return {"checked": checked, "mismatches": mismatches}


def verify_png(path: str | Path) -> dict[str, Any]:
    absolute = abs_path(path)
    with absolute.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Not a readable PNG: {absolute}")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid PNG dimensions: {absolute}")
    return {"path": relative_posix(absolute), "width": width, "height": height}


def assert_close(observed: float, expected: float, tolerance: float = 1e-9) -> None:
    if not math.isclose(float(observed), float(expected), rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"Values differ: observed={observed}, expected={expected}")


def verify_stage4_chain(bundle: dict[str, Any]) -> dict[str, Any]:
    if bundle["data_manifest"]["build_id"] != bundle["model_manifest"]["data_build_id"]:
        raise RuntimeError("Stage 4 data and model build identifiers disagree")
    if not all(item["hash_match"] for item in bundle["protected"].values()):
        raise RuntimeError("A protected V14/Stage 3 asset has changed")
    return {
        "data_build_id": bundle["data_manifest"]["build_id"],
        "model_build_id": bundle["model_manifest"]["build_id"],
        "data_validation": bundle["data_validation"]["status"],
        "model_validation": bundle["model_validation"]["status"],
        "protected_assets": len(bundle["protected"]),
        "verified_outputs": len(bundle["verified_outputs"]),
    }


def verify_inventory(
    bundle: dict[str, Any], config: dict[str, Any], geometries: dict[str, Any]
) -> dict[str, Any]:
    occurrences = bundle["occurrences"]
    components = bundle["components"]
    weights = bundle["weights"]
    model_table = bundle["model_table"]
    expected = config["inventory"]

    if len(occurrences) != int(expected["expected_source_records"]):
        raise AssertionError("Source-record count changed")
    if not occurrences["record_id"].is_unique:
        raise AssertionError("Source record identifiers are not unique")
    status_counts = occurrences["SiteStatus"].value_counts().to_dict()
    if {str(key): int(value) for key, value in status_counts.items()} != {
        str(key): int(value) for key, value in expected["expected_status_counts"].items()
    }:
        raise AssertionError(f"Status counts changed: {status_counts}")
    in_tract = occurrences.loc[occurrences["in_tract"].astype(bool)].copy()
    if len(in_tract) != int(expected["expected_in_tract"]):
        raise AssertionError("In-tract record count changed")
    total_expansion = int(occurrences["record_expansion"].sum())
    in_tract_expansion = int(in_tract["record_expansion"].sum())
    if total_expansion != 247 or in_tract_expansion != 244:
        raise AssertionError("Expanded record total differs from the audited total")

    groups = occurrences.loc[occurrences["SiteStatus"].eq("group")].copy()
    if len(groups) != 20:
        raise AssertionError("Group-record count changed")
    if len(components) != 80:
        raise AssertionError("Group-component count changed")
    if int(groups["n_group_components"].sum()) != len(components):
        raise AssertionError("Group multiplicity does not reproduce the component table")
    component_groups = components.groupby("record_id").size()
    if not component_groups.reindex(groups["record_id"]).notna().all():
        raise AssertionError("A group record has no component rows")
    coordinates = groups.set_index("record_id")[["x_m", "y_m"]]
    component_coordinates = components.set_index("record_id")[["x_m", "y_m"]]
    joined = component_coordinates.join(coordinates, how="left", rsuffix="_group")
    if not np.allclose(
        joined[["x_m", "y_m"]].to_numpy(dtype=float),
        joined[["x_m_group", "y_m_group"]].to_numpy(dtype=float),
        atol=1e-9, rtol=0.0,
    ):
        raise AssertionError("A group component was moved away from its group coordinate")

    probability_sums = weights.groupby("record_id", sort=False)["probability"].sum()
    if len(probability_sums) != len(occurrences):
        raise AssertionError("Positional weights do not cover every source record")
    if not np.allclose(probability_sums.to_numpy(dtype=float), 1.0, atol=1e-9, rtol=0.0):
        raise AssertionError("A record's positional probabilities do not sum to one")
    if weights.duplicated(["record_id", "support_id"]).any():
        raise AssertionError("Duplicate record-support positional rows")
    outside_rows = weights.loc[weights["assignment_type"].eq("outside_tract")]
    expected_outside = set(occurrences.loc[~occurrences["in_tract"].astype(bool), "record_id"])
    source_outside_rows = outside_rows.loc[outside_rows["record_id"].isin(expected_outside)]
    if set(source_outside_rows["record_id"]) != expected_outside:
        raise AssertionError("Source-outside records lack an outside-tract probability row")
    if not source_outside_rows["probability"].eq(1.0).all():
        raise AssertionError("A source-outside record is not wholly outside the tract")
    if not source_outside_rows["support_id"].isna().all():
        raise AssertionError("A source-outside record was assigned to a catchment")

    points = gpd.GeoSeries(
        gpd.points_from_xy(occurrences["x_m"], occurrences["y_m"]), crs=PROJECT_CRS
    )
    hexagons = geometries["nominal_hexagons"][["support_id", "geometry"]]
    membership = gpd.GeoDataFrame(
        {"record_id": occurrences["record_id"].to_numpy()},
        geometry=points.to_numpy(), crs=PROJECT_CRS,
    ).sjoin(hexagons, how="left", predicate="intersects")
    matched = membership.dropna(subset=["support_id"]).groupby("record_id").size()
    n_outside_hex = int(len(occurrences) - matched.reindex(occurrences["record_id"]).notna().sum())
    n_outside_hex_in_tract = int(len(in_tract) - matched.reindex(in_tract["record_id"]).notna().sum())
    if n_outside_hex != 47 or n_outside_hex_in_tract != 45:
        raise AssertionError(
            f"Nominal-hex membership changed: outside={n_outside_hex}, "
            f"in_tract_outside={n_outside_hex_in_tract}"
        )

    tract_area = float(geometries["tract_union"].area / 1e6)
    area_sum = float(model_table["tract_area_km2"].sum())
    if not math.isclose(tract_area, area_sum, rel_tol=1e-9, abs_tol=1e-6):
        raise AssertionError("Catchment area does not reproduce the tract area")
    area = model_table["tract_area_km2"].astype(float)
    area_context = {
        "min": float(area.min()), "max": float(area.max()),
        "median": float(area.median()), "q01": float(area.quantile(0.01)),
        "q99": float(area.quantile(0.99)), "sum": area_sum,
    }
    table_area = model_table[["x_m", "y_m"]]
    boundary_distance = distance_to_boundary_km(
        gpd.GeoSeries(gpd.points_from_xy(table_area["x_m"], table_area["y_m"]), crs=PROJECT_CRS),
        geometries["tract_union"],
    )
    area_boundary = stats.spearmanr(area.to_numpy(), boundary_distance)
    area_records = stats.spearmanr(area.to_numpy(), model_table["n_records"].to_numpy())

    return {
        "n_records": int(len(occurrences)),
        "n_in_tract": int(len(in_tract)),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "n_group_components": int(len(components)),
        "n_expanded_records": total_expansion,
        "n_expanded_records_in_tract": in_tract_expansion,
        "outside_tract_record_ids": sorted(expected_outside),
        "n_outside_nominal_hex": n_outside_hex,
        "n_outside_nominal_hex_in_tract": n_outside_hex_in_tract,
        "tract_area_km2": tract_area,
        "area_context": area_context,
        "area_vs_boundary_spearman_r": float(area_boundary.statistic),
        "area_vs_boundary_spearman_p": float(area_boundary.pvalue),
        "area_vs_records_spearman_r": float(area_records.statistic),
        "area_vs_records_spearman_p": float(area_records.pvalue),
        "positional_probability_sums_valid": True,
    }


def verify_summary_tables(
    bundle: dict[str, Any], config: dict[str, Any], geometries: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    model_table = bundle["model_table"]
    audit = pd.read_csv(abs_path(config["outputs"]["processed_dir"]) / config["outputs"]["inventory_record_audit"])
    status = pd.read_csv(abs_path(config["outputs"]["processed_dir"]) / config["outputs"]["inventory_status_summary"])
    fold_domain = pd.read_csv(
        abs_path(config["outputs"]["processed_dir"])
        / config["outputs"]["inventory_fold_domain_summary"]
    )
    fold_domain = fold_domain.loc[~fold_domain["domain_abbreviation"].eq("ALL")].copy()
    support = pd.read_csv(abs_path(config["outputs"]["processed_dir"]) / config["outputs"]["inventory_support_summary"])
    boundary = pd.read_csv(abs_path(config["outputs"]["processed_dir"]) / config["outputs"]["inventory_boundary_summary"])

    if len(audit) != 187 or not audit["record_id"].is_unique:
        raise AssertionError("Record audit does not preserve 187 unique records")
    recomputed_status = audit.groupby("SiteStatus", sort=True).agg(
        n_records=("record_id", "size"), n_in_tract=("in_tract", "sum")
    )
    recorded_status = status.set_index("site_status")
    for name, row in recomputed_status.iterrows():
        if int(recorded_status.loc[name, "n_records"]) != int(row["n_records"]):
            raise AssertionError(f"Status summary count changed for {name}")
        if int(recorded_status.loc[name, "n_in_tract"]) != int(row["n_in_tract"]):
            raise AssertionError(f"Status summary in-tract count changed for {name}")

    expected_fold_domain = model_table.groupby(
        ["new_fold", "new_fold_label", "domain_abbreviation"], sort=True
    ).agg(
        n_support_cells=("support_id", "size"),
        positive_cells=("any_record", "sum"),
        n_records=("n_records", "sum"),
        n_deposits=("n_deposits", "sum"),
        n_prospects=("n_prospects", "sum"),
        n_groups=("n_groups", "sum"),
        n_expanded_records=("n_expanded_records", "sum"),
    ).reset_index()
    comparison = fold_domain.merge(
        expected_fold_domain,
        on=["new_fold", "new_fold_label", "domain_abbreviation"],
        how="outer", suffixes=("_audit", "_recomputed"), validate="one_to_one",
    )
    for column in ("n_support_cells", "positive_cells", "n_records", "n_deposits", "n_prospects", "n_groups", "n_expanded_records"):
        if not np.array_equal(
            comparison[f"{column}_audit"].fillna(-1).to_numpy(dtype=int),
            comparison[f"{column}_recomputed"].fillna(-1).to_numpy(dtype=int),
        ):
            raise AssertionError(f"Fold/domain composition differs for {column}")

    bins = [float(value) for value in config["inventory"]["area_classes_km2"]]
    labels = [f"{bins[i]:g}-{bins[i+1]:g}" for i in range(len(bins) - 1)]
    area_class = pd.cut(
        model_table["tract_area_km2"], bins=bins, labels=labels,
        include_lowest=True, right=False,
    )
    support_all = support.loc[support["grouping"].eq("ALL") & support["group_value"].eq("ALL")].iloc[0]
    if int(support_all["n_support_cells"]) != len(model_table):
        raise AssertionError("Support summary total cell count changed")
    if int(support_all["n_records"]) != int(model_table["n_records"].sum()):
        raise AssertionError("Support summary total record count changed")
    area_rows = support.loc[support["grouping"].eq("area_class_km2")].set_index("group_value")
    for label in labels:
        subset = model_table.loc[area_class.eq(label)]
        if subset.empty:
            continue
        if int(area_rows.loc[label, "n_support_cells"]) != len(subset):
            raise AssertionError(f"Area-bin cell count changed for {label}")
        if int(area_rows.loc[label, "n_records"]) != int(subset["n_records"].sum()):
            raise AssertionError(f"Area-bin record count changed for {label}")

    source_occurrences = bundle["occurrences"]
    source_points = gpd.GeoSeries(
        gpd.points_from_xy(source_occurrences["x_m"], source_occurrences["y_m"]),
        crs=PROJECT_CRS,
    )
    tract_distance = distance_to_boundary_km(source_points, geometries["tract_union"])
    hex_union = unary_union(list(geometries["nominal_hexagons"].geometry))
    hex_boundary = unary_union(list(geometries["nominal_hexagons"].geometry.boundary))
    hex_distance = np.asarray(
        [shapely.distance(point, hex_boundary) / 1000.0 for point in source_points],
        dtype=float,
    )
    threshold_source = boundary.loc[
        boundary["site_status"].eq("ALL") & boundary["boundary_type"].eq("tract_boundary")
    ].set_index("threshold_m")
    for threshold in config["inventory"]["boundary_thresholds_m"]:
        observed = int(np.sum(tract_distance <= float(threshold) / 1000.0))
        if int(threshold_source.loc[int(threshold), "n_within_threshold"]) != observed:
            raise AssertionError(f"Tract-boundary count changed at {threshold} m")
    threshold_hex = boundary.loc[
        boundary["site_status"].eq("ALL") & boundary["boundary_type"].eq("nominal_hex_boundary")
    ].set_index("threshold_m")
    for threshold in config["inventory"]["boundary_thresholds_m"]:
        observed = int(np.sum(hex_distance <= float(threshold) / 1000.0))
        if int(threshold_hex.loc[int(threshold), "n_within_threshold"]) != observed:
            raise AssertionError(f"Nominal-hex boundary count changed at {threshold} m")

    context = manifest["support_context"]
    recomputed = verify_inventory(bundle, config, geometries)
    for key in ("catchment_area_min", "catchment_area_max", "catchment_area_median", "catchment_area_q01", "catchment_area_q99"):
        context_key = key.replace("catchment_area_", "")
        assert_close(context[key], recomputed["area_context"][context_key], tolerance=1e-9)
    assert_close(context["tract_area_km2"], recomputed["tract_area_km2"], tolerance=1e-9)
    return {
        "record_audit_rows": len(audit),
        "fold_domain_rows_checked": len(comparison),
        "area_classes_checked": len(labels),
        "boundary_thresholds_checked": len(config["inventory"]["boundary_thresholds_m"]) * 2,
        "support_cells": len(model_table),
        "hex_union_vertices": int(shapely.get_num_coordinates(hex_union)),
    }


def stage5_input_records(config: dict[str, Any], stage4_config: dict[str, Any]) -> dict[str, Any]:
    inputs = {}
    for key, spec in config["inputs"].items():
        if "path" not in spec:
            continue
        path = spec["path"]
        record = file_record(path, role=f"stage5 input: {key}")
        if "layer" in spec:
            record["layer"] = spec["layer"]
        inputs[key] = record
    for key, spec in stage4_config["inputs"].items():
        if not isinstance(spec, dict) or "path" not in spec:
            continue
        record = file_record(spec["path"], role=f"stage4 input: {key}")
        if "layer" in spec:
            record["layer"] = spec["layer"]
        if "sha256" in spec and str(spec["sha256"]).lower() != str(record["sha256"]).lower():
            raise RuntimeError(f"Stage 4 input hash declaration disagrees for {key}")
        inputs[f"stage4_{key}"] = record
    return inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage5_v1.json")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest_path = abs_path(config["outputs"]["processed_dir"]) / config["outputs"]["inventory_manifest"]
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    original_manifest_sha = manifest.get("original_manifest_sha256") or sha256_file(manifest_path)

    bundle = load_stage4_bundle(config)
    geometries = load_geometries(config)
    checks: list[dict[str, Any]] = []
    input_records = capture(
        checks, "stage5_input_hashes", lambda: stage5_input_records(config, bundle["stage4_config"])
    )
    capture(checks, "stage4_chain", lambda: verify_stage4_chain(bundle))
    capture(checks, "stage5a_output_hashes", lambda: verify_records(manifest["outputs"]))
    for key in ("inventory_figure", "inventory_positional_figure", "inventory_support_figure"):
        capture(checks, f"png_{key}", lambda key=key: verify_png(manifest["outputs"][key]["path"]))
    inventory = capture(checks, "inventory_recomputation", lambda: verify_inventory(bundle, config, geometries))
    capture(
        checks, "summary_recomputation",
        lambda: verify_summary_tables(bundle, config, geometries, manifest),
    )

    expected_validation = manifest["inventory_validation"]
    if inventory is not None:
        for key in ("n_records", "n_in_tract", "n_group_components", "n_expanded_records", "n_expanded_records_in_tract", "outside_tract_record_ids"):
            if inventory[key] != expected_validation[key]:
                record_check(checks, f"manifest_inventory_{key}", False, {
                    "manifest": expected_validation[key], "recomputed": inventory[key],
                })
            else:
                record_check(checks, f"manifest_inventory_{key}", True, inventory[key])

    n_failed = sum(check["status"] == "fail" for check in checks)
    validation = {
        "schema_version": "1.0.0",
        "status": "pass" if n_failed == 0 else "fail",
        "n_checks": len(checks),
        "n_failed": n_failed,
        "checks": checks,
        "stage5_build_id": manifest["build_id"],
        "stage4_build_ids": manifest["stage4_build_ids"],
        "original_manifest_sha256": original_manifest_sha,
        "software": software_stack(),
        "input_records": input_records,
    }
    output_dir = abs_path(config["outputs"]["processed_dir"])
    validation_path = output_dir / "stage5_inventory_validation_v1.json"
    validation_record = write_json(validation, validation_path)

    manifest["original_manifest_sha256"] = original_manifest_sha
    manifest["validation"] = {
        **file_record(validation_path, role="independent Stage 5A verification"),
        "status": validation["status"],
        "n_checks": validation["n_checks"],
        "n_failed": validation["n_failed"],
    }
    manifest["stage5_inputs"] = input_records or {}
    manifest["status"] = "verified" if n_failed == 0 else "verification_failed"
    manifest["notes"] = list(dict.fromkeys(manifest.get("notes", []) + [
        "Independent verification recomputed inventory, group, positional, nominal-hex, area, fold/domain and boundary facts.",
        "Nominal-hexagon and Daly-domain input files are hash-recorded in stage5_inputs.",
    ]))
    write_json(manifest, manifest_path)
    print(json.dumps({
        "status": validation["status"], "n_checks": len(checks),
        "n_failed": n_failed, "manifest_status": manifest["status"],
        "validation": relative_posix(validation_path),
    }, indent=2), flush=True)
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
