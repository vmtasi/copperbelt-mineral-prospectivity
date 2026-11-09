"""Independently verify the Stage 5B support-sensitivity experiment.

The verifier rebuilds support membership and record retention from the source
geometries, refits every S0_NO_GRAVITY scenario from its retained-record table,
and compares the result with the serialized outputs. It does not import the
Stage 5B support-construction functions, so the audit cannot simply reproduce
its own internal errors.
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
from shapely.strtree import STRtree


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modeling.fit_stage4_models import fit_experiment  # noqa: E402
from src.modeling.stage4_common import (  # noqa: E402
    PROJECT_CRS,
    abs_path,
    file_record,
    relative_posix,
    sha256_file,
    write_json,
)
from src.modeling.stage5_common import (  # noqa: E402
    load_config as load_stage5_config,
    load_geometries,
    load_stage4_bundle,
    software_stack,
)
from src.modeling.stage5b_common import (  # noqa: E402
    load_config,
    output_paths,
    s0_experiment,
)


def record_check(checks: list[dict[str, Any]], name: str, passed: bool, details: Any) -> None:
    checks.append({"name": name, "status": "pass" if passed else "fail", "details": details})


def capture(checks: list[dict[str, Any]], name: str, function: Callable[[], Any]) -> Any:
    try:
        details = function()
        record_check(checks, name, True, details)
        return details
    except Exception as error:  # noqa: BLE001 - verify each check independently
        record_check(checks, name, False, {"error": f"{type(error).__name__}: {error}"})
        return None


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.DataFrame):
        return {"rows": int(len(value)), "columns": [str(column) for column in value.columns]}
    if isinstance(value, pd.Series):
        return {"length": int(len(value)), "name": str(value.name)}
    if isinstance(value, pd.Index):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        item = value.item()
        if not isinstance(item, type(value)):
            return json_safe(item)
    return value


def assert_close(observed: float, expected: float, tolerance: float = 1e-8, label: str = "value") -> None:
    if not math.isclose(float(observed), float(expected), rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"{label} differs: observed={observed}, expected={expected}")


def verify_records(records: dict[str, Any]) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    for name, record in records.items():
        path = abs_path(record["path"])
        if not path.exists():
            mismatches.append({"name": name, "path": record["path"], "error": "missing"})
            continue
        if path.stat().st_size != int(record.get("bytes", -1)):
            mismatches.append({
                "name": name, "path": record["path"], "error": "byte-size mismatch",
                "expected": int(record.get("bytes", -1)), "observed": path.stat().st_size,
            })
        observed = sha256_file(path)
        if observed.lower() != str(record["sha256"]).lower():
            mismatches.append({
                "name": name, "path": record["path"], "error": "sha256 mismatch",
                "expected": record["sha256"], "observed": observed,
            })
    if mismatches:
        raise RuntimeError(f"{len(mismatches)} manifest records failed verification")
    return {"checked": len(records), "mismatches": mismatches}


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


def assign_points(points: gpd.GeoSeries, selected: gpd.GeoDataFrame) -> pd.Series:
    geometries = list(selected.geometry)
    tree = STRtree(geometries)
    centers = np.column_stack([selected.geometry.centroid.x, selected.geometry.centroid.y])
    identifiers = selected["support_id"].astype(str).to_numpy()
    assigned: list[str | None] = []
    for point in points:
        candidates = np.asarray(tree.query(point, predicate="intersects"), dtype=int)
        if candidates.size == 0:
            assigned.append(None)
            continue
        if candidates.size == 1:
            assigned.append(str(identifiers[int(candidates[0])]))
            continue
        distances = np.linalg.norm(centers[candidates] - np.array([point.x, point.y]), axis=1)
        order = np.lexsort((identifiers[candidates], distances))
        assigned.append(str(identifiers[int(candidates[int(order[0])])]))
    return pd.Series(assigned, index=points.index, dtype=object)


def selected_hexes(hexagons: gpd.GeoDataFrame, threshold: float) -> gpd.GeoDataFrame:
    selected = hexagons.loc[
        hexagons["frame_overlap_fraction"].astype(float).ge(float(threshold))
    ].copy()
    if selected.empty:
        raise AssertionError(f"No hexagons selected at threshold {threshold}")
    return selected


def independent_scenarios(
    config: dict[str, Any], bundle: dict[str, Any], geometries: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    occurrences = bundle["occurrences"].copy()
    model_table = bundle["model_table"].copy()
    points = gpd.GeoSeries(
        gpd.points_from_xy(occurrences["x_m"], occurrences["y_m"]), crs=PROJECT_CRS
    )
    in_tract = occurrences["in_tract"].astype(bool)
    scenarios: dict[str, dict[str, Any]] = {}
    for spec in config["support_scenarios"]:
        family = spec["family"]
        hex_threshold = spec.get("hex_threshold")
        filter_threshold = spec.get("record_filter_threshold")
        if family == "hex":
            selected = selected_hexes(geometries["nominal_hexagons"], float(hex_threshold))
            assigned = assign_points(points, selected)
            retained = in_tract & assigned.notna()
            support_ids = selected["support_id"].astype(str).tolist()
            area_lookup = selected.set_index("support_id")["frame_overlap_area_km2"].astype(float)
            support_area = model_table.set_index("support_id")["tract_area_km2"].copy()
            support_area = area_lookup.reindex(support_ids).astype(float)
        elif family == "voronoi":
            selected = (
                selected_hexes(geometries["nominal_hexagons"], 0.0)
                if filter_threshold is None else
                selected_hexes(geometries["nominal_hexagons"], float(filter_threshold))
            )
            contained = assign_points(points, selected)
            if filter_threshold is None:
                assigned = occurrences["assigned_support_id"].astype(object)
            else:
                assigned = contained
            retained = in_tract & pd.Series(assigned, index=occurrences.index).notna()
            support_ids = model_table["support_id"].astype(str).tolist()
            support_area = model_table.set_index("support_id")["tract_area_km2"].astype(float)
        else:
            raise ValueError(f"Unknown support family {family}")
        assigned = pd.Series(assigned, index=occurrences.index).where(in_tract)
        assignment = pd.DataFrame({
            "record_id": occurrences["record_id"].astype(str),
            "support_id": pd.Series(assigned, index=occurrences.index).astype(object),
            "point_retained": retained.astype(bool),
            "support_area_km2": pd.Series(assigned, index=occurrences.index).map(support_area),
            "SiteStatus": occurrences["SiteStatus"].astype(str),
            "record_expansion": occurrences["record_expansion"].astype(float),
        })
        scenarios[spec["id"]] = {
            "spec": spec,
            "support_ids": support_ids,
            "support_area": support_area,
            "assignment": assignment,
            "retained": assignment.loc[assignment["point_retained"]].copy(),
        }
    return scenarios


def verify_support_tables(
    config: dict[str, Any], bundle: dict[str, Any], geometries: dict[str, Any],
    scenarios: dict[str, dict[str, Any]], comparison: pd.DataFrame,
    assignments: pd.DataFrame, fold_composition: pd.DataFrame,
) -> dict[str, Any]:
    comparison_index = comparison.set_index("support_scheme")
    assignments_index = assignments.set_index(["support_scheme", "record_id"])
    fold_index = fold_composition.set_index(["support_scheme", "fold"])
    details: dict[str, Any] = {}
    for scheme, scenario in scenarios.items():
        retained = scenario["retained"]
        support_ids = set(scenario["support_ids"])
        if not retained["support_id"].isin(support_ids).all():
            raise AssertionError(f"{scheme} retains a record in a missing support cell")
        counts = retained.groupby("support_id").size()
        expected_positive = int(counts.size)
        expected_area = float(scenario["support_area"].sum())
        row = comparison_index.loc[scheme]
        assert_close(row["n_support"], len(support_ids), label=f"{scheme} support cells")
        assert_close(row["support_area_sum_km2"], expected_area, label=f"{scheme} support area")
        assert_close(row["n_retained_records"], len(retained), label=f"{scheme} retained records")
        assert_close(row["n_positive_cells"], expected_positive, label=f"{scheme} positive cells")
        assert_close(
            row["prevalence"], expected_positive / max(len(support_ids), 1),
            label=f"{scheme} prevalence",
        )
        for column, status in (("n_deposits", "deposit"), ("n_prospects", "prospect"), ("n_groups", "group")):
            assert_close(
                row[column], int(retained["SiteStatus"].eq(status).sum()),
                label=f"{scheme} {column}",
            )
        saved = assignments.loc[assignments["support_scheme"].eq(scheme)].set_index("record_id")
        expected = scenario["assignment"].set_index("record_id")
        if set(saved.index) != set(expected.index):
            raise AssertionError(f"{scheme} record identifiers differ")
        if not saved.loc[expected.index, "point_retained"].astype(bool).equals(
            expected["point_retained"].astype(bool)
        ):
            raise AssertionError(f"{scheme} point-retention decisions differ")
        saved_support = saved.loc[expected.index, "support_id"]
        expected_support = expected["support_id"]
        mismatch = ~(
            saved_support.isna() & expected_support.isna()
        ) & (saved_support.astype(str) != expected_support.astype(str))
        if mismatch.any():
            raise AssertionError(f"{scheme} assigned support identifiers differ in {int(mismatch.sum())} records")
        for fold in sorted(bundle["model_table"]["new_fold"].unique()):
            fold_ids = set(
                bundle["model_table"].loc[
                    bundle["model_table"]["new_fold"].eq(fold), "support_id"
                ].astype(str)
            ) & support_ids
            fold_saved = fold_index.loc[(scheme, fold)]
            fold_assignment = retained.loc[retained["support_id"].isin(fold_ids)]
            fold_counts = fold_assignment.groupby("support_id").size()
            assert_close(fold_saved["n_support"], len(fold_ids), label=f"{scheme} fold {fold} cells")
            assert_close(fold_saved["records"], len(fold_assignment), label=f"{scheme} fold {fold} records")
            assert_close(fold_saved["positive_cells"], len(fold_counts), label=f"{scheme} fold {fold} positive cells")
        details[scheme] = {
            "cells": len(support_ids),
            "retained_records": int(len(retained)),
            "positive_cells": expected_positive,
            "support_area_km2": expected_area,
        }
    return details


def verify_positional(
    config: dict[str, Any], bundle: dict[str, Any], geometries: dict[str, Any],
    scenarios: dict[str, dict[str, Any]], positional: pd.DataFrame,
    positional_summary: pd.DataFrame, comparison: pd.DataFrame,
) -> dict[str, Any]:
    if not positional["p_retained_mc"].between(0.0, 1.0).all():
        raise AssertionError("Positional retention probability outside [0, 1]")
    if not np.allclose(
        positional["p_retained_mc"] + positional["p_lost_mc"], 1.0, atol=1e-12
    ):
        raise AssertionError("Positional retained and lost probabilities do not sum to one")
    if positional.groupby(["coverage_variant", "record_id"]).size().max() != 1:
        raise AssertionError("Positional probabilities are not unique per variant/record")
    summary_index = positional_summary.set_index(["coverage_variant", "site_status"])
    for variant, group in positional.groupby("coverage_variant", sort=True):
        for status, subset in group.groupby("site_status", sort=True):
            row = summary_index.loc[(variant, status)]
            assert_close(row["n_records"], len(subset), label=f"{variant}/{status} records")
            assert_close(row["expected_records_retained_mc"], subset["p_retained_mc"].sum(), label=f"{variant}/{status} expected retention")
    occurrences = bundle["occurrences"].copy()
    accuracy_m = float(bundle["stage4_config"]["inputs"]["occurrences"]["positional_accuracy_m"])
    draws = int(bundle["stage4_config"]["inputs"]["occurrences"]["positional_monte_carlo_draws"])
    seed = int(bundle["stage4_config"]["inputs"]["occurrences"]["positional_seed"])
    support_unions: dict[str, Any] = {"VORONOI_ALL": None}
    for variant in positional["coverage_variant"].unique():
        if variant == "VORONOI_ALL":
            continue
        threshold = float(variant.removeprefix("HEX_"))
        support_unions[variant] = unary_union(list(selected_hexes(geometries["nominal_hexagons"], threshold).geometry))
    expected_probability = pd.DataFrame(index=occurrences["record_id"].astype(str), columns=sorted(positional["coverage_variant"].unique()), dtype=float)
    for row in occurrences.itertuples():
        rng = np.random.default_rng([seed, int(row.source_fid)])
        radius = accuracy_m * np.sqrt(rng.random(draws))
        angle = 2.0 * np.pi * rng.random(draws)
        draw_points = shapely.points(
            float(row.x_m) + radius * np.cos(angle),
            float(row.y_m) + radius * np.sin(angle),
        )
        inside_tract = np.asarray(shapely.covers(geometries["tract_union"], draw_points), dtype=bool)
        for variant in expected_probability.columns:
            if not bool(row.in_tract):
                probability = 0.0
            elif support_unions[variant] is None:
                probability = float(np.mean(inside_tract))
            else:
                inside_support = np.asarray(
                    shapely.covers(support_unions[variant], draw_points), dtype=bool
                )
                probability = float(np.mean(inside_tract & inside_support))
            expected_probability.loc[str(row.record_id), variant] = probability
    for variant in expected_probability.columns:
        observed = positional.loc[
            positional["coverage_variant"].eq(variant)
        ].set_index("record_id")["p_retained_mc"]
        expected = expected_probability[variant].reindex(observed.index).astype(float)
        if not np.allclose(observed.to_numpy(dtype=float), expected.to_numpy(dtype=float), atol=1e-12):
            maximum = float(np.max(np.abs(observed.to_numpy(dtype=float) - expected.to_numpy(dtype=float))))
            raise AssertionError(f"{variant} positional probabilities differ by {maximum}")
    comparison_index = comparison.set_index("support_scheme")
    for scheme in scenarios:
        threshold = scenarios[scheme]["spec"].get("record_filter_threshold")
        variant = "VORONOI_ALL" if threshold is None else f"HEX_{float(threshold):g}"
        assert_close(
            comparison_index.loc[scheme, "expected_records_retained_mc"],
            expected_probability[variant].sum(), label=f"{scheme} expected retained records",
        )
    return {
        "variants": sorted(positional["coverage_variant"].unique().tolist()),
        "records": int(positional["record_id"].nunique()),
        "rows": int(len(positional)),
    }


def rebuild_model_frame(
    scenario: dict[str, Any], bundle: dict[str, Any],
) -> pd.DataFrame:
    model_table = bundle["model_table"].copy()
    ids = pd.Index(scenario["support_ids"], name="support_id")
    frame = model_table.set_index("support_id").loc[ids].reset_index()
    frame["tract_area_km2"] = scenario["support_area"].reindex(frame["support_id"]).to_numpy(dtype=float)
    frame["log_tract_area_km2"] = np.log(frame["tract_area_km2"].clip(lower=1e-12))
    retained = scenario["retained"]
    count = retained.groupby("support_id").size()
    expanded = retained.groupby("support_id")["record_expansion"].sum()
    frame["n_records"] = frame["support_id"].map(count).fillna(0.0).astype(float)
    frame["n_expanded_records"] = frame["support_id"].map(expanded).fillna(0.0).astype(float)
    frame["any_record"] = frame["n_records"].gt(0).astype(int)
    return frame.sort_values("support_id").reset_index(drop=True)


def verify_models(
    config: dict[str, Any], bundle: dict[str, Any], scenarios: dict[str, dict[str, Any]],
    predictions: pd.DataFrame, metrics: pd.DataFrame,
) -> dict[str, Any]:
    saved_predictions = predictions.set_index(["support_scheme", "support_id"])
    saved_metrics = metrics.loc[metrics["aggregation"].eq("pooled_oof")].set_index("support_scheme")
    details: dict[str, Any] = {}
    for scheme, scenario in scenarios.items():
        frame = rebuild_model_frame(scenario, bundle)
        regenerated_predictions, regenerated_metrics, _, _ = fit_experiment(
            frame, s0_experiment(bundle["stage4_config"]), bundle["stage4_config"]
        )
        regenerated_predictions = regenerated_predictions.set_index("support_id")
        saved = saved_predictions.loc[scheme]
        if set(saved.index.astype(str)) != set(regenerated_predictions.index.astype(str)):
            raise AssertionError(f"{scheme} prediction support ids differ")
        saved = saved.set_index(saved.index.astype(str))
        regenerated_predictions.index = regenerated_predictions.index.astype(str)
        for field in (
            "predicted_count", "predicted_rate_per_km2", "predicted_any_probability",
            "pearson_residual", "observed_count",
        ):
            difference = float(np.max(np.abs(
                saved.loc[regenerated_predictions.index, field].to_numpy(dtype=float)
                - regenerated_predictions[field].to_numpy(dtype=float)
            )))
            if difference > 1e-8:
                raise AssertionError(f"{scheme} {field} differs by {difference}")
        regenerated = regenerated_metrics.set_index("aggregation").loc["pooled_oof"]
        for field in (
            "roc_auc", "average_precision", "brier_score", "brier_skill",
            "nb_deviance", "calibration_slope", "calibration_intercept",
        ):
            assert_close(
                saved_metrics.loc[scheme, field], regenerated[field],
                tolerance=1e-8, label=f"{scheme} {field}",
            )
        details[scheme] = {"support_cells": len(frame), "prediction_rows": len(regenerated_predictions)}
    return details


def verify_reference(bundle: dict[str, Any], predictions: pd.DataFrame, metrics: pd.DataFrame) -> dict[str, Any]:
    stage4_predictions = bundle["predictions"].loc[
        bundle["predictions"]["model_id"].eq("S0_NO_GRAVITY")
    ].set_index("support_id").sort_index()
    reference = predictions.loc[predictions["support_scheme"].eq("VORONOI_ALL")].set_index("support_id").sort_index()
    differences: dict[str, float] = {}
    for field in ("predicted_rate_per_km2", "predicted_any_probability", "observed_count", "pearson_residual"):
        difference = float(np.max(np.abs(
            stage4_predictions[field].to_numpy(dtype=float) - reference[field].to_numpy(dtype=float)
        )))
        differences[field] = difference
        if difference > 1e-8:
            raise AssertionError(f"VORONOI_ALL reference differs from Stage 4 for {field}: {difference}")
    stage4_metric = bundle["model_metrics"].loc[
        bundle["model_metrics"]["model_id"].eq("S0_NO_GRAVITY")
        & bundle["model_metrics"]["aggregation"].eq("pooled_oof")
    ].iloc[0]
    reference_metric = metrics.loc[
        metrics["support_scheme"].eq("VORONOI_ALL")
        & metrics["aggregation"].eq("pooled_oof")
    ].iloc[0]
    metric_differences: dict[str, float] = {}
    for field in ("roc_auc", "average_precision", "brier_score", "brier_skill", "calibration_slope", "calibration_intercept"):
        difference = abs(float(stage4_metric[field]) - float(reference_metric[field]))
        metric_differences[field] = difference
        if difference > 1e-8:
            raise AssertionError(f"VORONOI_ALL metric differs from Stage 4 for {field}: {difference}")
    return {"max_prediction_differences": differences, "max_metric_differences": metric_differences}


def verify_area_diagnostics(predictions: pd.DataFrame, area: pd.DataFrame) -> dict[str, Any]:
    details: dict[str, Any] = {}
    for scheme, group in predictions.groupby("support_scheme", sort=True):
        row = area.set_index("support_scheme").loc[scheme]
        rho = stats.spearmanr(group["tract_area_km2"], group["pearson_residual"])
        assert_close(row["area_vs_pearson_residual_spearman_r"], rho.statistic, tolerance=1e-8, label=f"{scheme} area-residual rho")
        assert_close(row["observed_over_predicted_total"], group["observed_count"].sum() / group["predicted_count"].sum(), tolerance=1e-8, label=f"{scheme} observed/predicted")
        details[scheme] = {"area_residual_rho": float(rho.statistic)}
    return details


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage5b_v1.json")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = output_paths(config)
    manifest_path = abs_path(paths["manifest"])
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    original_manifest_sha = manifest.get("original_manifest_sha256") or sha256_file(manifest_path)
    bundle = load_stage4_bundle(load_stage5_config(config["stage5"]["config"]))
    geometries = load_geometries(load_stage5_config(config["stage5"]["config"]))

    checks: list[dict[str, Any]] = []
    capture(checks, "manifest_output_hashes", lambda: verify_records(manifest["outputs"]))
    capture(checks, "manifest_input_hashes", lambda: verify_records(manifest["inputs"]))
    capture(checks, "config_hash", lambda: verify_records({"config": manifest["config"]}))
    for key in ("scenario_figure", "retention_figure", "metric_figure", "diagnostic_figure"):
        capture(checks, f"png_{key}", lambda key=key: verify_png(manifest["outputs"][key]["path"]))
    capture(checks, "stage4_protection", lambda: {
        "protected_assets": len(bundle["protected"]),
        "all_hash_matches": all(item["hash_match"] for item in bundle["protected"].values()),
    })

    comparison = pd.read_csv(paths["support_comparison"])
    assignments = pd.read_csv(paths["record_assignments"])
    fold_composition = pd.read_csv(paths["fold_composition"])
    positional = pd.read_csv(paths["positional_propagation"])
    positional_summary = pd.read_csv(paths["positional_summary"])
    predictions = pd.read_csv(paths["model_predictions"])
    metrics = pd.read_csv(paths["model_metrics"])
    area = pd.read_csv(paths["area_diagnostics"])

    scenarios = capture(
        checks, "independent_support_rebuild",
        lambda: independent_scenarios(config, bundle, geometries),
    )
    if scenarios is not None:
        capture(
            checks, "support_and_fold_recomputation",
            lambda: verify_support_tables(
                config, bundle, geometries, scenarios, comparison, assignments,
                fold_composition,
            ),
        )
        capture(
            checks, "positional_recomputation",
            lambda: verify_positional(
                config, bundle, geometries, scenarios, positional,
                positional_summary, comparison,
            ),
        )
        capture(
            checks, "model_prediction_recomputation",
            lambda: verify_models(config, bundle, scenarios, predictions, metrics),
        )
    capture(checks, "stage4_reference_reproduction", lambda: verify_reference(bundle, predictions, metrics))
    capture(checks, "area_diagnostic_recomputation", lambda: verify_area_diagnostics(predictions, area))

    for scheme in comparison["support_scheme"].astype(str):
        check_name = f"scenario_present_{scheme}"
        record_check(
            checks, check_name,
            scheme in set(manifest["scenarios"]) and scheme in set(predictions["support_scheme"]),
            {"scheme": scheme},
        )

    n_failed = sum(check["status"] == "fail" for check in checks)
    validation = {
        "schema_version": "1.0.0",
        "status": "pass" if n_failed == 0 else "fail",
        "n_checks": len(checks),
        "n_failed": n_failed,
        "checks": checks,
        "stage5b_build_id": manifest["build_id"],
        "stage4_build_ids": manifest["stage4_build_ids"],
        "original_manifest_sha256": original_manifest_sha,
        "software": software_stack(),
    }
    validation_path = abs_path(config["outputs"]["processed_dir"]) / "stage5b_validation_v1.json"
    validation = json_safe(validation)
    write_json(validation, validation_path)
    manifest["original_manifest_sha256"] = original_manifest_sha
    manifest["validation"] = {
        **file_record(validation_path, role="independent Stage 5B verification"),
        "status": validation["status"],
        "n_checks": validation["n_checks"],
        "n_failed": validation["n_failed"],
    }
    manifest["status"] = "verified" if n_failed == 0 else "verification_failed"
    write_json(manifest, manifest_path)
    print(json.dumps({
        "status": validation["status"],
        "n_checks": len(checks),
        "n_failed": n_failed,
        "manifest_status": manifest["status"],
        "validation": relative_posix(validation_path),
    }, indent=2), flush=True)
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
