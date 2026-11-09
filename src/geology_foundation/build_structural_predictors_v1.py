"""Build structural predictors on a rebuilt equal-area hexagonal support.

This module is intentionally read-only with respect to the V14 training table
and the v2 named-structure GeoPackage. It reads the V14 identifiers and
centroids only, for a nearest-cell crosswalk, and never reads
``deposit_present`` or any other occurrence field.

Outputs are written to new paths under ``data/processed/structural_predictors``
and ``outputs/structural_predictors``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import pyogrio
import shapely
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation_strategies import get_along_belt_folds
from src.geology_foundation.common import file_record, relative_posix, software_versions, write_json


PROJECT_CRS = "EPSG:32735"
BUILD_SCHEMA_VERSION = "1.0.0"
PARTIAL_OVERLAP_TOLERANCE = 1e-6
FEATURE_PREFIX = "STR_"


def _abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with _abs(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    with _abs(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_gdf(path: str | Path, layer: str) -> gpd.GeoDataFrame:
    frame = gpd.read_file(_abs(path), layer=layer, engine="pyogrio")
    if frame.crs is None:
        raise ValueError(f"Layer {layer} has no CRS: {path}")
    if frame.crs.to_string() != PROJECT_CRS:
        frame = frame.to_crs(PROJECT_CRS)
    return frame


def _hexagon_offsets(radius_m: float) -> list[tuple[float, float]]:
    return [
        (radius_m, 0.0),
        (radius_m / 2.0, math.sqrt(3.0) * radius_m / 2.0),
        (-radius_m / 2.0, math.sqrt(3.0) * radius_m / 2.0),
        (-radius_m, 0.0),
        (-radius_m / 2.0, -math.sqrt(3.0) * radius_m / 2.0),
        (radius_m / 2.0, -math.sqrt(3.0) * radius_m / 2.0),
    ]


def build_support(
    frame_cells: gpd.GeoDataFrame,
    nominal_area_km2: float,
    full_support_threshold: float,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Build a full-hexagon quadrature support with centroids inside the frame.

    The hexagons are equal nominal area. The clipped intersection with the
    tract footprint is retained only as a diagnostic overlap and display
    geometry; it is not used to change the nominal equal-area support weight.
    """

    radius_m = math.sqrt(2.0 * nominal_area_km2 * 1e6 / (3.0 * math.sqrt(3.0)))
    dx = 1.5 * radius_m
    dy = math.sqrt(3.0) * radius_m
    offsets = _hexagon_offsets(radius_m)
    minx, miny, maxx, maxy = frame_cells.total_bounds
    cols = range(0, int(math.ceil((maxx - minx) / dx)) + 2)
    rows = range(0, int(math.ceil((maxy - miny) / dy)) + 2)

    records: list[dict[str, Any]] = []
    for col in cols:
        cx = minx + radius_m + col * dx
        for row in rows:
            cy = miny + math.sqrt(3.0) * radius_m / 2.0 + row * dy + (col % 2) * dy / 2.0
            polygon = Polygon([(cx + x, cy + y) for x, y in offsets])
            records.append({"hex_key": f"RAW_{len(records):06d}", "geometry": polygon})

    candidates = gpd.GeoDataFrame(records, crs=PROJECT_CRS)
    centres = candidates.copy()
    centres["geometry"] = centres.geometry.centroid
    inside = gpd.sjoin(centres, frame_cells[["geometry"]], how="inner", predicate="within")
    inside = inside[~inside.index.duplicated(keep="first")]
    kept = candidates[candidates["hex_key"].isin(inside["hex_key"])].copy()
    if kept.empty:
        raise RuntimeError("No hexagon centres fall inside the tract frame")

    pair = gpd.sjoin(kept[["hex_key", "geometry"]], frame_cells[["geometry"]], how="inner", predicate="intersects")
    clipped_records: list[dict[str, Any]] = []
    for hex_key, group in pair.groupby("hex_key", sort=False):
        poly = kept.loc[kept["hex_key"] == hex_key, "geometry"].iloc[0]
        local_cells = unary_union(list(frame_cells.loc[group["index_right"], "geometry"]))
        clipped = poly.intersection(local_cells)
        if clipped.is_empty:
            continue
        overlap_area_km2 = float(clipped.area) / 1e6
        clipped_records.append(
            {
                "hex_key": hex_key,
                "frame_overlap_area_km2": overlap_area_km2,
                "frame_overlap_fraction": overlap_area_km2 / nominal_area_km2,
                "display_geometry": clipped,
            }
        )

    overlap = gpd.GeoDataFrame(clipped_records, geometry="display_geometry", crs=PROJECT_CRS)
    kept = kept.merge(overlap.drop(columns="display_geometry"), on="hex_key", how="left")
    kept["clipped_geometry"] = kept["hex_key"].map(
        dict(zip(overlap["hex_key"], overlap["display_geometry"]))
    )
    if kept["frame_overlap_fraction"].isna().any():
        missing = kept.loc[kept["frame_overlap_fraction"].isna(), "hex_key"].tolist()
        raise RuntimeError(f"Support hexagons without tract overlap: {missing[:5]}")
    kept["x_m"] = kept.geometry.centroid.x.to_numpy()
    kept["y_m"] = kept.geometry.centroid.y.to_numpy()
    kept["nominal_area_km2"] = float(nominal_area_km2)
    kept["full_support"] = kept["frame_overlap_fraction"] >= full_support_threshold
    kept["edge_hex"] = ~kept["full_support"]
    kept = kept.sort_values(["y_m", "x_m"], kind="stable").reset_index(drop=True)
    kept["support_id"] = [f"HEX10K_{index + 1:05d}" for index in range(len(kept))]

    metadata = kept.drop(columns=["geometry", "hex_key", "clipped_geometry"]).copy()
    points = gpd.GeoDataFrame(
        metadata.copy(),
        geometry=gpd.points_from_xy(kept["x_m"], kept["y_m"]),
        crs=PROJECT_CRS,
    )
    polygons = gpd.GeoDataFrame(
        metadata.copy(),
        geometry=kept["geometry"],
        crs=PROJECT_CRS,
    )
    overlaps = gpd.GeoDataFrame(
        metadata.copy(),
        geometry=kept["clipped_geometry"],
        crs=PROJECT_CRS,
    )
    return points, polygons, overlaps


def assign_domains(points: gpd.GeoDataFrame, domains: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    joined = gpd.sjoin(
        points[["support_id", "geometry"]],
        domains[["domain_id", "abbreviation", "domain_name", "geometry"]],
        how="left",
        predicate="within",
    )
    duplicate_count = int(joined["support_id"].duplicated().sum())
    joined = joined.sort_values(["support_id", "domain_id"], kind="stable").drop_duplicates("support_id", keep="first")
    joined = joined.set_index("support_id")
    out = points.copy()
    out["domain_id"] = out["support_id"].map(joined["domain_id"]).fillna("unassigned")
    out["domain_abbreviation"] = out["support_id"].map(joined["abbreviation"]).fillna("unassigned")
    out["domain_name"] = out["support_id"].map(joined["domain_name"]).fillna("Outside reconstructed Daly domain envelope")
    out.attrs["domain_join_duplicate_count"] = duplicate_count
    return out


def add_v14_crosswalk(points: gpd.GeoDataFrame, v14: pd.DataFrame, n_folds: int) -> gpd.GeoDataFrame:
    required = {"id", "centroid_x", "centroid_y"}
    missing = required.difference(v14.columns)
    if missing:
        raise ValueError(f"V14 crosswalk input is missing columns: {sorted(missing)}")
    v14 = v14[["id", "centroid_x", "centroid_y"]].copy()
    folds = get_along_belt_folds(v14, n_folds=n_folds)
    v14["v14_fold"] = folds
    v14["v14_fold_label"] = [f"Fold {int(fold) + 1}" for fold in folds]
    tree = cKDTree(v14[["centroid_x", "centroid_y"]].to_numpy())
    distances, indices = tree.query(points[["x_m", "y_m"]].to_numpy(), k=1)
    out = points.copy()
    out["v14_nearest_id"] = v14["id"].to_numpy()[indices]
    out["v14_nearest_distance_m"] = distances
    out["v14_fold"] = v14["v14_fold"].to_numpy()[indices]
    out["v14_fold_label"] = v14["v14_fold_label"].to_numpy()[indices]
    return out


def load_structures(config: dict[str, Any]) -> dict[str, Any]:
    source = config["structures"]
    segments = _read_gdf(source["path"], source["segments_layer"])
    centrelines = _read_gdf(source["path"], source["centrelines_layer"])
    bridges = _read_gdf(source["path"], source["bridges_layer"])
    catalogue = _read_gdf(source["path"], source["catalogue_layer"])

    for column in ["attributed_structure_id", "attributed_structure_abbreviation", "attributed_status"]:
        if column not in segments.columns:
            raise ValueError(f"Missing v2 segment field: {column}")

    high = segments[segments["attributed_status"] == "named_high"].copy()
    medium = segments[segments["attributed_status"] == "named_medium"].copy()
    hm = segments[segments["attributed_status"].isin(["named_high", "named_medium"])].copy()
    all_named = segments[segments["attributed_structure_id"].notna()].copy()
    unnamed = segments[segments["attributed_structure_id"].isna()].copy()

    def geom_union(frame: gpd.GeoDataFrame) -> Any:
        if frame.empty:
            raise ValueError("Cannot build a union from an empty frame")
        return unary_union(list(frame.geometry))

    system_unions: dict[str, dict[str, Any]] = {}
    for structure_id, group in all_named.groupby("attributed_structure_id", sort=True):
        system_unions[str(structure_id)] = {
            "structure_id": str(structure_id),
            "abbreviation": str(group["attributed_structure_abbreviation"].iloc[0]),
            "name": str(group["attributed_structure_name"].iloc[0]),
            "geometry": geom_union(group),
            "segments": group.copy(),
        }

    hm_system_unions: dict[str, Any] = {}
    for structure_id, group in hm.groupby("attributed_structure_id", sort=True):
        hm_system_unions[str(structure_id)] = geom_union(group)

    hm_ids = set(hm["structure_id"].astype(str))
    bridge_hm = bridges[
        bridges["from_segment_id"].astype(str).isin(hm_ids)
        & bridges["to_segment_id"].astype(str).isin(hm_ids)
    ].copy()
    hm_bridge_geoms = list(hm.geometry) + list(bridge_hm.geometry)
    hm_bridged_union = unary_union(hm_bridge_geoms)
    parent_axis_union = geom_union(centrelines)

    pair_records: list[dict[str, Any]] = []
    pair_ids = sorted(system_unions)
    for index_a, structure_a in enumerate(pair_ids):
        for structure_b in pair_ids[index_a + 1 :]:
            a = system_unions[structure_a]
            b = system_unions[structure_b]
            intersection = a["geometry"].intersection(b["geometry"])
            if intersection.is_empty:
                continue
            points_found: list[Point] = []
            if intersection.geom_type == "Point":
                points_found = [intersection]
            elif intersection.geom_type == "MultiPoint":
                points_found = list(intersection.geoms)
            else:
                raise RuntimeError(
                    f"Named-system intersection is not a point geometry: {a['abbreviation']} / "
                    f"{b['abbreviation']} -> {intersection.geom_type}"
                )
            for point in points_found:
                segment_a = a["segments"][a["segments"].geometry.distance(point) <= 0.01]
                segment_b = b["segments"][b["segments"].geometry.distance(point) <= 0.01]
                pair_records.append(
                    {
                        "system_a_id": a["structure_id"],
                        "system_a_abbreviation": a["abbreviation"],
                        "system_a_name": a["name"],
                        "system_b_id": b["structure_id"],
                        "system_b_abbreviation": b["abbreviation"],
                        "system_b_name": b["name"],
                        "pair_label": f"{a['name']}–{b['name']}",
                        "candidate_status": "candidate_contact_not_established_geological_node",
                        "segment_a_ids": ";".join(sorted(segment_a["structure_id"].astype(str))),
                        "segment_b_ids": ";".join(sorted(segment_b["structure_id"].astype(str))),
                        "x_m": float(point.x),
                        "y_m": float(point.y),
                        "geometry": point,
                    }
                )

    intersections = gpd.GeoDataFrame(pair_records, geometry="geometry", crs=PROJECT_CRS)
    intersections = intersections.sort_values(["pair_label", "x_m", "y_m"], kind="stable").reset_index(drop=True)
    intersections["candidate_id"] = [f"XSEC_{index + 1:03d}" for index in range(len(intersections))]
    intersections["qualification"] = (
        "Cartographic contact between dissolved observed named systems; no independent evidence "
        "establishes a through-going fault intersection or a geological node."
    )
    expected_pairs = {frozenset(("Kf", "Mb")), frozenset(("Ky", "LB"))}
    observed_pairs = {
        frozenset((str(row.system_a_abbreviation), str(row.system_b_abbreviation)))
        for row in intersections.itertuples()
    }
    if observed_pairs != expected_pairs or len(intersections) != 2:
        raise RuntimeError(
            "The v2 named-system intersection set changed; expected exactly Kafue-Mubulashi "
            f"and Kanyama-Lwawa, observed {sorted(tuple(sorted(pair)) for pair in observed_pairs)}"
        )

    return {
        "segments": segments,
        "centrelines": centrelines,
        "bridges": bridges,
        "catalogue": catalogue,
        "high": high,
        "medium": medium,
        "hm": hm,
        "all_named": all_named,
        "unnamed": unnamed,
        "hm_union": geom_union(hm),
        "high_union": geom_union(high),
        "all_named_union": geom_union(all_named),
        "unnamed_union": geom_union(unnamed),
        "hm_bridged_union": hm_bridged_union,
        "parent_axis_union": parent_axis_union,
        "hm_bridge_count": int(len(bridge_hm)),
        "system_unions": system_unions,
        "hm_system_unions": hm_system_unions,
        "intersections": intersections,
    }


def _line_length_in_radius(
    point: Point,
    radius_m: float,
    lines: list[Any],
    tree: STRtree,
) -> float:
    circle = point.buffer(radius_m)
    indices = tree.query(circle)
    total_m = 0.0
    for index in indices:
        total_m += lines[int(index)].intersection(circle).length
    return total_m / 1000.0


def _km(geometry_distance_m: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(geometry_distance_m) / 1000.0


def compute_features(
    points: gpd.GeoDataFrame,
    structures: dict[str, Any],
    boundaries: gpd.GeoDataFrame,
    radii_km: list[int],
    intersection_radius_km: int,
    system_count_radius_km: int,
) -> gpd.GeoDataFrame:
    out = points.copy()
    point_array = np.asarray(out.geometry.values)
    point_objects = list(out.geometry)

    def distance_km(geometry: Any) -> np.ndarray:
        return np.asarray(shapely.distance(point_array, geometry)) / 1000.0

    out["STR_DIST_NAMED_HM_KM"] = distance_km(structures["hm_union"])
    out["STR_DIST_NAMED_HIGH_KM"] = distance_km(structures["high_union"])
    out["STR_DIST_NAMED_ALL_KM"] = distance_km(structures["all_named_union"])
    out["STR_DIST_NAMED_BRIDGED_KM"] = distance_km(structures["hm_bridged_union"])
    out["STR_DIST_NAMED_PARENT_AXIS_KM"] = distance_km(structures["parent_axis_union"])
    out["STR_DIST_NAMED_BRIDGED_DELTA_KM"] = out["STR_DIST_NAMED_HM_KM"] - out["STR_DIST_NAMED_BRIDGED_KM"]
    out["STR_DIST_NAMED_PARENT_AXIS_DELTA_KM"] = (
        out["STR_DIST_NAMED_HM_KM"] - out["STR_DIST_NAMED_PARENT_AXIS_KM"]
    )

    for structure_id, info in structures["system_unions"].items():
        abbreviation = "".join(character if character.isalnum() else "_" for character in info["abbreviation"])
        out[f"STR_DIST_{abbreviation}_KM"] = distance_km(info["geometry"])

    intersection_points = [Point(x, y) for x, y in zip(structures["intersections"]["x_m"], structures["intersections"]["y_m"])]
    intersection_geometry = MultiPoint(intersection_points)
    out["STR_DIST_MAJOR_INTERSECTION_KM"] = distance_km(intersection_geometry)
    out["STR_MAJOR_INTERSECTION_COUNT_25KM"] = [
        sum(1 for candidate in intersection_points if point.distance(candidate) <= intersection_radius_km * 1000.0)
        for point in point_objects
    ]

    hm_lines = list(structures["hm"].geometry)
    unnamed_lines = list(structures["unnamed"].geometry)
    hm_tree = STRtree(hm_lines)
    unnamed_tree = STRtree(unnamed_lines)
    for radius in radii_km:
        lengths = [
            _line_length_in_radius(point, radius * 1000.0, hm_lines, hm_tree)
            for point in point_objects
        ]
        area_1000_km2 = math.pi * radius * radius / 1000.0
        out[f"STR_NAMED_LINE_DENSITY_{radius}KM"] = [length / area_1000_km2 for length in lengths]

    unnamed_lengths = [
        _line_length_in_radius(point, 25_000.0, unnamed_lines, unnamed_tree)
        for point in point_objects
    ]
    unnamed_area_1000_km2 = math.pi * 25.0 * 25.0 / 1000.0
    out["STR_UNNAMED_LINE_DENSITY_25KM"] = [length / unnamed_area_1000_km2 for length in unnamed_lengths]

    hm_system_items = list(structures["hm_system_unions"].items())
    out["STR_NAMED_SYSTEM_COUNT_25KM"] = [
        sum(1 for _, geometry in hm_system_items if point.distance(geometry) <= system_count_radius_km * 1000.0)
        for point in point_objects
    ]

    internal_boundaries = boundaries[
        boundaries["boundary_type"].astype(str) != "external"
    ].copy()
    boundary_lines = list(internal_boundaries.geometry)
    boundary_tree = STRtree(boundary_lines)
    boundary_distances = []
    for point in point_objects:
        nearest_index = int(boundary_tree.nearest(point))
        boundary_distances.append(point.distance(boundary_lines[nearest_index]) / 1000.0)
    out["STR_DIST_DALY_DOMAIN_BOUNDARY_KM"] = boundary_distances

    out["occurrence_data_used"] = "no"
    return out


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith(FEATURE_PREFIX)]


def feature_dictionary(structures: dict[str, Any], radii_km: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "feature_id": "STR_DIST_NAMED_HM_KM",
            "definition": "Distance to the nearest observed high-or-medium-confidence named segment",
            "units": "km",
            "source_layer": "v1_segments_attributed",
            "source_selection": "attributed_status in {named_high, named_medium}",
            "operation": "minimum planar distance to the dissolved HM linework",
            "role": "primary observed-structure proximity",
            "notes": "Primary named-structure variable; does not include bridges or lower-confidence components.",
        },
        {
            "feature_id": "STR_DIST_NAMED_HIGH_KM",
            "definition": "Distance to the nearest observed high-confidence named segment",
            "units": "km",
            "source_layer": "v1_segments_attributed",
            "source_selection": "attributed_status = named_high",
            "operation": "minimum planar distance to the dissolved HIGH linework",
            "role": "confidence sensitivity",
            "notes": "Excludes medium-confidence components.",
        },
        {
            "feature_id": "STR_DIST_NAMED_ALL_KM",
            "definition": "Distance to the nearest observed named segment of any confidence",
            "units": "km",
            "source_layer": "v1_segments_attributed",
            "source_selection": "attributed_structure_id is not null",
            "operation": "minimum planar distance to all 54 attributed segments",
            "role": "broad named-structure sensitivity",
            "notes": "Includes medium-low and low-confidence junction fragments.",
        },
        {
            "feature_id": "STR_DIST_NAMED_BRIDGED_KM",
            "definition": "Distance to HM linework plus bridges whose two endpoints are HM segments",
            "units": "km",
            "source_layer": "v1_segments_attributed; bridge_connectors",
            "source_selection": "HM segments and HM-to-HM bridge connectors",
            "operation": "minimum planar distance to the bridged HM union",
            "role": "bridge-construction sensitivity",
            "notes": "Bridges are cartographic reconstruction aids, not observed fault traces.",
        },
        {
            "feature_id": "STR_DIST_NAMED_PARENT_AXIS_KM",
            "definition": "Distance to the union of all reconstructed named parent axes",
            "units": "km",
            "source_layer": "named_structure_centrelines",
            "source_selection": "all 15 centreline parts",
            "operation": "minimum planar distance to the parent-axis union",
            "role": "parent-axis construction sensitivity",
            "notes": "Includes disconnected and lower-confidence components and all explicit bridges.",
        },
        {
            "feature_id": "STR_DIST_NAMED_BRIDGED_DELTA_KM",
            "definition": "HM distance minus HM-bridged distance",
            "units": "km",
            "source_layer": "derived from the two preceding distance fields",
            "source_selection": "HM and HM-to-HM bridges",
            "operation": "difference of distances",
            "role": "bridge-sensitivity diagnostic",
            "notes": "Positive values mean a bridge lies closer than the nearest observed HM segment.",
        },
        {
            "feature_id": "STR_DIST_NAMED_PARENT_AXIS_DELTA_KM",
            "definition": "HM distance minus parent-axis distance",
            "units": "km",
            "source_layer": "derived from the two preceding distance fields",
            "source_selection": "HM versus all parent-axis parts",
            "operation": "difference of distances",
            "role": "parent-axis sensitivity diagnostic",
            "notes": "Positive values mean the reconstructed parent axis is closer than observed HM linework.",
        },
        {
            "feature_id": "STR_DIST_MAJOR_INTERSECTION_KM",
            "definition": "Distance to the nearest candidate named-system contact",
            "units": "km",
            "source_layer": "intersection of dissolved observed named systems",
            "source_selection": "two exact contacts: Kafue–Mubulashi and Kanyama–Lwawa",
            "operation": "minimum planar distance to the candidate intersection points",
            "role": "secondary/sensitivity intersection feature",
            "notes": "Candidate contacts only; not established geological nodes.",
        },
        {
            "feature_id": "STR_MAJOR_INTERSECTION_COUNT_25KM",
            "definition": "Number of candidate named-system contacts within 25 km",
            "units": "count",
            "source_layer": "candidate intersection points",
            "source_selection": "the two exact contacts above",
            "operation": "count of candidate points within 25 km",
            "role": "secondary/sensitivity intersection feature",
            "notes": "The maximum possible value is two; this is not an independent intersection-density measure.",
        },
        {
            "feature_id": "STR_DIST_DALY_DOMAIN_BOUNDARY_KM",
            "definition": "Distance to the nearest internal Daly domain boundary",
            "units": "km",
            "source_layer": "daly_domain_boundaries",
            "source_selection": "boundary_type = internal",
            "operation": "minimum planar distance to the internal boundary linework",
            "role": "geological-context sensitivity",
            "notes": "Domain boundaries are cartographic class changes, not mapped faults or stratigraphic contacts.",
        },
        {
            "feature_id": "STR_UNNAMED_LINE_DENSITY_25KM",
            "definition": "Length of unnamed heavy linework within 25 km per 1,000 km2",
            "units": "km per 1,000 km2",
            "source_layer": "v1_segments_attributed",
            "source_selection": "attributed_structure_id is null (194 segments)",
            "operation": "exact line-length intersection with a 25 km circular window",
            "role": "mapping-texture and source-coverage sensitivity",
            "notes": "Reflects what Daly mapped as well as structural complexity; not a primary geological predictor.",
        },
        {
            "feature_id": "STR_NAMED_SYSTEM_COUNT_25KM",
            "definition": "Number of distinct HM named systems within 25 km",
            "units": "count",
            "source_layer": "v1_segments_attributed",
            "source_selection": "HM segments dissolved by attributed_structure_id",
            "operation": "count of system unions within 25 km",
            "role": "structural-complexity context",
            "notes": "Counts systems, not individual segments; the maximum is the number of HM systems.",
        },
    ]
    for radius in radii_km:
        rows.append(
            {
                "feature_id": f"STR_NAMED_LINE_DENSITY_{radius}KM",
                "definition": f"Length of HM linework within {radius} km per 1,000 km2",
                "units": "km per 1,000 km2",
                "source_layer": "v1_segments_attributed",
                "source_selection": "attributed_status in {named_high, named_medium}",
                "operation": f"exact line-length intersection with a {radius} km circular window",
                "role": "structural density"
                + (" (primary scale)" if radius == 25 else " (scale sensitivity)"),
                "notes": "The circular window is centred on the support point; edge windows include linework outside the tract frame.",
            }
        )
    for structure_id, info in sorted(structures["system_unions"].items()):
        abbreviation = "".join(character if character.isalnum() else "_" for character in info["abbreviation"])
        rows.append(
            {
                "feature_id": f"STR_DIST_{abbreviation}_KM",
                "definition": f"Distance to the observed {info['name']} system",
                "units": "km",
                "source_layer": "v1_segments_attributed",
                "source_selection": f"attributed_structure_id = {structure_id}",
                "operation": "minimum planar distance to the dissolved system linework",
                "role": "exploratory per-system distance",
                "notes": "Not intended for simultaneous inclusion with the other per-system distances.",
            }
        )
    return pd.DataFrame(rows)


def describe_feature(series: pd.Series, feature_id: str) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return {
            "feature_id": feature_id,
            "n": int(len(values)),
            "n_missing": int(values.isna().sum()),
            "n_zero": 0,
            "n_unique": 0,
        }
    quantiles = valid.quantile([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    return {
        "feature_id": feature_id,
        "n": int(len(values)),
        "n_missing": int(values.isna().sum()),
        "n_zero": int((valid.abs() <= 1e-12).sum()),
        "n_unique": int(valid.nunique()),
        "min": float(quantiles.loc[0.0]),
        "p01": float(quantiles.loc[0.01]),
        "p05": float(quantiles.loc[0.05]),
        "q25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.5]),
        "q75": float(quantiles.loc[0.75]),
        "p95": float(quantiles.loc[0.95]),
        "p99": float(quantiles.loc[0.99]),
        "max": float(quantiles.loc[1.0]),
        "mean": float(valid.mean()),
        "sd": float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
    }


def support_tables(
    frame: gpd.GeoDataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    support_rows = [describe_feature(frame[feature], feature) for feature in features]
    support = pd.DataFrame(support_rows)

    group_frames: list[pd.DataFrame] = []
    for group_field, label in [("domain_abbreviation", "domain"), ("v14_fold_label", "v14_fold")]:
        for group_value, group in frame.groupby(group_field, dropna=False, sort=True):
            for feature in features:
                row = describe_feature(group[feature], feature)
                row[label] = str(group_value)
                row["n_support"] = int(len(group))
                row["n_full_support"] = int(group["full_support"].sum())
                row["n_edge_support"] = int(group["edge_hex"].sum())
                group_frames.append(row)
    by_group = pd.DataFrame(group_frames)
    by_domain = by_group[by_group["domain"].notna()].copy() if "domain" in by_group else pd.DataFrame()
    by_fold = by_group[by_group["v14_fold"].notna()].copy() if "v14_fold" in by_group else pd.DataFrame()
    return support, by_domain, by_fold


def correlation_table(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = frame[features].apply(pd.to_numeric, errors="coerce")
    pearson = data.corr(method="pearson")
    spearman = data.corr(method="spearman")
    rows: list[dict[str, Any]] = []
    for index_a, feature_a in enumerate(features):
        for feature_b in features[index_a + 1 :]:
            pair = data[[feature_a, feature_b]].dropna()
            rows.append(
                {
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "n": int(len(pair)),
                    "pearson_r": float(pearson.loc[feature_a, feature_b]),
                    "spearman_r": float(spearman.loc[feature_a, feature_b]),
                }
            )
    table = pd.DataFrame(rows)
    high_pairs = table[(table["pearson_r"].abs() >= 0.8) | (table["spearman_r"].abs() >= 0.8)].copy()
    high_pairs = high_pairs.sort_values("pearson_r", key=lambda values: values.abs(), ascending=False)
    return table, {
        "pearson_matrix": pearson.round(6).to_dict(),
        "spearman_matrix": spearman.round(6).to_dict(),
        "high_pairs": high_pairs.round(6).to_dict(orient="records"),
    }


def _safe_correlation(left: pd.Series, right: pd.Series, method: str) -> float | None:
    pair = pd.concat([left, right], axis=1).apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return None
    value = pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method)
    return None if pd.isna(value) else float(value)


def _pair_distance_metrics(
    frame: pd.DataFrame,
    base_feature: str,
    alternative_feature: str,
) -> dict[str, Any]:
    base = pd.to_numeric(frame[base_feature], errors="coerce")
    alternative = pd.to_numeric(frame[alternative_feature], errors="coerce")
    delta = alternative - base
    absolute = delta.abs().dropna()
    if absolute.empty:
        return {"n": 0}
    return {
        "n": int(len(absolute)),
        "pearson_r": _safe_correlation(base, alternative, "pearson"),
        "spearman_r": _safe_correlation(base, alternative, "spearman"),
        "median_signed_difference_km": float(delta.median()),
        "median_abs_difference_km": float(absolute.median()),
        "p95_abs_difference_km": float(absolute.quantile(0.95)),
        "max_abs_difference_km": float(absolute.max()),
        "n_alternative_shorter_gt_1m": int((delta < -0.001).sum()),
        "n_effect_within_1m": int((absolute <= 0.001).sum()),
        "n_alternative_longer_gt_1m": int((delta > 0.001).sum()),
    }


def confidence_sensitivity_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare the fixed HM, HIGH and ALL_NAMED distance representations."""

    comparisons = [
        ("HM_vs_HIGH", "STR_DIST_NAMED_HM_KM", "STR_DIST_NAMED_HIGH_KM"),
        ("HM_vs_ALL_NAMED", "STR_DIST_NAMED_HM_KM", "STR_DIST_NAMED_ALL_KM"),
        ("HIGH_vs_ALL_NAMED", "STR_DIST_NAMED_HIGH_KM", "STR_DIST_NAMED_ALL_KM"),
    ]
    scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "all_support", frame)]
    for fold_label, group in frame.groupby("v14_fold_label", sort=True):
        scopes.append(("v14_fold", str(fold_label), group))

    rows: list[dict[str, Any]] = []
    for scope_type, scope, scope_frame in scopes:
        for comparison, base_feature, alternative_feature in comparisons:
            metrics = _pair_distance_metrics(scope_frame, base_feature, alternative_feature)
            rows.append(
                {
                    "comparison": comparison,
                    "base_feature": base_feature,
                    "alternative_feature": alternative_feature,
                    "scope_type": scope_type,
                    "scope": scope,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def bridge_sensitivity_table(frame: pd.DataFrame, structures: dict[str, Any]) -> pd.DataFrame:
    """Summarise observed, bridged and parent-axis distances for bridged systems."""

    segments = structures["segments"]
    bridges = structures["bridges"]
    centrelines = structures["centrelines"]
    point_array = np.asarray(frame.geometry.values)
    scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "all_support", frame)]
    for fold_label, group in frame.groupby("v14_fold_label", sort=True):
        scopes.append(("v14_fold", str(fold_label), group))

    rows: list[dict[str, Any]] = []
    named = segments[segments["attributed_structure_id"].notna()].copy()
    for structure_id, group in named.groupby("attributed_structure_id", sort=True):
        structure_id = str(structure_id)
        system_bridges = bridges[bridges["structure_id"].astype(str) == structure_id].copy()
        if system_bridges.empty:
            continue
        parent_parts = centrelines[centrelines["structure_id"].astype(str) == structure_id]
        observed_geometry = unary_union(list(group.geometry))
        bridged_geometry = unary_union(list(group.geometry) + list(system_bridges.geometry))
        parent_geometry = (
            unary_union(list(parent_parts.geometry)) if not parent_parts.empty else observed_geometry
        )
        abbreviation = str(group["attributed_structure_abbreviation"].iloc[0])
        name = str(group["attributed_structure_name"].iloc[0])
        for scope_type, scope, scope_frame in scopes:
            subset = np.asarray(scope_frame.geometry.values)
            observed_km = np.asarray(shapely.distance(subset, observed_geometry)) / 1000.0
            bridged_km = np.asarray(shapely.distance(subset, bridged_geometry)) / 1000.0
            parent_km = np.asarray(shapely.distance(subset, parent_geometry)) / 1000.0
            bridge_effect = observed_km - bridged_km
            parent_effect = observed_km - parent_km
            absolute_bridge = np.abs(bridge_effect)
            rows.append(
                {
                    "structure_id": structure_id,
                    "abbreviation": abbreviation,
                    "structure_name": name,
                    "scope_type": scope_type,
                    "scope": scope,
                    "n_support": int(len(scope_frame)),
                    "n_bridges": int(len(system_bridges)),
                    "bridge_length_km": float(system_bridges.geometry.length.sum() / 1000.0),
                    "median_observed_km": float(np.median(observed_km)),
                    "median_bridged_km": float(np.median(bridged_km)),
                    "median_parent_axis_km": float(np.median(parent_km)),
                    "median_abs_bridge_effect_km": float(np.median(absolute_bridge)),
                    "p95_abs_bridge_effect_km": float(np.quantile(absolute_bridge, 0.95)),
                    "max_bridge_effect_km": float(absolute_bridge.max()),
                    "n_bridge_closer_gt_1m": int((bridge_effect > 0.001).sum()),
                    "n_effect_within_1m": int((absolute_bridge <= 0.001).sum()),
                    "median_bridge_effect_km": float(np.median(bridge_effect)),
                    "median_parent_axis_effect_km": float(np.median(parent_effect)),
                    "n_parent_axis_closer_gt_1m": int((parent_effect > 0.001).sum()),
                }
            )
    return pd.DataFrame(rows)


def _redundancy_clusters(features: list[str], correlations: pd.DataFrame) -> dict[str, str]:
    adjacency = {feature: set() for feature in features}
    for row in correlations.itertuples(index=False):
        if abs(row.pearson_r) >= 0.8 or abs(row.spearman_r) >= 0.8:
            adjacency[row.feature_a].add(row.feature_b)
            adjacency[row.feature_b].add(row.feature_a)

    assignments: dict[str, str] = {}
    cluster_index = 0
    for feature in features:
        if feature in assignments:
            continue
        cluster_index += 1
        cluster_id = f"RC{cluster_index:02d}"
        stack = [feature]
        while stack:
            current = stack.pop()
            if current in assignments:
                continue
            assignments[current] = cluster_id
            stack.extend(sorted(adjacency[current], reverse=True))
    return assignments


def predictor_shortlist(
    features: list[str],
    correlations: pd.DataFrame,
) -> pd.DataFrame:
    """Assign fixed, occurrence-blind roles and flag redundant feature groups."""

    primary_candidates = {
        "STR_DIST_NAMED_HM_KM": (
            "primary_candidate_distance",
            "Primary observed named-structure proximity; use as the principal structural distance candidate.",
        ),
        "STR_NAMED_LINE_DENSITY_25KM": (
            "primary_candidate_density_alternative",
            "Primary district-scale density representation; compare with HM distance rather than assuming both are independent.",
        ),
    }
    sensitivity_variants = {
        "STR_DIST_NAMED_HIGH_KM": "Confidence sensitivity: excludes medium-confidence segments.",
        "STR_DIST_NAMED_ALL_KM": "Confidence sensitivity: includes low-confidence support and disconnected fragments.",
        "STR_DIST_NAMED_BRIDGED_KM": "Geometry sensitivity: includes inferred HM-to-HM bridge connectors.",
        "STR_DIST_NAMED_PARENT_AXIS_KM": "Geometry sensitivity: reconstructed parent axes rather than observed segments.",
        "STR_DIST_NAMED_BRIDGED_DELTA_KM": "Diagnostic only: quantifies bridge influence on nearest distance.",
        "STR_DIST_NAMED_PARENT_AXIS_DELTA_KM": "Diagnostic only: quantifies parent-axis influence on nearest distance.",
        "STR_NAMED_LINE_DENSITY_10KM": "Scale sensitivity for structural density.",
        "STR_NAMED_LINE_DENSITY_50KM": "Scale sensitivity for structural density.",
        "STR_UNNAMED_LINE_DENSITY_25KM": "Mapping-texture and source-coverage sensitivity; not a primary geological predictor.",
    }
    context_only = {
        "STR_DIST_MAJOR_INTERSECTION_KM": "Context only: two candidate cartographic contacts, not established geological nodes.",
        "STR_MAJOR_INTERSECTION_COUNT_25KM": "Context only: maximum possible count is two; not an intersection-density measure.",
        "STR_DIST_DALY_DOMAIN_BOUNDARY_KM": "Context only: Daly class boundaries are cartographic, not mapped geological contacts.",
        "STR_NAMED_SYSTEM_COUNT_25KM": "Context only: counts only the fixed HM named-system set, not all faults.",
    }

    cluster_assignments = _redundancy_clusters(features, correlations)
    rows: list[dict[str, Any]] = []
    for feature in features:
        if feature in primary_candidates:
            role, rationale = primary_candidates[feature]
            proposed_use = "candidate"
        elif feature in sensitivity_variants:
            role, rationale = "sensitivity_variant", sensitivity_variants[feature]
            proposed_use = "sensitivity"
        elif feature in context_only:
            role, rationale = "context_only", context_only[feature]
            proposed_use = "context"
        elif feature.startswith("STR_DIST_") and feature.endswith("_KM"):
            role = "exploratory_parent_system"
            rationale = "Exploratory per-system distance; do not enter all system distances simultaneously."
            proposed_use = "exploratory"
        else:
            role, rationale = "unclassified", "Role must be assigned explicitly before modelling."
            proposed_use = "hold"

        related = correlations[
            (correlations["feature_a"] == feature) | (correlations["feature_b"] == feature)
        ].copy()
        high_related = related[
            (related["pearson_r"].abs() >= 0.8) | (related["spearman_r"].abs() >= 0.8)
        ].copy()
        high_partners = []
        for row in high_related.itertuples(index=False):
            partner = row.feature_b if row.feature_a == feature else row.feature_a
            high_partners.append(str(partner))
        cluster_id = cluster_assignments[feature]
        cluster_members = sorted(
            member for member, assigned in cluster_assignments.items() if assigned == cluster_id
        )
        rows.append(
            {
                "feature_id": feature,
                "shortlist_role": role,
                "proposed_model_use": proposed_use,
                "selection_basis": "fixed feature definition; no occurrence or validation performance used",
                "rationale": rationale,
                "redundancy_cluster_id": cluster_id,
                "redundancy_cluster_size": len(cluster_members),
                "redundancy_cluster_members": ";".join(cluster_members),
                "high_correlation_partners": ";".join(sorted(high_partners)),
                "max_abs_pearson": float(related["pearson_r"].abs().max()) if not related.empty else None,
                "max_abs_spearman": float(related["spearman_r"].abs().max()) if not related.empty else None,
            }
        )
    return pd.DataFrame(rows)


def _frame_outline(frame_cells: gpd.GeoDataFrame) -> Any:
    return unary_union(list(frame_cells.geometry))


def _map_extent(polygons: gpd.GeoDataFrame, padding_fraction: float = 0.03) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = polygons.total_bounds
    padding = max(maxx - minx, maxy - miny) * padding_fraction
    return minx - padding, miny - padding, maxx + padding, maxy + padding


def _extent_including_points(
    polygons: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
    padding_fraction: float = 0.03,
) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = polygons.total_bounds
    minx = min(minx, float(points.geometry.x.min()))
    miny = min(miny, float(points.geometry.y.min()))
    maxx = max(maxx, float(points.geometry.x.max()))
    maxy = max(maxy, float(points.geometry.y.max()))
    padding = max(maxx - minx, maxy - miny) * padding_fraction
    return minx - padding, miny - padding, maxx + padding, maxy + padding


def _label_domains(ax: plt.Axes, domains: gpd.GeoDataFrame) -> None:
    for row in domains.itertuples():
        point = row.geometry.representative_point()
        ax.text(
            point.x,
            point.y,
            str(row.abbreviation),
            fontsize=6.5,
            color="#4c4c4c",
            ha="center",
            va="center",
            alpha=0.9,
            zorder=2,
        )


def _north_arrow_and_scale(ax: plt.Axes, extent: tuple[float, float, float, float]) -> None:
    minx, miny, maxx, maxy = extent
    width = maxx - minx
    height = maxy - miny
    ax.annotate(
        "N",
        xy=(0.965, 0.955),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        zorder=20,
    )
    ax.annotate(
        "",
        xy=(0.965, 0.935),
        xytext=(0.965, 0.865),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "color": "#222222", "lw": 1.2},
        zorder=20,
    )
    scale_km = 50.0
    x0 = minx + 0.065 * width
    y0 = miny + 0.045 * height
    ax.plot(
        [x0, x0 + scale_km * 1000.0],
        [y0, y0],
        color="#222222",
        linewidth=3.0,
        solid_capstyle="butt",
        zorder=20,
    )
    ax.plot([x0, x0], [y0 - 0.004 * height, y0 + 0.004 * height], color="#222222", lw=1.0, zorder=20)
    ax.plot(
        [x0 + scale_km * 1000.0, x0 + scale_km * 1000.0],
        [y0 - 0.004 * height, y0 + 0.004 * height],
        color="#222222",
        lw=1.0,
        zorder=20,
    )
    ax.text(
        x0 + scale_km * 500.0,
        y0 + 0.018 * height,
        f"{int(scale_km)} km",
        ha="center",
        va="bottom",
        fontsize=7,
        color="#222222",
        zorder=20,
    )


def _prepare_map_axes(
    ax: plt.Axes,
    outline: Any,
    domains: gpd.GeoDataFrame,
    extent: tuple[float, float, float, float],
    *,
    label_domains: bool = True,
) -> None:
    gpd.GeoSeries([outline], crs=PROJECT_CRS).boundary.plot(
        ax=ax, color="#2f2f2f", linewidth=0.8, zorder=3
    )
    domains.boundary.plot(ax=ax, color="#b5b5b5", linewidth=0.35, zorder=1)
    if label_domains:
        _label_domains(ax, domains)
    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    ax.set_aspect("equal")
    ax.set_axis_off()
    _north_arrow_and_scale(ax, extent)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_support_coverage(
    polygons: gpd.GeoDataFrame,
    outline: Any,
    domains: gpd.GeoDataFrame,
    extent: tuple[float, float, float, float],
    path: Path,
    support_summary: dict[str, Any],
) -> None:
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    cmap = LinearSegmentedColormap.from_list(
        "overlap",
        ["#d73027", "#fee08b", "#66bd63", "#1a9850"],
    )
    fig, ax = plt.subplots(figsize=(9.0, 8.6))
    norm = Normalize(vmin=0.0, vmax=1.0)
    polygons.plot(
        column="frame_overlap_fraction",
        cmap=cmap,
        norm=norm,
        ax=ax,
        edgecolor="white",
        linewidth=0.12,
        alpha=0.9,
        zorder=0,
    )
    _prepare_map_axes(ax, outline, domains, extent, label_domains=True)
    ax.scatter(
        polygons.geometry.centroid.x,
        polygons.geometry.centroid.y,
        s=0.8,
        c="#222222",
        alpha=0.45,
        linewidths=0,
        zorder=5,
    )
    ax.set_title(
        "Equal-area hexagonal support on the Copperbelt tract footprint",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    ax.text(
        0.0,
        1.01,
        "Full 10 km$^2$ hexagons; colour is the fraction inside the historical tract footprint.",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        color="#333333",
    )
    n_total = int(support_summary["n_support"])
    n_full = int(support_summary["n_full_support"])
    n_80 = int(support_summary["n_overlap_ge_80pct"])
    ax.text(
        0.01,
        0.02,
        f"n = {n_total:,} support cells\n"
        f"{n_full:,} cells >= 99% inside footprint\n"
        f"{n_80:,} cells >= 80% inside footprint\n"
        "Clipped overlap is diagnostic only; model support remains equal-area.",
        transform=ax.transAxes,
        fontsize=7.2,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.88, "pad": 5},
        zorder=30,
    )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.035, pad=0.02, aspect=35)
    cbar.set_label("Fraction of full 10 km$^2$ hexagon inside tract footprint", fontsize=8)
    _save_figure(fig, path)


def plot_feature_map(
    points: gpd.GeoDataFrame,
    structures: dict[str, Any],
    outline: Any,
    domains: gpd.GeoDataFrame,
    extent: tuple[float, float, float, float],
    path: Path,
    feature: str,
    title: str,
    colorbar_label: str,
    *,
    cmap: str,
    show_candidate_intersections: bool = False,
    subtitle: str | None = None,
) -> None:
    from matplotlib.colors import Normalize

    values = pd.to_numeric(points[feature], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"Cannot map a feature with no finite values: {feature}")
    vmin = float(np.floor(np.nanpercentile(finite, 0.5)))
    vmax = float(np.ceil(np.nanpercentile(finite, 99.5)))
    if vmax <= vmin:
        vmax = vmin + 1.0
    fig, ax = plt.subplots(figsize=(9.0, 8.6))
    if feature != "STR_UNNAMED_LINE_DENSITY_25KM":
        structures["unnamed"].plot(ax=ax, color="#b8b8b8", linewidth=0.35, alpha=0.55, zorder=1)
    structures["hm"].plot(ax=ax, color="#3b2a2a", linewidth=0.75, alpha=0.9, zorder=4)
    scatter = ax.scatter(
        points["x_m"],
        points["y_m"],
        c=values,
        s=18,
        marker="h",
        cmap=cmap,
        norm=Normalize(vmin=vmin, vmax=vmax),
        linewidths=0,
        alpha=0.95,
        zorder=5,
    )
    _prepare_map_axes(ax, outline, domains, extent, label_domains=True)
    if show_candidate_intersections:
        intersections = structures["intersections"]
        ax.scatter(
            intersections.geometry.x,
            intersections.geometry.y,
            marker="*",
            s=190,
            facecolor="#fff200",
            edgecolor="#111111",
            linewidth=0.7,
            zorder=25,
        )
        for row in intersections.itertuples():
            short_label = str(row.pair_label).replace("Kafue", "Kf").replace("Mubulashi", "Mb")
            short_label = short_label.replace("Kanyama", "Ky").replace("Lwawa", "LB")
            ax.annotate(
                short_label,
                xy=(row.x_m, row.y_m),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=6.5,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
                zorder=26,
            )
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    if subtitle:
        ax.text(
            0.0,
            1.01,
            subtitle,
            transform=ax.transAxes,
            fontsize=7.8,
            va="bottom",
            color="#333333",
        )
    n_clipped = int((finite > vmax).sum())
    ax.text(
        0.01,
        0.02,
        f"n = {len(points):,} hexagon-centroid support points"
        + (f"; {n_clipped} values above colour limit" if n_clipped else ""),
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85, "pad": 4},
        zorder=30,
    )
    fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02, label=colorbar_label)
    _save_figure(fig, path)


def plot_feature_distributions(
    points: gpd.GeoDataFrame,
    dictionary: pd.DataFrame,
    path: Path,
) -> None:
    selected = [
        "STR_DIST_NAMED_HM_KM",
        "STR_DIST_NAMED_HIGH_KM",
        "STR_DIST_NAMED_ALL_KM",
        "STR_DIST_NAMED_BRIDGED_KM",
        "STR_DIST_NAMED_PARENT_AXIS_KM",
        "STR_NAMED_LINE_DENSITY_10KM",
        "STR_NAMED_LINE_DENSITY_25KM",
        "STR_NAMED_LINE_DENSITY_50KM",
        "STR_NAMED_SYSTEM_COUNT_25KM",
        "STR_DIST_MAJOR_INTERSECTION_KM",
        "STR_MAJOR_INTERSECTION_COUNT_25KM",
        "STR_DIST_DALY_DOMAIN_BOUNDARY_KM",
        "STR_UNNAMED_LINE_DENSITY_25KM",
    ]
    selected = [feature for feature in selected if feature in points.columns]
    labels = dict(zip(dictionary["feature_id"], dictionary["definition"]))
    units = dict(zip(dictionary["feature_id"], dictionary["units"]))
    figure, axes = plt.subplots(4, 4, figsize=(15.0, 11.0))
    axes = axes.flatten()
    for ax, feature in zip(axes, selected):
        values = pd.to_numeric(points[feature], errors="coerce").dropna().to_numpy(dtype=float)
        discrete = feature in {"STR_NAMED_SYSTEM_COUNT_25KM", "STR_MAJOR_INTERSECTION_COUNT_25KM"}
        if discrete:
            lower = int(np.floor(values.min()))
            upper = int(np.ceil(values.max()))
            bins: Any = np.arange(lower - 0.5, upper + 1.5, 1.0)
        else:
            bins = 45
        ax.hist(values, bins=bins, color="#4b7f9f", edgecolor="white", linewidth=0.3)
        ax.axvline(float(np.median(values)), color="#c0392b", linewidth=1.3)
        if discrete:
            ax.set_xticks(np.arange(lower, upper + 1, 1))
        ax.set_title(feature.replace("STR_", ""), fontsize=8.5, loc="left", fontweight="bold")
        ax.text(
            0.01,
            0.98,
            f"median = {np.median(values):.2f} {units.get(feature, '')}\n"
            f"p95 = {np.quantile(values, 0.95):.2f}; n = {len(values):,}",
            transform=ax.transAxes,
            fontsize=6.5,
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2},
        )
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel(units.get(feature, ""), fontsize=7)
        ax.set_ylabel("support cells", fontsize=7)
    for ax in axes[len(selected) :]:
        ax.set_axis_off()
    figure.suptitle(
        "Distributions of structural predictors across hexagonal support",
        fontsize=13,
        x=0.01,
        y=0.995,
        ha="left",
    )
    figure.text(
        0.01,
        0.968,
        "Each panel uses all support points; red line is the median. No occurrence labels are used in these summaries.",
        fontsize=8,
        color="#444444",
    )
    figure.subplots_adjust(left=0.05, right=0.99, bottom=0.06, top=0.91, wspace=0.24, hspace=0.42)
    _save_figure(figure, path)


def _short_feature_label(feature: str) -> str:
    return (
        feature.removeprefix("STR_")
        .replace("DIST_NAMED_", "D_")
        .replace("LINE_DENSITY_", "L_")
        .replace("_KM", "")
        .replace("_25KM", "_25")
        .replace("_10KM", "_10")
        .replace("_50KM", "_50")
    )


def plot_correlation_heatmap(
    points: gpd.GeoDataFrame,
    features: list[str],
    path: Path,
) -> None:
    data = points[features].apply(pd.to_numeric, errors="coerce")
    pearson = data.corr(method="pearson")
    spearman = data.corr(method="spearman")
    labels = [_short_feature_label(feature) for feature in features]
    fig, axes = plt.subplots(1, 2, figsize=(22.0, 9.0))
    for ax, matrix, title in [
        (axes[0], pearson, "Pearson correlation"),
        (axes[1], spearman, "Spearman rank correlation"),
    ]:
        image = ax.imshow(
            matrix.to_numpy(),
            vmin=-1.0,
            vmax=1.0,
            cmap="RdBu_r",
            interpolation="nearest",
            aspect="auto",
        )
        ax.set_xticks(range(len(features)))
        ax.set_yticks(range(len(features)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6.0)
        ax.set_yticklabels(labels, fontsize=6.0)
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        ax.tick_params(length=0)
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.20, top=0.86, wspace=0.26)
    colorbar_axis = fig.add_axes([0.92, 0.22, 0.012, 0.55])
    fig.colorbar(image, cax=colorbar_axis, orientation="vertical", label="Correlation coefficient")
    fig.suptitle(
        "Structural-predictor correlation diagnostics",
        fontsize=13,
        x=0.01,
        y=0.995,
        ha="left",
    )
    fig.text(
        0.01,
        0.956,
        "High absolute correlations identify alternative and redundant representations; they are not model-selection evidence.",
        fontsize=8,
        color="#444444",
    )
    fig.text(
        0.01,
        0.928,
        "White or pale cells near zero indicate weak correlation; an undefined correlation would arise only from a constant feature.",
        fontsize=7.5,
        color="#666666",
    )
    _save_figure(fig, path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return relative_posix(value)
    return value


def _output_path(config: dict[str, Any], key: str) -> Path:
    value = Path(config["outputs"][key])
    if value.is_absolute() or len(value.parts) > 1:
        return _abs(value)
    return _abs(config["outputs"]["directory"]) / value


def _figure_path(config: dict[str, Any], key: str) -> Path:
    return _abs(config["outputs"]["figure_directory"]) / config["outputs"]["figures"][key]


def _source_record(path: str | Path, role: str, expected_sha256: str | None = None) -> dict[str, Any]:
    record = file_record(_abs(path), role=role)
    if expected_sha256 is not None:
        record["expected_sha256"] = expected_sha256
        record["hash_match"] = str(record["sha256"]).lower() == str(expected_sha256).lower()
    return record


def _assert_expected_hash(path: str | Path, expected: str, label: str) -> str:
    actual = sha256_file(_abs(path))
    if actual.lower() != str(expected).lower():
        raise RuntimeError(
            f"Source hash mismatch for {label}: expected {expected}, observed {actual} ({relative_posix(_abs(path))})"
        )
    return actual


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.9f",
        na_rep="",
    )


def _write_layers(path: Path, layers: list[tuple[str, gpd.GeoDataFrame]], overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {relative_posix(path)}. Use --overwrite to rebuild.")
        path.unlink()
    for index, (layer_name, frame) in enumerate(layers):
        pyogrio.write_dataframe(
            frame,
            path,
            layer=layer_name,
            driver="GPKG",
            append=index > 0,
        )


def _output_records(paths: list[tuple[Path, str]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path, role in paths:
        if not path.exists():
            raise FileNotFoundError(f"Expected output is missing: {relative_posix(path)}")
        records.append(file_record(path, role=role))
    return records


def _support_summary(
    points: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    support_config = config["support"]
    nominal_area = float(support_config["nominal_cell_area_km2"])
    area_km2 = polygons.geometry.area / 1e6
    overlap_fraction = polygons["frame_overlap_fraction"].astype(float)
    expected_n = int(support_config.get("expected_n_support", len(points)))
    expected_full = int(support_config.get("expected_n_full_support", -1))
    expected_80 = int(support_config.get("expected_n_overlap_ge_80pct", -1))
    if len(points) != expected_n:
        raise RuntimeError(f"Support count changed: expected {expected_n}, observed {len(points)}")
    n_full = int(polygons["full_support"].sum())
    n_80 = int((overlap_fraction >= float(support_config["edge_overlap_threshold"])).sum())
    if expected_full >= 0 and n_full != expected_full:
        raise RuntimeError(f"Full-support count changed: expected {expected_full}, observed {n_full}")
    if expected_80 >= 0 and n_80 != expected_80:
        raise RuntimeError(f"80%-overlap count changed: expected {expected_80}, observed {n_80}")
    if not np.allclose(area_km2.to_numpy(), nominal_area, atol=1e-6):
        raise RuntimeError("Support polygons are not equal-area")
    if not polygons.geometry.centroid.within(polygons.geometry).all():
        raise RuntimeError("A support centroid lies outside its full support hexagon")

    quantiles = overlap_fraction.quantile([0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    nearest_distance = points["v14_nearest_distance_m"].astype(float)
    return {
        "n_support": int(len(points)),
        "nominal_cell_area_km2": nominal_area,
        "nominal_total_area_km2": float(len(points) * nominal_area),
        "sum_frame_overlap_area_km2": float(polygons["frame_overlap_area_km2"].sum()),
        "n_full_support": n_full,
        "n_edge_support": int((~polygons["full_support"]).sum()),
        "n_overlap_ge_80pct": n_80,
        "full_support_threshold": float(support_config["full_support_threshold"]),
        "edge_overlap_threshold": float(support_config["edge_overlap_threshold"]),
        "frame_overlap_fraction_quantiles": {str(key): float(value) for key, value in quantiles.items()},
        "frame_overlap_fraction_min": float(overlap_fraction.min()),
        "frame_overlap_fraction_median": float(overlap_fraction.median()),
        "frame_overlap_fraction_max": float(overlap_fraction.max()),
        "n_unassigned_domains": int((points["domain_abbreviation"] == "unassigned").sum()),
        "n_domain_join_duplicates": int(points.attrs.get("domain_join_duplicate_count", 0)),
        "domain_counts": {str(key): int(value) for key, value in points["domain_abbreviation"].value_counts().sort_index().items()},
        "v14_fold_counts": {str(key): int(value) for key, value in points["v14_fold_label"].value_counts().sort_index().items()},
        "v14_nearest_distance_m": {
            "min": float(nearest_distance.min()),
            "median": float(nearest_distance.median()),
            "p95": float(nearest_distance.quantile(0.95)),
            "max": float(nearest_distance.max()),
        },
    }


def build(config_path: str | Path = "configs/structural_predictors_v1.json", *, overwrite: bool = False) -> dict[str, Any]:
    config_file = _abs(config_path)
    config = load_config(config_file)
    if config["project_crs"] != PROJECT_CRS:
        raise ValueError(f"Expected project CRS {PROJECT_CRS}, found {config['project_crs']}")

    expected_sources = {
        "frame": config["frame"]["expected_sha256"],
        "v14_crosswalk": config["v14_crosswalk"]["expected_sha256"],
        "structures": config["structures"]["expected_sha256"],
        "domains": config["domains"]["expected_sha256"],
        "boundaries": config["boundaries"]["expected_sha256"],
    }
    observed_expected: dict[str, str] = {}
    for key, value in expected_sources.items():
        source_section = config[key if key != "v14_crosswalk" else "v14_crosswalk"]
        observed_expected[key] = _assert_expected_hash(source_section["path"], value, key)
    protected_files = config["protected_files"]
    protected_hashes = config["protected_hashes"]
    protected_observed: dict[str, Any] = {}
    for key, path in protected_files.items():
        actual = _assert_expected_hash(path, protected_hashes[key], key)
        protected_observed[key] = {
            "path": relative_posix(_abs(path)),
            "expected_sha256": protected_hashes[key],
            "observed_sha256": actual,
            "hash_match": True,
        }

    output_gpkg = _output_path(config, "geopackage")
    if output_gpkg.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {relative_posix(output_gpkg)}. Use --overwrite to rebuild."
        )

    frame_cells = _read_gdf(config["frame"]["path"], config["frame"]["layer"])
    frame_id_field = config["frame"]["id_field"]
    frame_cells = frame_cells[[frame_id_field, "geometry"]].copy()
    if not frame_cells.geometry.is_valid.all():
        raise ValueError("The tract footprint contains invalid geometries")

    points, polygons, overlaps = build_support(
        frame_cells,
        nominal_area_km2=float(config["support"]["nominal_cell_area_km2"]),
        full_support_threshold=float(config["support"]["full_support_threshold"]),
    )
    domains = _read_gdf(config["domains"]["path"], config["domains"]["layer"])
    points = assign_domains(points, domains)
    v14_columns = list(config["v14_crosswalk"]["usecols"])
    v14 = pd.read_csv(_abs(config["v14_crosswalk"]["path"]), usecols=v14_columns)
    if list(v14.columns) != v14_columns:
        raise RuntimeError(f"V14 crosswalk columns changed: expected {v14_columns}, observed {list(v14.columns)}")
    points = add_v14_crosswalk(points, v14, int(config["v14_crosswalk"]["n_folds"]))

    for field in ["domain_id", "domain_abbreviation", "domain_name", "v14_fold", "v14_fold_label"]:
        mapping = points.set_index("support_id")[field]
        polygons[field] = polygons["support_id"].map(mapping)
        overlaps[field] = overlaps["support_id"].map(mapping)

    structures = load_structures(config)
    boundaries = _read_gdf(config["boundaries"]["path"], config["boundaries"]["layer"])
    feature_parameters = config["feature_parameters"]
    features = compute_features(
        points,
        structures,
        boundaries,
        radii_km=[int(radius) for radius in feature_parameters["density_radii_km"]],
        intersection_radius_km=int(feature_parameters["intersection_radius_km"]),
        system_count_radius_km=int(feature_parameters["system_count_radius_km"]),
    )
    feature_names = feature_columns(features)
    if len(feature_names) != 27:
        raise RuntimeError(f"Unexpected structural feature count: {len(feature_names)}")
    if features["STR_DIST_NAMED_HM_KM"].isna().any():
        raise RuntimeError("At least one support point has a missing HM distance")
    if (features["STR_DIST_NAMED_HM_KM"] < 0).any():
        raise RuntimeError("Negative HM distance encountered")

    dictionary = feature_dictionary(structures, [int(radius) for radius in feature_parameters["density_radii_km"]])
    if set(dictionary["feature_id"]) != set(feature_names):
        missing = sorted(set(feature_names).difference(dictionary["feature_id"]))
        extra = sorted(set(dictionary["feature_id"]).difference(feature_names))
        raise RuntimeError(f"Feature dictionary mismatch; missing={missing}; extra={extra}")

    support, by_domain, by_fold = support_tables(features, feature_names)
    correlations, correlation_summary = correlation_table(features, feature_names)
    confidence_sensitivity = confidence_sensitivity_table(features)
    bridge_sensitivity = bridge_sensitivity_table(features, structures)
    shortlist = predictor_shortlist(feature_names, correlations)
    if (shortlist["shortlist_role"] == "unclassified").any():
        unclassified = shortlist.loc[shortlist["shortlist_role"] == "unclassified", "feature_id"].tolist()
        raise RuntimeError(f"Unclassified structural features: {unclassified}")

    support_summary = _support_summary(points, polygons, config)
    outline = _frame_outline(frame_cells)
    map_extent = _map_extent(polygons)
    intersection_extent = _extent_including_points(polygons, structures["intersections"])
    point_array_for_candidates = np.asarray(points.geometry.values)
    candidate_distance_matrix = np.column_stack(
        [
            np.asarray(shapely.distance(point_array_for_candidates, row.geometry)) / 1000.0
            for row in structures["intersections"].itertuples()
        ]
    )
    nearest_candidate_index = np.argmin(candidate_distance_matrix, axis=1)
    nearest_candidate_counts = {
        str(structures["intersections"].iloc[index]["candidate_id"]): int((nearest_candidate_index == index).sum())
        for index in range(len(structures["intersections"]))
    }
    ky_lb_index = int(
        structures["intersections"].index[
            structures["intersections"]["pair_label"].str.contains("Kanyama", regex=False)
        ][0]
    )
    ky_lb_nearest_count = int((nearest_candidate_index == ky_lb_index).sum())

    output_directory = _abs(config["outputs"]["directory"])
    figure_directory = _abs(config["outputs"]["figure_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)

    feature_table_columns = [
        "support_id",
        "x_m",
        "y_m",
        "nominal_area_km2",
        "frame_overlap_area_km2",
        "frame_overlap_fraction",
        "full_support",
        "edge_hex",
        "domain_id",
        "domain_abbreviation",
        "domain_name",
        "v14_fold",
        "v14_fold_label",
        *feature_names,
        "occurrence_data_used",
    ]
    feature_table = features[feature_table_columns].copy()
    v14_crosswalk = features[
        ["support_id", "x_m", "y_m", "v14_nearest_id", "v14_nearest_distance_m", "v14_fold", "v14_fold_label"]
    ].copy()

    _write_layers(
        output_gpkg,
        [
            (config["outputs"]["layers"]["support_polygons"], polygons),
            (config["outputs"]["layers"]["support_points"], features),
            (config["outputs"]["layers"]["candidate_intersections"], structures["intersections"]),
        ],
        overwrite=overwrite,
    )

    table_paths: dict[str, Path] = {
        "feature_table": _output_path(config, "feature_table"),
        "feature_dictionary": _output_path(config, "feature_dictionary"),
        "feature_support": _output_path(config, "feature_support"),
        "feature_by_domain": _output_path(config, "feature_by_domain"),
        "feature_by_fold": _output_path(config, "feature_by_fold"),
        "correlations": _output_path(config, "correlations"),
        "bridge_sensitivity": _output_path(config, "bridge_sensitivity"),
        "confidence_sensitivity": _output_path(config, "confidence_sensitivity"),
        "v14_crosswalk": _output_path(config, "v14_crosswalk"),
        "shortlist": _output_path(config, "shortlist"),
        "candidate_intersections_csv": _output_path(config, "candidate_intersections_csv"),
    }
    _write_table(feature_table, table_paths["feature_table"])
    _write_table(dictionary, table_paths["feature_dictionary"])
    _write_table(support, table_paths["feature_support"])
    _write_table(by_domain, table_paths["feature_by_domain"])
    _write_table(by_fold, table_paths["feature_by_fold"])
    _write_table(correlations, table_paths["correlations"])
    _write_table(bridge_sensitivity, table_paths["bridge_sensitivity"])
    _write_table(confidence_sensitivity, table_paths["confidence_sensitivity"])
    _write_table(v14_crosswalk, table_paths["v14_crosswalk"])
    _write_table(shortlist, table_paths["shortlist"])
    _write_table(structures["intersections"].drop(columns="geometry"), table_paths["candidate_intersections_csv"])

    figure_paths: dict[str, Path] = {
        key: _figure_path(config, key)
        for key in [
            "support_coverage",
            "distance_hm",
            "density_named_25km",
            "intersection_distance",
            "domain_boundary_distance",
            "unnamed_density_25km",
            "feature_distributions",
            "correlation_heatmap",
        ]
    }
    plot_support_coverage(polygons, outline, domains, map_extent, figure_paths["support_coverage"], support_summary)
    plot_feature_map(
        features,
        structures,
        outline,
        domains,
        map_extent,
        figure_paths["distance_hm"],
        "STR_DIST_NAMED_HM_KM",
        "Distance to HM named major structures",
        "km (centroid sample; lower = closer)",
        cmap="viridis",
        subtitle="Primary structural-distance candidate. HM = named_high + named_medium observed linework.",
    )
    plot_feature_map(
        features,
        structures,
        outline,
        domains,
        map_extent,
        figure_paths["density_named_25km"],
        "STR_NAMED_LINE_DENSITY_25KM",
        "HM named-structure line density within 25 km",
        "km of line per 1,000 km$^2$",
        cmap="YlGnBu",
        subtitle="District-scale density alternative to nearest-structure distance; scales are not independent.",
    )
    plot_feature_map(
        features,
        structures,
        outline,
        domains,
        intersection_extent,
        figure_paths["intersection_distance"],
        "STR_DIST_MAJOR_INTERSECTION_KM",
        "Distance to candidate named-system contacts",
        "km; stars are the two candidate contacts",
        cmap="viridis",
        show_candidate_intersections=True,
        subtitle=(
            f"Nearest of two candidates. Ky-LB lies west of the tract and is nearest for {ky_lb_nearest_count} support cells; "
            "both are candidate cartographic contacts, not established geological nodes."
        ),
    )
    plot_feature_map(
        features,
        structures,
        outline,
        domains,
        map_extent,
        figure_paths["domain_boundary_distance"],
        "STR_DIST_DALY_DOMAIN_BOUNDARY_KM",
        "Distance to internal Daly domain boundaries",
        "km (cartographic class boundary)",
        cmap="magma",
        subtitle="Context only: Daly boundaries are reconstructed colour-class boundaries, not mapped geological contacts.",
    )
    plot_feature_map(
        features,
        structures,
        outline,
        domains,
        map_extent,
        figure_paths["unnamed_density_25km"],
        "STR_UNNAMED_LINE_DENSITY_25KM",
        "Unnamed heavy-linework density within 25 km",
        "km of line per 1,000 km$^2$",
        cmap="Greys",
        subtitle="Sensitivity for mapping texture and source coverage: measured linework is not a fault census.",
    )
    plot_feature_distributions(features, dictionary, figure_paths["feature_distributions"])
    plot_correlation_heatmap(features, feature_names, figure_paths["correlation_heatmap"])

    output_paths: list[tuple[Path, str]] = [
        (output_gpkg, "structural predictor GeoPackage"),
        (table_paths["feature_table"], "support-keyed structural feature table"),
        (table_paths["feature_dictionary"], "feature definitions and provenance"),
        (table_paths["feature_support"], "feature distributions over all support"),
        (table_paths["feature_by_domain"], "feature distributions by Daly domain"),
        (table_paths["feature_by_fold"], "feature distributions by historical V14 fold"),
        (table_paths["correlations"], "pairwise Pearson and Spearman correlations"),
        (table_paths["bridge_sensitivity"], "observed versus bridged and parent-axis sensitivity"),
        (table_paths["confidence_sensitivity"], "HM/HIGH/ALL_NAMED confidence sensitivity"),
        (table_paths["v14_crosswalk"], "support-to-V14 nearest-cell and fold crosswalk"),
        (table_paths["shortlist"], "fixed-role predictor shortlist and redundancy groups"),
        (table_paths["candidate_intersections_csv"], "candidate intersection qualifications"),
        (figure_paths["support_coverage"], "hexagonal support and footprint overlap map"),
        (figure_paths["distance_hm"], "HM nearest-distance map"),
        (figure_paths["density_named_25km"], "HM 25 km density map"),
        (figure_paths["intersection_distance"], "candidate intersection distance map"),
        (figure_paths["domain_boundary_distance"], "Daly boundary distance map"),
        (figure_paths["unnamed_density_25km"], "unnamed linework density map"),
        (figure_paths["feature_distributions"], "feature distribution panel"),
        (figure_paths["correlation_heatmap"], "feature correlation heatmap"),
    ]
    output_records = _output_records(output_paths)

    intersection_records = structures["intersections"].drop(columns="geometry").to_dict(orient="records")
    role_counts = shortlist["shortlist_role"].value_counts().sort_index().to_dict()
    run_id = build_id()
    manifest = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "build_id": run_id,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "built",
        "purpose": "Structural predictors on a rebuilt equal-area hexagonal support; no model fitted.",
        "config": _source_record(config_file, "build configuration"),
        "builder": _source_record(Path(__file__), "feature-builder source"),
        "sources": [
            _source_record(config["frame"]["path"], "tract footprint geometry only", config["frame"]["expected_sha256"]),
            _source_record(config["domains"]["path"], "Daly tectonic domains for assignment and context", config["domains"]["expected_sha256"]),
            _source_record(config["boundaries"]["path"], "internal Daly boundaries for context distance", config["boundaries"]["expected_sha256"]),
            _source_record(config["structures"]["path"], "named major-structure source", config["structures"]["expected_sha256"]),
            _source_record(config["v14_crosswalk"]["path"], "V14 ids, centroids and fold crosswalk only", config["v14_crosswalk"]["expected_sha256"]),
        ],
        "protected": protected_observed,
        "input_fields": {
            "frame": [frame_id_field, "geometry"],
            "v14_crosswalk": v14_columns,
            "v14_occurrence_fields_read": [],
            "structures_layers": [
                config["structures"]["segments_layer"],
                config["structures"]["centrelines_layer"],
                config["structures"]["bridges_layer"],
                config["structures"]["catalogue_layer"],
            ],
            "domains": ["domain_id", "abbreviation", "domain_name", "geometry"],
            "boundaries": ["boundary_type", "geometry"],
        },
        "support": support_summary,
        "features": {
            "n_features": len(feature_names),
            "feature_ids": feature_names,
            "n_missing_values": int(feature_table[feature_names].isna().sum().sum()),
            "n_support": int(len(feature_table)),
            "density_radii_km": feature_parameters["density_radii_km"],
            "intersection_radius_km": feature_parameters["intersection_radius_km"],
            "system_count_radius_km": feature_parameters["system_count_radius_km"],
            "role_counts": role_counts,
            "n_high_correlation_pairs": len(correlation_summary["high_pairs"]),
        },
        "named_structures": {
            "n_source_segments": int(len(structures["segments"])),
            "n_hm_segments": int(len(structures["hm"])),
            "n_high_segments": int(len(structures["high"])),
            "n_all_named_segments": int(len(structures["all_named"])),
            "n_unnamed_segments": int(len(structures["unnamed"])),
            "n_hm_systems": int(len(structures["hm_system_unions"])),
            "n_named_systems_with_geometry": int(len(structures["system_unions"])),
            "n_hm_bridges": int(structures["hm_bridge_count"]),
        },
        "candidate_intersections": _json_safe(intersection_records),
        "candidate_intersection_nearest_counts": nearest_candidate_counts,
        "layers": {
            config["outputs"]["layers"]["support_polygons"]: {
                "rows": int(len(polygons)),
                "geometry_type": "Polygon",
                "crs": PROJECT_CRS,
                "note": "Full equal-area support geometry; clipped overlap retained only as diagnostic fields.",
            },
            config["outputs"]["layers"]["support_points"]: {
                "rows": int(len(features)),
                "geometry_type": "Point",
                "crs": PROJECT_CRS,
                "note": "Hexagon centroids carrying the structural feature values.",
            },
            config["outputs"]["layers"]["candidate_intersections"]: {
                "rows": int(len(structures["intersections"])),
                "geometry_type": "Point",
                "crs": PROJECT_CRS,
                "note": "Cartographic candidate contacts, not established geological nodes.",
            },
        },
        "outputs": output_records,
        "software": software_versions(),
        "occurrence_data_used": False,
        "model_fitted": False,
        "limitations": [
            "Structural geometry is a cartographic reconstruction from Daly et al. (2025); it is not a field-validated fault map.",
            "The heavy printed line is a fault-zone symbol; its width is not physical fault-zone width or displacement.",
            "Bridge connectors and parent axes are reconstruction aids and are restricted to sensitivity variants.",
            "Only two candidate named-system contacts exist; they are not established geological nodes.",
            "Named and unnamed line density partly reflects map compilation and source coverage.",
            "Distances are centroid samples on equal-area hexagons, not polygon-to-line areal averages.",
            "Distances to mapped features do not establish geological absence where mapping coverage is incomplete.",
            "The V14 four-fold partitions are used only as a historical crosswalk; they are not evidence of geographic transferability.",
        ],
    }
    manifest_path = _output_path(config, "manifest")
    write_json(manifest_path, _json_safe(manifest))

    print(f"Structural predictor build complete: {run_id}")
    print(f"Support: {len(points):,} equal-area hexagon centroids; full support {support_summary['n_full_support']:,}")
    print(f"Features: {len(feature_names)}; missing values: {manifest['features']['n_missing_values']}")
    print(f"GeoPackage: {relative_posix(output_gpkg)}")
    print(f"Manifest: {relative_posix(manifest_path)}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/structural_predictors_v1.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    build(args.config, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
