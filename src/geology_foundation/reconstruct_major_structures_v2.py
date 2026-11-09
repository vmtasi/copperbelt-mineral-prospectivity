"""Reconstruct named Daly et al. (2025) major structures from the v1 extraction.

The v1 layer is read-only. This module writes a new v2 GeoPackage containing:

* every original v1 segment, unchanged, with attribution fields appended;
* approximate parent-axis centrelines built only from ordered printed components;
* explicit straight bridge connectors for cartographic gaps;
* a 15-name catalogue, including names with no defensible geometry;
* segment, structure and provenance tables plus cartographic review maps.

No mineral-occurrence locations are read or used.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import pyogrio
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pyproj import Transformer
from shapely.geometry import LineString, Point

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import (
    file_record,
    load_config,
    relative_posix,
    resolve_path,
    sha256_file,
    software_versions,
    write_csv,
    write_json,
)
from .fig12 import geometry_to_pixels, pixel_to_geographic


FIGURE12 = "Daly et al. (2025) Figure 12"
FIGURE14 = "Daly et al. (2025) Figure 14"
SOURCE_DOI = "10.1029/2025TC008955"
PROJECT_CRS = "EPSG:32735"
GEOGRAPHIC_CRS = "EPSG:4326"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _label_point(anchor_px: list[float], georeference: dict[str, Any]) -> tuple[float, float, float, float]:
    longitude, latitude = pixel_to_geographic(float(anchor_px[0]), float(anchor_px[1]), georeference)
    transformer = Transformer.from_crs(GEOGRAPHIC_CRS, PROJECT_CRS, always_xy=True)
    x, y = transformer.transform(longitude, latitude)
    return float(x), float(y), float(longitude), float(latitude)


def _segment_label_distance_px(
    geometry: LineString,
    anchor_px: Iterable[float],
    georeference: dict[str, Any],
) -> float:
    point = Point(float(anchor_px[0]), float(anchor_px[1]))
    distances = []
    for part in geometry_to_pixels(geometry, georeference):
        if len(part) >= 2:
            distances.append(point.distance(LineString(part)))
    return float(min(distances)) if distances else float("nan")


def _oriented_coords(geometry: LineString, reverse: bool) -> list[tuple[float, float]]:
    coords = [(float(x), float(y)) for x, y in geometry.coords]
    return list(reversed(coords)) if reverse else coords


def _endpoint_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _build_chain(
    part: dict[str, Any],
    segments_by_id: dict[str, LineString],
    *,
    structure_id: str,
    contiguous_tolerance_m: float,
) -> tuple[LineString, list[dict[str, Any]], dict[str, Any]]:
    """Orient and join ordered printed components without altering any source feature."""

    segment_ids = list(part["ordered_segment_ids"])
    if not segment_ids:
        raise ValueError(f"Centerline part {part['part_id']} has no segments")
    missing = [segment_id for segment_id in segment_ids if segment_id not in segments_by_id]
    if missing:
        raise KeyError(f"Centerline part {part['part_id']} references missing segments: {missing}")

    all_coords: list[tuple[float, float]] = []
    bridges: list[dict[str, Any]] = []
    previous_endpoint: tuple[float, float] | None = None
    previous_segment_id: str | None = None
    mapped_length_m = 0.0
    bridge_length_m = 0.0

    for index, segment_id in enumerate(segment_ids):
        geometry = segments_by_id[segment_id]
        coords = _oriented_coords(geometry, reverse=False)
        mapped_length_m += float(geometry.length)

        if index == 0:
            if len(segment_ids) > 1:
                next_geometry = segments_by_id[segment_ids[1]]
                next_coords = _oriented_coords(next_geometry, reverse=False)
                forward_gap = min(
                    _endpoint_distance(coords[-1], next_coords[0]),
                    _endpoint_distance(coords[-1], next_coords[-1]),
                )
                reverse_gap = min(
                    _endpoint_distance(coords[0], next_coords[0]),
                    _endpoint_distance(coords[0], next_coords[-1]),
                )
                if reverse_gap < forward_gap:
                    coords = list(reversed(coords))
            all_coords.extend(coords)
        else:
            if previous_endpoint is None or previous_segment_id is None:
                raise RuntimeError("Previous endpoint was not initialised")
            forward_gap = _endpoint_distance(previous_endpoint, coords[0])
            reverse_gap = _endpoint_distance(previous_endpoint, coords[-1])
            if reverse_gap < forward_gap:
                coords = list(reversed(coords))
            gap_m = _endpoint_distance(previous_endpoint, coords[0])
            if gap_m > contiguous_tolerance_m:
                bridges.append(
                    {
                        "structure_id": structure_id,
                        "part_id": part["part_id"],
                        "from_segment_id": previous_segment_id,
                        "to_segment_id": segment_id,
                        "from_x": previous_endpoint[0],
                        "from_y": previous_endpoint[1],
                        "to_x": coords[0][0],
                        "to_y": coords[0][1],
                        "gap_m": gap_m,
                        "gap_km": gap_m / 1000.0,
                        "geometry": LineString([previous_endpoint, coords[0]]),
                    }
                )
                bridge_length_m += gap_m
            if _endpoint_distance(all_coords[-1], coords[0]) > 0.0:
                all_coords.append(coords[0])
            all_coords.extend(coords[1:])

        previous_endpoint = all_coords[-1]
        previous_segment_id = segment_id

    centerline = LineString(all_coords)
    if centerline.is_empty or centerline.length <= 0.0:
        raise ValueError(f"Centerline part {part['part_id']} is empty")
    metrics = {
        "n_segments": len(segment_ids),
        "mapped_length_km": mapped_length_m / 1000.0,
        "bridge_length_km": bridge_length_m / 1000.0,
        "total_length_km": float(centerline.length) / 1000.0,
        "n_bridges": len(bridges),
        "max_bridge_km": max((bridge["gap_km"] for bridge in bridges), default=0.0),
    }
    return centerline, bridges, metrics


def _category_row(
    structure: dict[str, Any],
    *,
    distances: dict[str, float],
    centerline_metrics: list[dict[str, Any]],
    bridge_count: int,
    bridge_length_km: float,
    geometry: Point,
    anchor_xy_m: tuple[float, float],
    anchor_lon_lat: tuple[float, float],
) -> dict[str, Any]:
    attributed_segments = {
        segment_id
        for part in structure.get("centerline_parts", [])
        for segment_id in part.get("ordered_segment_ids", [])
    }
    supporting_segments = {
        item["segment_id"] for item in structure.get("supporting_segments", [])
    }
    mapped_length = sum(item["mapped_length_km"] for item in centerline_metrics)
    centerline_length = sum(item["total_length_km"] for item in centerline_metrics)
    aliases = structure.get("aliases", [])
    return {
        "structure_id": structure["structure_id"],
        "abbreviation": structure["abbreviation"],
        "structure_name": structure["name"],
        "aliases": "|".join(aliases),
        "attribution_status": "named" if structure.get("centerline_parts") else "unresolved_reference_only",
        "attribution_confidence": structure["attribution_confidence"],
        "geometry_status": structure["geometry_status"],
        "n_attributed_segments": len(attributed_segments) + len(supporting_segments),
        "n_centerline_parts": len(structure.get("centerline_parts", [])),
        "n_supporting_segments": len(supporting_segments),
        "mapped_length_km": round(mapped_length, 3),
        "bridge_length_km": round(bridge_length_km, 3),
        "centerline_length_km": round(centerline_length, 3),
        "n_bridges": int(bridge_count),
        "max_bridge_km": round(max((item["max_bridge_km"] for item in centerline_metrics), default=0.0), 3),
        "candidate_segments_not_assigned": len(structure.get("candidate_segments_not_assigned", [])),
        "label_anchor_key": structure["label_anchor_key"],
        "label_anchor_px_x": float(structure["_anchor_px"][0]),
        "label_anchor_px_y": float(structure["_anchor_px"][1]),
        "label_longitude": anchor_lon_lat[0],
        "label_latitude": anchor_lon_lat[1],
        "label_x_m": anchor_xy_m[0],
        "label_y_m": anchor_xy_m[1],
        "source_name_figure": FIGURE12,
        "source_geometry_figure": FIGURE14,
        "source_doi": SOURCE_DOI,
        "figure14_cross_check": structure["figure14_cross_check"],
        "published_context": structure["published_context"],
        "review_status": "reviewed_cartographic_attribution",
        "occurrence_evidence_used": "no",
        "notes": " ".join(structure.get("notes", [])),
        "geometry": geometry,
    }


def _validate_evidence(
    evidence: dict[str, Any],
    v1: gpd.GeoDataFrame,
) -> list[dict[str, Any]]:
    structures = evidence.get("structures", [])
    if len(structures) != 15:
        raise ValueError(f"Expected all 15 published names, found {len(structures)}")
    abbreviations = [item["abbreviation"] for item in structures]
    names = [item["name"] for item in structures]
    structure_ids = [item["structure_id"] for item in structures]
    if len(set(abbreviations)) != 15:
        raise ValueError(f"Published abbreviations are not unique: {abbreviations}")
    if len(set(names)) != 15:
        raise ValueError(f"Published names are not unique: {names}")
    if len(set(structure_ids)) != 15:
        raise ValueError(f"Structure identifiers are not unique: {structure_ids}")

    known_segments = set(v1["structure_id"].astype(str))
    accepted: list[str] = []
    anchors = evidence["label_anchors_px"]
    for structure in structures:
        for key in [structure["label_anchor_key"], *structure.get("additional_label_anchor_keys", [])]:
            if key not in anchors:
                raise KeyError(f"Missing label anchor {key!r} for {structure['structure_id']}")
        for part in structure.get("centerline_parts", []):
            for segment_id in part.get("ordered_segment_ids", []):
                if segment_id not in known_segments:
                    raise KeyError(f"Unknown segment {segment_id} in {structure['structure_id']}")
                accepted.append(segment_id)
        for supporting in structure.get("supporting_segments", []):
            segment_id = supporting["segment_id"]
            if segment_id not in known_segments:
                raise KeyError(f"Unknown supporting segment {segment_id}")
            accepted.append(segment_id)
        for candidate in structure.get("candidate_segments_not_assigned", []):
            segment_id = candidate["segment_id"]
            if segment_id not in known_segments:
                raise KeyError(f"Unknown candidate segment {segment_id}")

    duplicates = sorted(segment_id for segment_id, count in Counter(accepted).items() if count > 1)
    if duplicates:
        raise ValueError(f"Segments assigned more than once: {duplicates}")
    return structures


def _build_products(
    v1: gpd.GeoDataFrame,
    evidence: dict[str, Any],
    georeference: dict[str, Any],
    contiguous_tolerance_m: float,
) -> dict[str, Any]:
    structures = _validate_evidence(evidence, v1)
    segments_by_id = {
        str(row.structure_id): row.geometry for row in v1.itertuples(index=False)
    }
    anchor_xy: dict[str, tuple[float, float, float, float]] = {}
    for key, pixel in evidence["label_anchors_px"].items():
        anchor_xy[key] = _label_point(pixel, georeference)

    segment_assignments: dict[str, dict[str, Any]] = {}
    centerline_rows: list[dict[str, Any]] = []
    bridge_rows: list[dict[str, Any]] = []
    catalogue_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    bridge_sequence = 0

    for structure in structures:
        structure_id = structure["structure_id"]
        primary_anchor_key = structure["label_anchor_key"]
        primary_x, primary_y, primary_lon, primary_lat = anchor_xy[primary_anchor_key]
        structure["_anchor_px"] = evidence["label_anchors_px"][primary_anchor_key]

        part_metrics: list[dict[str, Any]] = []
        part_evidence: list[dict[str, Any]] = []
        structure_bridges: list[dict[str, Any]] = []

        for part in structure.get("centerline_parts", []):
            centerline, bridges, metrics = _build_chain(
                part,
                segments_by_id,
                structure_id=structure_id,
                contiguous_tolerance_m=contiguous_tolerance_m,
            )
            bridge_limit = float(part.get("bridge_max_km", 0.0))
            if metrics["max_bridge_km"] > bridge_limit + 1e-9:
                raise ValueError(
                    f"Bridge limit exceeded for {part['part_id']}: "
                    f"{metrics['max_bridge_km']:.3f} km > {bridge_limit:.3f} km"
                )
            part_metrics.append(metrics)
            structure_bridges.extend(bridges)
            centerline_rows.append(
                {
                    "structure_id": structure_id,
                    "abbreviation": structure["abbreviation"],
                    "structure_name": structure["name"],
                    "part_id": part["part_id"],
                    "part_role": part["role"],
                    "part_confidence": part.get("confidence", structure["attribution_confidence"]),
                    "n_source_segments": metrics["n_segments"],
                    "source_segment_ids": "|".join(part["ordered_segment_ids"]),
                    "n_bridges": metrics["n_bridges"],
                    "mapped_length_km": round(metrics["mapped_length_km"], 3),
                    "bridge_length_km": round(metrics["bridge_length_km"], 3),
                    "total_length_km": round(metrics["total_length_km"], 3),
                    "maximum_gap_km": round(metrics["max_bridge_km"], 3),
                    "geometry_class": "approximate_parent_axis",
                    "construction": "ordered printed components plus explicit straight gap bridges",
                    "source_name_figure": FIGURE12,
                    "source_geometry_figure": FIGURE14,
                    "source_doi": SOURCE_DOI,
                    "notes": part.get("reason", ""),
                    "geometry": centerline,
                }
            )
            part_evidence.append(
                {
                    "part_id": part["part_id"],
                    "role": part["role"],
                    "confidence": part.get("confidence", structure["attribution_confidence"]),
                    "ordered_segment_ids": list(part["ordered_segment_ids"]),
                    "n_segments": metrics["n_segments"],
                    "mapped_length_km": round(metrics["mapped_length_km"], 3),
                    "bridge_length_km": round(metrics["bridge_length_km"], 3),
                    "centerline_length_km": round(metrics["total_length_km"], 3),
                    "n_bridges": metrics["n_bridges"],
                    "maximum_gap_km": round(metrics["max_bridge_km"], 3),
                    "reason": part.get("reason", ""),
                }
            )

            for segment_id in part["ordered_segment_ids"]:
                anchor_key = primary_anchor_key
                if structure["abbreviation"] == "Ky" and part["part_id"] == "KY_SW_PROBABLE":
                    anchor_key = "Ky_sw"
                anchor_pixel = evidence["label_anchors_px"][anchor_key]
                distance_px = _segment_label_distance_px(
                    segments_by_id[segment_id], anchor_pixel, georeference
                )
                confidence = part.get("confidence", structure["attribution_confidence"])
                segment_assignments[segment_id] = {
                    "structure_id": structure_id,
                    "abbreviation": structure["abbreviation"],
                    "structure_name": structure["name"],
                    "attribution_status": f"named_{confidence}",
                    "attribution_confidence": confidence,
                    "mapping_role": part["role"],
                    "parent_part_id": part["part_id"],
                    "label_anchor_key": anchor_key,
                    "label_distance_px": distance_px,
                    "attribution_evidence": part.get("reason", ""),
                }

        for supporting in structure.get("supporting_segments", []):
            segment_id = supporting["segment_id"]
            anchor_key = primary_anchor_key
            anchor_pixel = evidence["label_anchors_px"][anchor_key]
            distance_px = _segment_label_distance_px(
                segments_by_id[segment_id], anchor_pixel, georeference
            )
            confidence = supporting.get("confidence", structure["attribution_confidence"])
            segment_assignments[segment_id] = {
                "structure_id": structure_id,
                "abbreviation": structure["abbreviation"],
                "structure_name": structure["name"],
                "attribution_status": f"named_{confidence}",
                "attribution_confidence": confidence,
                "mapping_role": supporting["role"],
                "parent_part_id": "",
                "label_anchor_key": anchor_key,
                "label_distance_px": distance_px,
                "attribution_evidence": supporting["reason"],
            }

        bridge_count = len(structure_bridges)
        bridge_length_km = sum(item["gap_km"] for item in structure_bridges)
        for bridge in structure_bridges:
            bridge_sequence += 1
            bridge["bridge_id"] = f"DALY2025_BRG_{bridge_sequence:04d}"
            bridge_rows.append(bridge)
        catalogue_rows.append(
            _category_row(
                structure,
                distances={},
                centerline_metrics=part_metrics,
                bridge_count=bridge_count,
                bridge_length_km=bridge_length_km,
                geometry=Point(primary_x, primary_y),
                anchor_xy_m=(primary_x, primary_y),
                anchor_lon_lat=(primary_lon, primary_lat),
            )
        )

        label_evidence = []
        for key in [primary_anchor_key, *structure.get("additional_label_anchor_keys", [])]:
            x, y, longitude, latitude = anchor_xy[key]
            label_evidence.append(
                {
                    "anchor_key": key,
                    "pixel_x": float(evidence["label_anchors_px"][key][0]),
                    "pixel_y": float(evidence["label_anchors_px"][key][1]),
                    "x_m": x,
                    "y_m": y,
                    "longitude": longitude,
                    "latitude": latitude,
                }
            )

        accepted_evidence = []
        for part in structure.get("centerline_parts", []):
            for segment_id in part["ordered_segment_ids"]:
                assignment = segment_assignments[segment_id]
                accepted_evidence.append(
                    {
                        "segment_id": segment_id,
                        "mapping_role": part["role"],
                        "confidence": assignment["attribution_confidence"],
                        "label_anchor_key": assignment["label_anchor_key"],
                        "label_distance_px": round(assignment["label_distance_px"], 3),
                        "reason": part.get("reason", ""),
                    }
                )
        for supporting in structure.get("supporting_segments", []):
            assignment = segment_assignments[supporting["segment_id"]]
            accepted_evidence.append(
                {
                    "segment_id": supporting["segment_id"],
                    "mapping_role": supporting["role"],
                    "confidence": assignment["attribution_confidence"],
                    "label_anchor_key": assignment["label_anchor_key"],
                    "label_distance_px": round(assignment["label_distance_px"], 3),
                    "reason": supporting["reason"],
                }
            )

        candidate_evidence = []
        for candidate in structure.get("candidate_segments_not_assigned", []):
            distance_px = _segment_label_distance_px(
                segments_by_id[candidate["segment_id"]],
                evidence["label_anchors_px"][primary_anchor_key],
                georeference,
            )
            candidate_evidence.append(
                {
                    "segment_id": candidate["segment_id"],
                    "label_distance_px": round(distance_px, 3),
                    "reason_not_assigned": candidate["reason"],
                }
            )

        evidence_rows.append(
            {
                "structure_id": structure_id,
                "abbreviation": structure["abbreviation"],
                "structure_name": structure["name"],
                "aliases": structure.get("aliases", []),
                "attribution_confidence": structure["attribution_confidence"],
                "geometry_status": structure["geometry_status"],
                "published_context": structure["published_context"],
                "figure14_cross_check": structure["figure14_cross_check"],
                "label_anchors": label_evidence,
                "centerline_parts": part_evidence,
                "accepted_segments": accepted_evidence,
                "candidate_segments_not_assigned": candidate_evidence,
                "notes": structure.get("notes", []),
            }
        )

    attributed = v1.copy().reset_index(drop=True)
    original_columns = [column for column in v1.columns if column != "geometry"]
    new_columns = [
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
    attributed["attributed_structure_id"] = pd.Series(index=attributed.index, dtype="object")
    attributed["attributed_structure_abbreviation"] = pd.Series(index=attributed.index, dtype="object")
    attributed["attributed_structure_name"] = pd.Series(index=attributed.index, dtype="object")
    attributed["attributed_status"] = "unnamed_no_supported_attribution"
    attributed["attribution_confidence"] = pd.Series(index=attributed.index, dtype="object")
    attributed["mapping_role"] = "unattributed"
    attributed["parent_part_id"] = pd.Series(index=attributed.index, dtype="object")
    attributed["label_anchor_key"] = pd.Series(index=attributed.index, dtype="object")
    attributed["label_anchor_px_x"] = np.nan
    attributed["label_anchor_px_y"] = np.nan
    attributed["label_distance_px"] = np.nan
    attributed["attribution_evidence"] = (
        "No published-label association met the cartographic evidence rule; retained as unnamed major linework."
    )
    attributed["source_name_figure"] = FIGURE12
    attributed["source_geometry_figure"] = FIGURE14
    attributed["occurrence_evidence_used"] = "no"

    for row_index, segment_id in attributed["structure_id"].astype(str).items():
        assignment = segment_assignments.get(segment_id)
        if assignment is None:
            continue
        attributed.at[row_index, "attributed_structure_id"] = assignment["structure_id"]
        attributed.at[row_index, "attributed_structure_abbreviation"] = assignment["abbreviation"]
        attributed.at[row_index, "attributed_structure_name"] = assignment["structure_name"]
        attributed.at[row_index, "attributed_status"] = assignment["attribution_status"]
        attributed.at[row_index, "attribution_confidence"] = assignment["attribution_confidence"]
        attributed.at[row_index, "mapping_role"] = assignment["mapping_role"]
        attributed.at[row_index, "parent_part_id"] = assignment["parent_part_id"]
        attributed.at[row_index, "label_anchor_key"] = assignment["label_anchor_key"]
        attributed.at[row_index, "label_anchor_px_x"] = float(
            evidence["label_anchors_px"][assignment["label_anchor_key"]][0]
        )
        attributed.at[row_index, "label_anchor_px_y"] = float(
            evidence["label_anchors_px"][assignment["label_anchor_key"]][1]
        )
        attributed.at[row_index, "label_distance_px"] = round(assignment["label_distance_px"], 3)
        attributed.at[row_index, "attribution_evidence"] = assignment["attribution_evidence"]

    attributed = attributed[original_columns + new_columns + ["geometry"]]
    attributed = gpd.GeoDataFrame(attributed, geometry="geometry", crs=v1.crs)
    centreline_gdf = gpd.GeoDataFrame(centerline_rows, geometry="geometry", crs=v1.crs)
    bridge_gdf = gpd.GeoDataFrame(bridge_rows, geometry="geometry", crs=v1.crs) if bridge_rows else gpd.GeoDataFrame(
        columns=["bridge_id", "structure_id", "part_id", "gap_m", "gap_km", "geometry"],
        geometry="geometry",
        crs=v1.crs,
    )
    catalogue_gdf = gpd.GeoDataFrame(catalogue_rows, geometry="geometry", crs=v1.crs)

    expected_bridge_count = int(centreline_gdf["n_bridges"].sum())
    if expected_bridge_count != len(bridge_gdf):
        raise ValueError(
            "Bridge layer is inconsistent with centreline metrics: "
            f"{len(bridge_gdf)} bridge rows for {expected_bridge_count} declared bridges"
        )
    expected_bridge_length_km = float(centreline_gdf["bridge_length_km"].sum())
    actual_bridge_length_km = float(bridge_gdf.geometry.length.sum()) / 1000.0
    if not np.isclose(expected_bridge_length_km, actual_bridge_length_km, atol=1e-6):
        raise ValueError(
            "Bridge length is inconsistent with centreline metrics: "
            f"{actual_bridge_length_km:.6f} km layer vs {expected_bridge_length_km:.6f} km declared"
        )

    for gdf in (centreline_gdf, bridge_gdf, catalogue_gdf):
        if not gdf.empty and not gdf.geometry.is_valid.all():
            invalid = gdf.loc[~gdf.geometry.is_valid, gdf.columns[0]].tolist()
            raise ValueError(f"Invalid output geometries: {invalid}")

    return {
        "v1_segments_attributed": attributed,
        "named_structure_centrelines": centreline_gdf,
        "bridge_connectors": bridge_gdf,
        "named_structure_catalogue": catalogue_gdf,
        "evidence_rows": evidence_rows,
        "segment_assignments": segment_assignments,
    }


def _domain_label_points(domains: gpd.GeoDataFrame) -> list[tuple[float, float, str]]:
    labels: list[tuple[float, float, str]] = []
    for abbreviation, group in domains.groupby("abbreviation", sort=True):
        if not abbreviation or str(abbreviation).lower() == "nan":
            continue
        union = group.geometry.union_all()
        if union.is_empty:
            continue
        point = union.representative_point()
        labels.append((float(point.x), float(point.y), str(abbreviation)))
    return labels


def _plot_context(
    ax: plt.Axes,
    domains: gpd.GeoDataFrame,
    attributed: gpd.GeoDataFrame,
    centerlines: gpd.GeoDataFrame,
) -> None:
    palette = [
        "#f7f1df",
        "#dcebf3",
        "#e4eed3",
        "#f3dfd2",
        "#d9e2ef",
        "#eadfe9",
        "#e1e1d5",
        "#d8ebdf",
        "#efe3c8",
    ]
    for index, (_, row) in enumerate(domains.iterrows()):
        gpd.GeoSeries([row.geometry], crs=domains.crs).plot(
            ax=ax,
            color=palette[index % len(palette)],
            edgecolor="#a9a9a9",
            linewidth=0.35,
            alpha=0.55,
            zorder=0,
        )

    for x, y, abbreviation in _domain_label_points(domains):
        ax.text(
            x,
            y,
            abbreviation,
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color="#4a4a4a",
            alpha=0.28,
            zorder=1,
        )

    unnamed = attributed[attributed["attributed_structure_id"].isna()]
    if not unnamed.empty:
        unnamed.plot(
            ax=ax,
            color="#b8b8b8",
            linewidth=0.65,
            alpha=0.9,
            zorder=2,
            label="Unnamed heavy linework (v1)",
        )
    if not centerlines.empty:
        centerlines.plot(
            ax=ax,
            color="#222222",
            linewidth=0.75,
            linestyle=(0, (4, 2)),
            alpha=0.65,
            zorder=3,
            label="Approximate parent axis",
        )


def _plot_named_segments(ax: plt.Axes, attributed: gpd.GeoDataFrame) -> None:
    styles = {
        "named_high": ("#083b66", 2.1),
        "named_medium": ("#1f7a8c", 1.8),
        "named_medium_low": ("#d48a13", 1.5),
        "named_low": ("#7d5a9a", 1.1),
    }
    for status, (color, width) in styles.items():
        subset = attributed[attributed["attributed_status"] == status]
        if subset.empty:
            continue
        subset.plot(ax=ax, color=color, linewidth=width, zorder=6)


def _style_map_axes(ax: plt.Axes, domains: gpd.GeoDataFrame) -> None:
    xmin, ymin, xmax, ymax = domains.total_bounds
    margin_x = (xmax - xmin) * 0.025
    margin_y = (ymax - ymin) * 0.035
    ax.set_xlim(xmin - margin_x, xmax + margin_x)
    ax.set_ylim(ymin - margin_y, ymax + margin_y)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m, EPSG:32735)", fontsize=8)
    ax.set_ylabel("Northing (m, EPSG:32735)", fontsize=8)
    ax.tick_params(labelsize=7, length=2, colors="#555555")
    ax.grid(True, color="#d9d9d9", linewidth=0.35, linestyle=":", alpha=0.65, zorder=99)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#777777")


def _scale_bar(ax: plt.Axes, domains: gpd.GeoDataFrame) -> None:
    xmin, ymin, xmax, ymax = domains.total_bounds
    length_m = 200_000.0
    x0 = xmin + 0.055 * (xmax - xmin)
    y0 = ymin + 0.045 * (ymax - ymin)
    height = 3_000.0
    for index in range(2):
        ax.add_patch(
            plt.Rectangle(
                (x0 + index * length_m / 2, y0),
                length_m / 2,
                height,
                facecolor="#222222" if index % 2 == 0 else "white",
                edgecolor="#222222",
                linewidth=0.7,
                zorder=120,
            )
        )
    ax.text(x0, y0 + height * 2.2, "0", ha="center", va="bottom", fontsize=7, zorder=121)
    ax.text(
        x0 + length_m / 2,
        y0 + height * 2.2,
        "100",
        ha="center",
        va="bottom",
        fontsize=7,
        zorder=121,
    )
    ax.text(
        x0 + length_m,
        y0 + height * 2.2,
        "200 km",
        ha="center",
        va="bottom",
        fontsize=7,
        zorder=121,
    )


def _north_arrow(ax: plt.Axes, domains: gpd.GeoDataFrame) -> None:
    xmin, ymin, xmax, ymax = domains.total_bounds
    x = xmax - 0.045 * (xmax - xmin)
    y = ymax - 0.10 * (ymax - ymin)
    ax.annotate(
        "N",
        xy=(x, y + 30_000),
        xytext=(x, y),
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        arrowprops={"facecolor": "#222222", "edgecolor": "#222222", "width": 1.3, "headwidth": 7, "headlength": 10},
        zorder=130,
    )


def _draw_named_map(
    path: Path,
    domains: gpd.GeoDataFrame,
    attributed: gpd.GeoDataFrame,
    centerlines: gpd.GeoDataFrame,
    bridges: gpd.GeoDataFrame,
    catalogue: gpd.GeoDataFrame,
    build_id: str,
) -> None:
    fig, ax = plt.subplots(figsize=(15.5, 15.2))
    _plot_context(ax, domains, attributed, centerlines)
    if not bridges.empty:
        bridges.plot(
            ax=ax,
            color="#b23a48",
            linewidth=0.8,
            linestyle=(0, (2, 2)),
            alpha=0.8,
            zorder=4,
        )
    _plot_named_segments(ax, attributed)

    unresolved = catalogue[catalogue["geometry_status"] == "unresolved_reference_only"]
    if not unresolved.empty:
        unresolved.plot(
            ax=ax,
            marker="^",
            color="#d95f02",
            markersize=35,
            edgecolor="white",
            linewidth=0.6,
            zorder=20,
        )

    for row in catalogue.itertuples():
        is_unresolved = row.geometry_status == "unresolved_reference_only"
        label = f"{row.abbreviation}\n{row.structure_name}"
        if is_unresolved:
            label += "\nunresolved"
        ax.annotate(
            label,
            xy=(row.label_x_m, row.label_y_m),
            xytext=(row.label_x_m, row.label_y_m),
            ha="center",
            va="center",
            fontsize=6.7 if not is_unresolved else 6.4,
            color="#d95f02" if is_unresolved else "#123047",
            fontweight="bold" if not is_unresolved else "normal",
            fontstyle="italic" if is_unresolved else "normal",
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
            },
            zorder=25,
        )

    southwest_parts = centerlines[centerlines["part_role"] == "disconnected_southwest_component"]
    for row in southwest_parts.itertuples():
        midpoint = row.geometry.interpolate(0.5, normalized=True)
        ax.annotate(
            f"{row.abbreviation} (SW)\n{row.structure_name}",
            xy=(midpoint.x, midpoint.y),
            xytext=(midpoint.x, midpoint.y),
            ha="center",
            va="center",
            fontsize=6.2,
            color="#9a5a00",
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
            zorder=26,
        )

    handles = [
        Line2D([0], [0], color="#083b66", lw=2.2, label="Named structure: high confidence"),
        Line2D([0], [0], color="#1f7a8c", lw=1.8, label="Named structure: medium confidence"),
        Line2D([0], [0], color="#d48a13", lw=1.5, label="Named structure: medium-low component"),
        Line2D([0], [0], color="#7d5a9a", lw=1.1, label="Named structure: low-confidence detail"),
        Line2D([0], [0], color="#b8b8b8", lw=0.8, label="Unnamed heavy linework"),
        Line2D([0], [0], color="#222222", lw=0.8, linestyle=(0, (4, 2)), label="Approximate parent axis"),
        Line2D([0], [0], color="#b23a48", lw=0.8, linestyle=(0, (2, 2)), label="Explicit gap bridge"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#d95f02", markeredgecolor="white", markersize=8, label="Unresolved published name"),
        Patch(facecolor="#d9e2ef", edgecolor="#a9a9a9", alpha=0.55, label="Daly domain context"),
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        fontsize=6.8,
        title="Structural attribution",
        title_fontsize=7.5,
    )
    _style_map_axes(ax, domains)
    _scale_bar(ax, domains)
    _north_arrow(ax, domains)
    ax.set_title(
        "Daly et al. (2025) named major structures reconstructed from printed heavy linework",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    fig.text(
        0.012,
        0.012,
        (
            "Names: Daly et al. (2025) Figure 12 caption and labels. Geometry: heavy linework in Figure 14, cross-checked against Figure 12.\n"
            f"Attribution build {build_id}. Occurrence locations were not used. Unresolved names retain label anchors but no invented geometry."
        ),
        fontsize=6.2,
        color="#444444",
        ha="left",
        va="bottom",
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.955, bottom=0.055)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=230, facecolor="white")
    plt.close(fig)


def _draw_attribution_map(
    path: Path,
    domains: gpd.GeoDataFrame,
    attributed: gpd.GeoDataFrame,
    centerlines: gpd.GeoDataFrame,
    bridges: gpd.GeoDataFrame,
    catalogue: gpd.GeoDataFrame,
    build_id: str,
) -> None:
    fig, ax = plt.subplots(figsize=(17.5, 17.0))
    _plot_context(ax, domains, attributed, centerlines)
    if not bridges.empty:
        bridges.plot(ax=ax, color="#b23a48", linewidth=0.9, linestyle=(0, (2, 2)), zorder=5)
    _plot_named_segments(ax, attributed)

    named = attributed[attributed["attributed_structure_id"].notna()]
    for row in named.itertuples():
        midpoint = row.geometry.interpolate(0.5, normalized=True)
        segment_number = str(row.structure_id).replace("DALY2025_MSC_", "")
        ax.annotate(
            segment_number,
            xy=(midpoint.x, midpoint.y),
            xytext=(midpoint.x, midpoint.y),
            ha="center",
            va="center",
            fontsize=3.7,
            color="#111111",
            bbox={"boxstyle": "round,pad=0.05", "facecolor": "white", "edgecolor": "none", "alpha": 0.65},
            zorder=30,
        )

    for row in catalogue.itertuples():
        if row.geometry_status == "unresolved_reference_only":
            ax.scatter([row.label_x_m], [row.label_y_m], marker="^", s=45, color="#d95f02", edgecolor="white", linewidth=0.6, zorder=31)
        ax.text(
            row.label_x_m,
            row.label_y_m,
            row.abbreviation,
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="#d95f02" if row.geometry_status == "unresolved_reference_only" else "#083b66",
            bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
            zorder=32,
        )

    southwest_parts = centerlines[centerlines["part_role"] == "disconnected_southwest_component"]
    for row in southwest_parts.itertuples():
        midpoint = row.geometry.interpolate(0.5, normalized=True)
        ax.text(
            midpoint.x,
            midpoint.y,
            f"{row.abbreviation} (SW)",
            ha="center",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            color="#9a5a00",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
            zorder=33,
        )

    handles = [
        Line2D([0], [0], color="#083b66", lw=2.1, label="High-confidence named segment"),
        Line2D([0], [0], color="#1f7a8c", lw=1.8, label="Medium-confidence named segment"),
        Line2D([0], [0], color="#d48a13", lw=1.5, label="Medium-low component"),
        Line2D([0], [0], color="#7d5a9a", lw=1.1, label="Low-confidence detail"),
        Line2D([0], [0], color="#b8b8b8", lw=0.8, label="Unnamed v1 segment"),
        Line2D([0], [0], color="#b23a48", lw=0.9, linestyle=(0, (2, 2)), label="Gap bridge"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#d95f02", markeredgecolor="white", markersize=8, label="Unresolved name"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, facecolor="white", framealpha=0.92, fontsize=6.8, title="Attribution audit", title_fontsize=7.5)
    _style_map_axes(ax, domains)
    _scale_bar(ax, domains)
    _north_arrow(ax, domains)
    ax.set_title(
        "Segment-level attribution audit for Daly major structures (v2)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    fig.text(
        0.012,
        0.012,
        (
            f"Build {build_id}. Segment numbers are the numeric suffixes of DALY2025_MSC_* identifiers. "
            "Grey linework is retained but not confidently attributable to a published name.\n"
            "The map is cartographic evidence only; it does not encode displacement, kinematics or physical fault-zone width."
        ),
        fontsize=6.0,
        color="#444444",
        ha="left",
        va="bottom",
    )
    fig.subplots_adjust(left=0.06, right=0.985, top=0.955, bottom=0.05)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=230, facecolor="white")
    plt.close(fig)


def _assert_protected_inputs(config: dict[str, Any]) -> dict[str, str]:
    hashes = config["protected_source_hashes"]
    sources = config["sources"]
    observed: dict[str, str] = {}
    failures: list[str] = []
    for key, expected in hashes.items():
        path = resolve_path(sources[key])
        if not path.exists():
            failures.append(f"{key}: missing {relative_posix(path)}")
            continue
        actual = sha256_file(path)
        observed[key] = actual
        if actual.lower() != str(expected).lower():
            failures.append(f"{key}: expected {expected}, observed {actual}")
    v14_path = resolve_path(sources["v14_training_csv"])
    v14_actual = sha256_file(v14_path)
    observed["v14_training_csv"] = v14_actual
    if v14_actual.lower() != str(sources["v14_training_sha256"]).lower():
        failures.append(
            f"v14_training_csv: expected {sources['v14_training_sha256']}, observed {v14_actual}"
        )
    if failures:
        raise RuntimeError("Protected input hash check failed:\n" + "\n".join(failures))
    return observed


def _write_layers(path: Path, products: dict[str, Any]) -> None:
    layers = [
        ("v1_segments_attributed", products["v1_segments_attributed"]),
        ("named_structure_centrelines", products["named_structure_centrelines"]),
        ("bridge_connectors", products["bridge_connectors"]),
        ("named_structure_catalogue", products["named_structure_catalogue"]),
    ]
    for index, (layer_name, gdf) in enumerate(layers):
        pyogrio.write_dataframe(
            gdf,
            path,
            layer=layer_name,
            driver="GPKG",
            append=index > 0,
        )


def _csv_records(gdf: gpd.GeoDataFrame) -> list[dict[str, Any]]:
    frame = gdf.drop(columns="geometry").copy() if "geometry" in gdf.columns else gdf.copy()
    records = frame.to_dict(orient="records")
    for record in records:
        for key, value in list(record.items()):
            if isinstance(value, (np.integer, np.floating)):
                record[key] = value.item()
            elif value is pd.NA:
                record[key] = None
            elif isinstance(value, float) and math.isnan(value):
                record[key] = None
    return records


def build(config_path: str | Path, *, overwrite: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    sources = config["sources"]
    outputs = config["outputs"]
    target_crs = config["target_crs"]
    if target_crs != PROJECT_CRS:
        raise ValueError(f"Expected {PROJECT_CRS}, found {target_crs}")

    protected_hashes = _assert_protected_inputs(config)
    v1_path = resolve_path(sources["v1_gpkg"])
    output_gpkg = resolve_path(Path(outputs["directory"]) / outputs["named_gpkg"])
    if output_gpkg.resolve() == v1_path.resolve():
        raise ValueError("The v2 output path resolves to the immutable v1 GeoPackage")

    if output_gpkg.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {relative_posix(output_gpkg)}. Use --overwrite for a v2 rebuild."
        )
    if output_gpkg.exists() and overwrite:
        output_gpkg.unlink()

    v1 = gpd.read_file(v1_path, layer=sources["v1_layer"], engine="pyogrio")
    if v1.crs is None or v1.crs.to_string() != PROJECT_CRS:
        raise ValueError(f"v1 must be in {PROJECT_CRS}, found {v1.crs}")
    if not v1.geometry.geom_type.eq("LineString").all():
        raise ValueError("v1 contains non-LineString geometry")
    if not v1["structure_id"].is_unique:
        raise ValueError("v1 structure_id values are not unique")

    with resolve_path(sources["v1_manifest"]).open("r", encoding="utf-8") as handle:
        v1_manifest = json.load(handle)
    with resolve_path(v1_manifest["summary_file"]).open("r", encoding="utf-8") as handle:
        v1_summary = json.load(handle)
    expected_segments = int(v1_summary["features"])
    if len(v1) != expected_segments:
        raise ValueError(f"Expected {expected_segments} v1 segments, found {len(v1)}")

    with resolve_path(sources["attribution_config"]).open("r", encoding="utf-8") as handle:
        evidence = json.load(handle)

    domains = gpd.read_file(
        resolve_path(sources["domain_gpkg"]),
        layer=sources["domain_layer"],
        engine="pyogrio",
    )
    if domains.crs is None or domains.crs.to_string() != PROJECT_CRS:
        domains = domains.to_crs(PROJECT_CRS)

    products = _build_products(
        v1,
        evidence,
        config["figure12_georeference"],
        float(evidence["policy"]["contiguous_tolerance_m"]),
    )

    build_id = _utc_stamp()
    generated_utc = _utc_iso()
    attributed = products["v1_segments_attributed"]
    centerlines = products["named_structure_centrelines"]
    bridges = products["bridge_connectors"]
    catalogue = products["named_structure_catalogue"]
    resolved = catalogue[catalogue["geometry_status"] != "unresolved_reference_only"]
    unresolved = catalogue[catalogue["geometry_status"] == "unresolved_reference_only"]

    _write_layers(output_gpkg, products)

    output_directory = resolve_path(outputs["directory"])
    segment_csv = output_directory / outputs["segment_attribution_csv"]
    summary_csv = output_directory / outputs["structure_summary_csv"]
    attribution_json = output_directory / outputs["attribution_json"]
    summary_json = output_directory / outputs["summary_json"]
    manifest_json = output_directory / outputs["manifest_json"]
    validation_json = output_directory / outputs["validation_json"]
    named_map = resolve_path(outputs["named_map_png"])
    attribution_map = resolve_path(outputs["attribution_map_png"])

    summary_columns = [
        "structure_id",
        "abbreviation",
        "structure_name",
        "aliases",
        "attribution_status",
        "attribution_confidence",
        "geometry_status",
        "n_attributed_segments",
        "n_centerline_parts",
        "n_supporting_segments",
        "mapped_length_km",
        "bridge_length_km",
        "centerline_length_km",
        "n_bridges",
        "max_bridge_km",
        "candidate_segments_not_assigned",
        "label_anchor_key",
        "label_anchor_px_x",
        "label_anchor_px_y",
        "label_longitude",
        "label_latitude",
        "label_x_m",
        "label_y_m",
        "figure14_cross_check",
        "published_context",
        "review_status",
        "notes",
    ]
    write_csv(segment_csv, _csv_records(attributed))
    write_csv(summary_csv, _csv_records(catalogue[summary_columns]))

    anchors_payload: dict[str, Any] = {}
    for key, pixel in evidence["label_anchors_px"].items():
        x, y, longitude, latitude = _label_point(pixel, config["figure12_georeference"])
        anchors_payload[key] = {
            "pixel_x": float(pixel[0]),
            "pixel_y": float(pixel[1]),
            "x_m": x,
            "y_m": y,
            "longitude": longitude,
            "latitude": latitude,
        }
    attribution_payload = {
        "schema_version": "1.0.0",
        "build_id": build_id,
        "generated_utc": generated_utc,
        "source_doi": SOURCE_DOI,
        "method": {
            "primary_naming_figure": FIGURE12,
            "geometry_cross_check_figure": FIGURE14,
            "v1_source": relative_posix(v1_path),
            "v1_layer": sources["v1_layer"],
            "naming_policy": evidence["policy"]["naming_rule"],
            "connectivity_policy": evidence["policy"]["connectivity_rule"],
            "prohibited_evidence": evidence["policy"]["prohibited_evidence"],
            "bridge_geometry": evidence["policy"]["bridge_geometry"],
            "contiguous_tolerance_m": float(evidence["policy"]["contiguous_tolerance_m"]),
        },
        "label_anchors": anchors_payload,
        "structures": products["evidence_rows"],
        "unresolved_reference_names": unresolved["abbreviation"].tolist(),
        "unnamed_linework_policy": (
            "All 248 v1 segments remain in v1_segments_attributed. Segments without a supported published-name link "
            "are retained as unnamed heavy linework and are not used as a named-fault predictor."
        ),
        "occurrence_evidence_used": False,
    }
    write_json(attribution_json, attribution_payload)

    status_counts = attributed["attributed_status"].value_counts().to_dict()
    summary = {
        "schema_version": "1.0.0",
        "build_id": build_id,
        "generated_utc": generated_utc,
        "source_doi": SOURCE_DOI,
        "target_crs": PROJECT_CRS,
        "source_v1": {
            "path": relative_posix(v1_path),
            "layer": sources["v1_layer"],
            "features": int(len(v1)),
            "sha256": protected_hashes["v1_gpkg"],
        },
        "published_names": int(len(catalogue)),
        "named_structures_with_geometry": int(len(resolved)),
        "unresolved_reference_only": int(len(unresolved)),
        "attributed_segments": int((attributed["attributed_structure_id"].notna()).sum()),
        "unnamed_segments": int((attributed["attributed_structure_id"].isna()).sum()),
        "attribution_status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "centerline_parts": int(len(centerlines)),
        "bridge_connectors": int(len(bridges)),
        "mapped_named_length_km": round(float(centerlines["mapped_length_km"].sum()), 3),
        "bridge_length_km": round(float(centerlines["bridge_length_km"].sum()), 3),
        "centerline_total_length_km": round(float(centerlines["total_length_km"].sum()), 3),
        "maximum_bridge_km": round(float(centerlines["maximum_gap_km"].max()), 3),
        "unresolved_names": unresolved["structure_name"].tolist(),
        "quality_status": {
            "all_248_v1_segments_preserved": bool(len(attributed) == 248),
            "source_ids_unique": bool(attributed["structure_id"].is_unique),
            "all_published_names_catalogued": bool(len(catalogue) == 15),
            "attributed_segment_ids_unique": bool(
                attributed.loc[attributed["attributed_structure_id"].notna(), "structure_id"].is_unique
            ),
            "unresolved_names_have_no_centerline": bool(
                set(unresolved["structure_id"]).isdisjoint(set(centerlines["structure_id"]))
            ),
            "all_centerlines_valid": bool(centerlines.geometry.is_valid.all()),
            "all_bridges_valid": bool(bridges.geometry.is_valid.all()) if not bridges.empty else True,
            "crs": PROJECT_CRS,
            "occurrence_inputs_used": False,
            "v1_sha256_unchanged": protected_hashes["v1_gpkg"]
            == config["protected_source_hashes"]["v1_gpkg"],
            "v14_sha256_unchanged": protected_hashes["v14_training_csv"]
            == sources["v14_training_sha256"],
        },
        "interpretation": (
            "The named centreline is an approximate parent axis reconstructed from heavy printed components. "
            "It is not a surveyed fault surface and does not encode kinematics, displacement or physical width. "
            "Unresolved names retain catalogue label anchors but no invented geometry."
        ),
    }
    write_json(summary_json, summary)

    _draw_named_map(
        named_map,
        domains,
        attributed,
        centerlines,
        bridges,
        catalogue,
        build_id,
    )
    _draw_attribution_map(
        attribution_map,
        domains,
        attributed,
        centerlines,
        bridges,
        catalogue,
        build_id,
    )

    manifest = {
        "schema_version": "1.0.0",
        "build_id": build_id,
        "generated_utc": generated_utc,
        "build_command": (
            "python -m src.geology_foundation.reconstruct_major_structures_v2 "
            f"--config {Path(config_path).as_posix()} --overwrite"
        ),
        "software": software_versions(),
        "config": file_record(resolve_path(config_path), role="v2 reconstruction configuration"),
        "attribution_config": file_record(
            resolve_path(sources["attribution_config"]),
            role="segment and structure attribution evidence",
        ),
        "sources": [
            file_record(resolve_path(sources["figure12_raster"]), role="published naming and label-placement reference"),
            file_record(resolve_path(sources["figure14_raster"]), role="georeferenced heavy-linework geometry source"),
            file_record(resolve_path(sources["caption_pdf"]), role="published caption and body-text evidence"),
            file_record(resolve_path(sources["v1_gpkg"]), role="immutable v1 segment source"),
            file_record(resolve_path(sources["v1_manifest"]), role="immutable v1 provenance"),
            file_record(resolve_path(sources["domain_gpkg"]), role="tectonic-domain map context"),
        ],
        "protected_inputs": {
            **protected_hashes,
            "v1_expected_sha256": config["protected_source_hashes"]["v1_gpkg"],
            "v14_expected_sha256": sources["v14_training_sha256"],
        },
        "outputs": [
            file_record(output_gpkg, role="named major-structure GeoPackage v2"),
            file_record(segment_csv, role="segment-to-structure attribution table"),
            file_record(summary_csv, role="structure-level summary table"),
            file_record(attribution_json, role="attribution evidence and unresolved-case register"),
            file_record(summary_json, role="reconstruction summary and quality status"),
            file_record(named_map, role="named, unnamed and unresolved structure map"),
            file_record(attribution_map, role="segment-level attribution audit map"),
        ],
        "layers": [
            "v1_segments_attributed",
            "named_structure_centrelines",
            "bridge_connectors",
            "named_structure_catalogue",
        ],
        "expected_validation_output": relative_posix(validation_json),
        "occurrence_data_used": False,
        "v14_benchmark_modified": False,
        "manuscript_modified": False,
    }
    write_json(manifest_json, manifest)
    print(f"Major-structure reconstruction v2 complete: {build_id}")
    print(f"GeoPackage: {relative_posix(output_gpkg)}")
    print(
        f"Published names: {len(catalogue)}; with geometry: {len(resolved)}; "
        f"unresolved: {len(unresolved)}"
    )
    print(
        f"Attributed v1 segments: {summary['attributed_segments']}; "
        f"unnamed: {summary['unnamed_segments']}; bridges: {summary['bridge_connectors']}"
    )
    print(f"Map: {relative_posix(named_map)}")
    print(f"Manifest: {relative_posix(manifest_json)}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/major_structures_v2.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    build(args.config, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
