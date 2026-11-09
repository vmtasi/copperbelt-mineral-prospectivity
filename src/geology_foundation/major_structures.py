from __future__ import annotations

import argparse
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import matplotlib
import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer
from scipy import ndimage as ndi
from shapely.geometry import LineString

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from .common import (
    file_record,
    load_config,
    relative_posix,
    resolve_path,
    sha256_file,
    software_versions,
    write_json,
)
from .domains import read_rgb_tiff


OFFSETS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _neighbours(point: tuple[int, int], pixels: set[tuple[int, int]]) -> list[tuple[int, int]]:
    row, col = point
    found: list[tuple[int, int]] = []
    for delta_row, delta_col in OFFSETS:
        candidate = (row + delta_row, col + delta_col)
        if candidate not in pixels:
            continue
        if delta_row and delta_col:
            if (row + delta_row, col) in pixels or (row, col + delta_col) in pixels:
                continue
        found.append(candidate)
    return found


def _zhang_suen(mask: np.ndarray) -> tuple[np.ndarray, int]:
    image = mask.astype(np.uint8, copy=True)
    removed_total = 0
    for _ in range(200):
        removed_iteration = 0
        for first_pass in (True, False):
            padded = np.pad(image, 1)
            p2 = padded[0:-2, 1:-1]
            p3 = padded[0:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, 0:-2]
            p8 = padded[1:-1, 0:-2]
            p9 = padded[0:-2, 0:-2]
            neighbour_count = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
            )
            if first_pass:
                corner_conditions = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                corner_conditions = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = (
                (image == 1)
                & (neighbour_count >= 2)
                & (neighbour_count <= 6)
                & (transitions == 1)
                & corner_conditions
            )
            count = int(remove.sum())
            if count:
                image[remove] = 0
                removed_iteration += count
        removed_total += removed_iteration
        if removed_iteration == 0:
            break
    return image.astype(bool), removed_total


def _trace_network(
    skeleton: np.ndarray,
    min_pixels: int,
) -> tuple[list[list[tuple[int, int]]], dict[int, int]]:
    pixels = {(int(row), int(col)) for row, col in zip(*np.nonzero(skeleton))}
    neighbourhoods = {point: _neighbours(point, pixels) for point in pixels}
    degree = {point: len(neighbours) for point, neighbours in neighbourhoods.items()}
    special = {point for point in pixels if degree[point] != 2}
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    paths: list[list[tuple[int, int]]] = []

    def edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        return (a, b) if a <= b else (b, a)

    for start in sorted(special):
        for neighbour in neighbourhoods[start]:
            key = edge_key(start, neighbour)
            if key in visited_edges:
                continue
            path = [start, neighbour]
            visited_edges.add(key)
            previous, current = start, neighbour
            while current not in special:
                following = [point for point in neighbourhoods[current] if point != previous]
                if not following:
                    break
                next_point = following[0]
                key = edge_key(current, next_point)
                if key in visited_edges:
                    break
                visited_edges.add(key)
                path.append(next_point)
                previous, current = current, next_point
            if len(path) - 1 >= min_pixels:
                paths.append(path)

    used = {point for path in paths for point in path}
    for start in sorted(pixels - used):
        for neighbour in neighbourhoods[start]:
            key = edge_key(start, neighbour)
            if key in visited_edges:
                continue
            path = [start, neighbour]
            visited_edges.add(key)
            previous, current = start, neighbour
            while current != start:
                following = [point for point in neighbourhoods[current] if point != previous]
                if not following:
                    break
                next_point = following[0]
                key = edge_key(current, next_point)
                if key in visited_edges:
                    break
                visited_edges.add(key)
                path.append(next_point)
                previous, current = current, next_point
            if path[-1] == start and len(path) - 1 >= min_pixels:
                paths.append(path)

    network_labels, _ = ndi.label(skeleton, structure=np.ones((3, 3), dtype=np.uint8))
    path_networks = {
        index: int(network_labels[path[0][0], path[0][1]])
        for index, path in enumerate(paths, start=1)
    }
    return paths, path_networks


def _path_length_pixels(path: Iterable[tuple[int, int]]) -> float:
    points = list(path)
    if len(points) < 2:
        return 0.0
    differences = np.diff(np.asarray(points, dtype=float), axis=0)
    return float(np.hypot(differences[:, 0], differences[:, 1]).sum())


def _pixel_to_coordinates(
    path: list[tuple[int, int]],
    geotransform: tuple[float, ...] | list[float],
    transformer: Transformer,
    simplify_tolerance_px: float,
) -> LineString:
    pixel_line = LineString([(float(col), float(row)) for row, col in path])
    simplified = pixel_line.simplify(simplify_tolerance_px, preserve_topology=False)
    coordinates = []
    for col, row in simplified.coords:
        longitude = geotransform[0] + (col + 0.5) * geotransform[1] + (row + 0.5) * geotransform[2]
        latitude = geotransform[3] + (col + 0.5) * geotransform[4] + (row + 0.5) * geotransform[5]
        easting, northing = transformer.transform(longitude, latitude)
        coordinates.append((easting, northing))
    return LineString(coordinates)


def _extraction_confidence(length_px: float, width_px: float) -> str:
    if length_px >= 50 and width_px >= 4.0:
        return "high"
    if length_px >= 15:
        return "medium"
    return "low"


def _segment_class(length_px: float) -> str:
    if length_px >= 50:
        return "major_ribbon_long"
    if length_px >= 15:
        return "major_ribbon_medium"
    return "major_fragment_or_dash"


def _source_extent_m(metadata: dict[str, Any], target_crs: str) -> tuple[float, float, float, float]:
    geotransform = metadata["geotransform"]
    width = int(metadata["width"])
    height = int(metadata["height"])
    transformer = Transformer.from_crs(metadata["source_crs"], target_crs, always_xy=True)
    corners = []
    for col, row in ((0, 0), (width, 0), (0, height), (width, height)):
        longitude = geotransform[0] + col * geotransform[1] + row * geotransform[2]
        latitude = geotransform[3] + col * geotransform[4] + row * geotransform[5]
        corners.append(transformer.transform(longitude, latitude))
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return min(xs), min(ys), max(xs), max(ys)


def _write_overlays(
    rgb: np.ndarray,
    paths: list[list[tuple[int, int]]],
    outputs: dict[str, Path],
) -> None:
    original = Image.fromarray(rgb)
    overlay = original.copy()
    draw = ImageDraw.Draw(overlay)
    for path in paths:
        draw.line([(col, row) for row, col in path], fill=(214, 12, 32), width=2)
    overlay.save(outputs["overlay_png"], optimize=True)

    audit = original.copy()
    audit_draw = ImageDraw.Draw(audit)
    colours = [(214, 12, 32), (20, 82, 180), (0, 122, 92), (173, 74, 0)]
    for index, path in enumerate(paths):
        audit_draw.line(
            [(col, row) for row, col in path],
            fill=colours[index % len(colours)],
            width=2,
        )
    audit.save(outputs["segments_png"], optimize=True)

    detail = overlay.crop((180, 500, 1780, 1740))
    detail.save(outputs["overlay_detail_png"], optimize=True)


def _write_map(
    segments: gpd.GeoDataFrame,
    domains_path: Path,
    boundaries_path: Path,
    foundation_config: dict[str, Any],
    output_path: Path,
) -> None:
    domains = gpd.read_file(domains_path, layer="daly_domains", engine="pyogrio").to_crs("EPSG:4326")
    boundaries = gpd.read_file(
        boundaries_path,
        layer="daly_domain_boundaries",
        engine="pyogrio",
    ).to_crs("EPSG:4326")
    segment_geographic = segments.to_crs("EPSG:4326")
    colour_by_domain = {
        item["domain_id"]: tuple(channel / 255.0 for channel in item["rgb"])
        for item in foundation_config["daly"]["classes"]
    }
    fill_colours = [colour_by_domain[value] for value in domains["domain_id"]]

    figure, axis = plt.subplots(figsize=(11, 9), dpi=300)
    domains.plot(ax=axis, color=fill_colours, edgecolor="none", zorder=1)
    boundaries.boundary.plot(ax=axis, color="#4f4f4f", linewidth=0.45, zorder=2)
    long_segments = segment_geographic[segment_geographic["line_class"] != "major_fragment_or_dash"]
    short_segments = segment_geographic[segment_geographic["line_class"] == "major_fragment_or_dash"]
    long_segments.plot(ax=axis, color="#111111", linewidth=0.85, zorder=5)
    short_segments.plot(ax=axis, color="#222222", linewidth=0.55, alpha=0.9, zorder=4)

    xmin, ymin, xmax, ymax = domains.total_bounds
    axis.set_xlim(xmin - 0.35, xmax + 0.35)
    axis.set_ylim(ymin - 0.55, ymax + 0.45)
    axis.set_aspect("equal")
    axis.set_xlabel("Longitude", fontsize=9)
    axis.set_ylabel("Latitude", fontsize=9)
    axis.tick_params(labelsize=8)
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:.0f}°E"))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{abs(value):.0f}°S"))
    axis.grid(color="white", linewidth=0.55, alpha=0.65, zorder=3)
    axis.set_axisbelow(False)

    mean_latitude = float(np.mean([ymin, ymax]))
    bar_length_degrees = 100.0 / (111.32 * np.cos(np.radians(mean_latitude)))
    bar_y = ymin - 0.33
    bar_x = xmax - bar_length_degrees - 0.15
    axis.plot([bar_x, bar_x + bar_length_degrees], [bar_y, bar_y], color="black", linewidth=2.5)
    axis.plot([bar_x, bar_x], [bar_y - 0.06, bar_y + 0.06], color="black", linewidth=1.2)
    axis.plot(
        [bar_x + bar_length_degrees, bar_x + bar_length_degrees],
        [bar_y - 0.06, bar_y + 0.06],
        color="black",
        linewidth=1.2,
    )
    axis.text(bar_x + bar_length_degrees / 2, bar_y - 0.13, "100 km", ha="center", va="top", fontsize=8)

    legend_handles = [
        Line2D([0], [0], color="#111111", linewidth=2.0, label="Extracted major structural linework"),
        Line2D([0], [0], color="#4f4f4f", linewidth=0.7, label="Reconstructed tectonic-domain boundary"),
        Patch(facecolor="#d9d9d9", edgecolor="#777777", linewidth=0.3, label="Daly et al. tectonic domains"),
    ]
    axis.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=8,
        frameon=True,
        framealpha=0.92,
        title="Map content",
        title_fontsize=8,
    )
    axis.set_title(
        "Major structural linework digitised from Daly et al. (2025), Figure 14",
        fontsize=11,
        pad=10,
    )
    figure.text(
        0.5,
        0.01,
        "Automated centreline extraction of the emboldened linework; fault names and hierarchy are not assigned.",
        ha="center",
        fontsize=7.5,
        color="#333333",
    )
    figure.text(
        0.012,
        0.018,
        "Source: Daly et al. (2025), Tectonics, doi:10.1029/2025TC008955, CC BY 4.0.",
        fontsize=6.5,
        color="#444444",
    )
    figure.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.10)
    figure.savefig(output_path, dpi=300, facecolor="white")
    plt.close(figure)


def build(config_path: str | Path, overwrite: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    source_path = resolve_path(config["source"]["raster"])
    output_directory = resolve_path(config["outputs"]["directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "gpkg": output_directory / config["outputs"]["gpkg"],
        "attributes_csv": output_directory / config["outputs"]["attributes_csv"],
        "summary_json": output_directory / config["outputs"]["summary_json"],
        "manifest_json": output_directory / config["outputs"]["manifest_json"],
        "overlay_png": resolve_path(config["outputs"]["overlay_png"]),
        "segments_png": resolve_path(config["outputs"]["segments_png"]),
        "overlay_detail_png": resolve_path(config["outputs"]["overlay_detail_png"]),
        "map_png": resolve_path(config["outputs"]["map_png"]),
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Major-structure outputs already exist; pass --overwrite after review: "
            + ", ".join(relative_posix(path) for path in existing)
        )

    rgb, geotransform, projection, metadata = read_rgb_tiff(source_path)
    metadata["source_crs"] = "EPSG:4326"
    params = config["extraction"]
    dark = rgb.max(axis=2) < int(params["dark_max_rgb"])
    distance = ndi.distance_transform_edt(dark)
    core = distance >= float(params["core_distance_px"])
    core_pixels_before_masking = int(core.sum())
    for rectangle in params["exclusion_rectangles_px"]:
        core[
            int(rectangle["y0"]) : int(rectangle["y1"]),
            int(rectangle["x0"]) : int(rectangle["x1"]),
        ] = False
    core = ndi.binary_closing(core, structure=np.ones((3, 3), dtype=np.uint8), iterations=1)
    component_labels, component_count = ndi.label(core, structure=np.ones((3, 3), dtype=np.uint8))
    component_sizes = np.bincount(component_labels.ravel())[1:]
    retained_components = np.flatnonzero(component_sizes >= int(params["min_component_pixels"])) + 1
    core = np.isin(component_labels, retained_components)

    skeleton, removed_skeleton_pixels = _zhang_suen(core)
    paths, path_networks = _trace_network(skeleton, int(params["min_polyline_pixels"]))
    if not paths:
        raise RuntimeError("No major-structure centrelines survived extraction")

    transformer = Transformer.from_crs("EPSG:4326", config["target_crs"], always_xy=True)
    rows: list[dict[str, Any]] = []
    for segment_number, path in enumerate(paths, start=1):
        length_pixels = _path_length_pixels(path)
        widths = np.asarray([2.0 * distance[row, col] for row, col in path], dtype=float)
        median_width = float(np.median(widths))
        mean_width = float(np.mean(widths))
        geometry = _pixel_to_coordinates(
            path,
            geotransform,
            transformer,
            float(params["simplify_tolerance_px"]),
        )
        if geometry.is_empty or geometry.length <= 0:
            continue
        rows.append(
            {
                "structure_id": f"DALY2025_MSC_{segment_number:04d}",
                "network_id": f"DALY2025_NET_{path_networks[segment_number]:04d}",
                "feature_class": "major_fault_zone_candidate",
                "line_class": _segment_class(length_pixels),
                "structure_name": None,
                "structure_type": None,
                "figure_class": "major_fault_zone",
                "tier": None,
                "hierarchy_status": "not assigned",
                "source_figure": config["source"]["figure"],
                "source_doi": config["source"]["doi"],
                "source_reference": config["source"]["reference"],
                "extraction_method": "Embolder linework; dark-ribbon distance core; Zhang-Suen centreline; pixel graph trace",
                "source_pixel_length": round(length_pixels, 3),
                "median_stroke_width_px": round(median_width, 3),
                "mean_stroke_width_px": round(mean_width, 3),
                "length_km": round(geometry.length / 1000.0, 3),
                "extraction_confidence": _extraction_confidence(length_pixels, median_width),
                "name_status": "unassigned",
                "attribution_status": "caption-supported linework; name and tectonic role not assigned",
                "review_status": "automated_candidate",
                "notes": "Thick black ribbon in Daly et al. (2025) Figure 14; Figure 12 caption identifies emboldened linework as major fault zones.",
                "geometry": geometry,
            }
        )
    if not rows:
        raise RuntimeError("All extracted geometries were empty")

    segments = gpd.GeoDataFrame(rows, geometry="geometry", crs=config["target_crs"])
    segments = segments.sort_values(
        ["length_km", "network_id", "structure_id"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    segments["structure_id"] = [
        f"DALY2025_MSC_{index:04d}" for index in range(1, len(segments) + 1)
    ]

    for path, target in (
        (output_directory / config["outputs"]["gpkg"], "gpkg"),
        (resolve_path(config["outputs"]["overlay_png"]), "overlay_png"),
        (resolve_path(config["outputs"]["segments_png"]), "segments_png"),
        (resolve_path(config["outputs"]["overlay_detail_png"]), "overlay_detail_png"),
        (resolve_path(config["outputs"]["map_png"]), "map_png"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    segments.to_file(
        output_paths["gpkg"],
        layer=config["outputs"]["layer"],
        driver="GPKG",
        engine="pyogrio",
        index=False,
    )
    attribute_columns = [column for column in segments.columns if column != "geometry"]
    segments[attribute_columns].to_csv(output_paths["attributes_csv"], index=False, encoding="utf-8-sig")
    _write_overlays(rgb, paths, output_paths)
    foundation_config = load_config(config["source"]["foundation_config"])
    _write_map(
        segments,
        resolve_path(config["source"]["domain_gpkg"]),
        resolve_path(config["source"]["boundary_gpkg"]),
        foundation_config,
        output_paths["map_png"],
    )

    class_counts = Counter(segments["line_class"])
    confidence_counts = Counter(segments["extraction_confidence"])
    lengths = segments["length_km"].to_numpy(dtype=float)
    bounds = _source_extent_m(metadata, config["target_crs"])
    summary = {
        "schema_version": config["schema_version"],
        "build_id": _utc_stamp(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_raster": relative_posix(source_path),
        "source_sha256": sha256_file(source_path),
        "source_figure": config["source"]["figure"],
        "source_doi": config["source"]["doi"],
        "target_crs": config["target_crs"],
        "parameters": params,
        "pixel_counts": {
            "dark_pixels": int(dark.sum()),
            "core_pixels_before_exclusions": core_pixels_before_masking,
            "core_pixels_after_exclusions_and_filtering": int(core.sum()),
            "skeleton_pixels": int(skeleton.sum()),
            "skeleton_pixels_removed_by_thinning": removed_skeleton_pixels,
            "candidate_components_before_filtering": int(component_count),
            "retained_components": int(len(retained_components)),
        },
        "features": int(len(segments)),
        "networks": int(segments["network_id"].nunique()),
        "total_length_km": round(float(lengths.sum()), 3),
        "length_km_quantiles": {
            str(percentile): round(float(np.percentile(lengths, percentile)), 3)
            for percentile in (0, 25, 50, 75, 90, 95, 99, 100)
        },
        "line_class_counts": dict(sorted(class_counts.items())),
        "extraction_confidence_counts": dict(sorted(confidence_counts.items())),
        "named_features": int(segments["structure_name"].notna().sum()),
        "typed_features": int(segments["structure_type"].notna().sum()),
        "source_extent_m": {
            "xmin": bounds[0],
            "ymin": bounds[1],
            "xmax": bounds[2],
            "ymax": bounds[3],
        },
        "outputs": {
            "geopackage": config["outputs"]["gpkg"],
            "layer": config["outputs"]["layer"],
            "attributes_csv": config["outputs"]["attributes_csv"],
            "overlay_png": config["outputs"]["overlay_png"],
            "segments_png": config["outputs"]["segments_png"],
            "overlay_detail_png": config["outputs"]["overlay_detail_png"],
            "map_png": config["outputs"]["map_png"],
        },
        "quality_status": {
            "all_geometries_valid": bool(segments.geometry.is_valid.all()),
            "all_lengths_positive": bool((segments.geometry.length > 0).all()),
            "crs": segments.crs.to_string(),
            "all_names_unassigned": bool(segments["structure_name"].isna().all()),
            "all_types_unassigned": bool(segments["structure_type"].isna().all()),
        },
        "interpretation": (
            "The layer digitises the thick black structural ribbons visible in Figure 14. "
            "The Figure 12 caption explicitly identifies emboldened linework as major fault zones. "
            "Extraction is cartographic rather than author-vector data, and names, kinematics and hierarchy "
            "are not inferred from proximity."
        ),
    }
    write_json(output_paths["summary_json"], summary)

    manifest = {
        "schema_version": config["schema_version"],
        "build_id": summary["build_id"],
        "generated_utc": summary["generated_utc"],
        "build_command": f"python -m src.geology_foundation.major_structures --config {Path(config_path).as_posix()} --overwrite",
        "software": software_versions(),
        "config": file_record(resolve_path(config_path), role="major-structure extraction configuration"),
        "source": file_record(source_path, role="Daly et al. (2025) Figure 14 georeferenced raster"),
        "parameters": params,
        "outputs": [
            file_record(output_paths["gpkg"], role="major-structure candidate vector layer"),
            file_record(output_paths["attributes_csv"], role="major-structure attributes"),
            file_record(output_paths["summary_json"], role="major-structure summary"),
            file_record(output_paths["overlay_png"], role="full-figure extraction overlay"),
            file_record(output_paths["segments_png"], role="segment-identity overlay"),
            file_record(output_paths["overlay_detail_png"], role="detail extraction overlay"),
            file_record(output_paths["map_png"], role="clean cartographic major-structure map"),
        ],
        "summary_file": relative_posix(output_paths["summary_json"]),
        "protected_v14_reference": {
            "path": config["source"]["v14_training_csv"],
            "sha256": config["source"]["v14_training_sha256"],
        },
    }
    write_json(output_paths["manifest_json"], manifest)
    print(f"Major-structure build complete: {summary['build_id']}")
    print(f"Vector layer: {output_paths['gpkg']}")
    print(f"Features: {summary['features']}; total length: {summary['total_length_km']:.1f} km")
    print(f"Overlay: {output_paths['overlay_png']}")
    print(f"Map: {output_paths['map_png']}")
    print(f"Manifest: {output_paths['manifest_json']}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/major_structures.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    build(args.config, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
