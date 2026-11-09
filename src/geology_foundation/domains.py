from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import geopandas as gpd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from shapely import wkb
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Polygon
from shapely.ops import unary_union
from osgeo import gdal, ogr, osr

from .common import disk, keep_components


def read_rgb_tiff(path: str | Path) -> tuple[np.ndarray, tuple[float, ...], str, dict[str, Any]]:
    dataset = gdal.Open(str(path))
    if dataset is None:
        raise RuntimeError(f"Could not open RGB raster: {path}")
    if dataset.RasterCount < 3:
        raise ValueError(f"Expected at least three bands in {path}")
    bands = [dataset.GetRasterBand(index).ReadAsArray() for index in (1, 2, 3)]
    array = np.dstack(bands).astype(np.uint8, copy=False)
    metadata = {
        "width": dataset.RasterXSize,
        "height": dataset.RasterYSize,
        "band_count": dataset.RasterCount,
        "crs_wkt": dataset.GetProjection(),
        "geotransform": list(dataset.GetGeoTransform()),
        "source_band_dtype": gdal.GetDataTypeName(dataset.GetRasterBand(1).DataType),
    }
    return array, dataset.GetGeoTransform(), dataset.GetProjection(), metadata


def _remove_legend_components(
    seed: np.ndarray,
    zones: list[dict[str, Any]],
    max_pixels: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    labels, count = ndi.label(seed, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel())
    keep = np.ones(count + 1, dtype=bool)
    keep[0] = False
    removed_components = 0
    removed_pixels = 0
    for label, window in enumerate(ndi.find_objects(labels), start=1):
        if window is None or sizes[label] >= max_pixels:
            continue
        y0, y1 = window[0].start, window[0].stop
        x0, x1 = window[1].start, window[1].stop
        for zone in zones:
            inside = (
                x0 >= zone["x0"]
                and x1 <= zone["x1"]
                and y0 >= zone["y0"]
                and y1 <= zone["y1"]
            )
            if inside:
                keep[label] = False
                removed_components += 1
                removed_pixels += int(sizes[label])
                break
    return keep[labels], {
        "removed_components": removed_components,
        "removed_pixels": removed_pixels,
    }


def classify_figure14(
    rgb: np.ndarray,
    daly_config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    params = daly_config["pixel_parameters"]
    classes = daly_config["classes"]
    palette = np.asarray([item["rgb"] for item in classes], dtype=np.int32)
    class_values = np.asarray([item["class_value"] for item in classes], dtype=np.uint8)

    distances = np.sqrt(
        ((rgb.astype(np.int32)[:, :, None, :] - palette[None, None, :, :]) ** 2).sum(axis=3)
    )
    nearest_index = distances.argmin(axis=2)
    nearest_class = class_values[nearest_index]
    seed = distances.min(axis=2) <= int(params["color_tolerance_rgb"])

    seed, legend_info = _remove_legend_components(
        seed,
        zones=params["legend_exclusion_zones_px"],
        max_pixels=int(params["legend_component_max_pixels"]),
    )

    seed_connect_radius = int(params.get("seed_connect_radius_px", 0))
    envelope_seed_input = seed
    if seed_connect_radius > 0:
        envelope_seed_input = ndi.binary_closing(
            seed,
            structure=disk(seed_connect_radius),
        )
    envelope_seed, _ = keep_components(
        envelope_seed_input,
        min_area=int(params["envelope_component_min_pixels"]),
    )
    if not envelope_seed.any():
        raise RuntimeError("No figure colour components survived legend and size filtering")
    envelope = ndi.binary_closing(
        envelope_seed,
        structure=disk(int(params["envelope_close_radius_px"])),
    )
    envelope = ndi.binary_fill_holes(envelope)
    envelope, envelope_components = keep_components(
        envelope,
        min_area=int(params["envelope_keep_min_pixels"]),
    )
    if not envelope.any():
        raise RuntimeError("The reconstructed map envelope is empty")

    class_seeds = np.zeros_like(seed, dtype=bool)
    class_seed_counts: dict[str, int] = {}
    for item in classes:
        value = int(item["class_value"])
        members = seed & (nearest_class == value) & envelope
        if item["domain_id"] == "6":
            members, _ = keep_components(
                members,
                min_area=int(params["domain_6_seed_min_pixels"]),
                min_side=int(params["domain_6_seed_min_side_px"]),
            )
        else:
            members, _ = keep_components(
                members,
                min_area=int(params["class_seed_min_pixels"]),
                min_side=int(params["class_seed_min_side_px"]),
            )
        class_seed_counts[item["domain_id"]] = int(members.sum())
        if not members.any():
            raise RuntimeError(f"No colour seed pixels remain for domain {item['domain_id']}")
        class_seeds |= members

    indices = ndi.distance_transform_edt(
        ~class_seeds,
        return_distances=False,
        return_indices=True,
    )
    assigned = nearest_class[tuple(indices)].astype(np.uint8)
    assigned[~envelope] = 0
    assigned = ndi.median_filter(
        assigned,
        size=int(params["median_filter_px"]),
        mode="nearest",
    )
    assigned[~envelope] = 0

    for value in class_values:
        members, _ = keep_components(
            assigned == value,
            min_area=int(params["min_domain_component_pixels"]),
        )
        assigned[(assigned == value) & ~members] = 0

    if (assigned == 0).any():
        fill_indices = ndi.distance_transform_edt(
            assigned == 0,
            return_distances=False,
            return_indices=True,
        )
        filled = assigned[tuple(fill_indices)]
        assigned[(assigned == 0) & envelope] = filled[(assigned == 0) & envelope]

    missing = sorted(set(class_values.tolist()) - set(np.unique(assigned).tolist()))
    if missing:
        raise RuntimeError(f"Reconstruction lost required class values: {missing}")

    stats: dict[str, Any] = {
        "legend_removal": legend_info,
        "envelope_pixels": int(envelope.sum()),
        "envelope_components_retained": envelope_components,
        "class_seed_pixels": class_seed_counts,
        "class_pixels": {},
        "class_components": {},
    }
    for item in classes:
        value = int(item["class_value"])
        members = assigned == value
        _, component_count = ndi.label(members, structure=np.ones((3, 3), dtype=np.uint8))
        stats["class_pixels"][item["domain_id"]] = int(members.sum())
        stats["class_components"][item["domain_id"]] = int(component_count)
    return assigned, stats


def write_diagnostic(
    source_rgb: np.ndarray,
    assigned: np.ndarray,
    daly_config: dict[str, Any],
    path: str | Path,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    classified = np.full_like(source_rgb, 255, dtype=np.uint8)
    for item in daly_config["classes"]:
        classified[assigned == int(item["class_value"])] = np.asarray(item["rgb"], dtype=np.uint8)

    left = Image.fromarray(source_rgb)
    right = Image.fromarray(classified)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(right)
    for item in daly_config["classes"]:
        members = assigned == int(item["class_value"])
        labels, count = ndi.label(members, structure=np.ones((3, 3), dtype=np.uint8))
        if count == 0:
            continue
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        largest = int(sizes.argmax())
        center_y, center_x = ndi.center_of_mass(labels == largest)
        text = f"{item['domain_id']} {item['abbreviation']}"
        draw.text(
            (center_x, center_y),
            text,
            anchor="mm",
            fill=(0, 0, 0),
            font=font,
            stroke_width=2,
            stroke_fill=(255, 255, 255),
        )

    canvas = Image.new("RGB", (left.width + right.width + 8, left.height), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + 8, 0))
    canvas.save(target, optimize=True)
    return target


def write_categorical_raster(
    assigned: np.ndarray,
    geotransform: tuple[float, ...],
    projection: str,
    destination: str | Path,
    *,
    target_crs: str,
    resolution_m: int,
) -> dict[str, Any]:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    memory = gdal.GetDriverByName("MEM").Create(
        "", assigned.shape[1], assigned.shape[0], 1, gdal.GDT_Byte
    )
    memory.SetGeoTransform(geotransform)
    memory.SetProjection(projection)
    band = memory.GetRasterBand(1)
    band.WriteArray(assigned)
    band.SetNoDataValue(0)
    band.SetDescription("Daly et al. (2025) Figure 14 domain class")

    options = gdal.WarpOptions(
        dstSRS=target_crs,
        xRes=int(resolution_m),
        yRes=int(resolution_m),
        resampleAlg=gdal.GRA_NearestNeighbour,
        dstNodata=0,
        outputType=gdal.GDT_Byte,
        creationOptions=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"],
    )
    result = gdal.Warp(str(destination), memory, options=options)
    if result is None:
        raise RuntimeError(f"Failed to write categorical raster: {destination}")
    width, height = result.RasterXSize, result.RasterYSize
    geotransform_out = list(result.GetGeoTransform())
    result = None

    update = gdal.Open(str(destination), gdal.GA_Update)
    if update is not None:
        out_band = update.GetRasterBand(1)
        out_band.SetDescription("Daly et al. (2025) Figure 14 tectonic-domain class")
        out_band.SetMetadataItem("CLASS_CROSSWALK", "domain_crosswalk.csv")
        out_band.SetMetadataItem("SOURCE_DOI", "10.1029/2025TC008955")
        update = None
    return {
        "width": width,
        "height": height,
        "resolution_m": int(resolution_m),
        "geotransform": geotransform_out,
        "crs": target_crs,
    }


def _as_multipolygon(geometry: Any) -> MultiPolygon:
    if geometry.geom_type == "Polygon":
        return MultiPolygon([geometry])
    if geometry.geom_type == "MultiPolygon":
        return geometry
    if geometry.geom_type == "GeometryCollection":
        parts = [part for part in geometry.geoms if part.geom_type in {"Polygon", "MultiPolygon"}]
        polygons: list[Polygon] = []
        for part in parts:
            polygons.extend(list(part.geoms) if part.geom_type == "MultiPolygon" else [part])
        return MultiPolygon(polygons)
    raise TypeError(f"Cannot convert {geometry.geom_type} to MultiPolygon")


def polygonize_domains(
    raster_path: str | Path,
    daly_config: dict[str, Any],
) -> gpd.GeoDataFrame:
    dataset = gdal.Open(str(raster_path))
    if dataset is None:
        raise RuntimeError(f"Could not open categorical raster: {raster_path}")
    band = dataset.GetRasterBand(1)
    mask = band.GetMaskBand()
    memory = ogr.GetDriverByName("Memory").CreateDataSource("domain_polygonize")
    layer = memory.CreateLayer(
        "domains",
        srs=dataset.GetSpatialRef(),
        geom_type=ogr.wkbPolygon,
    )
    layer.CreateField(ogr.FieldDefn("class_value", ogr.OFTInteger))
    gdal.Polygonize(band, mask, layer, 0, [], callback=None)

    rows: list[dict[str, Any]] = []
    for feature in layer:
        geometry = feature.GetGeometryRef()
        if geometry is None:
            continue
        rows.append(
            {
                "class_value": int(feature.GetField(0)),
                "geometry": wkb.loads(bytes(geometry.ExportToWkb())),
            }
        )
    memory = None
    if not rows:
        raise RuntimeError("No polygons were produced from the categorical raster")
    raw = gpd.GeoDataFrame(rows, crs=dataset.GetSpatialRef().ExportToWkt())

    by_value = {item["class_value"]: item for item in daly_config["classes"]}
    output_rows: list[dict[str, Any]] = []
    for class_value, group in raw.groupby("class_value", sort=True):
        item = by_value[int(class_value)]
        geometry = _as_multipolygon(unary_union(group.geometry.tolist()))
        output_rows.append(
            {
                "class_value": int(class_value),
                "domain_id": item["domain_id"],
                "domain_number": item["domain_number"],
                "is_subdomain": bool(item["is_subdomain"]),
                "abbreviation": item["abbreviation"],
                "domain_name": item["name"],
                "parent_domain_id": item["parent_domain_id"],
                "figure_legend_text": item["figure_legend_text"],
                "source_doi": daly_config["source_doi"],
                "source_figure": "Figure 14",
                "geometry_method": "600 m categorical reconstruction from the georeferenced published figure",
                "geometry_confidence": "class interior verified; boundary positions map-derived and approximate",
                "notes": item["notes"],
                "geometry": geometry,
            }
        )
    output_rows.sort(key=lambda row: int(row["class_value"]))
    domains = gpd.GeoDataFrame(output_rows, crs=raw.crs).to_crs("EPSG:32735")
    domains["area_km2"] = domains.geometry.area / 1_000_000
    return domains


def _iter_line_parts(geometry: Any) -> Iterator[LineString]:
    if geometry is None or geometry.is_empty:
        return
    if geometry.geom_type in {"LineString", "LinearRing"}:
        if len(geometry.coords) >= 2:
            yield LineString(geometry.coords)
        return
    if isinstance(geometry, (MultiLineString, GeometryCollection)):
        for part in geometry.geoms:
            yield from _iter_line_parts(part)


def _as_multiline(part: LineString) -> MultiLineString:
    return MultiLineString([part])


def derive_boundaries(
    domains: gpd.GeoDataFrame,
    daly_config: dict[str, Any],
) -> tuple[gpd.GeoDataFrame, dict[str, float]]:
    records = domains.reset_index(drop=True)
    internal_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    outer_boundary = unary_union(records.geometry.tolist()).boundary

    for left_index in range(len(records)):
        left = records.iloc[left_index]
        for right_index in range(left_index + 1, len(records)):
            right = records.iloc[right_index]
            shared = left.geometry.boundary.intersection(right.geometry.boundary)
            for part in _iter_line_parts(shared):
                if part.length < 1.0:
                    continue
                internal_rows.append(
                    {
                        "boundary_type": "internal",
                        "domain_a": left.domain_id,
                        "domain_b": right.domain_id,
                        "length_km": part.length / 1000.0,
                        "source_doi": daly_config["source_doi"],
                        "source_figure": "Figure 14",
                        "geometry_method": "shared boundary of the 600 m categorical domain reconstruction",
                        "geometry_confidence": "map-derived; not a mapped fault",
                        "geometry": _as_multiline(part),
                    }
                )

    for _, row in records.iterrows():
        outside = row.geometry.boundary.intersection(outer_boundary)
        for part in _iter_line_parts(outside):
            if part.length < 1.0:
                continue
            external_rows.append(
                {
                    "boundary_type": "external",
                    "domain_a": row.domain_id,
                    "domain_b": None,
                    "length_km": part.length / 1000.0,
                    "source_doi": daly_config["source_doi"],
                    "source_figure": "Figure 14",
                    "geometry_method": "outer boundary of the 600 m categorical domain reconstruction",
                    "geometry_confidence": "map-derived; not a mapped geological contact",
                    "geometry": _as_multiline(part),
                }
            )

    rows = internal_rows + external_rows
    rows.sort(
        key=lambda row: (
            row["boundary_type"],
            str(row["domain_a"]),
            str(row["domain_b"] or ""),
            row["geometry"].bounds[0],
            row["geometry"].bounds[1],
        )
    )
    for index, row in enumerate(rows, start=1):
        row["boundary_id"] = f"DB{index:04d}"
    columns = [
        "boundary_id",
        "boundary_type",
        "domain_a",
        "domain_b",
        "length_km",
        "source_doi",
        "source_figure",
        "geometry_method",
        "geometry_confidence",
        "geometry",
    ]
    boundaries = gpd.GeoDataFrame(rows, columns=columns, crs=domains.crs)
    stats = {
        "internal_length_km": sum(row["length_km"] for row in internal_rows),
        "external_length_km": sum(row["length_km"] for row in external_rows),
        "internal_segments": len(internal_rows),
        "external_segments": len(external_rows),
    }
    return boundaries, stats
