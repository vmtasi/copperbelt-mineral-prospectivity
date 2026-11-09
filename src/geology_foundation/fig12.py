from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer

from .common import load_config, resolve_path


def pixel_to_geographic(
    x: float,
    y: float,
    georeference: dict[str, Any],
) -> tuple[float, float]:
    lon_a, lon_b = georeference["longitude_ticks"]
    lat_a, lat_b = georeference["latitude_ticks"]
    longitude = lon_a["longitude"] + (x - lon_a["pixel_x"]) * (
        (lon_b["longitude"] - lon_a["longitude"]) / (lon_b["pixel_x"] - lon_a["pixel_x"])
    )
    latitude = lat_a["latitude"] + (y - lat_a["pixel_y"]) * (
        (lat_b["latitude"] - lat_a["latitude"]) / (lat_b["pixel_y"] - lat_a["pixel_y"])
    )
    return float(longitude), float(latitude)


def geographic_to_pixel(
    longitude: float,
    latitude: float,
    georeference: dict[str, Any],
) -> tuple[float, float]:
    lon_a, lon_b = georeference["longitude_ticks"]
    lat_a, lat_b = georeference["latitude_ticks"]
    x = lon_a["pixel_x"] + (longitude - lon_a["longitude"]) * (
        (lon_b["pixel_x"] - lon_a["pixel_x"]) / (lon_b["longitude"] - lon_a["longitude"])
    )
    y = lat_a["pixel_y"] + (latitude - lat_a["latitude"]) * (
        (lat_b["pixel_y"] - lat_a["pixel_y"]) / (lat_b["latitude"] - lat_a["latitude"])
    )
    return float(x), float(y)


def geometry_to_pixels(
    geometry,
    georeference: dict[str, Any],
) -> list[list[tuple[float, float]]]:
    transformer = Transformer.from_crs("EPSG:32735", "EPSG:4326", always_xy=True)
    parts = geometry.geoms if geometry.geom_type == "MultiLineString" else [geometry]
    projected_parts = []
    for part in parts:
        longitudes, latitudes = transformer.transform(
            np.asarray([point[0] for point in part.coords]),
            np.asarray([point[1] for point in part.coords]),
        )
        projected_parts.append(
            [
                geographic_to_pixel(longitude, latitude, georeference)
                for longitude, latitude in zip(longitudes, latitudes)
            ]
        )
    return projected_parts


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)


def draw_alignment(
    figure12_path: str | Path,
    segments: gpd.GeoDataFrame,
    georeference: dict[str, Any],
    output_path: str | Path,
) -> None:
    base = Image.open(figure12_path).convert("RGB")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for geometry in segments.geometry:
        for part in geometry_to_pixels(geometry, georeference):
            draw.line(part, fill=(220, 10, 30, 210), width=3, joint="curve")
    composed = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    composed.save(output_path, optimize=True)


def draw_segment_ids(
    figure12_path: str | Path,
    segments: gpd.GeoDataFrame,
    georeference: dict[str, Any],
    output_path: str | Path,
) -> None:
    base = Image.open(figure12_path).convert("RGB")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _font(10)
    for row in segments.itertuples():
        parts = geometry_to_pixels(row.geometry, georeference)
        for part in parts:
            draw.line(part, fill=(220, 10, 30, 190), width=2, joint="curve")
        midpoint = row.geometry.interpolate(0.5, normalized=True)
        x, y = geometry_to_pixels(midpoint, georeference)[0][0]
        label = row.structure_id.replace("DALY2025_MSC_", "")
        draw.text((x + 3, y + 2), label, font=font, fill=(0, 25, 125, 255), stroke_width=2, stroke_fill=(255, 255, 255, 235))
    composed = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    composed.save(output_path, optimize=True)


if __name__ == "__main__":
    config = load_config("configs/major_structures_v2.json")
    source = config["sources"]
    outputs = config["outputs"]
    v1 = gpd.read_file(
        resolve_path(source["v1_gpkg"]),
        layer=source["v1_layer"],
        engine="pyogrio",
    )
    draw_alignment(
        resolve_path(source["figure12_raster"]),
        v1,
        config["figure12_georeference"],
        resolve_path(outputs["figure12_alignment_png"]),
    )
    draw_segment_ids(
        resolve_path(source["figure12_raster"]),
        v1,
        config["figure12_georeference"],
        resolve_path(outputs["figure12_segment_ids_png"]),
    )
