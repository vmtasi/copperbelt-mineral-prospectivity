"""Preprocess and assess the three alternative Zambian geological maps.

Outputs are written only below
data/raw/zambia_acquisition/derived/alternative_geological_maps/.
No model, Phase 6, Phase 7 or frozen V14 artefact is modified.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image
from pyproj import Transformer
from scipy import ndimage
from shapely import wkb
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import linemerge, transform as shapely_transform, unary_union

from osgeo import gdal, ogr, osr


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/raw/zambia_acquisition/derived/alternative_geological_maps"
EXTRACTED = OUT / "extracted"
GEO = OUT / "georeferenced"
LAYERS = OUT / "source_layers"
CROSSWALK = OUT / "crosswalk"
REPORTS = OUT / "reports"
TARGET_CRS = "EPSG:32735"
WGS84 = "EPSG:4326"
RENDER_DPI = 600

PDFS = {
    "dewaele_2006": ROOT / "literature/The_Mesoproterozoic_Irumide_belt_of_Zamb.pdf",
    "hanson_1994": ROOT / "literature/Geologic_evolution_of_the_neoproterozoic.pdf",
    "zambia_overview": ROOT / "literature/3-540-29145-8_60.pdf",
}

SOURCES = {
    "dewaele_2006": {
        "citation": "De Waele et al. (2006), Journal of African Earth Sciences 46, 36-70",
        "figure": "Figure 2",
        "pdf_page": 3,
        "printed_page": 38,
        "map_crop": (370, 570, 4580, 3722),
        "figure_crop": (360, 565, 4585, 3885),
        "legend_crop": (2870, 690, 4570, 1690),
        "frame": (383.0, 586.0, 4566.5, 3714.0),
        "x_controls": [(645.0, 28.0), (1187.5, 29.0), (2815.5, 32.0),
                       (3358.5, 33.0), (3901.5, 34.0), (4444.0, 35.0)],
        "y_controls": [(1088.0, -8.0), (1630.0, -9.0), (2225.0, -10.0),
                       (2768.0, -11.0), (3310.0, -12.0)],
        "mask_rects": [(2860, 680, 4570, 1695), (2600, 2180, 4360, 3360)],
        "uncertainty_m": 5000,
        "scale_note": "Graticule spacing implies approximately 1:4.9 million; linework is generalised at this scale.",
    },
    "hanson_1994": {
        "citation": "Hanson, Wilson and Munyanyiwa (1994), Journal of African Earth Sciences 18(2), 135-150",
        "figure": "Figure 2",
        "pdf_page": 3,
        "printed_page": 137,
        "map_crop": (530, 610, 4450, 2815),
        "figure_crop": (280, 600, 4470, 3070),
        "legend_crop": (285, 880, 1040, 1910),
        "frame_corners": [(560.0, 875.0), (4435.0, 625.0),
                          (4445.0, 2809.0), (1064.0, 2809.0)],
        "x_controls": [(1374.0, 26.0), (2214.0, 28.0), (3056.0, 30.0), (3898.0, 32.0)],
        "y_controls": [(987.0, -14.0), (1835.5, -16.0), (2690.5, -18.0)],
        "place_controls": [
            (2330.0, 1600.0, 28.3228, -15.3875, "Lusaka (L)"),
            (3370.0, 2690.0, 31.0522, -17.8252, "Harare (H)"),
        ],
        "bottom_border_y": 2798.0,
        "right_border_x": 4435.0,
        "mask_rects": [(285, 880, 1050, 1910)],
        "uncertainty_m": 25000,
        "scale_note": "Graticule spacing implies approximately 1:6 million; frame is skewed and the scan has no independent projection statement. Control residuals reach 25 km, so 25 km is treated as a minimum positional uncertainty.",
    },
    "zambia_overview": {
        "citation": "Schlüter (2006), Geological Atlas of Africa, Springer, chapter 'Zambia', pp. 244-247; Figure 239 (printed page 245); modified after Drysdall et al. (1961, 1981) and Ng'Ambi et al. (1994)",
        "figure": "Figure 239",
        "pdf_page": 2,
        "printed_page": 245,
        "map_crop": (400, 485, 4088, 3845),
        "figure_crop": (390, 470, 4200, 5760),
        "legend_crop": (390, 3900, 4200, 5600),
        "frame": (409.5, 499.5, 4077.5, 3837.5),
        "x_controls": [(1413.5, 25.0), (2819.5, 30.0)],
        "y_controls": [(1221.2, -10.0), (2626.5, -15.0)],
        "mask_rects": [],
        "uncertainty_m": 10000,
        "scale_note": "Graticule spacing implies approximately 1:9 million; this is a national overview, not a local map.",
    },
}

CLASSES = {
    "zambia_overview": [
        ("katanga_system", "Conglomerates, shales, quartzites, greywackes, iron formations of the Katanga System", "Katanga System (Cambrian? - Neoproterozoic)", (255, 231, 196), "stratigraphic_unit_katanga", "partial", "Only the Katanga System family is shared with the Daly study; formal member equivalence is not established."),
        ("mozambique_belt", "Metamorphic rocks of the Mozambique Belt", "Neoproterozoic", (254, 218, 164), "basement_metamorphic_complex", "partial", "Regional metamorphic family only; no unit-level equivalence to Phase 6 evidence."),
        ("kibarani_system", "Metasediments, metavolcanics of the Kibarani System", "Mesoproterozoic", (251, 197, 113), "supracrustal_metasedimentary_unit", "not_equivalent", "A regional unit with no demonstrated one-to-one Phase 6 equivalent."),
        ("lusenga_syenites", "Syenites of the Lusenga Group", "Mesoproterozoic", (246, 168, 82), "intrusive_igneous_unit", "partial", "Intrusive family only."),
        ("ubendian_system", "High-grade metamorphic rocks of the Ubendian System", "Palaeoproterozoic", (248, 179, 172), "basement_metamorphic_complex", "partial", "Metamorphic basement family only."),
        ("pre_katanga_basement", "High-grade metamorphic rocks of the Pre-Katanga Basement (remobilised in part)", "Palaeoproterozoic", (245, 98, 134), "basement_metamorphic_complex", "partial", "Broad basement family; remobilisation is explicitly retained in the source term."),
        ("plateau_series", "Platform sediments of the Plateau Series, Abercorn Sandstones, Luapula-Luitikila Beds", "Palaeoproterozoic", (195, 80, 107), "cover_or_platform_sequence", "not_equivalent", "Composite legend term; no single Phase 6 equivalent."),
        ("granites_gneisses_volcanics", "Granites, gneisses, volcanic rocks", "Undifferentiated", (220, 125, 172), "undifferentiated_crystalline_basement", "not_equivalent", "Undifferentiated source class cannot be assigned a unique project class."),
        ("sandstones_mudstones", "Sandstones and mudstones", "Phanerozoic", (243, 244, 183), "cover_sequence", "partial", "Lithology-level cover class only."),
        ("karoo_supergroup", "Continental sediments of the Dwyka, Ecca, Beaufort, Stormberg Groups; Karoo Supergroup", "Lower Jurassic - Upper Carboniferous", (169, 211, 209), "cover_sequence", "partial", "Karoo basin fill is represented only as a cover family."),
    ],
    "dewaele_2006": [
        ("katanga_supergroup_younger_cover", "Katanga Supergroup and younger cover", "Katanga Supergroup and younger cover", (255, 252, 197), "stratigraphic_unit_katanga", "partial", "The Katanga Supergroup is the closest regional analogue, but the source combines it with younger cover."),
        ("kasama_formation", "Kasama Formation", "Muva Supergroup", (255, 190, 0), "stratigraphic_unit_muva", "partial", "Mapped at formation level; only the Muva Supergroup family is cross-walked."),
        ("manshya_mafingi_mitoba_groups", "Manshya River, Mafingi and Mitoba River Groups", "Muva Supergroup", (255, 255, 1), "stratigraphic_unit_muva", "partial", "Multiple groups are combined in one source legend item."),
        ("mporokoso_group", "Mporokoso Group", "Muva Supergroup", (164, 104, 52), "stratigraphic_unit_muva", "partial", "Only supergroup-level correspondence is asserted."),
        ("granitoids_group_iv_ii", "Granitoids (Group IV and II suites)", "Igneous rocks", (254, 0, 3), "intrusive_igneous_unit", "partial", "Granitoid family only."),
        ("bangweulu_granitoids", "Bangweulu granitoids (Group Ib suite, 1.88-1.83 Ga)", "Igneous rocks", (255, 166, 166), "intrusive_igneous_unit", "partial", "Granitoid family only."),
        ("granulites_migmatites_mulungwizi", "Granulites, migmatites and schist belts including Mulungwizi Gneiss (M) and Luwalizi Granite (L) (2.09-1.93 Ga)", "Igneous and metamorphic rocks", (255, 109, 110), "basement_metamorphic_complex", "partial", "Composite high-grade metamorphic class; no unit-level equivalence."),
    ],
}

CROSSWALK_EXTRA = [
    ("dewaele_2006", "Syenite with whole rock Rb-Sr date", "Igneous rocks", "point_symbol", "intrusive_igneous_marker", "partial", "Point symbol only; mapped where the magenta datum is printed."),
    ("dewaele_2006", "Luapula volcanic sequence (Group Ib suite, 1.88-1.82 Ga)", "Igneous rocks", "polygon_unit", "volcanic_unit", "not_equivalent", "Black unit polygons were not separated from black linework and labels; retained in the crosswalk only."),
    ("zambia_overview", "Kimberlites", "Phanerozoic", "point_symbol", "intrusive_igneous_marker", "partial", "Source uses triangle symbols; marker positions were not digitised in this pass."),
    ("zambia_overview", "Fault", "Structures", "line_symbol", "fault_unclassified", "partial", "Source legend defines a fault line symbol, but individual traces are not separable from administrative and hydrographic linework at this scale."),
    ("dewaele_2006", "Shear zone (named on figure)", "Structures", "line_symbol", "fault_shear_zone", "partial", "Named shear-zone bands are manually traced; kinematics are not restated."),
    ("hanson_1994", "Great Dyke", "Structures", "line_symbol", "intrusive_linear_feature", "not_equivalent", "Dyke trace is explicit on the source map; no Phase 6 equivalent is asserted."),
    ("hanson_1994", "Mwembeshi dislocation", "Structures", "line_symbol", "fault_shear_zone", "not_equivalent", "Named trace is explicit on the source map; continuity outside the figure is not inferred."),
    ("hanson_1994", "Phanerozoic cover", "Legend unit", "polygon_unit", "cover_sequence", "partial", "Source pattern not digitally separated from the underlying linework in this pass."),
    ("hanson_1994", "Katangan Supergroup", "Legend unit", "polygon_unit", "stratigraphic_unit_katanga", "partial", "Source pattern not digitally separated from the underlying linework in this pass."),
    ("hanson_1994", "Orthogneiss", "Zambezi Belt", "polygon_unit", "basement_metamorphic_complex", "not_equivalent", "Source pattern not digitally separated; unit-level equivalence is not asserted."),
    ("hanson_1994", "Supracrustal rocks", "Zambezi Belt", "polygon_unit", "supracrustal_metasedimentary_unit", "not_equivalent", "Source pattern not digitally separated; unit-level equivalence is not asserted."),
    ("hanson_1994", "Remobilized basement", "Zambezi Belt", "polygon_unit", "basement_metamorphic_complex", "partial", "Source pattern not digitally separated; broad basement family only."),
    ("hanson_1994", "Middle Proterozoic", "Legend unit", "polygon_unit", "undifferentiated_crystalline_basement", "not_equivalent", "Broad age-defined source class; no demonstrated Phase 6 equivalent."),
    ("hanson_1994", "Lower Proterozoic", "Legend unit", "polygon_unit", "undifferentiated_crystalline_basement", "not_equivalent", "Broad age-defined source class; no demonstrated Phase 6 equivalent."),
    ("hanson_1994", "Archean", "Legend unit", "polygon_unit", "undifferentiated_crystalline_basement", "not_equivalent", "Broad age-defined source class; no demonstrated Phase 6 equivalent."),
]


def clean_dir(path: Path, keep_suffixes: tuple[str, ...] = ()) -> None:
    if path.exists():
        for child in path.iterdir():
            if child.is_file() or child.is_symlink():
                if child.suffix.lower() in keep_suffixes:
                    continue
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
    path.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def render_page(pdf: Path, page: int, target: Path) -> Image.Image:
    exe = os.environ.get("POPPLER_EXE") or shutil.which("pdftoppm")
    if not exe:
        raise RuntimeError("pdftoppm is required to regenerate the source rasters")
    prefix = target.with_suffix("")
    subprocess.run(
        [exe, "-f", str(page), "-l", str(page), "-r", str(RENDER_DPI), "-png", "-singlefile", str(pdf), str(prefix)],
        check=True,
    )
    return Image.open(target).convert("RGB")


def disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return x * x + y * y <= radius * radius


def fill_small_holes(mask: np.ndarray, max_pixels: int) -> np.ndarray:
    filled = ndimage.binary_fill_holes(mask)
    holes = filled & ~mask
    labels, count = ndimage.label(holes)
    if count:
        sizes = np.bincount(labels.ravel())
        remove = sizes < max_pixels
        remove[0] = False
        mask = mask | remove[labels]
    return mask


def fit_axis(pixel: list[float], values: list[float]) -> tuple[float, float, np.ndarray]:
    if len(pixel) < 2:
        raise ValueError("at least two graticule controls are required")
    slope, intercept = np.polyfit(np.asarray(values), np.asarray(pixel), 1)
    predicted = slope * np.asarray(values) + intercept
    return float(slope), float(intercept), np.asarray(pixel) - predicted


def source_transform(source_id: str) -> dict:
    src = SOURCES[source_id]
    if source_id == "hanson_1994":
        # The frame is a projected graticule: neither border is a parallel or a
        # meridian.  Fit one affine per coordinate using every labelled control,
        # with its true pixel position, then use place labels as consistency checks.
        bottom_y = src["bottom_border_y"]
        right_x = src["right_border_x"]
        lon_rows = [(x, bottom_y, lon) for x, lon in src["x_controls"]]
        lon_rows += [(x, y, lon) for x, y, lon, _, _ in src["place_controls"]]
        lon_coef = np.linalg.lstsq(
            np.asarray([[x, y, 1.0] for x, y, _ in lon_rows]),
            np.asarray([lon for _, _, lon in lon_rows]), rcond=None)[0]
        lat_rows = [(right_x, y, lat) for y, lat in src["y_controls"]]
        lat_rows += [(x, y, lat) for x, y, _, lat, _ in src["place_controls"]]
        lat_coef = np.linalg.lstsq(
            np.asarray([[x, y, 1.0] for x, y, _ in lat_rows]),
            np.asarray([lat for _, _, lat in lat_rows]), rcond=None)[0]
        lon_right = float(lon_coef @ np.asarray([right_x, bottom_y, 1.0]))
        lat_bottom = float(lat_coef @ np.asarray([right_x, bottom_y, 1.0]))
        points = [(x, bottom_y, lon, lat_bottom, "border longitude tick")
                  for x, lon in src["x_controls"]]
        points += [(right_x, y, lon_right, lat, "border latitude tick")
                   for y, lat in src["y_controls"]]
        points += [(x, y, lon, lat, name) for x, y, lon, lat, name in src["place_controls"]]
        # Local scale at the frame centre, used only to express residuals in pixels.
        x_slope = 1.0 / float(lon_coef[0])
        y_slope = 1.0 / float(lat_coef[1])
        return {
            "kind": "affine",
            "lon_coef": lon_coef,
            "lat_coef": lat_coef,
            "points": points,
            "x_slope": float(x_slope),
            "y_slope": float(y_slope),
            "lon_right": lon_right,
            "lat_bottom": lat_bottom,
        }
    x_slope, x_intercept, x_res = fit_axis(
        [p[0] for p in src["x_controls"]], [p[1] for p in src["x_controls"]]
    )
    y_slope, y_intercept, y_res = fit_axis(
        [p[0] for p in src["y_controls"]], [p[1] for p in src["y_controls"]]
    )
    return {
        "kind": "axis",
        "x_slope": x_slope,
        "x_intercept": x_intercept,
        "y_slope": y_slope,
        "y_intercept": y_intercept,
        "x_residuals": x_res,
        "y_residuals": y_res,
    }


def px_to_lonlat(x: float, y: float, tr: dict) -> tuple[float, float]:
    if tr["kind"] == "axis":
        return (x - tr["x_intercept"]) / tr["x_slope"], (y - tr["y_intercept"]) / tr["y_slope"]
    lon = tr["lon_coef"][0] * x + tr["lon_coef"][1] * y + tr["lon_coef"][2]
    lat = tr["lat_coef"][0] * x + tr["lat_coef"][1] * y + tr["lat_coef"][2]
    return float(lon), float(lat)


def crop_geotransform(box: tuple[int, int, int, int], tr: dict) -> tuple[float, ...]:
    x0, y0, _, _ = box
    lon0, lat0 = px_to_lonlat(x0, y0, tr)
    lonx, latx = px_to_lonlat(x0 + 1, y0, tr)
    lony, laty = px_to_lonlat(x0, y0 + 1, tr)
    return lon0, lonx - lon0, lony - lon0, lat0, latx - lat0, laty - lat0


def write_geotiffs(source_id: str, page: Image.Image, tr: dict) -> tuple[Path, Path]:
    box = SOURCES[source_id]["map_crop"]
    crop = masked_map_crop(page, source_id)
    arr = np.asarray(crop).copy()
    h, w = arr.shape[:2]
    mem = gdal.GetDriverByName("MEM").Create("", w, h, 3, gdal.GDT_Byte)
    for band in range(3):
        mem.GetRasterBand(band + 1).WriteArray(arr[:, :, band])
    srs = osr.SpatialReference(); srs.ImportFromEPSG(4326)
    mem.SetProjection(srs.ExportToWkt())
    mem.SetGeoTransform(crop_geotransform(box, tr))
    out_ll = GEO / f"{source_id}_map_epsg4326.tif"
    gdal.Translate(str(out_ll), mem, options=["-co", "COMPRESS=LZW", "-co", "TILED=YES"])
    out_utm = GEO / f"{source_id}_map_epsg32735_250m.tif"
    gdal.Warp(
        str(out_utm), mem, dstSRS=TARGET_CRS, xRes=250, yRes=250,
        resampleAlg="near", creationOptions=["COMPRESS=LZW", "TILED=YES"],
    )
    return out_ll, out_utm


def masked_map_crop(page: Image.Image, source_id: str) -> Image.Image:
    box = SOURCES[source_id]["map_crop"]
    crop = page.crop(box).convert("RGB")
    x0, y0, _, _ = box
    draw = Image.new("L", crop.size, 255)
    from PIL import ImageDraw
    if source_id == "hanson_1994":
        draw = Image.new("L", crop.size, 0)
        ImageDraw.Draw(draw).polygon([(x - x0, y - y0) for x, y in SOURCES[source_id]["frame_corners"]], fill=255)
    arr = np.array(crop, dtype=np.uint8, copy=True)
    for left, top, right, bottom in SOURCES[source_id]["mask_rects"]:
        draw.paste(0, (max(0, left - x0), max(0, top - y0), max(0, right - x0), max(0, bottom - y0)))
    arr[np.asarray(draw) == 0] = 255
    return Image.fromarray(arr)


def make_class_raster(source_id: str, page: Image.Image) -> tuple[np.ndarray, tuple[float, ...]]:
    box = SOURCES[source_id]["map_crop"]
    x0, y0, x1, y1 = box
    arr = np.asarray(page.crop(box)).astype(np.int32)
    palette = np.asarray([item[3] for item in CLASSES[source_id]], dtype=np.int32)
    distance = ((arr[:, :, None, :] - palette[None, None, :, :]) ** 2).sum(axis=3)
    nearest = distance.argmin(axis=2) + 1
    root_distance = np.sqrt(distance.min(axis=2))
    classes = np.where(root_distance <= 42, nearest, 0).astype(np.uint8)
    for left, top, right, bottom in SOURCES[source_id]["mask_rects"]:
        classes[max(0, top - y0):max(0, bottom - y0), max(0, left - x0):max(0, right - x0)] = 0
    cleaned = np.zeros_like(classes)
    for class_id in range(1, len(CLASSES[source_id]) + 1):
        mask = classes == class_id
        mask = ndimage.binary_closing(mask, structure=disk(5))
        mask = fill_small_holes(mask, 3000)
        labels, count = ndimage.label(mask)
        if count:
            sizes = np.bincount(labels.ravel())
            remove = sizes < 220
            remove[0] = False
            mask[remove[labels]] = False
        cleaned[mask] = class_id
    return cleaned, crop_geotransform(box, source_transform(source_id))


def transform_geometry(geom, transformer: Transformer):
    return shapely_transform(transformer.transform, geom)


def polygonize_units(source_id: str, page: Image.Image, transformer: Transformer) -> gpd.GeoDataFrame:
    classes, gt = make_class_raster(source_id, page)
    h, w = classes.shape
    mem = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte)
    mem.GetRasterBand(1).WriteArray(classes)
    mem.SetGeoTransform(gt)
    srs = osr.SpatialReference(); srs.ImportFromEPSG(4326)
    mem.SetProjection(srs.ExportToWkt())
    vector = ogr.GetDriverByName("Memory").CreateDataSource("")
    layer = vector.CreateLayer("units", srs=srs, geom_type=ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("class_id", ogr.OFTInteger))
    gdal.Polygonize(mem.GetRasterBand(1), mem.GetRasterBand(1), layer, 0, [], callback=None)
    records = []
    meta = {i + 1: item for i, item in enumerate(CLASSES[source_id])}
    for feature in layer:
        class_id = feature.GetField("class_id")
        if class_id not in meta:
            continue
        geom = transform_geometry(wkb.loads(bytes(feature.GetGeometryRef().ExportToWkb())), transformer)
        if geom.is_empty:
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
        geom = geom.simplify(100, preserve_topology=True)
        if geom.is_empty or geom.area / 1e6 < 0.5:
            continue
        source_class, legend_term, parent, rgb, harmonised, status, rationale = meta[class_id]
        records.append({
            "source_id": source_id,
            "source_class": source_class,
            "legend_term": legend_term,
            "source_parent": parent,
            "source_color": "#%02x%02x%02x" % rgb,
            "harmonised_class": harmonised,
            "equivalence_status": status,
            "geometry_role": "geological_unit_polygon",
            "digitisation_method": "colour segmentation at 600 dpi; morphological cleanup; polygonised with GDAL",
            "positional_uncertainty_m": SOURCES[source_id]["uncertainty_m"],
            "confidence": "medium" if source_id == "dewaele_2006" else "low",
            "source_pdf_page": SOURCES[source_id]["pdf_page"],
            "source_figure": SOURCES[source_id]["figure"],
            "source_pdf_sha256": sha256(PDFS[source_id]),
            "area_km2": geom.area / 1e6,
            "notes": rationale,
            "geometry": geom,
        })
    if not records:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=TARGET_CRS)
    result = gpd.GeoDataFrame(records, geometry="geometry", crs=TARGET_CRS)
    result["unit_id"] = [f"{source_id}_u{i:05d}" for i in range(1, len(result) + 1)]
    return result


def derive_contacts(units: gpd.GeoDataFrame, source_id: str) -> gpd.GeoDataFrame:
    if units.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=TARGET_CRS)
    merged = unary_union([geom for geom in units.geometry if geom is not None])
    boundary = merged.boundary
    merged_lines = linemerge(boundary) if not boundary.is_empty else boundary
    parts = list(merged_lines.geoms) if hasattr(merged_lines, "geoms") else [merged_lines]
    rows = []
    for geom in parts:
        if geom.is_empty or geom.length == 0:
            continue
        rows.append({
            "source_id": source_id,
            "contact_type": "contact_derived_from_adjacent_unit_polygons",
            "source_class": "derived boundary",
            "harmonised_class": "contact_stratigraphic",
            "equivalence_status": "unmapped",
            "geometry_role": "geological_contact",
            "digitisation_method": "boundary of colour-class polygons; may include map-edge segments",
            "positional_uncertainty_m": SOURCES[source_id]["uncertainty_m"],
            "confidence": "low",
            "length_km": geom.length / 1000,
            "notes": "Derived contact, not an independently traced source line. Map-edge segments are retained and flagged.",
            "geometry": geom,
        })
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=TARGET_CRS)


def write_layers(path: Path, layers: list[tuple[str, gpd.GeoDataFrame, list[tuple[str, int]]]]) -> None:
    if path.exists():
        path.unlink()
    ds = ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
    srs = osr.SpatialReference(); srs.ImportFromEPSG(32735)
    for layer_name, frame, field_spec in layers:
        geom_type = ogr.wkbPolygon if layer_name == "geological_units" else ogr.wkbLineString
        if not frame.empty:
            sample = frame.geometry.iloc[0]
            geom_type = ogr.wkbPolygon if sample.geom_type in {"Polygon", "MultiPolygon"} else ogr.wkbLineString
        layer = ds.CreateLayer(layer_name, srs=srs, geom_type=geom_type)
        for name, ogr_type in field_spec:
            layer.CreateField(ogr.FieldDefn(name, ogr_type))
        for record in frame.to_dict("records"):
            feature = ogr.Feature(layer.GetLayerDefn())
            for name, _ in field_spec:
                value = record.get(name)
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    continue
                feature.SetField(name, value)
            feature.SetGeometry(ogr.CreateGeometryFromWkb(record["geometry"].wkb))
            layer.CreateFeature(feature)
            feature = None
    ds = None


UNIT_FIELDS = [
    ("unit_id", ogr.OFTString), ("source_id", ogr.OFTString), ("source_class", ogr.OFTString),
    ("legend_term", ogr.OFTString), ("source_parent", ogr.OFTString), ("source_color", ogr.OFTString),
    ("harmonised_class", ogr.OFTString), ("equivalence_status", ogr.OFTString),
    ("geometry_role", ogr.OFTString), ("digitisation_method", ogr.OFTString),
    ("positional_uncertainty_m", ogr.OFTInteger), ("confidence", ogr.OFTString),
    ("source_pdf_page", ogr.OFTInteger), ("source_figure", ogr.OFTString),
    ("source_pdf_sha256", ogr.OFTString), ("area_km2", ogr.OFTReal), ("notes", ogr.OFTString),
]
CONTACT_FIELDS = [
    ("source_id", ogr.OFTString), ("contact_type", ogr.OFTString), ("source_class", ogr.OFTString),
    ("harmonised_class", ogr.OFTString), ("equivalence_status", ogr.OFTString),
    ("geometry_role", ogr.OFTString), ("digitisation_method", ogr.OFTString),
    ("positional_uncertainty_m", ogr.OFTInteger), ("confidence", ogr.OFTString),
    ("length_km", ogr.OFTReal), ("notes", ogr.OFTString),
]
STRUCT_FIELDS = [
    ("source_id", ogr.OFTString), ("feature_name", ogr.OFTString), ("feature_type", ogr.OFTString),
    ("source_legend_term", ogr.OFTString), ("harmonised_class", ogr.OFTString),
    ("equivalence_status", ogr.OFTString), ("geometry_role", ogr.OFTString),
    ("digitisation_method", ogr.OFTString), ("positional_uncertainty_m", ogr.OFTInteger),
    ("confidence", ogr.OFTString), ("length_km", ogr.OFTReal), ("notes", ogr.OFTString),
]


def manual_structures(source_id: str, tr: dict) -> gpd.GeoDataFrame:
    transformer = Transformer.from_crs(WGS84, TARGET_CRS, always_xy=True)
    features = []
    if source_id == "dewaele_2006":
        features.extend([
            ("Luongo Shear Zone", "shear_zone", [(1120, 2600), (1300, 2500), (1480, 2400), (1680, 2310), (1880, 2220), (2080, 2140), (2260, 2080)]),
            ("Mugesse Shear Zone", "shear_zone", [(2500, 770), (2620, 940), (2780, 1120), (2920, 1300), (3050, 1470)]),
        ])
    elif source_id == "hanson_1994":
        features.extend([
            ("Great Dyke", "dyke", [(3335, 2050), (3380, 2150), (3395, 2200), (3360, 2250), (3335, 2300), (3313, 2350), (3280, 2400), (3273, 2450), (3260, 2500), (3244, 2550), (3230, 2600), (3220, 2650), (3208, 2700), (3200, 2750), (3195, 2795)]),
        ])
    rows = []
    for name, kind, pts in features:
        ll = [px_to_lonlat(x, y, tr) for x, y in pts]
        line = LineString(ll)
        line = transform_geometry(line, transformer)
        rows.append({
            "source_id": source_id,
            "feature_name": name,
            "feature_type": kind,
            "source_legend_term": "Named structural feature" if name.endswith("Shear Zone") else "Explicitly mapped structure",
            "harmonised_class": "fault_shear_zone" if kind == "shear_zone" else "intrusive_linear_feature",
            "equivalence_status": "partial" if kind == "shear_zone" else "not_equivalent",
            "geometry_role": "structural_trace",
            "digitisation_method": "manual screen trace of explicit named feature at 600 dpi",
            "positional_uncertainty_m": SOURCES[source_id]["uncertainty_m"],
            "confidence": "low",
            "length_km": line.length / 1000,
            "notes": "Derived from the printed figure; no kinematic or continuity inference.",
            "geometry": line,
        })
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=TARGET_CRS) if rows else gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=TARGET_CRS)


def source_footprint(source_id: str, tr: dict) -> Polygon:
    src = SOURCES[source_id]
    if source_id == "hanson_1994":
        pts = [px_to_lonlat(x, y, tr) for x, y in src["frame_corners"]]
    else:
        left, top, right, bottom = src["frame"]
        pts = [px_to_lonlat(x, y, tr) for x, y in [(left, top), (right, top), (right, bottom), (left, bottom)]]
    transformer = Transformer.from_crs(WGS84, TARGET_CRS, always_xy=True)
    return transform_geometry(Polygon(pts), transformer)


def integrate(source_id: str, units: gpd.GeoDataFrame, contacts: gpd.GeoDataFrame,
              structures: gpd.GeoDataFrame, footprint: Polygon) -> dict:
    union_gdf = gpd.read_file(ROOT / "data/raw/rdc_rocks/derived/integration/phase7/supplementary/daly_study_area_v1.gpkg", layer="daly_study_area").to_crs(TARGET_CRS)
    domains = gpd.read_file(ROOT / "data/raw/rdc_rocks/derived/integration/phase7/supplementary/daly_study_area_v1.gpkg", layer="daly_domains").to_crs(TARGET_CRS)
    grid = gpd.read_file(ROOT / "data/raw/rdc_rocks/derived/integration/phase7/supplementary/daly_support_5km_v1.gpkg", layer="daly_support_5km").to_crs(TARGET_CRS)
    union_geom = unary_union(union_gdf.geometry)
    footprint_area = footprint.area / 1e6
    clip_area = footprint.intersection(union_geom).area / 1e6
    unit_union = unary_union(units.geometry) if not units.empty else Polygon()
    unit_total_area = unit_union.area / 1e6 if not unit_union.is_empty else 0.0
    unit_area = unit_union.intersection(union_geom).area / 1e6 if not unit_union.is_empty else 0.0
    grid_hits = int(grid[grid.geometry.intersects(footprint)].shape[0])
    unit_grid_hits = int(grid[grid.geometry.intersects(unit_union)].shape[0]) if not unit_union.is_empty else 0
    class_rows = []
    if not units.empty:
        clipped = units.copy()
        clipped["geometry"] = clipped.geometry.intersection(union_geom)
        clipped = clipped[~clipped.geometry.is_empty]
        clipped["area_m2"] = clipped.geometry.area
        clipped_area = clipped.groupby("source_class")["area_m2"].sum() / 1e6
        for cls, group in units.groupby("source_class"):
            class_rows.append({
                "source_id": source_id,
                "source_class": cls,
                "total_area_km2": round(group.geometry.area.sum() / 1e6, 2),
                "area_in_daly_union_km2": round(float(clipped_area.get(cls, 0.0)), 2),
                "feature_count": int(len(group)),
            })
    domain_rows = []
    if not units.empty:
        units_by_class = units.dissolve(by="source_class", as_index=False)
        for _, domain in domains.iterrows():
            inter = units_by_class.geometry.intersection(domain.geometry)
            area = inter.area.sum() / 1e6
            if area > 0:
                domain_rows.append({
                    "source_id": source_id,
                    "domain_id": domain["domain_id"],
                    "domain_name": domain["domain_name"],
                    "source_unit_area_km2": round(area, 2),
                    "fraction_of_domain_covered": round(area / domain["area_km2"], 4),
                })
    return {
        "source_id": source_id,
        "footprint_area_km2": round(footprint_area, 2),
        "footprint_in_daly_union_km2": round(clip_area, 2),
        "digitised_unit_area_total_km2": round(unit_total_area, 2),
        "digitised_unit_area_in_daly_union_km2": round(unit_area, 2),
        "digitised_fraction_of_footprint": round(unit_total_area / footprint_area, 4) if footprint_area else None,
        "digitised_fraction_of_daly_union_footprint": round(unit_area / clip_area, 4) if clip_area else None,
        "unit_features": int(len(units)),
        "contact_features": int(len(contacts)),
        "structural_features": int(len(structures)),
        "valid_unit_geometries": int(units.geometry.is_valid.sum()) if not units.empty else 0,
        "grid_cells_intersecting_footprint": grid_hits,
        "grid_cells_intersecting_digitised_units": unit_grid_hits,
        "class_area": class_rows,
        "domain_area": domain_rows,
    }


def predictor_eligibility(source_id: str) -> str:
    return {
        "dewaele_2006": "no_5km_predictor; regional_context_only",
        "zambia_overview": "no_5km_predictor; national_context_only",
        "hanson_1994": "not_digitised; source_only",
    }[source_id]


def main() -> None:
    for directory in (OUT, EXTRACTED, GEO, LAYERS, CROSSWALK, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)
    for directory in (EXTRACTED, GEO, LAYERS, CROSSWALK, REPORTS):
        clean_dir(directory, keep_suffixes=(".md",) if directory == REPORTS else ())
    generated = dt.datetime.now(dt.timezone.utc)
    transformer = Transformer.from_crs(WGS84, TARGET_CRS, always_xy=True)
    report_data = {}
    gcp_rows = []
    crosswalk_rows = []
    metric_rows = []
    all_units = {}
    all_contacts = {}
    all_structures = {}
    with tempfile.TemporaryDirectory(prefix="zambia_alt_maps_") as temporary:
        tempdir = Path(temporary)
        for source_id, src in SOURCES.items():
            page_path = tempdir / f"{source_id}.png"
            page = render_page(PDFS[source_id], src["pdf_page"], page_path)
            page.crop(src["figure_crop"]).save(EXTRACTED / f"{source_id}_figure.png")
            masked_map_crop(page, source_id).save(EXTRACTED / f"{source_id}_map.png")
            page.crop(src["legend_crop"]).save(EXTRACTED / f"{source_id}_legend.png")
            tr = source_transform(source_id)
            if tr["kind"] == "axis":
                x_rmse = float(np.sqrt(np.mean(tr["x_residuals"] ** 2)))
                y_rmse = float(np.sqrt(np.mean(tr["y_residuals"] ** 2)))
                km_per_px_x = abs(1.0 / tr["x_slope"]) * 111.32 * math.cos(math.radians(10 if source_id == "zambia_overview" else 10.5))
                km_per_px_y = abs(1.0 / tr["y_slope"]) * 110.57
                for value, residual in zip([p[1] for p in src["x_controls"]], tr["x_residuals"]):
                    gcp_rows.append({
                        "source_id": source_id, "control_axis": "longitude", "coord_value": value,
                        "pixel_residual": round(float(residual), 2),
                        "ground_residual_m": round(float(abs(residual) * km_per_px_x * 1000), 1),
                        "uncertainty_basis": "printed longitude ticks at 600 dpi",
                    })
                for value, residual in zip([p[1] for p in src["y_controls"]], tr["y_residuals"]):
                    gcp_rows.append({
                        "source_id": source_id, "control_axis": "latitude", "coord_value": value,
                        "pixel_residual": round(float(residual), 2),
                        "ground_residual_m": round(float(abs(residual) * km_per_px_y * 1000), 1),
                        "uncertainty_basis": "printed latitude ticks at 600 dpi",
                    })
                report_data[source_id] = {
                    "transform": "independent linear fits of labelled longitudes and latitudes",
                    "x_slope_px_per_degree": round(tr["x_slope"], 3),
                    "x_intercept_px": round(tr["x_intercept"], 3),
                    "y_slope_px_per_degree": round(tr["y_slope"], 3),
                    "y_intercept_px": round(tr["y_intercept"], 3),
                    "x_rmse_px": round(x_rmse, 2),
                    "y_rmse_px": round(y_rmse, 2),
                    "x_max_residual_px": round(float(np.max(np.abs(tr["x_residuals"]))), 2),
                    "y_max_residual_px": round(float(np.max(np.abs(tr["y_residuals"]))), 2),
                    "positional_uncertainty_m": src["uncertainty_m"],
                    "scale_note": src["scale_note"],
                }
            else:
                points = tr["points"]
                residuals = []
                for x, y, lon, lat, label in points:
                    lon2, lat2 = px_to_lonlat(x, y, tr)
                    residuals.append((lon2 - lon, lat2 - lat))
                arr = np.asarray(residuals)
                report_data[source_id] = {
                    "transform": "affine fit to labelled bottom longitudes, right-edge latitudes and the printed L/H place labels; border treated as a local graticule",
                    "lon_coefficients": [round(float(v), 12) for v in tr["lon_coef"]],
                    "lat_coefficients": [round(float(v), 12) for v in tr["lat_coef"]],
                    "assumed_border_longitude": round(tr["lon_right"], 5),
                    "assumed_border_latitude": round(tr["lat_bottom"], 5),
                    "label_fit_rmse_px": round(float(np.sqrt(np.mean((arr[:, 0] * tr["x_slope"]) ** 2 + (arr[:, 1] * abs(tr["y_slope"])) ** 2))), 2),
                    "positional_uncertainty_m": src["uncertainty_m"],
                    "caveat": "No independent geodetic control is printed. The RMSE describes graticule-label consistency only; the stated uncertainty includes projection and scan distortion.",
                    "scale_note": src["scale_note"],
                }
                for x, y, lon, lat, label in points:
                    lon2, lat2 = px_to_lonlat(x, y, tr)
                    dx_px = (lon2 - lon) * tr["x_slope"]
                    dy_px = (lat2 - lat) * abs(tr["y_slope"])
                    dx_m = (lon2 - lon) * 111320.0 * math.cos(math.radians(lat2))
                    dy_m = (lat2 - lat) * 110570.0
                    gcp_rows.append({
                        "source_id": source_id, "control_axis": label,
                        "coord_value": f"{lon:.3f},{lat:.3f}",
                        "pixel_residual": round(math.hypot(dx_px, dy_px), 2),
                        "ground_residual_m": round(math.hypot(dx_m, dy_m), 1),
                        "uncertainty_basis": "printed graticule and frame intersection at 600 dpi",
                    })
            out_ll, out_utm = write_geotiffs(source_id, page, tr)
            units = polygonize_units(source_id, page, transformer) if source_id != "hanson_1994" else gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=TARGET_CRS)
            contacts = derive_contacts(units, source_id)
            structures = manual_structures(source_id, tr)
            gpkg = LAYERS / f"{source_id}.gpkg"
            write_layers(gpkg, [
                ("geological_units", units, UNIT_FIELDS),
                ("geological_contacts", contacts, CONTACT_FIELDS),
                ("structural_features", structures, STRUCT_FIELDS),
            ])
            all_units[source_id], all_contacts[source_id], all_structures[source_id] = units, contacts, structures
            footprint = source_footprint(source_id, tr)
            metrics = integrate(source_id, units, contacts, structures, footprint)
            metrics.update({
                "georeferenced_4326": str(out_ll.relative_to(OUT)),
                "georeferenced_32735": str(out_utm.relative_to(OUT)),
                "vector_gpkg": str(gpkg.relative_to(OUT)),
            })
            metric_rows.append(metrics)
            for row in units.to_dict("records") if not units.empty else []:
                crosswalk_rows.append({
                    "source_id": source_id, "source_legend_term": row["legend_term"],
                    "source_parent_class": row["source_parent"], "source_color_rgb": row["source_color"],
                    "source_geometry_role": row["geometry_role"], "project_feature_group": "stratigraphic",
                    "project_harmonised_class": row["harmonised_class"], "equivalence_status": row["equivalence_status"],
                    "existing_phase6_class": "not_present",
                    "schema_extension_required": "yes",
                    "rationale": row["notes"], "digitised_geometry": "yes",
                    "predictor_eligibility": predictor_eligibility(source_id),
                })
    for source_id, legend, parent, role, harmonised, status, rationale in CROSSWALK_EXTRA:
        digitised = "partial" if role == "line_symbol" and source_id != "zambia_overview" else "no"
        if source_id == "hanson_1994" and legend == "Mwembeshi dislocation":
            digitised = "no"
        crosswalk_rows.append({
            "source_id": source_id, "source_legend_term": legend, "source_parent_class": parent,
            "source_color_rgb": "", "source_geometry_role": role,
            "project_feature_group": "structural" if role == "line_symbol" else "stratigraphic",
            "project_harmonised_class": harmonised, "equivalence_status": status,
            "existing_phase6_class": "not_present",
            "schema_extension_required": "yes",
            "rationale": rationale, "digitised_geometry": digitised,
            "predictor_eligibility": predictor_eligibility(source_id) if source_id == "hanson_1994" else "no",
        })
    for source_id in SOURCES:
        crosswalk_rows.append({
            "source_id": source_id,
            "source_legend_term": "Contact derived from adjacent colour-class polygons",
            "source_parent_class": "Derived geometry",
            "source_color_rgb": "",
            "source_geometry_role": "derived_contact",
            "project_feature_group": "structural",
            "project_harmonised_class": "contact_stratigraphic",
            "equivalence_status": "unmapped",
            "existing_phase6_class": "not_present",
            "schema_extension_required": "yes",
            "rationale": "Boundary extracted from the class raster; not an independently printed contact trace and may include map-edge segments.",
            "digitised_geometry": "yes" if source_id != "hanson_1994" else "no",
            "predictor_eligibility": "no",
        })
    # Deduplicate repeated class entries introduced by multiple polygon parts.
    seen = set(); unique_crosswalk = []
    for row in crosswalk_rows:
        key = (row["source_id"], row["source_legend_term"], row["source_geometry_role"])
        if key not in seen:
            seen.add(key); unique_crosswalk.append(row)
    crosswalk_path = CROSSWALK / "classification_crosswalk_v1.csv"
    pd.DataFrame(unique_crosswalk).to_csv(crosswalk_path, index=False)
    pd.DataFrame(gcp_rows).to_csv(REPORTS / "georeferencing_control_points_v1.csv", index=False)
    (REPORTS / "georeferencing_residuals_v1.json").write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    metrics_path = REPORTS / "integration_metrics_v1.json"
    metrics_path.write_text(json.dumps(metric_rows, indent=2), encoding="utf-8")
    flat_rows = []
    for metric in metric_rows:
        flat = {k: v for k, v in metric.items() if not isinstance(v, list)}
        flat_rows.append(flat)
    pd.DataFrame(flat_rows).to_csv(REPORTS / "integration_metrics_summary_v1.csv", index=False)
    manifest = {
        "generated_utc": generated.isoformat(),
        "processing_script": str(Path(__file__).relative_to(ROOT)),
        "render_dpi": RENDER_DPI,
        "target_crs": TARGET_CRS,
        "sources": {
            source_id: {
                **{k: v for k, v in src.items() if k not in {"map_crop", "figure_crop", "legend_crop", "x_controls", "y_controls", "frame_corners", "frame", "mask_rects", "bottom_border_y", "right_border_x"}},
                "pdf_path": str(PDFS[source_id].relative_to(ROOT)),
                "pdf_sha256": sha256(PDFS[source_id]),
            }
            for source_id, src in SOURCES.items()
        },
        "outputs": sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file()),
    }
    (OUT / "processing_manifest_v1.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"sources": len(SOURCES), "metrics": metric_rows}, indent=2))


if __name__ == "__main__":
    main()
