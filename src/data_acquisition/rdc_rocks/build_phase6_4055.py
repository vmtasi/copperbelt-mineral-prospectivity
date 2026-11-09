"""Build the RDC_4055 structural GeoPackage in calibrated source-pixel space."""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from gpkg_writer import linestring_wkb, point_wkb, polygon_wkb, write_gpkg

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GEOREF = PROJECT_ROOT / "data/raw/rdc_rocks/derived/georef"
RASTER = GEOREF / "phase6/crops/RDC_4055_upright.png"
OUT_DIR = GEOREF / "phase6/4055"
GPKG = OUT_DIR / "rdc_4055_structural_v1.gpkg"
CSV_PATH = OUT_DIR / "rdc_4055_features_v1.csv"
CONTROL_CSV = OUT_DIR / "rdc_4055_control_points_v1.csv"
REPORT = OUT_DIR / "rdc_4055_transformation_report_v1.md"
OVERLAY = PROJECT_ROOT / "work/4055_qa_overlay.png"
TRANSFORMS = GEOREF / "transform_params_v1.json"

SOURCE_DOC = "RDC_4055"
SOURCE_PAGE = 1
COORDINATE_SPACE = (
    "RDC_4055_upright.png source pixels (x right, y down); easting calibrated "
    "from the bottom border; northing unresolved; no datum or projection"
)
PROVENANCE = "manual_envelope_on_100px_source_pixel_grid"

# Bottom-border easting labels, all at the same label row y=3070.
# Windows OCR read six labels whole and two in part; 170 km is absent.
CONTROL = [
    ("4055-E1", 130000, 344.5, "OCR '30.000' under 'X'; leading 1 verified visually"),
    ("4055-E2", 140000, 658.5, "OCR '140.000'"),
    ("4055-E3", 150000, 974.5, "OCR '150.000'"),
    ("4055-E4", 160000, 1297.5, "OCR fragment '000'; centre from label box, 160 verified visually"),
    ("4055-E5", 180000, 1924.0, "OCR '180.000'"),
    ("4055-E6", 190000, 2238.5, "OCR fragment '000'; centre from label box, 190 verified visually"),
    ("4055-E7", 200000, 2550.5, "OCR '200.000'"),
]

CARAPACES = [
    {
        "id": "4055-C01", "sheet_no": 1, "name_fr": "Carapace de Tombolo",
        "legend": "CARAPACE DE TOMBOLO", "confidence": "medium",
        "vertices": [[960, 1155], [1060, 1158], [1150, 1170], [1200, 1200], [1215, 1240],
                     [1185, 1275], [1100, 1295], [1000, 1305], [900, 1308], [840, 1290],
                     [820, 1255], [830, 1215], [870, 1185], [915, 1168]],
        "notes": "Envelope of the Tombolo overthrust remnant; west end includes a grid-hatched wedge.",
    },
    {
        "id": "4055-C02", "sheet_no": 2, "name_fr": "Carapace de la Senia",
        "legend": "CARAPACE DE LA SENIA", "confidence": "low",
        "vertices": [[1500, 780], [1540, 710], [1610, 665], [1700, 650], [1800, 655],
                     [1900, 680], [1980, 725], [2040, 800], [2070, 880], [2050, 960],
                     [1995, 1030], [1920, 1080], [1830, 1105], [1730, 1110], [1640, 1090],
                     [1560, 1050], [1505, 990], [1470, 910], [1475, 840]],
        "notes": "Regional envelope bounded by the dotted nappe front and the Tondo block; not a contact trace.",
    },
    {
        "id": "4055-C03", "sheet_no": 3, "name_fr": "Carapace de la Gule",
        "legend": "CARAPACE DE LA GULE", "confidence": "medium",
        "vertices": [[1590, 1210], [1640, 1190], [1700, 1180], [1770, 1180], [1840, 1195],
                     [1890, 1220], [1910, 1255], [1900, 1290], [1860, 1320], [1810, 1340],
                     [1750, 1355], [1690, 1360], [1640, 1350], [1605, 1330], [1585, 1295],
                     [1580, 1255]],
        "notes": "Stippled carapace lobe north of the Pungulume thrust sheet.",
    },
    {
        "id": "4055-C04", "sheet_no": 4, "name_fr": "Carapace de Kimungu",
        "legend": "CARAPACE DE KIMUNGU", "confidence": "low",
        "vertices": [[1895, 1330], [1925, 1318], [1960, 1325], [1978, 1345], [1960, 1362],
                     [1925, 1368], [1898, 1355], [1885, 1342]],
        "notes": "Small crescent east of La Gule; outline partly obscured by the carapace edge.",
    },
    {
        "id": "4055-C05", "sheet_no": 5, "name_fr": "Carapace de Pungulume",
        "legend": "CARAPACE DE PUNGULUME", "confidence": "medium",
        "vertices": [[1755, 1512], [1790, 1500], [1835, 1508], [1862, 1528], [1850, 1552],
                     [1815, 1568], [1775, 1560], [1748, 1538]],
        "notes": "Small lens immediately south of the Pungulume label.",
    },
    {
        "id": "4055-C06", "sheet_no": 6, "name_fr": "Carapace de Kalukundi Ouest",
        "legend": "CARAPACE DE KALUKUNDI OUEST", "confidence": "medium",
        "vertices": [[2255, 1555], [2295, 1542], [2335, 1555], [2352, 1585], [2340, 1620],
                     [2305, 1645], [2265, 1635], [2248, 1600]],
        "notes": "Western Kalukundi lens, adjacent to thrust sheet 10.",
    },
    {
        "id": "4055-C07", "sheet_no": 7, "name_fr": "Carapace de Kalukundi Est",
        "legend": "CARAPACE DE KALUKUNDI EST", "confidence": "medium",
        "vertices": [[2445, 1535], [2495, 1518], [2545, 1528], [2575, 1552], [2578, 1588],
                     [2550, 1615], [2500, 1622], [2458, 1608], [2435, 1578]],
        "notes": "Oval carapace north of Pumpi.",
    },
]

THRUST_SHEETS = [
    {
        "id": "4055-S08", "sheet_no": 8, "name_fr": "Lambeau de poussee de Kanunka",
        "legend": "LAMBEAU DE POUSSEE DE KANUNKA", "confidence": "medium",
        "vertices": [[1332, 1990], [1385, 1972], [1450, 1965], [1520, 1962], [1590, 1970],
                     [1650, 1985], [1685, 2008], [1670, 2030], [1610, 2048], [1530, 2055],
                     [1450, 2055], [1385, 2045], [1345, 2028], [1328, 2008]],
        "notes": "West end of the Kanunka sliver; the number 8 is printed on the western tip.",
    },
    {
        "id": "4055-S09", "sheet_no": 9, "name_fr": "Lambeau de poussee de la Kipwishi",
        "legend": "LAMBEAU DE POUSSEE DE LA KIPWISHI", "confidence": "medium",
        "vertices": [[1695, 1908], [1745, 1888], [1810, 1878], [1880, 1872], [1950, 1875],
                     [2010, 1888], [2040, 1908], [2045, 1932], [2010, 1948], [1945, 1958],
                     [1870, 1960], [1800, 1950], [1740, 1935], [1705, 1922]],
        "notes": "Elongate Kipwishi sliver south of the Kanunka sliver.",
    },
    {
        "id": "4055-S10", "sheet_no": 10, "name_fr": "Lambeau de poussee de Kalukundi",
        "legend": "LAMBEAU DE POUSSEE DE KALUKUNDI", "confidence": "medium",
        "vertices": [[2185, 1560], [2225, 1545], [2262, 1562], [2258, 1600], [2238, 1645],
                     [2232, 1705], [2212, 1665], [2192, 1620], [2175, 1585]],
        "notes": "Kalukundi sliver with a narrow southern tail; west of carapace 6.",
    },
    {
        "id": "4055-S11a", "sheet_no": 11, "name_fr": "Lambeau de poussee du Signal Kapiri (north window)",
        "legend": "LAMBEAU DE POUSSEE DU SIGNAL KAPIRI", "confidence": "low",
        "vertices": [[1948, 1172], [1975, 1160], [2002, 1175], [2005, 1202], [1985, 1222],
                     [1958, 1215], [1943, 1192]],
        "notes": "Northern of three printed windows carrying the number 11.",
    },
    {
        "id": "4055-S11b", "sheet_no": 11, "name_fr": "Lambeau de poussee du Signal Kapiri (central window)",
        "legend": "LAMBEAU DE POUSSEE DU SIGNAL KAPIRI", "confidence": "low",
        "vertices": [[1940, 1308], [1972, 1295], [2005, 1312], [2008, 1345], [1988, 1368],
                     [1955, 1360], [1935, 1335]],
        "notes": "Central of three printed windows carrying the number 11.",
    },
    {
        "id": "4055-S11c", "sheet_no": 11, "name_fr": "Lambeau de poussee du Signal Kapiri (south window)",
        "legend": "LAMBEAU DE POUSSEE DU SIGNAL KAPIRI", "confidence": "low",
        "vertices": [[1842, 1502], [1875, 1482], [1915, 1492], [1948, 1518], [1945, 1548],
                     [1910, 1570], [1872, 1560], [1842, 1535]],
        "notes": "Southern of three printed windows carrying the number 11.",
    },
    {
        "id": "4055-S12", "sheet_no": 12, "name_fr": "Lambeau de poussee de la Kaku'-O",
        "legend": "LAMBEAU DE POUSSEE DE LA KAKU'-O", "confidence": "low",
        "vertices": [[2008, 1310], [2065, 1295], [2130, 1300], [2178, 1330], [2200, 1378],
                     [2192, 1428], [2160, 1465], [2110, 1480], [2055, 1472], [2015, 1440],
                     [1995, 1385]],
        "notes": "Broad sheet west of the Tshilongo nappe front; internal linework not traced.",
    },
    {
        "id": "4055-S13", "sheet_no": 13, "name_fr": "Lambeau de poussee de la Gule",
        "legend": "LAMBEAU DE POUSSEE DE LA GULE", "confidence": "low",
        "vertices": [[2058, 1082], [2115, 1062], [2175, 1090], [2210, 1140], [2208, 1205],
                     [2180, 1265], [2135, 1298], [2082, 1288], [2050, 1235], [2045, 1155]],
        "notes": "Thrust sheet 13, not carapace 3; the shared La Gule name does not imply the same feature.",
    },
    {
        "id": "4055-S14", "sheet_no": 14, "name_fr": "Lambeau de poussee de la Mulungwishi",
        "legend": "LAMBEAU DE POUSSEE DE LA MULUNGWISHI", "confidence": "low",
        "vertices": [[2052, 850], [2088, 858], [2108, 898], [2102, 958], [2088, 1012],
                     [2092, 1082], [2087, 1150], [2072, 1200], [2054, 1162], [2050, 1088],
                     [2052, 1010], [2046, 930]],
        "notes": "Narrow north-south Mulungwishi sliver; width at the limit of manual tracing.",
    },
    {
        "id": "4055-S15", "sheet_no": 15, "name_fr": "Lambeau de poussee de la Kamilombo",
        "legend": "LAMBEAU DE POUSSEE DE LA KAMILOMBO", "confidence": "medium",
        "vertices": [[2308, 1228], [2338, 1202], [2390, 1190], [2448, 1194], [2502, 1218],
                     [2542, 1252], [2552, 1292], [2522, 1325], [2472, 1345], [2422, 1355],
                     [2372, 1345], [2332, 1315], [2306, 1272]],
        "notes": "Kamilombo sliver east of the Tshilongo axis.",
    },
    {
        "id": "4055-S16", "sheet_no": 16, "name_fr": "Lambeau de poussee de la Lukonde",
        "legend": "LAMBEAU DE POUSSEE DE LA LUKONDE", "confidence": "medium",
        "vertices": [[2502, 1415], [2562, 1400], [2625, 1400], [2685, 1415], [2722, 1440],
                     [2728, 1470], [2695, 1492], [2638, 1508], [2578, 1512], [2530, 1496],
                     [2500, 1468]],
        "notes": "Lukonde sliver between the Kamilombo and Kabwe sheets.",
    },
    {
        "id": "4055-S17", "sheet_no": 17, "name_fr": "Lambeau de poussee de la Kabwe",
        "legend": "LAMBEAU DE POUSSEE DE LA KABWE", "confidence": "medium",
        "vertices": [[2790, 1445], [2832, 1430], [2882, 1435], [2910, 1455], [2912, 1488],
                     [2888, 1515], [2848, 1535], [2808, 1525], [2784, 1495], [2778, 1465]],
        "notes": "Easternmost numbered thrust sheet, near the KIBAMBA margin.",
    },
]

FIELDS = [
    ("feature_id", "text"), ("feature_class", "text"), ("sheet_no", "integer"),
    ("name_fr", "text"), ("original_legend_term", "text"), ("confidence", "text"),
    ("geometry_provenance", "text"), ("coordinate_space", "text"),
    ("label_px_x", "real"), ("label_px_y", "real"), ("centroid_px_x", "real"),
    ("centroid_px_y", "real"), ("easting_centroid_m", "real"), ("easting_range_m", "text"),
    ("notes", "text"),
]
COLOURS = {"carapace": (180, 30, 120), "thrust_sheet": (20, 80, 200)}


def fit_easting() -> dict:
    n = len(CONTROL)
    sum_e = sum(row[1] / 1000 for row in CONTROL)
    sum_x = sum(row[2] for row in CONTROL)
    sum_ee = sum((row[1] / 1000) ** 2 for row in CONTROL)
    sum_ex = sum(row[1] / 1000 * row[2] for row in CONTROL)
    denominator = n * sum_ee - sum_e ** 2
    slope = (n * sum_ex - sum_e * sum_x) / denominator
    intercept = (sum_x - slope * sum_e) / n
    residuals = [px - (slope * easting / 1000 + intercept) for _, easting, px, _ in CONTROL]
    rmse_px = math.sqrt(sum(value * value for value in residuals) / n)
    return {
        "slope": slope, "intercept": intercept, "residuals": residuals,
        "rmse_px": rmse_px, "rmse_m": rmse_px / slope * 1000,
        "max_px": max(abs(value) for value in residuals),
        "max_m": max(abs(value) for value in residuals) / slope * 1000,
    }


def centroid(vertices: list[list[float]]) -> tuple[float, float]:
    return (sum(x for x, _ in vertices) / len(vertices), sum(y for _, y in vertices) / len(vertices))


def easting_at(px: float, fit: dict) -> float:
    return 1000.0 * (px - fit["intercept"]) / fit["slope"]


def polygon_wkt(vertices: list[list[float]]) -> str:
    ring = vertices + [vertices[0]]
    return "POLYGON ((" + ", ".join(f"{x} {y}" for x, y in ring) + "))"


def main() -> int:
    fit = fit_easting()
    features = []
    for feature_class, items in (("carapace", CARAPACES), ("thrust_sheet", THRUST_SHEETS)):
        for item in items:
            vertices = [[float(x), float(y)] for x, y in item["vertices"]]
            assert len(vertices) >= 3, item["id"]
            assert vertices[0] != vertices[-1], f"{item['id']}: ring must be open in source"
            for x, y in vertices:
                assert 0 <= x <= 2987 and 0 <= y <= 3223, f"{item['id']}: pixel {x},{y} outside raster"
            cx, cy = centroid(vertices)
            eastings = [easting_at(x, fit) for x, _ in vertices]
            features.append({
                **item, "feature_class": feature_class, "vertices": vertices,
                "centroid": (cx, cy), "easting_centroid": easting_at(cx, fit),
                "easting_range": f"{min(eastings):.0f}-{max(eastings):.0f}",
            })

    ids = [feature["id"] for feature in features]
    assert len(ids) == len(set(ids)), "duplicate feature ids"
    assert len(CARAPACES) == 7, "expected seven carapaces"
    assert sorted({item["sheet_no"] for item in THRUST_SHEETS}) == list(range(8, 18)), "numbered sheets must be 8-17"

    def attributes(feature: dict) -> dict:
        cx, cy = feature["centroid"]
        return {
            "feature_id": feature["id"], "feature_class": feature["feature_class"],
            "sheet_no": feature["sheet_no"], "name_fr": feature["name_fr"],
            "original_legend_term": feature["legend"], "confidence": feature["confidence"],
            "geometry_provenance": PROVENANCE, "coordinate_space": COORDINATE_SPACE,
            "label_px_x": round(cx, 1), "label_px_y": round(cy, 1),
            "centroid_px_x": round(cx, 1), "centroid_px_y": round(cy, 1),
            "easting_centroid_m": round(feature["easting_centroid"], 1),
            "easting_range_m": feature["easting_range"], "notes": feature["notes"],
        }

    def make_layer(name: str, feature_class: str, description: str) -> dict:
        built = []
        for feature in features:
            if feature["feature_class"] != feature_class:
                continue
            points = [tuple(vertex) for vertex in feature["vertices"]]
            ring = points + [points[0]]
            built.append({"geom": polygon_wkb([ring]), "points": ring, "attributes": attributes(feature)})
        return {"name": name, "geometry_type": "POLYGON", "fields": FIELDS, "features": built,
                "description": description}

    layers = [
        make_layer("carapaces", "carapace", "Carapaces 1-7, envelopes in calibrated source-pixel space"),
        make_layer("thrust_sheets", "thrust_sheet", "Numbered thrust sheets 8-17; repeated numbers split into windows"),
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = write_gpkg(GPKG, layers)

    with CONTROL_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["point_id", "axis", "grid_value_m", "pixel_x", "pixel_y", "residual_px", "residual_m", "source", "confidence", "note"])
        for (point_id, easting, px, note), residual in zip(CONTROL, fit["residuals"]):
            writer.writerow([point_id, "easting", easting, px, 3070.0, round(residual, 2),
                             round(residual / fit["slope"] * 1000, 1), "bottom margin label", "medium", note])

    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature_id", "feature_class", "sheet_no", "name_fr", "original_legend_term",
                         "confidence", "source_doc", "source_page", "geometry_wkt", "easting_centroid_m",
                         "easting_range_m", "notes"])
        for feature in features:
            writer.writerow([feature["id"], feature["feature_class"], feature["sheet_no"], feature["name_fr"],
                             feature["legend"], feature["confidence"], SOURCE_DOC, SOURCE_PAGE,
                             polygon_wkt(feature["vertices"]), round(feature["easting_centroid"], 1),
                             feature["easting_range"], feature["notes"]])

    raster = Image.open(RASTER).convert("RGB")
    scale = 0.5
    raster = raster.resize((int(raster.width * scale), int(raster.height * scale)), Image.LANCZOS)
    draw = ImageDraw.Draw(raster)
    font = ImageFont.truetype("arial.ttf", 16)
    for feature in features:
        pixels = [(int(x * scale), int(y * scale)) for x, y in feature["vertices"]]
        colour = COLOURS[feature["feature_class"]]
        draw.line(pixels + [pixels[0]], fill=colour, width=3)
        cx, cy = feature["centroid"]
        draw.text((int(cx * scale), int(cy * scale)), feature["id"].split("-")[1], fill=colour, font=font)
    raster.save(OVERLAY)

    nominal_px_per_km = 200 / 25.4 * 1000 / 250
    report = [
        "# RDC_4055 structural digitisation (v1)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- Features: **{len(features)}** in two layers ({len(CARAPACES)} carapaces, {len(THRUST_SHEETS)} thrust-sheet windows).",
        f"- Numbered thrust sheets: **10** (8-17), not 17. The printed sequence ends at 17; the old '17 sheets' count misread the last number.",
        "- Coordinate space: source-raster pixels, easting calibrated only. Northing labels were not found on any margin.",
        "- No datum, projection or WGS84 assignment is made.",
        f"- Source: {SOURCE_DOC} p.{SOURCE_PAGE}, Planche XV, 1:250,000, 200 dpi.",
        f"- GeoPackage: `{GPKG.relative_to(PROJECT_ROOT)}`.",
        f"- Feature table: `{CSV_PATH.relative_to(PROJECT_ROOT)}`.",
        f"- Control register: `{CONTROL_CSV.relative_to(PROJECT_ROOT)}`.",
        f"- QA overlay: `{OVERLAY.relative_to(PROJECT_ROOT)}`.",
        "",
        "## Easting calibration",
        "",
        f"- Controls: {len(CONTROL)} bottom-border labels at 130, 140, 150, 160, 180, 190 and 200 km; no 170 km label survived OCR.",
        f"- Fit: `px_x = {fit['slope']:.6f} * E_km {fit['intercept']:+.2f}`.",
        f"- Residuals: RMSE {fit['rmse_px']:.2f} px ({fit['rmse_m']:.1f} m), maximum {fit['max_px']:.2f} px ({fit['max_m']:.1f} m).",
        f"- Fitted scale {fit['slope']:.3f} px/km against nominal 1:250,000 at 200 dpi = {nominal_px_per_km:.3f} px/km ({(fit['slope'] / nominal_px_per_km - 1) * 100:+.2f}%).",
        "- This is an easting-only calibration. It does not establish a map northing, a datum or a projection.",
        "",
        "## What was digitised",
        "",
        "- Carapaces 1-7: Tombolo, La Senia, La Gule, Kimungu, Pungulume, Kalukundi Ouest, Kalukundi Est.",
        "- Thrust sheets 8-17: Kanunka, Kipwishi, Kalukundi, Signal Kapiri, Kaku'-O, La Gule, Mulungwishi, Kamilombo, Lukonde, Kabwe.",
        "- Signal Kapiri (11) is printed in three separate windows; these are three features with the same `sheet_no`.",
        "- Each feature keeps a stable ID, sheet number, original French legend term, class, confidence, provenance, coordinate-space identifier, pixel centroid, calibrated easting and notes.",
        "",
        "## Limitations",
        "",
        "- The polygons are manual envelopes of the printed outcrop belts, not contact-traced boundaries. Expect 1-3 km boundary uncertainty on long outlines.",
        "- Carapaces 2 and 4 are the weakest: carapace 2 is bounded largely by the dotted nappe front, and carapace 4 is a small, partly obscured crescent.",
        "- Thrust sheets 11, 12 and 14 are low confidence: sheet 11 is split into windows, sheet 12 contains untraced internal linework, and sheet 14 is a narrow sliver.",
        "- The shared name 'La Gule' on carapace 3 and thrust sheet 13 does not establish that they are the same geological feature.",
        "- No comparison with RDC_3863 was made here. Scale (1:250,000 vs 1:200,000), coverage and nomenclature differ, so similar names are not treated as identities.",
        "",
    ]
    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")

    transforms = json.loads(TRANSFORMS.read_text(encoding="utf-8"))
    transforms["4055"] = {
        "title": "Region de Kolwezi - structures tectoniques principales (Planche XV)",
        "status": "easting-only calibration in source pixel space; northing UNRESOLVED",
        "nominal_scale": 250000,
        "raster_dpi": 200,
        "transform": {
            "x_px": "a*E_km + c", "a": round(fit["slope"], 6), "c": round(fit["intercept"], 2),
            "y_px": "source pixel y; not calibrated",
        },
        "grid_extent": {"E_min": 128000, "E_max": 202000, "N_min": None, "N_max": None},
        "control": {
            "eastings": len(CONTROL), "northings": 0,
            "rmse_px": round(fit["rmse_px"], 2), "rmse_m": round(fit["rmse_m"], 1),
            "max_px": round(fit["max_px"], 2), "max_m": round(fit["max_m"], 1),
        },
        "nominal_px_per_km": round(nominal_px_per_km, 3),
        "fitted_px_per_km": round(fit["slope"], 4),
        "crs": "UNRESOLVED: no northing labels, datum or projection. Coordinates are source pixels; do not label WGS84.",
        "note": "Digitised 2026-11-08. Ten numbered sheets 8-17, not seventeen.",
    }
    TRANSFORMS.write_text(json.dumps(transforms, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "layers": summary, "features": len(features), "easting_rmse_m": round(fit["rmse_m"], 1),
        "gpkg": str(GPKG),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
