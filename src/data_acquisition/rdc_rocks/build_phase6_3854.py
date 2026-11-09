"""Build the corrected RDC_3854 stratigraphic GeoPackage and transform report."""
from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from gpkg_writer import linestring_wkb, point_wkb, polygon_wkb, write_gpkg

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GEOREF = PROJECT_ROOT / "data/raw/rdc_rocks/derived/georef"
SPEC_PATH = PROJECT_ROOT / "src/data_acquisition/rdc_rocks/3854_features_v1.json"
OUT_DIR = GEOREF / "phase6/3854"
GPKG = OUT_DIR / "rdc_3854_stratigraphy_v1.gpkg"
CSV_PATH = OUT_DIR / "rdc_3854_features_v1.csv"
CONTROL_CSV = OUT_DIR / "rdc_3854_control_points_v1.csv"
REPORT = OUT_DIR / "rdc_3854_transformation_report_v1.md"
OVERLAY = PROJECT_ROOT / "work/3854_qa_overlay.png"
TRANSFORMS = GEOREF / "transform_params_v1.json"
FIT_RAW = GEOREF / "georef_fit_raw.json"

# Calibration labels read from the 1:500,000 source page, with pixel centres.
# px = ax * easting + bx * py + cx; py = ay * northing + by * px + cy.
CONTROL_X = [
    ("3854-E1", 200000, 1572.5, 384.0, "top margin", "medium", "Label centre; 200.000 at top."),
    ("3854-E2", 220000, 1893.5, 377.0, "top margin", "medium", "Label centre; 220.000 at top."),
    ("3854-E3", 240000, 2211.5, 377.0, "top margin", "medium", "Label centre; 240.000 at top."),
    ("3854-E4", 260000, 2535.5, 379.0, "top margin", "low", "Partial OCR 260.; full width inferred from neighbouring labels."),
    ("3854-E5", 220000, 1879.0, 1950.5, "bottom margin", "medium", "Label centre; 220.000 at bottom."),
    ("3854-E6", 240000, 2201.5, 2012.5, "bottom margin", "medium", "Label centre; 240.000 at bottom."),
    ("3854-E7", 260000, 2524.0, 2016.0, "bottom margin", "medium", "Label centre; 260.000 at bottom."),
]
CONTROL_Y = [
    ("3854-N1", 340000, 3278.5, 648.5, "right margin", "medium", "340.000 on right margin."),
    ("3854-N2", 340000, 1631.5, 639.0, "interior grid line", "low", "340.000 at internal 200 km line."),
    ("3854-N3", 320000, 3291.0, 962.5, "right margin", "medium", "320.000 on right margin."),
    ("3854-N4", 320000, 1630.0, 951.0, "interior grid line", "low", "320.000 at internal 200 km line."),
    ("3854-N5", 300000, 3274.5, 1284.5, "right margin", "medium", "300.000 on right margin."),
    ("3854-N6", 300000, 1623.0, 1271.5, "interior grid line", "low", "300.000 at internal 200 km line."),
    ("3854-N7", 280000, 3231.5, 1603.0, "right margin", "medium", "280.000 on right margin."),
    ("3854-N8", 280000, 1624.5, 1588.0, "interior grid line", "low", "280.000 at internal 200 km line."),
]
WITHHELD_Y = ("3854-N9", 260000, 1635.0, 1923.5, "interior label", "excluded", "260.000 in map interior; incompatible with the margin series and not used.")

LEGEND = {
    "r4_kabul": "R.4 de facies Kabul",
    "r4_chabara": "R.4 de facies Chabara",
    "r4_luankonko": "R.4 de facies Luankonko",
    "r4_absent": "R.4 absent",
    "facies_boundary": "Limite entre facies",
    "thrust_fault": "Faille de charriage",
    "discordance": "Discordance",
    "isopach_observation": "Epaisseurs des Formations R.4.2 / R.4.1",
}
FACIES = {
    "r4_kabul": "Kabul",
    "r4_chabara": "Chabara",
    "r4_luankonko": "Luankonko",
    "r4_absent": "absent",
}
POLYGON_CLASSES = {"r4_kabul", "r4_chabara", "r4_luankonko", "r4_absent"}
LINE_CLASSES = {"facies_boundary", "thrust_fault", "discordance"}
FIELDS = [
    ("feature_id", "text"), ("feature_class", "text"), ("name_fr", "text"),
    ("original_legend_term", "text"), ("r4_facies", "text"),
    ("thickness_r42_m", "integer"), ("thickness_r41_m", "integer"),
    ("thickness_source_text", "text"), ("source_doc", "text"), ("source_page", "integer"),
    ("confidence", "text"), ("geometry_provenance", "text"),
    ("coordinate_space", "text"), ("notes", "text"),
]
COLOURS = {
    "r4_kabul": (210, 40, 40), "r4_chabara": (200, 120, 20),
    "r4_luankonko": (30, 100, 210), "r4_absent": (120, 60, 160),
    "facies_boundary": (0, 140, 80), "thrust_fault": (180, 0, 0),
    "discordance": (100, 100, 100), "isopach_observation": (0, 0, 200),
}


def fit_linear(rows: list[tuple[float, float, float]]) -> list[float]:
    """Least squares target = a*u + b*v + c."""
    normal_rows = [(u, v, 1.0, target) for target, u, v in rows]
    normal = [[sum(r[i] * r[j] for r in normal_rows) for j in range(3)] for i in range(3)]
    rhs = [sum(r[i] * r[3] for r in normal_rows) for i in range(3)]
    matrix = [normal[i][:] + [rhs[i]] for i in range(3)]
    for i in range(3):
        pivot = max(range(i, 3), key=lambda k: abs(matrix[k][i]))
        matrix[i], matrix[pivot] = matrix[pivot], matrix[i]
        divisor = matrix[i][i]
        for j in range(i, 4):
            matrix[i][j] /= divisor
        for k in range(3):
            if k == i:
                continue
            factor = matrix[k][i]
            for j in range(i, 4):
                matrix[k][j] -= factor * matrix[i][j]
    return [matrix[i][3] for i in range(3)]


def make_fit():
    ax, bx, cx = fit_linear([(px, E, py) for _, E, px, py, *_ in CONTROL_X])
    ay, by, cy = fit_linear([(py, N, px) for _, N, px, py, *_ in CONTROL_Y])

    def residuals(rows, coefficients, axis):
        a, b, c = coefficients
        values = []
        for _, value, px, py, *_ in rows:
            target = px if axis == "x" else py
            predictor = py if axis == "x" else px
            values.append(target - (a * value + b * predictor + c))
        return values

    rx = residuals(CONTROL_X, (ax, bx, cx), "x")
    ry = residuals(CONTROL_Y, (ay, by, cy), "y")
    sx, sy = abs(ax), abs(ay)

    def rmse(values):
        return math.sqrt(sum(value * value for value in values) / len(values))

    def loo(points, axis):
        errors = []
        for index in range(len(points)):
            training = points[:index] + points[index + 1 :]
            if axis == "x":
                a, b, c = fit_linear([(px, E, py) for _, E, px, py, *_ in training])
                _, value, px, py, *_ = points[index]
                errors.append((px - (a * value + b * py + c)) / sx)
            else:
                a, b, c = fit_linear([(py, N, px) for _, N, px, py, *_ in training])
                _, value, px, py, *_ = points[index]
                errors.append((py - (a * value + b * px + c)) / sy)
        return math.sqrt(sum(error * error for error in errors) / len(errors))

    withheld = WITHHELD_Y
    withheld_pred = ay * withheld[1] + by * withheld[2] + cy
    withheld_residual_px = withheld[3] - withheld_pred

    return {
        "ax": ax, "bx": bx, "cx": cx, "ay": ay, "by": by, "cy": cy,
        "rx": rx, "ry": ry, "rmse_x_px": rmse(rx), "rmse_y_px": rmse(ry),
        "rmse_x_m": rmse(rx) / sx, "rmse_y_m": rmse(ry) / sy,
        "max_x_m": max(abs(value) for value in rx) / sx,
        "max_y_m": max(abs(value) for value in ry) / sy,
        "loo_x_m": loo(CONTROL_X, "x"), "loo_y_m": loo(CONTROL_Y, "y"),
        "withheld_pred_py": withheld_pred, "withheld_residual_px": withheld_residual_px,
        "withheld_residual_m": abs(withheld_residual_px) / sy,
        "sx": sx, "sy": sy,
    }


def main() -> int:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    fit = make_fit()
    ax, bx, cx = fit["ax"], fit["bx"], fit["cx"]
    ay, by, cy = fit["ay"], fit["by"], fit["cy"]

    def pixel_to_en(px: float, py: float) -> tuple[float, float]:
        return ((px - bx * py - cx) / ax, (py - by * px - cy) / ay)

    def en_to_pixel(E: float, N: float) -> tuple[float, float]:
        px = (ax * E + bx * (ay * N + cy) + cx) / (1 - bx * by)
        return (px, ay * N + by * px + cy)

    features = spec["features"]
    isopachs = spec["isopachs"]
    ids = [feature["id"] for feature in features] + [feature["id"] for feature in isopachs]
    assert len(ids) == len(set(ids)), "duplicate feature ids"
    for feature in features:
        assert feature["class"] in LEGEND, feature["class"]
        assert feature["geometry"] in ("polygon", "line"), feature["geometry"]
        assert len(feature["vertices"]) >= (4 if feature["geometry"] == "polygon" else 2), feature["id"]
        assert feature["vertices"][0] != feature["vertices"][-1], f"{feature['id']}: ring must be open in source"
        for E, N in feature["vertices"]:
            assert 100000 <= E <= 330000 and 240000 <= N <= 370000, f"{feature['id']}: outside sheet {E},{N}"
            px, py = en_to_pixel(E, N)
            assert 0 <= px <= 3459 and 0 <= py <= 2339, f"{feature['id']}: pixel {px:.0f},{py:.0f}"
    for observation in isopachs:
        assert re.fullmatch(r"\d+\s*/\s*\d+", observation["text"]), observation["id"]
        E, N = pixel_to_en(observation["px"], observation["py"])
        assert 100000 <= E <= 330000 and 240000 <= N <= 370000, f"{observation['id']}: outside sheet {E},{N}"

    def base_attributes(feature_id: str, feature_class: str, name_fr: str, confidence: str, notes: str,
                        provenance: str, thickness_r42=None, thickness_r41=None,
                        thickness_text=None, r4_facies=None) -> dict:
        return {
            "feature_id": feature_id, "feature_class": feature_class, "name_fr": name_fr,
            "original_legend_term": LEGEND[feature_class],
            "r4_facies": r4_facies if r4_facies is not None else FACIES.get(feature_class),
            "thickness_r42_m": thickness_r42, "thickness_r41_m": thickness_r41,
            "thickness_source_text": thickness_text, "source_doc": "RDC_3854",
            "source_page": 1, "confidence": confidence, "geometry_provenance": provenance,
            "coordinate_space": spec["coordinate_space"], "notes": notes,
        }

    def build_line_or_polygon(feature: dict) -> dict:
        points = [(float(E), float(N)) for E, N in feature["vertices"]]
        attributes = base_attributes(
            feature["id"], feature["class"], feature["name_fr"], feature["confidence"],
            feature.get("notes", ""), "manual_trace_on_gridded_1:500000_raster",
        )
        if feature["geometry"] == "polygon":
            ring = points + [points[0]]
            return {"geom": polygon_wkb([ring]), "points": ring, "attributes": attributes}
        return {"geom": linestring_wkb(points), "points": points, "attributes": attributes}

    def layer(name: str, geometry_type: str, selected: list[dict], description: str) -> dict:
        return {
            "name": name, "geometry_type": geometry_type.upper(), "fields": FIELDS,
            "features": [build_line_or_polygon(feature) for feature in selected],
            "description": description,
        }

    facies = [f for f in features if f["class"] in FACIES and f["class"] != "r4_absent"]
    absent = [f for f in features if f["class"] == "r4_absent"]
    boundaries = [f for f in features if f["class"] == "facies_boundary"]
    faults = [f for f in features if f["class"] == "thrust_fault"]
    discordances = [f for f in features if f["class"] == "discordance"]

    isopach_layer = []
    for observation in isopachs:
        E, N = pixel_to_en(observation["px"], observation["py"])
        left, right = (int(value) for value in re.split(r"\s*/\s*", observation["text"]))
        attributes = base_attributes(
            observation["id"], "isopach_observation", f"Epaisseur {observation['text']}",
            observation["confidence"], observation.get("note", ""),
            "OCR_text_box_center_manual_check", left, right, observation["text"], "",
        )
        isopach_layer.append({"geom": point_wkb(E, N), "points": [(E, N)], "attributes": attributes})

    layers = [
        layer("r4_facies", "polygon", facies, "Approximate envelopes of the three named R.4 facies belts"),
        layer("r4_absent", "polygon", absent, "R.4 absent zones"),
        layer("facies_boundaries", "line", boundaries, "Limites entre facies"),
        {"name": "isopach_observations", "geometry_type": "POINT", "fields": FIELDS,
         "features": isopach_layer, "description": "R.4.2/R.4.1 thickness labels, positioned at their text centres"},
        layer("structural_faults", "line", faults, "Faille de charriage traces"),
        layer("discordances", "line", discordances, "Discordance traces"),
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = write_gpkg(GPKG, layers)

    with CONTROL_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["point_id", "axis", "grid_value_m", "pixel_x", "pixel_y", "role", "confidence", "residual_px", "residual_m", "source", "note"])
        for index, (point_id, value, px, py, source, confidence, note) in enumerate(CONTROL_X):
            writer.writerow([point_id, "easting", value, round(px, 1), round(py, 1), "train", confidence,
                             round(fit["rx"][index], 2), round(fit["rx"][index] / fit["sx"], 1), source, note])
        for index, (point_id, value, px, py, source, confidence, note) in enumerate(CONTROL_Y):
            writer.writerow([point_id, "northing", value, round(px, 1), round(py, 1), "train", confidence,
                             round(fit["ry"][index], 2), round(fit["ry"][index] / fit["sy"], 1), source, note])
        writer.writerow([WITHHELD_Y[0], "northing", WITHHELD_Y[1], WITHHELD_Y[2], WITHHELD_Y[3], WITHHELD_Y[4],
                         "low", round(fit["withheld_residual_px"], 2), round(fit["withheld_residual_m"], 1),
                         WITHHELD_Y[5], WITHHELD_Y[6]])

    csv_rows = []
    for feature in features:
        points = " ".join(f"{E} {N}" for E, N in feature["vertices"])
        wkt = f"POLYGON (({points}))" if feature["geometry"] == "polygon" else f"LINESTRING ({points})"
        csv_rows.append((feature["id"], feature["class"], feature["name_fr"], LEGEND[feature["class"]],
                         FACIES.get(feature["class"], ""), "", "", "", "RDC_3854", 1,
                         feature["confidence"], "manual_trace_on_gridded_1:500000_raster",
                         spec["coordinate_space"], wkt, feature.get("notes", "")))
    for observation in isopachs:
        E, N = pixel_to_en(observation["px"], observation["py"])
        left, right = (int(value) for value in re.split(r"\s*/\s*", observation["text"]))
        csv_rows.append((observation["id"], "isopach_observation", f"Epaisseur {observation['text']}",
                         LEGEND["isopach_observation"], "", left, right, observation["text"], "RDC_3854", 1,
                         observation["confidence"], "OCR_text_box_center_manual_check",
                         spec["coordinate_space"], f"POINT ({E:.1f} {N:.1f})", observation.get("note", "")))
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature_id", "feature_class", "name_fr", "original_legend_term", "r4_facies",
                         "thickness_r42_m", "thickness_r41_m", "thickness_source_text",
                         "source_doc", "source_page", "confidence", "geometry_provenance",
                         "coordinate_space", "geometry_wkt", "notes"])
        writer.writerows(csv_rows)

    raster = Image.open(PROJECT_ROOT / spec["document"]["source_raster"]).convert("RGB")
    scale = 0.62
    raster = raster.resize((int(raster.width * scale), int(raster.height * scale)), Image.LANCZOS)
    draw = ImageDraw.Draw(raster)
    font = ImageFont.truetype("arial.ttf", 16)
    for E in range(120000, 311000, 10000):
        start = en_to_pixel(E, 360000)
        end = en_to_pixel(E, 248000)
        draw.line([(start[0] * scale, start[1] * scale), (end[0] * scale, end[1] * scale)], fill=(220, 60, 60), width=1)
        draw.text((start[0] * scale + 2, 2), str(E // 1000), fill=(170, 0, 0), font=font)
    for N in range(250000, 361000, 10000):
        start = en_to_pixel(120000, N)
        end = en_to_pixel(311000, N)
        draw.line([(start[0] * scale, start[1] * scale), (end[0] * scale, end[1] * scale)], fill=(220, 60, 60), width=1)
        draw.text((2, start[1] * scale + 2), str(N // 1000), fill=(170, 0, 0), font=font)
    for feature in features:
        pixels = [(int(en_to_pixel(E, N)[0] * scale), int(en_to_pixel(E, N)[1] * scale)) for E, N in feature["vertices"]]
        colour = COLOURS[feature["class"]]
        if feature["geometry"] == "polygon":
            draw.line(pixels + [pixels[0]], fill=colour, width=2)
        else:
            draw.line(pixels, fill=colour, width=3)
        draw.text(pixels[0], feature["id"].split("-")[1], fill=colour, font=font)
    for observation in isopachs:
        px, py = en_to_pixel(*pixel_to_en(observation["px"], observation["py"]))
        draw.ellipse((px * scale - 3, py * scale - 3, px * scale + 3, py * scale + 3), fill=(0, 0, 180))
    raster.save(OVERLAY)

    report = [
        "# RDC_3854 stratigraphic digitisation and transformation (v1)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Transform",
        "",
        f"`px_x = {ax:.10f} * E {bx:+.10f} * px_y {cx:+.2f}`",
        "",
        f"`px_y = {ay:.10f} * N {by:+.10f} * px_x {cy:+.2f}`",
        "",
        f"- Easting controls: {len(CONTROL_X)} labels; RMSE {fit['rmse_x_px']:.2f} px ({fit['rmse_x_m']:.0f} m); max {fit['max_x_m']:.0f} m.",
        f"- Northing controls: {len(CONTROL_Y)} labels; RMSE {fit['rmse_y_px']:.2f} px ({fit['rmse_y_m']:.0f} m); max {fit['max_y_m']:.0f} m.",
        f"- Scales: {fit['sx']:.7f} px/m (E) and {fit['sy']:.7f} px/m (N); nominal 1:500,000 at 200 dpi is 0.015748 px/m (+2.0% and +0.9%).",
        f"- Leave-one-out checks: {fit['loo_x_m']:.0f} m (E), {fit['loo_y_m']:.0f} m (N).",
        f"- Withheld 260 km label: observed py {WITHHELD_Y[3]:.1f}; predicted {fit['withheld_pred_py']:.1f}; residual {fit['withheld_residual_m']:.0f} m. It was not used.",
        "- Datum and projection remain unresolved. Coordinates are the sheet metric grid, not WGS84.",
        "",
        "## What was digitised",
        "",
        f"- {len(facies)} R.4 facies envelopes: Kabul, Chabara and Luankonko.",
        f"- {len(absent)} R.4 absent zones.",
        f"- {len(boundaries)} facies boundaries.",
        f"- {len(faults)} thrust-fault traces and {len(discordances)} discordance traces.",
        f"- {len(isopachs)} R.4.2/R.4.1 thickness labels, kept in the source order.",
        "",
        "## Limitations",
        "",
        "- The facies polygons are envelopes of printed symbol belts, not contact-traced geological boundaries. Gaps, overlaps and internal undifferentiated R.4 remain; the package is not a topology-enforced facies map.",
        "- The 260 km northing annotation was excluded. The southern margin therefore depends on extrapolation from 280-340 km controls; it is the least certain part of the transform.",
        "- Manual trace uncertainty on this 1:500,000 sheet is about 1-2 km. `low` confidence marks features whose boundaries or class assignment are uncertain.",
        "- Thickness labels are point positions of annotations; they are not measured section locations and do not imply a continuous isopach surface.",
        "- Historic terms R.4, R.4.1 and R.4.2 are preserved exactly. No modern formation names are substituted.",
        "",
        f"GeoPackage: `{GPKG.relative_to(PROJECT_ROOT)}`",
        f"Feature table: `{CSV_PATH.relative_to(PROJECT_ROOT)}`",
        f"Control register: `{CONTROL_CSV.relative_to(PROJECT_ROOT)}`",
        f"QA overlay: `{OVERLAY.relative_to(PROJECT_ROOT)}`",
    ]
    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")

    transforms = json.loads(TRANSFORMS.read_text(encoding="utf-8"))
    transforms["3854"] = {
        "title": spec["document"]["title"],
        "status": "defensible in grid coordinates; datum UNRESOLVED",
        "nominal_scale": 500000,
        "raster_dpi": 200,
        "transform": {
            "x_px": "a*E + b*py + c", "y_px": "d*N + e*px + f",
            "a": round(ax, 10), "b": round(bx, 10), "c": round(cx, 2),
            "d": round(ay, 10), "e": round(by, 10), "f": round(cy, 2),
        },
        "grid_extent": {"E_min": 120000, "E_max": 310000, "N_min": 248000, "N_max": 360000},
        "control": {"eastings": len(CONTROL_X), "northings": len(CONTROL_Y),
                    "rmse_x_m": round(fit["rmse_x_m"], 1), "rmse_y_m": round(fit["rmse_y_m"], 1),
                    "loo_x_m": round(fit["loo_x_m"], 1), "loo_y_m": round(fit["loo_y_m"], 1),
                    "withheld_y_residual_m": round(fit["withheld_residual_m"], 1)},
        "nominal_px_per_m": 0.015748,
        "fitted_px_per_m": {"x": round(fit["sx"], 7), "y": round(fit["sy"], 7)},
        "crs": "UNRESOLVED: no datum/projection declaration. Metric grid treated as a local projected grid; do not label WGS84.",
        "note": "Corrected 2026-11-08. A 260 km interior label was rejected; south margin is extrapolated.",
    }
    TRANSFORMS.write_text(json.dumps(transforms, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    fits = json.loads(FIT_RAW.read_text(encoding="utf-8"))
    for index, item in enumerate(fits):
        if item.get("id") == "3854":
            fits[index] = {
                "id": "3854", "status": "fitted_affine_axis_models", "east": len(CONTROL_X), "north": len(CONTROL_Y),
                "scale_nominal": 500000, "sx": fit["sx"], "sy": fit["sy"],
                "ax": ax, "bx": bx, "cx": cx, "ay": ay, "by": by, "cy": cy,
                "ev": [row[1] for row in CONTROL_X], "ep": [row[2] for row in CONTROL_X],
                "nv": [row[1] for row in CONTROL_Y], "np": [row[3] for row in CONTROL_Y],
                "rx": fit["rx"], "ry": fit["ry"],
                "rmse_m": math.hypot(fit["rmse_x_m"], fit["rmse_y_m"]),
                "max_m": math.hypot(fit["max_x_m"], fit["max_y_m"]),
                "rmse_x": fit["rmse_x_m"], "rmse_y": fit["rmse_y_m"],
                "max_x": fit["max_x_m"], "max_y": fit["max_y_m"],
                "nominal_px_per_m": 0.015748031496062992,
                "title": spec["document"]["title"],
            }
            break
    FIT_RAW.write_text(json.dumps(fits, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "layers": summary, "features": len(features) + len(isopachs),
        "rmse_x_m": round(fit["rmse_x_m"], 1), "rmse_y_m": round(fit["rmse_y_m"], 1),
        "gpkg": str(GPKG),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
