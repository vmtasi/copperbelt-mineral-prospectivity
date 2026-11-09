"""Build the RDC_3863 structural GeoPackage from the digitised feature table."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from gpkg_writer import linestring_wkb, point_wkb, polygon_wkb, write_gpkg

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GEOREF = PROJECT_ROOT / "data/raw/rdc_rocks/derived/georef"
FEATURES = GEOREF / "phase6/3863_features_v1.json"
TRANSFORMS = GEOREF / "transform_params_v1.json"
FIT = GEOREF / "georef_fit_raw.json"
OUT_DIR = GEOREF / "phase6/3863"
GPKG = OUT_DIR / "rdc_3863_structural_v1.gpkg"
CSV_PATH = OUT_DIR / "rdc_3863_features_v1.csv"
OVERLAY = PROJECT_ROOT / "work/3863_qa_overlay.png"

LEGEND = {
    "thrust_fault": "Faille de chevauchement",
    "back_thrust": "Faille de retrochevauchement",
    "thrust_sheet_boundary": "Lambeau de poussee",
    "contact_r4_r3": "Contact R.4-R.3",
    "faulted_contact": "Contact par faille",
    "anticlinal_axis": "Axe anticlinal (avec ennoyage)",
    "synclinal_axis": "Axe synclinal (avec ennoyage)",
    "extrusion": "Extrusion",
    "excrescence": "Excrescence",
}
FIELDS = [
    ("feature_id", "text"), ("feature_class", "text"), ("name_fr", "text"),
    ("sheet_no", "integer"), ("confidence", "text"), ("source_doc", "text"),
    ("source_page", "integer"), ("original_legend_term", "text"),
    ("geometry_provenance", "text"), ("coordinate_space", "text"), ("notes", "text"),
]
COLOURS = {
    "thrust_fault": (200, 30, 30), "back_thrust": (220, 110, 30),
    "thrust_sheet_boundary": (20, 90, 200), "contact_r4_r3": (20, 150, 60),
    "faulted_contact": (120, 60, 160), "anticlinal_axis": (200, 40, 160),
    "synclinal_axis": (40, 160, 190), "extrusion": (170, 130, 0),
    "excrescence": (110, 110, 0),
}


def wkt(feature: dict) -> str:
    pts = " ".join(f"{x} {y}" for x, y in feature["vertices"])
    if feature["class"] in ("thrust_sheet_boundary", "extrusion", "excrescence"):
        return f"POLYGON (({pts}))"
    return f"LINESTRING ({pts})"


def main() -> int:
    spec = json.loads(FEATURES.read_text(encoding="utf-8"))
    transform = json.loads(TRANSFORMS.read_text(encoding="utf-8"))["3863"]
    fit = next(item for item in json.loads(FIT.read_text(encoding="utf-8")) if item["id"] == "3863")
    a, b = transform["transform"]["a"], transform["transform"]["b"]
    c, d = transform["transform"]["c"], transform["transform"]["d"]
    ext = transform["grid_extent"]

    features = spec["features"]
    ids = [f["id"] for f in features]
    assert len(ids) == len(set(ids)), "duplicate feature ids"
    for feature in features:
        assert feature["class"] in LEGEND, f"unknown class {feature['class']}"
        assert len(feature["vertices"]) >= 2, f"{feature['id']}: too few vertices"
        for e, n in feature["vertices"]:
            assert ext["E_min"] - 6000 <= e <= ext["E_max"] + 6000, f"{feature['id']}: E out of range {e}"
            assert ext["N_min"] - 6000 <= n <= ext["N_max"] + 6000, f"{feature['id']}: N out of range {n}"
            px, py = a * e + b, c * n + d
            assert -50 <= px <= 6900 and -50 <= py <= 4750, f"{feature['id']}: pixel out of raster {px:.0f},{py:.0f}"
        if feature["class"] in ("thrust_sheet_boundary", "extrusion", "excrescence"):
            assert feature["vertices"][0] != feature["vertices"][-1], f"{feature['id']}: ring must be open in source"

    def attributes(feature: dict) -> dict:
        return {
            "feature_id": feature["id"], "feature_class": feature["class"],
            "name_fr": feature["name_fr"], "sheet_no": feature.get("sheet_no"),
            "confidence": feature["confidence"], "source_doc": "RDC_3863",
            "source_page": 1, "original_legend_term": LEGEND[feature["class"]],
            "geometry_provenance": "manual_trace_on_upright_200dpi_raster",
            "coordinate_space": spec["coordinate_space"], "notes": feature.get("notes", ""),
        }

    def make_layer(name: str, geometry_type: str, selected: list[dict], description: str) -> dict:
        built = []
        for feature in selected:
            points = [tuple(vertex) for vertex in feature["vertices"]]
            if geometry_type == "polygon":
                ring = points + [points[0]]
                built.append({"geom": polygon_wkb([ring]), "points": ring, "attributes": attributes(feature)})
            else:
                built.append({"geom": linestring_wkb(points), "points": points, "attributes": attributes(feature)})
        return {"name": name, "geometry_type": geometry_type.upper(), "fields": FIELDS, "features": built, "description": description}

    faults = [f for f in features if f["class"] in ("thrust_fault", "back_thrust")]
    axes = [f for f in features if f["class"] in ("anticlinal_axis", "synclinal_axis")]
    sheets = [f for f in features if f["class"] == "thrust_sheet_boundary"]
    contacts = [f for f in features if f["class"] in ("contact_r4_r3", "faulted_contact")]
    extrusions = [f for f in features if f["class"] == "extrusion"]
    excrescences = [f for f in features if f["class"] == "excrescence"]

    layers = [
        make_layer("structural_faults", "line", faults, "Thrust and back-thrust traces"),
        make_layer("thrust_sheets", "polygon", sheets, "Outlines of the seven numbered thrust sheets"),
        make_layer("fold_axes", "line", axes, "Anticlinal and synclinal axes"),
        make_layer("contacts", "line", contacts, "R.4/R.3 and faulted contacts"),
        make_layer("extrusions", "polygon", extrusions, "Extrusion outlines"),
        make_layer("excrescences", "polygon", excrescences, "Excrescence outlines"),
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = write_gpkg(GPKG, layers)

    control_csv = OUT_DIR / "rdc_3863_control_points_v1.csv"
    with control_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["point_id", "axis", "grid_value_m", "pixel_x", "pixel_y", "residual_m"])
        for index, (value, px, residual) in enumerate(zip(fit["ev"], fit["ep"], fit["rx"])):
            writer.writerow([f"3863-E{index+1}", "easting", value, round(px, 1), "", round(residual, 1)])
        for index, (value, py, residual) in enumerate(zip(fit["nv"], fit["np"], fit["ry"])):
            writer.writerow([f"3863-N{index+1}", "northing", value, "", round(py, 1), round(residual, 1)])

    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature_id", "feature_class", "name_fr", "sheet_no", "confidence", "source_doc", "source_page", "geometry_wkt", "notes"])
        for feature in features:
            writer.writerow([feature["id"], feature["class"], feature["name_fr"], feature.get("sheet_no"),
                             feature["confidence"], "RDC_3863", 1, wkt(feature), feature.get("notes", "")])

    raster = Image.open(PROJECT_ROOT / spec["document"]["source_raster"]).convert("RGB")
    scale = 0.6
    raster = raster.resize((int(raster.width * scale), int(raster.height * scale)), Image.LANCZOS)
    draw = ImageDraw.Draw(raster)
    font = ImageFont.truetype("arial.ttf", 18)
    for feature in features:
        pixels = [(int((a * e + b) * scale), int((c * n + d) * scale)) for e, n in feature["vertices"]]
        colour = COLOURS[feature["class"]]
        if feature["class"] in ("thrust_sheet_boundary", "extrusion", "excrescence"):
            draw.line(pixels + [pixels[0]], fill=colour, width=2)
        else:
            draw.line(pixels, fill=colour, width=3)
        draw.text(pixels[0], feature["id"].split("-")[1], fill=colour, font=font)
    raster.save(OVERLAY)

    report = [
        "# RDC_3863 structural digitisation (v1)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- Features: **{len(features)}** across {len(layers)} layers.",
        f"- Coordinate space: `{spec['coordinate_space']}` (datum and projection unresolved).",
        f"- Transform: `x = {a:.8f}E {b:+.2f}`, `y = {c:.8f}N {d:+.2f}`; RMSE {fit['rmse_m']:.1f} m, max {fit['max_m']:.1f} m.",
        f"- Source: RDC_3863 p.1, Figure 17, 1:200,000.",
        f"- GeoPackage: `{GPKG.relative_to(PROJECT_ROOT)}`.",
        f"- QA overlay: `{OVERLAY.relative_to(PROJECT_ROOT)}`.",
        "",
        "## Layer counts",
        "",
    ]
    report += [f"- `{name}`: {count} features" for name, count in summary.items()]
    report += [
        "",
        "## Limitations",
        "",
        "- Vertices were read manually against a 10 km grid overlay; expect 1-2 km positional uncertainty on long traces, better near grid lines.",
        "- The R.4/R.3 contacts are inferred from thin solid lines on a scan and are marked `low` confidence. They must be checked against the original sheet before any contact-based analysis.",
        "- Thrust-sheet outlines are the numbered massifs in the Tenke-Kakanda area; they are not re-interpreted beyond the original numbering and names.",
        "- No datum or projection is asserted. Coordinates are the sheet's own metric grid.",
    ]
    report_path = OUT_DIR / "rdc_3863_digitising_report_v1.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"features": len(features), "layers": summary, "gpkg": str(GPKG)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
