"""Build the Phase 7 integrated geological, occurrence and observation evidence package."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path

from gpkg_writer import write_gpkg

PROJECT_ROOT = Path(__file__).resolve().parents[3]
P6 = PROJECT_ROOT / "data/raw/rdc_rocks/derived/georef/phase6"
OCC = PROJECT_ROOT / "data/raw/rdc_rocks/derived/occurrences"
EXP = PROJECT_ROOT / "data/raw/rdc_rocks/derived/exploration"
OUT = PROJECT_ROOT / "data/raw/rdc_rocks/derived/integration/phase7"
CANON = OUT / "canonical_phase6"
ACQUISITION_MANIFEST = PROJECT_ROOT / "data/raw/rdc_rocks/metadata/acquisition_manifest_v1.csv"
USGS_DIR = PROJECT_ROOT / "data/raw/usgs_sir2010_5090_t"

SOURCES = {
    "RDC_3863": {
        "group": "structural",
        "gpkg": P6 / "3863/rdc_3863_structural_v1.gpkg",
        "script": PROJECT_ROOT / "src/data_acquisition/rdc_rocks/build_phase6_3863.py",
        "layers": ["structural_faults", "thrust_sheets", "fold_axes", "contacts", "extrusions", "excrescences"],
    },
    "RDC_3854": {
        "group": "stratigraphic",
        "gpkg": P6 / "3854/rdc_3854_stratigraphy_v1.gpkg",
        "script": PROJECT_ROOT / "src/data_acquisition/rdc_rocks/build_phase6_3854.py",
        "layers": ["r4_facies", "r4_absent", "facies_boundaries", "isopach_observations", "structural_faults", "discordances"],
    },
    "RDC_4055": {
        "group": "structural",
        "gpkg": P6 / "4055/rdc_4055_structural_v1.gpkg",
        "script": PROJECT_ROOT / "src/data_acquisition/rdc_rocks/build_phase6_4055.py",
        "layers": ["carapaces", "thrust_sheets"],
    },
}

ARTIFACTS = [
    ("geopackage", "RDC_3863", "1", P6 / "3863/rdc_3863_structural_v1.gpkg"),
    ("geopackage", "RDC_3854", "1", P6 / "3854/rdc_3854_stratigraphy_v1.gpkg"),
    ("geopackage", "RDC_4055", "1", P6 / "4055/rdc_4055_structural_v1.gpkg"),
    ("report", "RDC_3863", "1", P6 / "3863/rdc_3863_digitising_report_v1.md"),
    ("report", "RDC_3854", "1", P6 / "3854/rdc_3854_transformation_report_v1.md"),
    ("report", "RDC_4055", "1", P6 / "4055/rdc_4055_transformation_report_v1.md"),
    ("validation", "RDC_3863;RDC_3854;RDC_4055", "1", PROJECT_ROOT / "data/raw/rdc_rocks/derived/phase6_validation_v1.md"),
    ("handoff", "RDC.ROCKS", "n/a", PROJECT_ROOT / "ACQUISITION_HANDOFF_v3.md"),
]

FIELDS = [
    ("evidence_id", "text"), ("feature_group", "text"), ("source_doc", "text"),
    ("source_page", "integer"), ("source_layer", "text"), ("feature_id", "text"),
    ("original_label", "text"), ("original_legend_term", "text"), ("harmonised_class", "text"),
    ("geological_interpretation", "text"), ("evidence_basis", "text"), ("is_inferred", "integer"),
    ("attribution_tier", "text"), ("confidence", "text"), ("coordinate_space", "text"),
    ("crs_status", "text"), ("geometry_type", "text"), ("geometry_provenance", "text"),
    ("digitisation_method", "text"), ("positional_uncertainty_m", "integer"),
    ("uncertainty_basis", "text"), ("source_pdf_sha256", "text"), ("processing_script", "text"),
    ("processing_script_sha256", "text"), ("source_attributes_json", "text"), ("notes", "text"),
]

SCHEMA_FIELDS = {
    "evidence_id": "Stable Phase 7 evidence identifier derived from source document and source feature order.",
    "feature_group": "structural or stratigraphic.",
    "source_doc": "Source RDC.ROCKS document identifier (RDC_3863, RDC_3854, RDC_4055).",
    "source_page": "Source page or plate number; all Phase 6 maps are page/plate 1.",
    "source_layer": "Original Phase 6 GeoPackage layer name.",
    "feature_id": "Stable source feature identifier, unchanged from Phase 6.",
    "original_label": "Original map label or name_fr value.",
    "original_legend_term": "Original French legend terminology; never replaced by a modern term.",
    "harmonised_class": "Cross-dataset class used for integration; original terms are retained separately.",
    "geological_interpretation": "What the feature represents, limited to the printed map interpretation.",
    "evidence_basis": "cartographic_observation or cartographic_interpretation.",
    "is_inferred": "1 where the feature is an interpreted boundary, envelope or axis; 0 for a directly printed trace or measurement.",
    "attribution_tier": "A_direct_cartographic, B_interpreted_cartographic or C_envelope_or_low_confidence.",
    "confidence": "Source digitising confidence.",
    "coordinate_space": "Original coordinate space text; grid metres or source pixels.",
    "crs_status": "Explicit unresolved-datum status; no EPSG assignment.",
    "geometry_type": "POINT, LINESTRING or POLYGON.",
    "geometry_provenance": "Original Phase 6 provenance string.",
    "digitisation_method": "Manual tracing method used in Phase 6.",
    "positional_uncertainty_m": "Conservative map-scale uncertainty in metres; 0 is not used.",
    "uncertainty_basis": "Report sentence or map-scale basis for the uncertainty value.",
    "source_pdf_sha256": "SHA-256 of the source PDF recorded in the acquisition manifest.",
    "processing_script": "Phase 6 builder that produced the source feature.",
    "processing_script_sha256": "SHA-256 of the Phase 6 builder at integration time.",
    "source_attributes_json": "All non-geometry source attributes preserved verbatim as JSON.",
    "notes": "Source notes and integration caveats.",
}

CLASS_RULES = {
    ("RDC_3863", "thrust_fault"): ("fault_thrust", "Mapped thrust-fault trace; kinematics as printed", "cartographic_observation", 0, "A_direct_cartographic", 2000),
    ("RDC_3863", "back_thrust"): ("fault_back_thrust", "Mapped back-thrust trace; kinematics as printed", "cartographic_observation", 0, "A_direct_cartographic", 2000),
    ("RDC_3863", "thrust_sheet_boundary"): ("thrust_sheet_outline", "Outline of one of the seven numbered thrust sheets", "cartographic_interpretation", 1, "B_interpreted_cartographic", 2000),
    ("RDC_3863", "contact_r4_r3"): ("contact_stratigraphic", "Mapped contact between R.4 and R.3", "cartographic_interpretation", 1, "B_interpreted_cartographic", 2000),
    ("RDC_3863", "faulted_contact"): ("contact_faulted", "Contact mapped as faulted", "cartographic_interpretation", 1, "B_interpreted_cartographic", 2000),
    ("RDC_3863", "anticlinal_axis"): ("fold_axis_anticline", "Anticlinal axis with plunge symbol where printed", "cartographic_interpretation", 1, "B_interpreted_cartographic", 2000),
    ("RDC_3863", "synclinal_axis"): ("fold_axis_syncline", "Synclinal axis with plunge symbol where printed", "cartographic_interpretation", 1, "B_interpreted_cartographic", 2000),
    ("RDC_3863", "extrusion"): ("extrusion_outline", "Mapped extrusion outline", "cartographic_interpretation", 1, "B_interpreted_cartographic", 2000),
    ("RDC_3863", "excrescence"): ("excrescence_outline", "Mapped excrescence outline", "cartographic_interpretation", 1, "B_interpreted_cartographic", 2000),
    ("RDC_3854", "r4_kabul"): ("r4_facies_belt", "Approximate envelope of the R.4 Kabul facies belt", "cartographic_interpretation", 1, "C_envelope_or_low_confidence", 2000),
    ("RDC_3854", "r4_chabara"): ("r4_facies_belt", "Approximate envelope of the R.4 Chabara facies belt", "cartographic_interpretation", 1, "C_envelope_or_low_confidence", 2000),
    ("RDC_3854", "r4_luankonko"): ("r4_facies_belt", "Approximate envelope of the R.4 Luankonko facies belt", "cartographic_interpretation", 1, "C_envelope_or_low_confidence", 2000),
    ("RDC_3854", "r4_absent"): ("r4_absent_zone", "Area where R.4 is mapped absent", "cartographic_interpretation", 1, "C_envelope_or_low_confidence", 2000),
    ("RDC_3854", "facies_boundary"): ("facies_boundary", "Boundary between R.4 facies belts", "cartographic_interpretation", 1, "C_envelope_or_low_confidence", 2000),
    ("RDC_3854", "isopach_observation"): ("isopach_observation", "Printed R.4.2/R.4.1 thickness annotation", "cartographic_observation", 0, "A_direct_cartographic", 2000),
    ("RDC_3854", "thrust_fault"): ("fault_thrust", "Mapped thrust-fault trace", "cartographic_observation", 0, "A_direct_cartographic", 2000),
    ("RDC_3854", "discordance"): ("discordance", "Mapped discordance trace", "cartographic_interpretation", 1, "B_interpreted_cartographic", 2000),
    ("RDC_4055", "carapace"): ("carapace_outline", "Outline of a numbered carapace (overthrust remnant)", "cartographic_interpretation", 1, "C_envelope_or_low_confidence", 3000),
    ("RDC_4055", "thrust_sheet"): ("thrust_sheet_outline", "Outline of a numbered thrust-sheet window", "cartographic_interpretation", 1, "C_envelope_or_low_confidence", 3000),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_gpkg_geometry(blob: bytes):
    assert blob[:2] == b"GP", "not a GeoPackage geometry"
    offset = 8
    endian = "<" if blob[offset] == 1 else ">"
    geometry_type = struct.unpack_from(endian + "I", blob, offset + 1)[0]
    cursor = offset + 5
    if geometry_type == 1:
        x, y = struct.unpack_from(endian + "dd", blob, cursor)
        return "POINT", (x, y)
    if geometry_type == 2:
        count = struct.unpack_from(endian + "I", blob, cursor)[0]
        cursor += 4
        points = []
        for _ in range(count):
            x, y = struct.unpack_from(endian + "dd", blob, cursor)
            cursor += 16
            points.append((x, y))
        return "LINESTRING", points
    if geometry_type == 3:
        ring_count = struct.unpack_from(endian + "I", blob, cursor)[0]
        cursor += 4
        rings = []
        for _ in range(ring_count):
            count = struct.unpack_from(endian + "I", blob, cursor)[0]
            cursor += 4
            ring = []
            for _ in range(count):
                x, y = struct.unpack_from(endian + "dd", blob, cursor)
                cursor += 16
                ring.append((x, y))
            rings.append(ring)
        return "POLYGON", rings
    raise ValueError(f"unsupported WKB geometry type {geometry_type}")


def flatten_points(geometry_type: str, parts) -> list[tuple[float, float]]:
    if geometry_type == "POINT":
        return [parts]
    if geometry_type == "LINESTRING":
        return list(parts)
    return [point for ring in parts for point in ring]


def geometry_wkt(geometry_type: str, parts) -> str:
    def fmt(point):
        return f"{point[0]:.6f} {point[1]:.6f}"

    if geometry_type == "POINT":
        return f"POINT ({fmt(parts)})"
    if geometry_type == "LINESTRING":
        return "LINESTRING (" + ", ".join(fmt(point) for point in parts) + ")"
    rings = ["(" + ", ".join(fmt(point) for point in ring) + ")" for ring in parts]
    return "POLYGON (" + ", ".join(rings) + ")"


def read_source_features() -> list[dict]:
    manifest_rows = list(csv.DictReader(ACQUISITION_MANIFEST.open(encoding="utf-8-sig")))
    pdf_hash = {row["id"]: row["sha256"] for row in manifest_rows}
    features = []
    for doc, spec in SOURCES.items():
        script_hash = sha256(spec["script"])
        connection = sqlite3.connect(spec["gpkg"])
        connection.row_factory = sqlite3.Row
        for layer in spec["layers"]:
            rows = connection.execute(f'SELECT * FROM "{layer}" ORDER BY fid').fetchall()
            for row in rows:
                attributes = {key: row[key] for key in row.keys() if key not in ("fid", "geom")}
                rule = CLASS_RULES[(doc, attributes["feature_class"])]
                harmonised, interpretation, basis, inferred, tier, uncertainty = rule
                geometry_type, parts = parse_gpkg_geometry(row["geom"])
                doc_id = doc.replace("RDC_", "")
                assert doc_id in pdf_hash, doc
                features.append({
                    "evidence_id": f"EV-{attributes['feature_id']}",
                    "feature_group": spec["group"],
                    "source_doc": doc,
                    "source_page": int(attributes.get("source_page") or 1),
                    "source_layer": layer,
                    "feature_id": attributes["feature_id"],
                    "original_label": attributes.get("name_fr", ""),
                    "original_legend_term": attributes.get("original_legend_term", ""),
                    "harmonised_class": harmonised,
                    "geological_interpretation": interpretation,
                    "evidence_basis": basis,
                    "is_inferred": inferred,
                    "attribution_tier": tier,
                    "confidence": attributes.get("confidence", ""),
                    "coordinate_space": attributes.get("coordinate_space", ""),
                    "crs_status": (
                        "datum_unresolved_no_projection_source_pixel_easting_calibrated"
                        if doc == "RDC_4055" else "datum_unresolved_no_projection_map_grid_metres"
                    ),
                    "geometry_type": geometry_type,
                    "geometry_provenance": attributes.get("geometry_provenance", ""),
                    "digitisation_method": (
                        "manual_envelope_on_source_pixel_grid" if doc == "RDC_4055"
                        else "manual_trace_on_gridded_raster"
                    ),
                    "positional_uncertainty_m": uncertainty,
                    "uncertainty_basis": (
                        "Phase 6 report: 1-3 km envelope uncertainty" if doc == "RDC_4055"
                        else "Phase 6 report: 1-2 km manual-trace uncertainty"
                    ),
                    "source_pdf_sha256": pdf_hash[doc_id],
                    "processing_script": rel(spec["script"]),
                    "processing_script_sha256": script_hash,
                    "source_attributes_json": json.dumps(attributes, ensure_ascii=False, sort_keys=True),
                    "notes": attributes.get("notes", ""),
                    "_geom": row["geom"],
                    "_parts": parts,
                    "_points": flatten_points(geometry_type, parts),
                })
        connection.close()
    return features


def build_canonical_artifacts() -> list[dict]:
    CANON.mkdir(parents=True, exist_ok=True)
    manifest = []
    script_path = PROJECT_ROOT / "src/data_acquisition/rdc_rocks/build_phase7_integration.py"
    script_hash = sha256(script_path)
    pdf_rows = {row["id"]: row["sha256"] for row in csv.DictReader(ACQUISITION_MANIFEST.open(encoding="utf-8-sig"))}
    for artifact_type, docs, pages, original in ARTIFACTS:
        canonical = CANON / original.name
        shutil.copy2(original, canonical)
        original_hash = sha256(original)
        canonical_hash = sha256(canonical)
        assert original_hash == canonical_hash, f"copy mismatch: {original}"
        source_docs = docs.split(";")
        source_hashes = []
        for doc in source_docs:
            if doc.startswith("RDC_") and doc[4:].isdigit():
                source_hashes.append(f"{doc}:{pdf_rows[doc[4:]]}")
        manifest.append({
            "artifact_type": artifact_type,
            "source_documents": docs,
            "source_pages": pages,
            "original_path": rel(original),
            "canonical_path": rel(canonical),
            "bytes": original.stat().st_size,
            "sha256_original": original_hash,
            "sha256_canonical": canonical_hash,
            "hash_match": "yes",
            "source_document_sha256": ";".join(source_hashes),
            "processing_script": rel(script_path),
            "processing_script_sha256": script_hash,
            "recorded_utc": utc_now(),
        })
    return manifest


def build_unified_evidence(features: list[dict]) -> dict:
    grouped = {}
    for feature in features:
        kind = {"POINT": "points", "LINESTRING": "lines", "POLYGON": "polygons"}[feature["geometry_type"]]
        doc_id = feature["source_doc"].replace("RDC_", "")
        grouped.setdefault((doc_id, kind), []).append(feature)

    layers = []
    for (doc_id, kind), items in sorted(grouped.items()):
        group_name = "structural" if items[0]["feature_group"] == "structural" else "stratigraphic"
        name = f"{group_name}_{doc_id}_{kind}"
        geometry_type = {"points": "POINT", "lines": "LINESTRING", "polygons": "POLYGON"}[kind]
        built = []
        for feature in items:
            attributes = {field: feature[field] for field, _ in FIELDS}
            built.append({"geom": feature["_geom"], "points": feature["_points"], "attributes": attributes})
        layers.append({
            "name": name, "geometry_type": geometry_type, "fields": FIELDS, "features": built,
            "description": f"Unified Phase 7 evidence for {doc_id}, {kind}; original geometry and unresolved CRS.",
        })
    summary = write_gpkg(OUT / "rdc_geological_evidence_v1.gpkg", layers)

    csv_fields = [field for field, _ in FIELDS] + ["geometry_wkt"]
    rows = []
    for feature in features:
        row = {field: feature[field] for field, _ in FIELDS}
        row["geometry_wkt"] = geometry_wkt(feature["geometry_type"], feature["_parts"])
        rows.append(row)
    write_csv(OUT / "geological_evidence_v1.csv", csv_fields, rows)
    return {"layers": summary, "features": len(features)}


def build_schema() -> None:
    schema = {
        "schema_version": "1.0.0",
        "generated_utc": utc_now(),
        "description": "Unified structural and stratigraphic evidence schema for Phase 6 RDC.ROCKS maps.",
        "coordinate_spaces": {
            "RDC_3863": "map grid metres; datum unresolved; no projection",
            "RDC_3854": "map grid metres; datum unresolved; no projection",
            "RDC_4055": "source pixels; easting calibrated; northing unresolved; no projection",
        },
        "fields": [
            {"name": field, "type": kind, "description": SCHEMA_FIELDS[field]}
            for field, kind in FIELDS
        ],
        "rules": [
            "Original legend terms are retained; harmonised classes never replace them.",
            "is_inferred=1 marks interpreted boundaries, envelopes and axes, not direct measurements.",
            "Map crossings and apparent continuities are not promoted to confirmed structures.",
            "No datum, projection or EPSG code is assigned.",
            "RDC_4055 remains in source-pixel space with easting metadata only.",
        ],
    }
    (OUT / "geological_evidence_schema_v1.json").write_text(
        json.dumps(schema, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Geological evidence schema (v1)", "",
        "Unified schema for the structural and stratigraphic Phase 6 evidence.", "",
        "## Fields", "", "| Field | Type | Description |", "|---|---|---|",
    ]
    for field, kind in FIELDS:
        lines.append(f"| `{field}` | {kind} | {SCHEMA_FIELDS[field]} |")
    lines += [
        "", "## Rules", "",
        "- Original legend terms are retained; harmonised classes never replace them.",
        "- `is_inferred=1` marks interpreted boundaries, envelopes and axes, not direct measurements.",
        "- Map crossings and apparent continuities are not promoted to confirmed structures.",
        "- No datum, projection or EPSG code is assigned.",
        "- RDC_4055 remains in source-pixel space with easting metadata only.",
        "",
    ]
    (OUT / "geological_evidence_schema_v1.md").write_text("\n".join(lines), encoding="utf-8")


def build_occurrence_reconciliation() -> dict:
    observations = list(csv.DictReader((OCC / "entities/occurrence_observations_v1.csv").open(encoding="utf-8-sig")))
    sites = list(csv.DictReader((OCC / "entities/site_entities_v1.csv").open(encoding="utf-8-sig")))
    crosswalk = list(csv.DictReader((OCC / "entities/occurrence_entity_crosswalk_v1.csv").open(encoding="utf-8-sig")))
    assert len(observations) == 147
    assert len(sites) == 113
    assert len(crosswalk) == 147
    site_counts = {row["site_entity_id"]: int(row["observation_count"]) for row in sites}
    candidate_counts = {}
    for row in crosswalk:
        if row["usgs_name"].strip():
            candidate_counts[row["usgs_name"]] = candidate_counts.get(row["usgs_name"], 0) + 1

    rows = []
    for row in crosswalk:
        candidate = row["usgs_name"].strip()
        group_size = candidate_counts.get(candidate, 0) if candidate else 0
        status = row["review_status"]
        if status == "ambiguous_many_to_one" or group_size > 1:
            priority = "high"
            action = "resolve_many_to_one_before_acceptance"
        elif status == "site_level_candidate_unverified":
            priority = "medium"
            action = "verify_name_coordinates_and_host_unit"
        else:
            priority = "low"
            action = "no_action_keep_unmatched"
        notes = []
        if group_size > 1:
            notes.append(f"{group_size} source rows share this USGS candidate")
        if site_counts.get(row["site_entity_id"], 0) > 1:
            notes.append(f"site has {site_counts[row['site_entity_id']]} observations")
        rows.append({
            "reconciliation_id": f"REC-{row['crosswalk_id']}",
            "occurrence_id": row["occurrence_id"],
            "site_entity_id": row["site_entity_id"],
            "original_name": row["original_name"],
            "commodity": row["commodity"],
            "candidate_usgs_name": candidate,
            "candidate_latitude": row["usgs_latitude"],
            "candidate_longitude": row["usgs_longitude"],
            "existing_match_confidence": row["match_confidence"],
            "existing_match_basis": row["match_basis"],
            "existing_review_status": status,
            "confirmed_identity": "no",
            "candidate_group_size": group_size,
            "site_observation_count": site_counts.get(row["site_entity_id"], 0),
            "review_priority": priority,
            "review_action": action,
            "verification_required": "yes" if candidate else "no",
            "verification_type": "geological_map_and_historical_gazetteer" if candidate else "",
            "match_rationale": row["match_rationale"],
            "use_as_model_positive": "no",
            "notes": "; ".join(notes),
        })
    write_csv(OUT / "occurrence_reconciliation_v1.csv", list(rows[0].keys()), rows)
    status_counts = {}
    for row in rows:
        status_counts[row["existing_review_status"]] = status_counts.get(row["existing_review_status"], 0) + 1
    return {
        "rows": len(rows), "status_counts": status_counts, "confirmed": 0,
        "candidates": sum(1 for row in rows if row["candidate_usgs_name"]),
    }


def build_exploration_register() -> dict:
    history = list(csv.DictReader((EXP / "exploration_history_v1.csv").open(encoding="utf-8-sig")))
    assert len(history) == 21
    names = {
        "EXP-002": ("Mission URAKAT (Fairchild campaign)", "Fairchild (as stated)", "explicit_in_text"),
        "EXP-003": ("Campagne Hunting", "Hunting (as stated)", "explicit_in_text"),
        "EXP-004": ("Association Uranium Zaire", "COGEMA-CGEA-GECAMINES", "explicit_in_text"),
        "EXP-005": ("Luiswishi borehole Lsw-1215", "not stated", "descriptive_record"),
        "EXP-006": ("Luiswishi borehole Lsw-1216", "not stated", "descriptive_record"),
        "EXP-008": ("UMHK/Gecamines Lufilian Arc concession", "UMHK/Gecamines", "explicit_in_text"),
        "EXP-009": ("CHEMAF Kolwezi exploration", "CHEMAF", "explicit_in_text"),
        "EXP-010": ("CHEMAF concession traverses", "CHEMAF", "explicit_in_text"),
        "EXP-015": ("Kamoto Etang Nord underground mapping", "KCC", "explicit_in_text"),
        "EXP-016": ("KCC operational support", "KCC", "explicit_in_text"),
        "EXP-017": ("1953 Mitwaba map compilation", "Van den Branden (source map); Comite Special du Katanga (topographic base)", "explicit_in_text"),
        "EXP-021": ("RDC.ROCKS corpus absence assessment", "RDC.ROCKS project", "project_assessment"),
    }
    rows = []
    for row in history:
        evidence_id = row["evidence_id"]
        name, operator, basis = names.get(evidence_id, ("not formally named", "not stated", "not_stated"))
        if evidence_id in ("EXP-005", "EXP-006"):
            footprint = "coordinate_text_local_datum_unresolved"
            effort = "partial_point_only"
            effort_note = "Local-grid borehole coordinates are present in prose but the datum is unresolved; no point was created."
        elif row["evidence_class"] == "modern_operational_context":
            footprint = "no_footprint_modern_context"
            effort = "no"
            effort_note = "Modern operational acknowledgement; not exploration effort."
        elif row["evidence_class"] == "documentary_absence":
            footprint = "no_footprint_documentary_absence"
            effort = "no"
            effort_note = "Negative evidence about the acquired corpus, not a mineralisation observation."
        elif evidence_id in ("EXP-003", "EXP-008"):
            footprint = "textual_area_only"
            effort = "partial_area_only"
            effort_note = "A survey or concession area is stated in text; no polygon is supplied."
        elif row["spatial_precision"] == "regional":
            footprint = "textual_regional_extent_only"
            effort = "partial_presence_only"
            effort_note = "Regional campaign presence is documented; no spatial footprint."
        else:
            footprint = "textual_place_name_only"
            effort = "partial_presence_only"
            effort_note = "Named locality documents activity, not its spatial extent or intensity."
        source_doc = "RDC.ROCKS corpus" if row["record_id"] == "ALL" else f"RDC_{row['record_id']}"
        rows.append({
            "evidence_id": evidence_id,
            "evidence_class": row["evidence_class"],
            "campaign_or_survey_name": name,
            "name_basis": basis,
            "operator_or_institution": operator,
            "date_start": row["date_start"],
            "date_end": row["date_end"],
            "date_text": row["date_text"],
            "commodity": row["commodity"],
            "geographic_coverage": row["geographic_scope"],
            "spatial_precision": row["spatial_precision"],
            "source_doc": source_doc,
            "source_page": row["source_page"],
            "footprint_status": footprint,
            "footprint_geometry_available": "no",
            "can_constrain_effort": effort,
            "effort_constraint_note": effort_note,
            "evidence_type_original": row["evidence_type_original"],
            "direct_observation": row["direct_observation"],
            "observation_or_inference": row["observation_or_inference"],
            "confidence": row["confidence"],
        })
    write_csv(OUT / "exploration_evidence_register_v1.csv", list(rows[0].keys()), rows)
    return {
        "rows": len(rows),
        "dated": sum(1 for row in rows if row["date_start"].strip()),
        "footprints": sum(1 for row in rows if row["footprint_geometry_available"] == "yes"),
        "classes": {row["evidence_class"] for row in rows},
    }


def build_predictor_table() -> list[dict]:
    rows = [
        ("P01", "Distance to mapped thrust fault", "primary_geological", "primary_candidate",
         "Thrust faults are first-order controls on Copperbelt mineralisation and fluid flow.",
         "RDC_3863, RDC_3854, RDC_4055 fault layers", "Phase 6 GPKGs; unresolved datum",
         "Three map footprints only; not the full basin", "2-3 km", "1-10 km distance",
         "Correlated with fault density and thrust-sheet proximity", "Eligible within map coverage; mask elsewhere."),
        ("P02", "Thrust-fault density (10 km)", "secondary_geological", "secondary_candidate",
         "Fault density may proxy strain and fluid-pathway abundance.",
         "RDC_3863, RDC_3854 fault layers", "Phase 6 GPKGs", "Same three map footprints", "2-3 km", "10 km neighbourhood",
         "Strongly correlated with P01; avoid joint primary use", "Secondary or sensitivity."),
        ("P03", "Distance to back-thrust", "secondary_geological", "secondary_candidate",
         "Back-thrusts bound some thrust sheets and may localise secondary structures.",
         "RDC_3863 fault layer", "Phase 6 GPKG", "RDC_3863 footprint only", "2 km", "1-10 km",
         "Correlated with P01 and thrust-sheet boundaries", "Use only with coverage mask."),
        ("P04", "Named fault-system membership", "context_only", "context",
         "Named systems support interpretation but names do not define geometry.",
         "Phase 6 labels and published nomenclature", "Map labels", "Textual only", "Not applicable", "Categorical",
         "Not a spatial variable without validated traces", "Context only; do not model as a predictor."),
        ("P05", "Fault-intersection density", "excluded_pending_evidence", "excluded",
         "Intersections may localise permeability, but map crossings are not confirmed intersections.",
         "Digitised traces", "Phase 6 GPKGs", "Map footprints", "2-3 km", "Point or 5 km neighbourhood",
         "Would duplicate P01/P02 and overstate certainty", "Excluded until structural validation."),
        ("P06", "Distance to thrust-sheet boundary", "secondary_geological", "secondary_candidate",
         "Thrust-sheet margins may control host-unit repetition and structural traps.",
         "RDC_3863 and RDC_4055 outlines", "Phase 6 GPKGs", "Two structural map footprints", "2-3 km", "1-10 km",
         "Correlated with fault distance; sheet outlines are envelopes", "Secondary with coverage mask."),
        ("P07", "Thrust-sheet membership", "context_only", "context",
         "Sheet assignment is a mapped interpretation, not an independent observation.",
         "RDC_3863 and RDC_4055 polygons", "Phase 6 GPKGs", "Two map footprints", "2-3 km", "Polygon membership",
         "Overlapping and multi-window outlines", "Context or sensitivity only."),
        ("P08", "Distance to carapace boundary", "sensitivity_only", "sensitivity",
         "Carapace margins may juxtapose allochthonous and autochthonous units.",
         "RDC_4055 carapace outlines", "Phase 6 GPKG; pixel space", "Kolwezi plate only", "3 km", "1-10 km",
         "Correlated with thrust-sheet and fault distance", "Sensitivity only; no northing calibration."),
        ("P09", "Carapace membership", "context_only", "context",
         "Carapace membership is a mapped interpretation with low boundary confidence.",
         "RDC_4055 carapace outlines", "Phase 6 GPKG; pixel space", "Kolwezi plate only", "3 km", "Polygon membership",
         "Overlaps sheet membership", "Context only."),
        ("P10", "R.4 facies membership (Kabul/Chabara/Luankonko)", "secondary_geological", "sensitivity",
         "R.4 facies belts may proxy host-rock chemistry and basin position.",
         "RDC_3854 facies envelopes", "Phase 6 GPKG", "RDC_3854 footprint", "2 km", "Polygon membership",
         "Envelopes are not contact-traced; classes may overlap or leave gaps", "Sensitivity until contact-traced."),
        ("P11", "R.4 absent-zone membership", "sensitivity_only", "sensitivity",
         "R.4 absence may indicate erosion, non-deposition or structural omission.",
         "RDC_3854 absent polygons", "Phase 6 GPKG", "RDC_3854 footprint", "2 km", "Polygon membership",
         "Depends on facies boundary interpretation", "Sensitivity only."),
        ("P12", "Distance to facies boundary", "sensitivity_only", "sensitivity",
         "Facies transitions may control host-unit juxtaposition.",
         "RDC_3854 facies boundaries", "Phase 6 GPKG", "RDC_3854 footprint", "2 km", "1-10 km",
         "Boundaries are envelopes; correlated with P10", "Sensitivity only."),
        ("P13", "R.4.2 thickness", "secondary_geological", "secondary_candidate",
         "Thickness variation may reflect basin architecture and host-unit volume.",
         "RDC_3854 23 isopach labels", "Phase 6 GPKG", "Sparse point coverage", "2 km", "Interpolated surface at 10-25 km",
         "Interpolation uncertainty; correlated with P14", "Secondary only; publish interpolation diagnostics."),
        ("P14", "R.4.1 thickness", "secondary_geological", "secondary_candidate",
         "Thickness variation may reflect basin architecture and host-unit volume.",
         "RDC_3854 23 isopach labels", "Phase 6 GPKG", "Sparse point coverage", "2 km", "Interpolated surface at 10-25 km",
         "Correlated with P13", "Secondary only; publish interpolation diagnostics."),
        ("P15", "R.4.2/R.4.1 thickness ratio", "sensitivity_only", "sensitivity",
         "Ratio may proxy relative accommodation or truncation.",
         "RDC_3854 paired thickness labels", "Phase 6 GPKG", "Sparse point coverage", "2 km", "Interpolated surface",
         "Ratio amplifies interpolation error", "Sensitivity only."),
        ("P16", "Distance to R.4/R.3 contact", "secondary_geological", "secondary_candidate",
         "The R.4/R.3 contact is a major stratigraphic boundary in the Copperbelt.",
         "RDC_3863 contact layer", "Phase 6 GPKG", "RDC_3863 footprint; 3 low-confidence traces", "2-3 km", "1-10 km",
         "Few traces; correlated with thrust faults", "Secondary or sensitivity; validate contacts first."),
        ("P17", "Distance to discordance", "sensitivity_only", "sensitivity",
         "Discordances may mark omission or erosional surfaces.",
         "RDC_3854 discordance layer", "Phase 6 GPKG", "RDC_3854 footprint", "2 km", "1-10 km",
         "Only three traces", "Sensitivity only."),
        ("P18", "Distance to fold axis", "sensitivity_only", "sensitivity",
         "Fold hinges may control fracture density and fluid pathways.",
         "RDC_3863 fold axes", "Phase 6 GPKG", "RDC_3863 footprint", "2 km", "1-10 km",
         "Axes are interpreted traces; correlated with faults", "Sensitivity only."),
        ("P19", "Fold-axis density", "sensitivity_only", "sensitivity",
         "Fold intensity may proxy strain.",
         "RDC_3863 fold axes", "Phase 6 GPKG", "RDC_3863 footprint", "2 km", "10 km neighbourhood",
         "Correlated with P18 and fault density", "Sensitivity only."),
        ("P20", "Extrusion/excrescence proximity", "sensitivity_only", "sensitivity",
         "These bodies may mark local structural or diapiric pathways.",
         "RDC_3863 extrusion and excrescence polygons", "Phase 6 GPKG", "RDC_3863 footprint", "2 km", "1-5 km",
         "Rare features; unstable at small sample sizes", "Sensitivity only."),
        ("P21", "Cover/exposure (Kalahari)", "excluded_pending_evidence", "excluded",
         "Cover thickness and exposure affect both geology and detection.",
         "Map legend includes Kalahari, but no cover polygon was digitised", "Phase 6 review", "Not available", "Not available", "Polygon or raster",
         "Not a validated surface", "Excluded pending a digitised cover/exposure layer."),
        ("P22", "Exploration campaign footprint", "observation_process_proxy", "sensitivity",
         "Campaign footprints could proxy observation effort if spatially documented.",
         "21 exploration records", "Phase 7 exploration register", "No spatial geometry", "Not applicable", "Polygon or distance",
         "Only textual areas and two unresolved boreholes", "Sensitivity only; cannot form a detection surface."),
        ("P23", "Document-source effect", "observation_process_proxy", "sensitivity",
         "Occurrence reporting is clustered by source document and compiler.",
         "147 observations and 5 source documents", "Phase 7 occurrence tables", "Categorical", "Not applicable", "Categorical offset",
         "Confounded with geography and era", "Estimable as a sensitivity stratification."),
        ("P24", "Occurrence spatial-precision stratum", "observation_process_proxy", "sensitivity",
         "Place-name-only records have different locational error from coordinate records.",
         "147 observations", "Phase 7 occurrence tables", "All records", "Variable", "Categorical or weighting",
         "Correlated with source document", "Sensitivity; never treat as geological predictor."),
        ("P25", "Reporting era", "observation_process_proxy", "excluded",
         "Reporting intensity varies through time.",
         "21 exploration records", "Phase 7 exploration register", "Only 5 dated records", "Not applicable", "Time slice",
         "Cannot date 147 occurrence records", "Excluded pending a dated occurrence inventory."),
        ("P26", "District/tenement context", "context_only", "context",
         "Administrative and concession boundaries shape access and reporting.",
         "Published concession descriptions", "Exploration register", "Textual", "Variable", "Categorical",
         "Not a geological variable", "Context only."),
        ("P27", "Distance to nearest occurrence", "excluded_pending_evidence", "excluded",
         "Would leak the target into the predictors.",
         "Occurrence tables", "Phase 7 occurrence tables", "Outcome-derived", "Not applicable", "Distance",
         "Direct target leakage", "Excluded from all models."),
        ("P28", "Mineral-occurrence density", "excluded_pending_evidence", "excluded",
         "This is the outcome, not a predictor.",
         "Occurrence tables", "Phase 7 occurrence tables", "Outcome-derived", "Not applicable", "Kernel density",
         "Direct target leakage", "Excluded from all models."),
        ("P29", "Nappe front/root distance", "excluded_pending_evidence", "excluded",
         "Nappe architecture is first-order but was not digitised in Phase 6.",
         "RDC_4055 and RDC_3863 legends", "Phase 6 review", "Not available as geometry", "Not applicable", "Distance",
         "Not digitised", "Excluded pending trace digitisation and validation."),
        ("P30", "Frozen-foundation lithology/stratigraphy", "context_only", "context",
         "Existing foundation provides basin context but is outside Phase 7 integration.",
         "Frozen geology foundation", "Existing V14-era data", "Basin-wide", "Foundation-specific", "Raster or polygon",
         "Must not be silently rebuilt or mixed with Phase 6", "Context only; assess separately before model use."),
    ]
    fields = ["predictor_id", "predictor_name", "category", "model_use", "geological_rationale",
              "data_source", "provenance", "spatial_coverage", "positional_uncertainty",
              "expected_spatial_support", "redundancy_collinearity", "eligibility_notes"]
    records = [dict(zip(fields, row)) for row in rows]
    write_csv(OUT / "predictor_eligibility_v1.csv", fields, records)
    return records


def build_observation_outputs() -> dict:
    scenarios = [
        ("OBS-S0", "No observation component", "None", "yes", "None",
         "Baseline model with geology only.", "Assumes occurrence intensity is proportional to prospectivity; cannot separate reporting bias."),
        ("OBS-S1", "District reporting intensity", "District-level occurrence counts", "no", "A consistent district polygon layer and dated inventory",
         "Not implementable from the current place-name records.", "Would be circular if districts are defined from the same occurrences."),
        ("OBS-S2", "Document-source fixed effect", "Categorical offset by source document", "yes", "Existing 147 observations with source document",
         "Stratify or offset by source document in sensitivity models.", "Confounded with geography, era and compiler; not a detection surface."),
        ("OBS-S3", "Campaign-footprint proximity", "Distance to documented survey polygons", "no", "Survey footprints",
         "No polygons exist; only textual areas and two local-grid boreholes.", "Cannot be implemented without inventing footprints."),
        ("OBS-S4", "Reporting-era stratification", "Time-slice intensity", "no", "Dated occurrence records",
         "Only 5 exploration records are dated; occurrences are undated.", "Cannot estimate temporal reporting intensity."),
        ("OBS-S5", "Spatial-precision restriction", "Restrict or down-weight place-name-only records", "yes", "Existing coordinate_format and positional_uncertainty fields",
         "Compare models on coordinate-bearing records versus all records.", "Changes the target population; not a correction surface."),
        ("OBS-S6", "Entity-duplication restriction", "Collapse repeated commodity observations to site entities for sensitivity", "yes", "Existing 113 site entities and 147 observations",
         "Compare observation-level and site-level sensitivities without merging source records.", "Site entities are candidate clusters, not confirmed deposits."),
    ]
    fields = ["scenario_id", "scenario_name", "observation_component", "estimable_now", "data_required", "implementation", "interpretation_limit"]
    write_csv(OUT / "observation_scenarios_v1.csv", fields, [dict(zip(fields, row)) for row in scenarios])
    estimable = sum(1 for row in scenarios if row[3] == "yes")
    text = f"""# Observation-process evidence assessment (v1)

Generated: {utc_now()}

## Corpus

- 21 exploration-history records from five source documents plus one corpus-level assessment.
- Evidence classes: 4 historical campaign, 2 airborne survey, 4 ground survey, 5 drilling, 4 sampling/assay, 1 modern operational context, 1 documentary absence.
- Only 5 records carry a numeric start and end year.
- **Zero records have a spatial geometry.** Footprints are textual only. Two boreholes carry local-grid coordinates in prose, but the datum is unresolved.

## Assessment

The evidence is **not sufficient to identify a spatial observation or detection surface**. Historical exploration is documented as presence, chronology and local activity, not as continuous coverage. The records do not supply survey polygons, line kilometres, drill density, sampling density or a dated occurrence inventory. Claiming an identifiable observation intensity from this corpus would overstate the evidence.

What can be estimated:

- A document-source fixed effect, because all 147 observations carry a source document.
- A spatial-precision stratum, because coordinate format and positional uncertainty are recorded.
- An entity-duplication sensitivity, because 147 observations resolve to 113 candidate site entities.

What cannot be estimated:

- A continuous detection or reporting-intensity surface.
- Temporal reporting intensity, because occurrence records are undated.
- Campaign-footprint proximity, because no campaign polygon exists.
- Any correction that assumes unobserved areas were explored at the same intensity as documented areas.

## Bounded scenarios

`observation_scenarios_v1.csv` defines seven scenarios; {estimable} are estimable now (S0, S2, S5, S6). S1, S3 and S4 require data that do not exist in the acquired corpus and must remain unimplemented rather than approximated by place-name buffers.

No scenario may be interpreted as a mineralisation probability. Observation-process terms are sensitivity controls, not geological evidence.
"""
    (OUT / "observation_process_assessment_v1.md").write_text(text, encoding="utf-8")
    return {"records": 21, "dated": 5, "footprints": 0, "estimable_scenarios": estimable}


def build_protected_baselines() -> None:
    usgs_rows = []
    for path in sorted(USGS_DIR.rglob("*")):
        if path.is_file():
            usgs_rows.append({"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_csv(OUT / "usgs_baseline_hashes_v1.csv", ["path", "bytes", "sha256"], usgs_rows)

    protected = [
        PROJECT_ROOT / "data/derived/canonical_training_manifest_v14_centroid_contact.json",
        PROJECT_ROOT / "data/derived/litho_contact_centroid_distance_v14.csv",
        PROJECT_ROOT / "data/derived/litho_contact_centroid_distance_v14.gpkg",
        PROJECT_ROOT / "configs/geology_foundation.json",
    ]
    protected_dirs = [
        PROJECT_ROOT / "data/processed/stage4",
        PROJECT_ROOT / "data/processed/stage5",
        PROJECT_ROOT / "data/processed/stage5b",
        PROJECT_ROOT / "data/processed/stage5c",
        PROJECT_ROOT / "data/processed/geology_foundation",
        PROJECT_ROOT / "data/processed/structural_predictors",
        PROJECT_ROOT / "src/geology_foundation",
        PROJECT_ROOT / "paper_1/stages",
        PROJECT_ROOT / "paper_1/manuscript",
    ]
    for directory in protected_dirs:
        protected.extend(path for path in directory.rglob("*") if path.is_file())
    rows = []
    for path in sorted({p for p in protected if p.exists()}):
        rows.append({"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_csv(OUT / "frozen_baseline_hashes_v1.csv", ["path", "bytes", "sha256"], rows)


def build_provenance() -> None:
    script = PROJECT_ROOT / "src/data_acquisition/rdc_rocks/build_phase7_integration.py"
    script_hash = sha256(script)
    pdf_hash = {
        str(int(row["id"])): row["sha256"]
        for row in csv.DictReader(ACQUISITION_MANIFEST.open(encoding="utf-8-sig"))
        if row["id"].isdigit()
    }
    rows = [
        {
            "dataset_id": "P7-GEOLOGY",
            "output_path": rel(OUT / "rdc_geological_evidence_v1.gpkg"),
            "source_documents": "RDC_3863;RDC_3854;RDC_4055",
            "source_pages": "1;1;1",
            "input_artifacts": ";".join(rel(item) for item in [P6 / "3863/rdc_3863_structural_v1.gpkg", P6 / "3854/rdc_3854_stratigraphy_v1.gpkg", P6 / "4055/rdc_4055_structural_v1.gpkg"]),
            "processing_script": rel(script),
            "processing_script_sha256": script_hash,
            "source_file_sha256": ";".join(f"RDC_{key}:{pdf_hash[key]}" for key in ("3863", "3854", "4055")),
            "generated_utc": utc_now(),
        },
        {
            "dataset_id": "P7-OCCURRENCE",
            "output_path": rel(OUT / "occurrence_reconciliation_v1.csv"),
            "source_documents": "RDC_3731;RDC_3399;RDC_1919",
            "source_pages": "as recorded in crosswalk",
            "input_artifacts": ";".join(rel(OCC / name) for name in ["entities/occurrence_observations_v1.csv", "entities/site_entities_v1.csv", "entities/occurrence_entity_crosswalk_v1.csv"]),
            "processing_script": rel(script),
            "processing_script_sha256": script_hash,
            "source_file_sha256": ";".join(f"RDC_{key}:{pdf_hash[str(int(key))]}" for key in ("3731", "3399", "1919")),
            "generated_utc": utc_now(),
        },
        {
            "dataset_id": "P7-EXPLORATION",
            "output_path": rel(OUT / "exploration_evidence_register_v1.csv"),
            "source_documents": "RDC_0100;RDC_0379;RDC_1388;RDC_1408;RDC_3461;RDC.ROCKS corpus",
            "source_pages": "as recorded in register",
            "input_artifacts": rel(EXP / "exploration_history_v1.csv"),
            "processing_script": rel(script),
            "processing_script_sha256": script_hash,
            "source_file_sha256": ";".join(f"RDC_{key}:{pdf_hash[str(int(key))]}" for key in ("0100", "0379", "1388", "1408", "3461")),
            "generated_utc": utc_now(),
        },
        {
            "dataset_id": "P7-PREDICTORS",
            "output_path": rel(OUT / "predictor_eligibility_v1.csv"),
            "source_documents": "RDC_3863;RDC_3854;RDC_4055;RDC.ROCKS occurrence and exploration tables",
            "source_pages": "1;1;1;various",
            "input_artifacts": ";".join(rel(OUT / name) for name in ["geological_evidence_v1.csv", "occurrence_reconciliation_v1.csv", "exploration_evidence_register_v1.csv"]),
            "processing_script": rel(script),
            "processing_script_sha256": script_hash,
            "source_file_sha256": ";".join(f"RDC_{key}:{pdf_hash[key]}" for key in ("3863", "3854", "4055")),
            "generated_utc": utc_now(),
        },
    ]
    write_csv(OUT / "phase7_provenance_v1.csv", list(rows[0].keys()), rows)


def build_report(features: list[dict], evidence: dict, occurrence: dict,
                 exploration: dict, predictors: list[dict], observation: dict) -> None:
    by_doc = {}
    for feature in features:
        key = feature["source_doc"]
        by_doc[key] = by_doc.get(key, 0) + 1
    classes = {}
    for feature in features:
        key = feature["harmonised_class"]
        classes[key] = classes.get(key, 0) + 1
    predictor_classes = {}
    for row in predictors:
        key = row["category"]
        predictor_classes[key] = predictor_classes.get(key, 0) + 1
    lines = [
        "# Phase 7 integration report (v1)",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Status",
        "",
        "Phase 6 outputs were copied without modification into the canonical package, "
        "then integrated into one traceable geological evidence table and GeoPackage. "
        "No model was fitted and no source document, frozen foundation file or USGS record was changed.",
        "",
        "## Geological evidence",
        "",
        f"- Source features integrated: **{evidence['features']}**.",
        f"- RDC_3863: {by_doc.get('RDC_3863', 0)} features.",
        f"- RDC_3854: {by_doc.get('RDC_3854', 0)} features.",
        f"- RDC_4055: {by_doc.get('RDC_4055', 0)} features.",
        "- Coordinate spaces remain unresolved: map-grid metres for RDC_3863 and RDC_3854; "
        "source pixels with calibrated easting only for RDC_4055.",
        "",
        "| Harmonised class | Features |",
        "|---|---:|",
    ]
    for key in sorted(classes):
        lines.append(f"| {key} | {classes[key]} |")
    lines += [
        "",
        "## Occurrence reconciliation",
        "",
        f"- Source observations preserved: **147**.",
        f"- Candidate site entities preserved: **113**.",
        f"- Crosswalk rows preserved: **147**.",
        f"- Rows carrying a candidate USGS match: {occurrence['candidates']}.",
        "- Confirmed identity matches: **0**. No row is a model positive.",
    ]
    for key in sorted(occurrence["status_counts"]):
        lines.append(f"- `{key}`: {occurrence['status_counts'][key]} rows.")
    lines += [
        "",
        "## Exploration evidence",
        "",
        f"- Evidence records: **{exploration['rows']}**.",
        f"- Records with numeric start and end years: {exploration['dated']}.",
        f"- Records with a spatial footprint: **{exploration['footprints']}**.",
        "- Textual areas and two local-grid boreholes were retained as text; no polygon or point was invented.",
        "",
        "## Predictors",
        "",
        f"- Candidate predictors specified: **{len(predictors)}**.",
    ]
    for key in sorted(predictor_classes):
        lines.append(f"- `{key}`: {predictor_classes[key]}.")
    lines += [
        "",
        "## Observation process",
        "",
        f"- Bounded scenarios: 7; estimable now: {observation['estimable_scenarios']}.",
        "- The corpus does not support a continuous detection or reporting-intensity surface. "
        "Document-source, spatial-precision and entity-duplication sensitivities are the defensible controls.",
        "",
        "## Canonical outputs",
        "",
        "- `canonical_phase6/`",
        "- `phase6_artifact_manifest_v1.csv`",
        "- `rdc_geological_evidence_v1.gpkg`",
        "- `geological_evidence_v1.csv`",
        "- `geological_evidence_schema_v1.json` / `.md`",
        "- `occurrence_reconciliation_v1.csv`",
        "- `exploration_evidence_register_v1.csv`",
        "- `predictor_eligibility_v1.csv`",
        "- `observation_scenarios_v1.csv`",
        "- `observation_process_assessment_v1.md`",
        "- `usgs_baseline_hashes_v1.csv`",
        "- `frozen_baseline_hashes_v1.csv`",
        "- `phase7_provenance_v1.csv`",
        "",
        "## Model-variable decision",
        "",
        "- `supplementary/model_variable_decision_v1.md` / `.csv`",
        "- Primary geological term: distance to high/medium-confidence mapped Daly major structures.",
        "- All 186 USGS deposits inside the Daly union fall in NRB subdomains 3a/3b; no nine-domain effect is estimable.",
        "- No independent observation-effort surface; only source, precision and duplication sensitivities are estimable.",
        "",
        "## Validation boundary",
        "",
        "Run `validate_phase7.py` for hash, count, reference-integrity and geometry checks. "
        "That report is the authoritative validation record; this report is the build summary.",
        "",
    ]
    (OUT / "phase7_report_v1.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    canonical = build_canonical_artifacts()
    write_csv(OUT / "phase6_artifact_manifest_v1.csv", list(canonical[0].keys()), canonical)
    features = read_source_features()
    assert len(features) == 136, f"expected 136 Phase 6 features, got {len(features)}"
    assert len({row["evidence_id"] for row in features}) == len(features), "duplicate evidence_id"
    evidence = build_unified_evidence(features)
    build_schema()
    occurrence = build_occurrence_reconciliation()
    exploration = build_exploration_register()
    predictors = build_predictor_table()
    observation = build_observation_outputs()
    build_protected_baselines()
    build_provenance()
    build_report(features, evidence, occurrence, exploration, predictors, observation)
    print(json.dumps({
        "features": evidence["features"],
        "occurrence_rows": occurrence["rows"],
        "confirmed_matches": occurrence["confirmed"],
        "exploration_rows": exploration["rows"],
        "predictors": len(predictors),
        "outputs": rel(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
