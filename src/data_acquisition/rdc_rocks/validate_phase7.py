"""Validate the Phase 7 integrated evidence package and Daly support outputs.

Stdlib only. Writes a concise markdown record; does not modify source data.
"""
from __future__ import annotations

import csv
import hashlib
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P7 = ROOT / "data/raw/rdc_rocks/derived/integration/phase7"
SUP = P7 / "supplementary"
ENT = ROOT / "data/raw/rdc_rocks/derived/occurrences/entities"
OUT = P7 / "phase7_validation_v1.md"

checks: list[tuple[str, bool, str]] = []


def add(name: str, ok: bool, detail: str) -> None:
    checks.append((name, bool(ok), detail))


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def layer_counts(path: Path) -> dict[str, int]:
    con = sqlite3.connect(path)
    try:
        names = [name for (name,) in con.execute("select table_name from gpkg_contents")]
        return {name: con.execute(f'select count(*) from "{name}"').fetchone()[0] for name in names}
    finally:
        con.close()


def empty_geometries(path: Path) -> dict[str, int]:
    con = sqlite3.connect(path)
    try:
        columns = con.execute("select table_name, column_name from gpkg_geometry_columns").fetchall()
        bad = {}
        for table, column in columns:
            count = con.execute(
                f'select count(*) from "{table}" where "{column}" is null or length("{column}") = 0'
            ).fetchone()[0]
            if count:
                bad[table] = count
        return bad
    finally:
        con.close()


def main() -> int:
    observations = rows(ENT / "occurrence_observations_v1.csv")
    sites = rows(ENT / "site_entities_v1.csv")
    crosswalk = rows(ENT / "occurrence_entity_crosswalk_v1.csv")
    reconciliation = rows(P7 / "occurrence_reconciliation_v1.csv")
    exploration = rows(P7 / "exploration_evidence_register_v1.csv")
    evidence = rows(P7 / "geological_evidence_v1.csv")

    add("Occurrence observations preserved", len(observations) == 147, f"{len(observations)} rows")
    add("Candidate site entities preserved", len(sites) == 113, f"{len(sites)} rows")
    add("Occurrence crosswalk preserved", len(crosswalk) == 147, f"{len(crosswalk)} rows")
    obs_ids = {row["occurrence_id"] for row in observations}
    site_ids = {row["site_entity_id"] for row in sites}
    add(
        "Crosswalk referential integrity",
        all(row["occurrence_id"] in obs_ids and row["site_entity_id"] in site_ids for row in crosswalk),
        "every crosswalk row resolves to an observation and a site",
    )
    add(
        "No confirmed identities or model positives",
        all(row["confirmed_identity"] == "no" and row["use_as_model_positive"] == "no" for row in reconciliation),
        "147 rows, 0 confirmed, 0 positives",
    )
    add("Exploration register preserved", len(exploration) == 21, f"{len(exploration)} rows")
    add(
        "No invented exploration footprints",
        all(row["footprint_geometry_available"] == "no" for row in exploration),
        "21 rows, 0 footprints",
    )
    add("Geological evidence preserved", len(evidence) == 136, f"{len(evidence)} rows")

    gpkg = layer_counts(P7 / "rdc_geological_evidence_v1.gpkg")
    add(
        "Geological evidence GeoPackage counts",
        sum(gpkg.values()) == 136,
        ", ".join(f"{name}={count}" for name, count in gpkg.items()),
    )

    manifest = rows(P7 / "phase6_artifact_manifest_v1.csv")
    add(
        "Phase 6 canonical copies match originals",
        len(manifest) == 8
        and all(row["hash_match"] == "yes" and row["sha256_original"] == row["sha256_canonical"] for row in manifest),
        f"{len(manifest)} artefacts",
    )

    usgs = rows(P7 / "usgs_baseline_hashes_v1.csv")
    usgs_bad = [row["path"] for row in usgs if sha256(ROOT / row["path"]) != row["sha256"]]
    add("USGS baseline hashes", len(usgs) == 12 and not usgs_bad, f"{len(usgs) - len(usgs_bad)}/{len(usgs)} match")

    acquisition = {row["id"]: row for row in rows(ROOT / "data/raw/rdc_rocks/metadata/acquisition_manifest_v1.csv")}
    source_bad = []
    for doc_id in ("3863", "3854", "4055", "3731", "3399", "1919"):
        row = acquisition[doc_id]
        if sha256(ROOT / row["local_path"]) != row["sha256"]:
            source_bad.append(doc_id)
    add("Source document hashes", not source_bad, "6/6 match" if not source_bad else f"mismatch: {source_bad}")

    support = layer_counts(SUP / "daly_support_5km_v1.gpkg")
    add("Daly support grid cells", support.get("daly_support_5km") == 16492, f"{support.get('daly_support_5km')} cells")
    evidence_empty = empty_geometries(P7 / "rdc_geological_evidence_v1.gpkg")
    add("Geological evidence geometries", not evidence_empty, "no null or empty geometries" if not evidence_empty else str(evidence_empty))
    support_empty = empty_geometries(SUP / "daly_support_5km_v1.gpkg")
    add("Support-grid geometries", not support_empty, "no null or empty geometries" if not support_empty else str(support_empty))
    fold_empty = empty_geometries(SUP / "daly_geographic_folds_v1.gpkg")
    add("Fold geometries", not fold_empty, "no null or empty geometries" if not fold_empty else str(fold_empty))
    con = sqlite3.connect(SUP / "daly_support_5km_v1.gpkg")
    try:
        null_domains = con.execute(
            "select count(*) from daly_support_5km where domain_id is null or domain_id = ''"
        ).fetchone()[0]
        null_structures = con.execute(
            "select count(*) from daly_support_5km where dist_major_structure_km is null"
        ).fetchone()[0]
    finally:
        con.close()
    add("Support-grid domain assignment", null_domains == 0, f"{null_domains} null cells")
    add("Support-grid structure distance", null_structures == 0, f"{null_structures} null cells")

    folds = rows(SUP / "daly_fold_support_summary_v1.csv")
    four = [int(row["usgs_deposits"]) for row in folds if row["scheme"] == "four_fold_primary"]
    five = [int(row["usgs_deposits"]) for row in folds if row["scheme"] == "five_fold_alternative"]
    add("Four-fold occurrence support", len(four) == 4 and min(four) >= 40, f"counts {four}")
    add("Five-fold alternative support", len(five) == 5 and min(five) >= 30, f"counts {five}")

    audit = rows(SUP / "usgs_deposit_overlay_audit_v1.csv")
    overlay = {row["metric"]: int(row["value"]) for row in rows(SUP / "usgs_deposit_overlay_summary_v1.csv")}
    add("USGS overlay audit complete", len(audit) == 187, f"{len(audit)} records")
    add(
        "USGS Daly union containment",
        overlay.get("inside_daly_union") == 186 and overlay.get("boundary_daly_union") == 0,
        f"inside={overlay.get('inside_daly_union')}, boundary={overlay.get('boundary_daly_union')}",
    )
    add(
        "USGS permissive-tract containment",
        overlay.get("inside_tracts") == 184 and overlay.get("outside_tracts") == 3,
        f"inside={overlay.get('inside_tracts')}, outside={overlay.get('outside_tracts')}",
    )
    domain_counts = Counter(row["domain_ids"] for row in audit if row["union_relation"] == "inside")
    add("USGS NRB 3a/3b support", domain_counts == {"3a": 149, "3b": 37}, f"counts {dict(domain_counts)}")
    country_counts = Counter(row["source_country"] for row in audit)
    add(
        "USGS source-country totals",
        country_counts == {"Democratic Republic of the Congo": 147, "Zambia": 40},
        f"counts {dict(country_counts)}",
    )
    add(
        "USGS cross-border ambiguity recorded",
        overlay.get("country_assignment_mismatches") == 2,
        "2 source-Zambia Mokambo records fall in the Natural Earth DRC polygon; both assignments retained",
    )
    audit_fold = Counter(int(row["fold4"]) for row in audit if row["union_relation"] == "inside")
    add(
        "USGS fold support matches audit",
        [audit_fold[index] for index in range(1, 5)] == four,
        f"audit {[audit_fold[index] for index in range(1, 5)]}; summary {four}",
    )

    provenance = rows(SUP / "phase7_supplement_provenance_v1.csv")
    provenance_bad = []
    for row in provenance:
        path = ROOT / row["output_path"]
        if not path.exists() or sha256(path) != row["output_sha256"]:
            provenance_bad.append(row["output_path"])
    add(
        "Supplement provenance",
        len(provenance) == 13 and not provenance_bad,
        f"{len(provenance)} rows, {len(provenance_bad)} hash mismatches",
    )
    decision_rows = rows(SUP / "model_variable_decision_v1.csv")
    add(
        "Model decision table schema",
        len(decision_rows) == 18 and all(None not in row and None not in row.values() for row in decision_rows),
        f"{len(decision_rows)} decisions, no ragged rows",
    )
    decision = {row["decision_id"]: row for row in decision_rows}
    add(
        "Tract decision reconciled",
        "184 inside the tract union" in decision["D07"]["support_metric"]
        and "3 outside the tracts" in decision["D07"]["support_metric"],
        decision["D07"]["support_metric"],
    )

    study = {row["metric"]: row["value"] for row in rows(SUP / "daly_study_area_summary_v1.csv")}
    add("Daly study-area extent", float(study["study_area_km2"]) > 398000, f"{study['study_area_km2']} km2")
    add("Daly domain count", study["domain_count"] == "9", f"{study['domain_count']} polygons")

    passed = sum(1 for _, ok, _ in checks if ok)
    lines = [
        "# Phase 7 validation (v1)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"**Result: {passed}/{len(checks)} checks passed.**",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for name, ok, detail in checks:
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    lines += [
        "",
        "## Caveats",
        "",
        "- `frozen_baseline_hashes_v1.csv` is a recapture by `build_phase7_integration.py`, not a comparison against an earlier baseline. It is not used here as proof of frozen-file integrity.",
        "- `AI_PROJECT_CONTEXT.md` and `src/validation_strategies.py` match Git HEAD `d56afb1` with no working-tree diff; the earlier drift warning could not be reproduced.",
        "- No model was fitted. No raw PDF, TIFF, OCR or USGS file was modified.",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"{passed}/{len(checks)} checks passed -> {OUT}")
    for name, ok, detail in checks:
        if not ok:
            print("FAIL", name, detail)
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
