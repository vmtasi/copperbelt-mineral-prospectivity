"""Record provenance for the Daly support and USGS overlay supplement.

Stdlib only. Hashes existing outputs; does not modify source or derived data.
"""
from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SUP = ROOT / "data/raw/rdc_rocks/derived/integration/phase7/supplementary"
OUT = SUP / "phase7_supplement_provenance_v1.csv"
BUILDER = "src/data_acquisition/rdc_rocks/build_phase7_supplement.py"
AUDITOR = "src/data_acquisition/rdc_rocks/audit_usgs_overlay.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_map(paths: list[Path]) -> str:
    return ";".join(f"{path.name}:{sha256(path)}" for path in paths)


def main() -> int:
    now = datetime.now(timezone.utc).isoformat()
    builder_hash = sha256(ROOT / BUILDER)
    auditor_hash = sha256(ROOT / AUDITOR)
    inputs = [
        ROOT / "data/processed/geology_foundation/daly_tectonic_domains_v1.gpkg",
        ROOT / "data/processed/geology_foundation/daly_major_structures_named_v2.gpkg",
        ROOT / "data/raw/usgs_sir2010_5090_t/CACB_tracts_utm.shp",
        ROOT / "data/raw/usgs_sir2010_5090_t/CACB_deposits_utm.shp",
        SUP / "daly_country_context_v1.gpkg",
    ]
    input_hashes = hash_map(inputs)
    rows = []

    def add(dataset_id: str, output: str, script: str, script_hash: str, input_hash: str = input_hashes) -> None:
        path = SUP / output
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append({
            "dataset_id": dataset_id,
            "output_path": path.relative_to(ROOT).as_posix(),
            "source_documents": "Daly et al. tectonic domains; USGS SIR 2010-5090-T",
            "source_pages": "Daly Figure 14; USGS inventory and tracts",
            "input_artifacts": ";".join(path.relative_to(ROOT).as_posix() for path in inputs),
            "processing_script": script,
            "processing_script_sha256": script_hash,
            "source_file_sha256": input_hash,
            "output_sha256": sha256(path),
            "generated_utc": now,
        })

    for output in ("daly_study_area_v1.gpkg", "daly_support_5km_v1.gpkg", "daly_geographic_folds_v1.gpkg"):
        add("P7S-DALY-GEOMETRY", output, BUILDER, builder_hash)
    for output in (
        "daly_study_area_summary_v1.csv",
        "daly_country_summary_v1.csv",
        "daly_domain_area_summary_v1.csv",
        "daly_fold_support_summary_v1.csv",
        "daly_support_5km_summary_v1.csv",
        "spatial_support_assessment_v1.md",
    ):
        add("P7S-DALY-TABLE", output, BUILDER, builder_hash)
    for output in ("usgs_deposit_overlay_audit_v1.csv", "usgs_deposit_overlay_summary_v1.csv"):
        add("P7S-USGS-OVERLAY", output, AUDITOR, auditor_hash)

    rows.append({
        "dataset_id": "P7S-MODEL-DECISION",
        "output_path": (SUP / "model_variable_decision_v1.csv").relative_to(ROOT).as_posix(),
        "source_documents": "Phase 7 synthesised evidence",
        "source_pages": "not applicable",
        "input_artifacts": ";".join(path.relative_to(ROOT).as_posix() for path in (
            SUP / "daly_support_5km_v1.gpkg",
            SUP / "daly_fold_support_summary_v1.csv",
            SUP / "usgs_deposit_overlay_summary_v1.csv",
            ROOT / "data/raw/rdc_rocks/derived/integration/phase7/predictor_eligibility_v1.csv",
            ROOT / "data/raw/rdc_rocks/derived/integration/phase7/exploration_evidence_register_v1.csv",
        )),
        "processing_script": "manual_reviewed_synthesis",
        "processing_script_sha256": "",
        "source_file_sha256": "",
        "output_sha256": sha256(SUP / "model_variable_decision_v1.csv"),
        "generated_utc": now,
    })
    rows.append({
        "dataset_id": "P7S-MODEL-DECISION",
        "output_path": (SUP / "model_variable_decision_v1.md").relative_to(ROOT).as_posix(),
        "source_documents": "Phase 7 synthesised evidence",
        "source_pages": "not applicable",
        "input_artifacts": (SUP / "model_variable_decision_v1.csv").relative_to(ROOT).as_posix(),
        "processing_script": "manual_reviewed_synthesis",
        "processing_script_sha256": "",
        "source_file_sha256": "",
        "output_sha256": sha256(SUP / "model_variable_decision_v1.md"),
        "generated_utc": now,
    })

    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} provenance rows -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
