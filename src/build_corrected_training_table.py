"""Build the canonical corrected modelling table.

The frozen V5 table supplies labels, domains, centroids, and all predictors
except distance to lithological contact. The validated V13 table supplies the
corrected minimum internal-contact distance. Every other V5 column, including
the secondary lithology-class field, is preserved byte-for-byte at the text
level. This builder is a provenance step; ordinary pipeline execution reads the
canonical output only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = (
    DEFAULT_ROOT
    / "archive"
    / "litho_contact_repair_2026-10-18"
    / "data"
    / "copperbelt_training_v5_with_tectonic_domain.csv"
)
DEFAULT_DISTANCE = DEFAULT_ROOT / "data" / "derived" / "litho_contact_distance_v13.csv"
DEFAULT_OUTPUT = DEFAULT_ROOT / "data" / "copperbelt_training_v13_litho_contact.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(base: Path, distance: Path, output: Path) -> dict:
    with distance.open(newline="", encoding="utf-8") as handle:
        distance_rows = list(csv.DictReader(handle))
    distance_by_id = {
        row["id"]: row["distance_to_lithology_contact"] for row in distance_rows
    }
    if len(distance_by_id) != len(distance_rows):
        raise ValueError("Duplicate ids in corrected distance table")

    with base.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if "distance_to_lithology_contact" not in fields:
        raise ValueError("Base table lacks distance_to_lithology_contact")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate ids in base table")
    missing = {row["id"] for row in rows} - set(distance_by_id)
    extra = set(distance_by_id) - {row["id"] for row in rows}
    if missing or extra:
        raise ValueError(f"ID mismatch: missing={sorted(missing)} extra={sorted(extra)}")

    for row in rows:
        row["distance_to_lithology_contact"] = distance_by_id[row["id"]]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "manifest_schema_version": 1,
        "dataset_version": "V13",
        "canonical_file": output.relative_to(DEFAULT_ROOT).as_posix(),
        "rows": len(rows),
        "unique_ids": len({row["id"] for row in rows}),
        "base_file": base.relative_to(DEFAULT_ROOT).as_posix(),
        "base_sha256": sha256(base),
        "distance_file": distance.relative_to(DEFAULT_ROOT).as_posix(),
        "distance_sha256": sha256(distance),
        "canonical_sha256": sha256(output),
        "columns": fields,
        "distance_definition": (
            "minimum Euclidean distance from each grid-cell polygon to the "
            "nearest mapped internal lithological contact; outer study-area "
            "perimeter excluded; EPSG:32735; metres"
        ),
    }
    manifest_path = output.parent / "derived" / "canonical_training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--distance", type=Path, default=DEFAULT_DISTANCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.base, args.distance, args.output), indent=2))


if __name__ == "__main__":
    main()
