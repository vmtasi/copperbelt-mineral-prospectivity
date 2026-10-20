"""End-to-end integrity checks for the corrected lithology-contact feature.

Run from the repository root and choose a new report file:

    python src/validate_litho_contact_pipeline.py --report figures/audit/recheck.json

The script exits non-zero on the first failed assertion and refuses to
overwrite an existing report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CANONICAL = ROOT / "data" / "copperbelt_training_v13_litho_contact.csv"
DISTANCE_CSV = ROOT / "data" / "derived" / "litho_contact_distance_v13.csv"
DISTANCE_GPKG = ROOT / "data" / "derived" / "litho_contact_distance_v13.gpkg"
MANIFEST_PATH = ROOT / "data" / "derived" / "canonical_training_manifest.json"
AUDIT_DIR = ROOT / "figures" / "audit"

MODEL_COLUMNS = [
    "centroid_x",
    "centroid_y",
    "litho_contact_litho_class",
    "domain",
    "distance_to_lithology_contact",
    "distance_to_fault",
    "bouguer",
]
LEGACY_PATTERNS = [
    "copperbelt_training_v5",
    "copperbelt_training_v6",
    "litho_contact_features_v12",
    "grid_features_litho_contact_v02",
    "copperbelt_dataset_clean",
    "distance_to_tract_boundary",
    "Mining_count",
    "v11_oof_posterior_uncertainty.gpkg",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_gpkg_distance(path: Path) -> pd.DataFrame:
    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "select table_name from gpkg_contents"
            )
        }
        check("litho_contact_distance" in tables, "GPKG layer missing")
        columns = {
            row[1]
            for row in connection.execute(
                "pragma table_info(litho_contact_distance)"
            )
        }
        check({"id", "distance_m"}.issubset(columns), "GPKG fields missing")
        frame = pd.read_sql_query(
            "select id, distance_m from litho_contact_distance order by id",
            connection,
        )
        srs = connection.execute(
            "select srs_id from gpkg_contents where table_name = 'litho_contact_distance'"
        ).fetchone()
    check(srs is not None and int(srs[0]) == 32735, f"GPKG CRS is {srs}, expected 32735")
    return frame


def scan_active_references() -> list[str]:
    offenders: list[str] = []
    roots = [ROOT / "src", ROOT / "tests", ROOT / "paper_1", ROOT / "README.md", ROOT / "AI_PROJECT_CONTEXT.md"]
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        iterator = root.rglob("*") if root.is_dir() else [root]
        for path in iterator:
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            relative = path.relative_to(ROOT).as_posix()
            if relative.startswith(("archive/", ".tmp/", ".venv", ".runtime_")):
                continue
            if path.suffix.lower() not in {".py", ".md", ".tex", ".json", ".yml", ".yaml", ".toml"}:
                continue
            if path.name in {
                "build_corrected_training_table.py",
                "validate_litho_contact_pipeline.py",
                "test_litho_contact_pipeline.py",
            }:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in LEGACY_PATTERNS:
                if pattern in text:
                    offenders.append(f"{relative}:{pattern}")
    return offenders


def main(report_path: Path) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    check(manifest.get("manifest_schema_version") == 1, "unsupported manifest schema")
    check(manifest.get("dataset_version") == "V13", "unexpected dataset version")

    canonical_manifest_path = (ROOT / manifest["canonical_file"]).resolve()
    distance_manifest_path = (ROOT / manifest["distance_file"]).resolve()
    base_manifest_path = (ROOT / manifest["base_file"]).resolve()
    for path in (canonical_manifest_path, distance_manifest_path, base_manifest_path):
        check(ROOT.resolve() in path.parents, f"manifest path escapes project: {path}")
    check(canonical_manifest_path == CANONICAL.resolve(), "manifest canonical path differs")
    check(distance_manifest_path == DISTANCE_CSV.resolve(), "manifest distance path differs")
    check(sha256(CANONICAL) == manifest["canonical_sha256"], "canonical hash differs from manifest")
    check(sha256(DISTANCE_CSV) == manifest["distance_sha256"], "distance hash differs from manifest")
    check(sha256(base_manifest_path) == manifest["base_sha256"], "base hash differs from manifest")

    canonical = pd.read_csv(CANONICAL)
    distance = pd.read_csv(DISTANCE_CSV)
    gpkg = read_gpkg_distance(DISTANCE_GPKG)

    check(len(canonical) == manifest["rows"], "canonical row count differs from manifest")
    check(canonical["id"].nunique() == manifest["unique_ids"], "canonical unique-id count differs from manifest")
    check(not canonical["id"].duplicated().any(), "duplicate canonical ids")
    check(len(distance) == manifest["rows"], "distance row count differs from manifest")
    check(distance["id"].nunique() == manifest["unique_ids"], "distance unique-id count differs from manifest")
    check(not distance["id"].duplicated().any(), "duplicate distance ids")
    check(set(canonical["id"]) == set(distance["id"]), "canonical/distance id sets differ")
    check(set(distance["id"]) == set(gpkg["id"]), "CSV/GPKG id sets differ")
    check(len(gpkg) == manifest["rows"] and gpkg["id"].is_unique, "GPKG id integrity failed")

    values = canonical["distance_to_lithology_contact"].to_numpy(dtype=float)
    check(np.isfinite(values).all(), "non-finite canonical distances")
    check((values >= 0).all(), "negative canonical distances")
    check(np.isfinite(distance["distance_to_lithology_contact"].astype(float)).all(),
          "non-finite V13 distances")

    merged = canonical[["id", "distance_to_lithology_contact"]].merge(
        distance[["id", "distance_to_lithology_contact"]],
        on="id",
        suffixes=("_canonical", "_v13"),
        validate="one_to_one",
    )
    max_csv_diff = float(
        np.max(np.abs(
            merged["distance_to_lithology_contact_canonical"]
            - merged["distance_to_lithology_contact_v13"]
        ))
    )
    check(max_csv_diff <= 1e-9, f"canonical/V13 max diff={max_csv_diff}")

    gpkg_merged = merged.merge(gpkg, on="id", validate="one_to_one")
    max_gpkg_diff = float(
        np.max(np.abs(
            gpkg_merged["distance_to_lithology_contact_canonical"]
            - gpkg_merged["distance_m"]
        ))
    )
    check(max_gpkg_diff <= 1e-9, f"CSV/GPKG max diff={max_gpkg_diff}")

    base = pd.read_csv(base_manifest_path, encoding="utf-8-sig")
    check(list(base.columns) == list(canonical.columns), "canonical schema changed")
    check(base["id"].equals(canonical["id"]), "canonical row order/ids changed")
    shared = [column for column in base.columns if column != "distance_to_lithology_contact"]
    check(base[shared].equals(canonical[shared]),
          "columns other than distance differ from the frozen base")
    check(base["litho_contact_litho_class"].equals(canonical["litho_contact_litho_class"]),
          "secondary class predictor was altered")

    frame = canonical.dropna(subset=MODEL_COLUMNS).copy()
    frame = frame[
        frame["domain"].map(lambda value: any(
            key in str(value).lower()
            for key in ("3a", "3b", "crz", "srb", "nkb", "mmsb")
        ))
    ].copy()
    positives = int(frame["deposit_present"].sum())
    backgrounds = int((frame["deposit_present"] == 0).sum())
    check(positives + backgrounds == len(frame), "label totals differ from complete-case rows")
    check(frame["deposit_present"].isin([0, 1]).all(), "labels are not binary")

    from src.validation_strategies import get_along_belt_folds

    blocks = get_along_belt_folds(frame, n_folds=4)
    check(len(blocks) == len(frame), "fold assignment length differs from complete-case rows")
    check(set(np.unique(blocks)) == set(range(4)), "fold assignment does not contain all expected folds")
    check(np.array_equal(blocks, get_along_belt_folds(frame, n_folds=4)),
          "fold assignment is not deterministic")
    frame["spatial_block"] = blocks
    fold_counts = frame.groupby("spatial_block")["deposit_present"].agg(["size", "sum"])
    check(int(fold_counts["size"].sum()) == len(frame), "fold sizes do not cover all rows")
    check(int(fold_counts["sum"].sum()) == positives, "fold positive counts do not reconcile")

    domain_counts = frame["domain"].value_counts().sort_index()
    base_frame = base.dropna(subset=MODEL_COLUMNS).copy()
    base_frame = base_frame[
        base_frame["domain"].map(lambda value: any(
            key in str(value).lower()
            for key in ("3a", "3b", "crz", "srb", "nkb", "mmsb")
        ))
    ].copy()
    check(base_frame["domain"].value_counts().sort_index().equals(domain_counts),
          "domain assignments changed")

    offenders = scan_active_references()
    check(not offenders, f"legacy references remain: {offenders}")

    report = {
        "status": "pass",
        "validation_schema_version": 1,
        "manifest_schema_version": manifest["manifest_schema_version"],
        "dataset_version": manifest["dataset_version"],
        "manifest_sha256": sha256(MANIFEST_PATH),
        "canonical_file": CANONICAL.relative_to(ROOT).as_posix(),
        "canonical_sha256": sha256(CANONICAL),
        "distance_csv_sha256": sha256(DISTANCE_CSV),
        "distance_gpkg_sha256": sha256(DISTANCE_GPKG),
        "rows": int(len(canonical)),
        "unique_ids": int(canonical["id"].nunique()),
        "distance_min_m": float(values.min()),
        "distance_max_m": float(values.max()),
        "zero_distance_cells": int((values == 0).sum()),
        "max_canonical_vs_csv_diff_m": max_csv_diff,
        "max_csv_vs_gpkg_diff_m": max_gpkg_diff,
        "crs_epsg": 32735,
        "complete_case_rows": int(len(frame)),
        "complete_case_positives": positives,
        "complete_case_background": int((frame["deposit_present"] == 0).sum()),
        "fold_sizes": fold_counts["size"].tolist(),
        "fold_positives": [int(value) for value in fold_counts["sum"].tolist()],
        "legacy_references": offenders,
        "checks": {
            "one_row_per_grid_cell": True,
            "no_missing_or_duplicate_ids": True,
            "csv_gpkg_agree": True,
            "crs_and_grid_preserved": True,
            "other_columns_unchanged": True,
            "class_predictor_unchanged": True,
            "frame_labels_domains_folds_unchanged": True,
            "no_active_legacy_references": True,
        },
    }
    report_path = report_path.resolve()
    check(report_path.parent == AUDIT_DIR.resolve(), "report must be directly under figures/audit")
    check(report_path.suffix.lower() == ".json", "report must use a .json extension")
    check(not report_path.exists(), f"refusing to overwrite existing report: {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="new JSON report path directly under figures/audit",
    )
    main(parser.parse_args().report)
