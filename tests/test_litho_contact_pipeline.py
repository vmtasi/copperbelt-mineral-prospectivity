"""Regression tests for the corrected lithology-contact feature and pipeline.

Run with ``python -m pytest tests/test_litho_contact_pipeline.py``. The tests
avoid pytest-specific fixtures so they also work in the project's QGIS runtime.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "data" / "copperbelt_training_v13_litho_contact.csv"
DISTANCE_CSV = ROOT / "data" / "derived" / "litho_contact_distance_v13.csv"
DISTANCE_GPKG = ROOT / "data" / "derived" / "litho_contact_distance_v13.gpkg"
MANIFEST = ROOT / "data" / "derived" / "canonical_training_manifest.json"
ARCHIVE_ROOT = ROOT / "archive" / "litho_contact_repair_2026-10-18"
ARCHIVE_HASHES = ARCHIVE_ROOT / "hashes.sha256"
pytestmark = pytest.mark.skipif(
    not all(
        path.is_file()
        for path in (
            CANONICAL,
            DISTANCE_CSV,
            DISTANCE_GPKG,
            ARCHIVE_HASHES,
        )
    ),
    reason="local research data and historical archive are not present",
)
# These two provenance scripts are preserved from Git/stash sources whose
# normalized content differs from the older digests in the archive manifest.
ARCHIVE_HASH_EXCEPTIONS = {
    "repro/compare_repro.py": (
        "6f6e31ac07734e40486d56e2b09df395c3d054145966ed5f0df954f97c2fcc95",
        "2ae394fe073d8a7f925aae0dab33ae66f0505ade288f50011ae370a2fe2a84ff",
    ),
    "repro/verify_repro_v13.py": (
        "ed931d5cfd8f49a55c31af48278caba89ac9106a239ea69cff1d5a239f4a2f82",
        "1147c50ea60929aca4387b5638be0c3b29cc9238160c24b19f73e4ef266df69b",
    ),
}


def _manifest() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["manifest_schema_version"] == 1
    assert manifest["dataset_version"] == "V13"
    return manifest


def _complete_case_frame(canonical: pd.DataFrame) -> pd.DataFrame:
    frame = canonical.dropna(
        subset=[
            "centroid_x",
            "centroid_y",
            "litho_contact_litho_class",
            "domain",
            "distance_to_lithology_contact",
            "distance_to_fault",
            "bouguer",
        ]
    ).copy()
    return frame[
        frame["domain"].map(
            lambda value: any(
                key in str(value).lower()
                for key in ("3a", "3b", "crz", "srb", "nkb", "mmsb")
            )
        )
    ].copy()


def _gpkg_distance() -> pd.DataFrame:
    with sqlite3.connect(DISTANCE_GPKG) as connection:
        frame = pd.read_sql_query(
            "select id, distance_m from litho_contact_distance order by id",
            connection,
        )
        srs_id = connection.execute(
            "select srs_id from gpkg_contents where table_name='litho_contact_distance'"
        ).fetchone()[0]
    assert int(srs_id) == 32735
    return frame


def test_canonical_table_has_one_row_per_grid_id():
    manifest = _manifest()
    canonical = pd.read_csv(CANONICAL)
    assert len(canonical) == manifest["rows"]
    assert canonical["id"].nunique() == manifest["unique_ids"]
    assert canonical["id"].is_unique
    assert not canonical["id"].isna().any()
    assert hashlib.sha256(CANONICAL.read_bytes()).hexdigest() == manifest["canonical_sha256"]


def test_historical_archive_matches_manifest_or_verified_source_hash():
    assert ARCHIVE_HASHES.is_file()
    entries = ARCHIVE_HASHES.read_text(encoding="utf-8").splitlines()
    assert entries
    known_mismatches = {}
    for entry in entries:
        expected_hash, relative_path = entry.split(maxsplit=1)
        relative_path = relative_path.strip().replace("\\", "/")
        archived_file = (ARCHIVE_ROOT / relative_path).resolve()
        assert ARCHIVE_ROOT.resolve() in archived_file.parents
        assert archived_file.is_file(), relative_path
        content = archived_file.read_bytes()
        actual_hash = hashlib.sha256(content).hexdigest()
        if actual_hash.lower() == expected_hash.lower():
            continue
        normalized_content = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        normalized_hash = hashlib.sha256(normalized_content).hexdigest()
        known_mismatches[relative_path] = (
            expected_hash.lower(),
            normalized_hash.lower(),
        )
    assert known_mismatches == ARCHIVE_HASH_EXCEPTIONS


def test_v13_has_no_arbitrary_class_column():
    distance = pd.read_csv(DISTANCE_CSV)
    assert "litho_contact_litho_class" not in distance.columns
    assert "representative_contact_fid" in distance.columns
    assert "tied_contact_classes" in distance.columns


def test_canonical_distance_equals_v13_and_gpkg():
    manifest = _manifest()
    canonical = pd.read_csv(CANONICAL)
    distance = pd.read_csv(DISTANCE_CSV)
    gpkg = _gpkg_distance()
    assert len(distance) == manifest["rows"]
    assert hashlib.sha256(DISTANCE_CSV.read_bytes()).hexdigest() == manifest["distance_sha256"]
    merged = canonical[["id", "distance_to_lithology_contact"]].merge(
        distance[["id", "distance_to_lithology_contact"]],
        on="id",
        suffixes=("_canonical", "_v13"),
        validate="one_to_one",
    )
    assert np.allclose(
        merged["distance_to_lithology_contact_canonical"],
        merged["distance_to_lithology_contact_v13"],
        atol=1e-9,
        rtol=0,
    )
    merged = merged.merge(gpkg, on="id", validate="one_to_one")
    assert np.allclose(
        merged["distance_to_lithology_contact_canonical"],
        merged["distance_m"],
        atol=1e-9,
        rtol=0,
    )


def test_canonical_distance_changed_from_retired_table_but_class_did_not():
    manifest = _manifest()
    archive_base = (ROOT / manifest["base_file"]).resolve()
    assert ROOT.resolve() in archive_base.parents
    assert hashlib.sha256(archive_base.read_bytes()).hexdigest() == manifest["base_sha256"]
    canonical = pd.read_csv(CANONICAL)
    base = pd.read_csv(archive_base, encoding="utf-8-sig")
    assert base["id"].equals(canonical["id"])
    changed = (
        np.abs(
            canonical["distance_to_lithology_contact"].to_numpy()
            - base["distance_to_lithology_contact"].to_numpy()
        )
        > 1e-6
    )
    assert changed.any()
    assert base["litho_contact_litho_class"].equals(
        canonical["litho_contact_litho_class"]
    )
    other = [
        column
        for column in base.columns
        if column != "distance_to_lithology_contact"
    ]
    assert base[other].equals(canonical[other])


def test_complete_case_frame_folds_unchanged():
    from src.validation_strategies import get_along_belt_folds

    canonical = pd.read_csv(CANONICAL)
    frame = _complete_case_frame(canonical)
    blocks = get_along_belt_folds(frame, n_folds=4)
    assert len(blocks) == len(frame)
    assert np.array_equal(blocks, get_along_belt_folds(frame, n_folds=4))
    assert set(np.unique(blocks)) == set(range(4))
    frame["spatial_block"] = blocks
    counts = frame.groupby("spatial_block")["deposit_present"].agg(["size", "sum"])
    assert int(counts["size"].sum()) == len(frame)
    assert int(counts["sum"].sum()) == int(frame["deposit_present"].sum())
    assert (counts["size"] > 0).all()
    assert (counts["sum"] > 0).all()
    assert (counts["sum"] < counts["size"]).all()


def test_retired_feature_files_are_not_active():
    active = list((ROOT / "data").rglob("*"))
    names = {path.name for path in active if path.is_file()}
    assert not any("v12" in name for name in names)
    assert not any("provisional" in name for name in names)
    assert "copperbelt_training_v13_litho_contact.csv" in names


def test_no_active_script_reads_retired_contact_tables():
    patterns = (
        "copperbelt_training_v5",
        "copperbelt_training_v6",
        "litho_contact_features_v12",
        "grid_features_litho_contact_v02",
        "copperbelt_dataset_clean",
        "distance_to_tract_boundary",
        "Mining_count",
        "v11_oof_posterior_uncertainty.gpkg",
    )
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        if path.name in {
            "build_corrected_training_table.py",
            "validate_litho_contact_pipeline.py",
        }:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        offenders.extend(
            f"{path.relative_to(ROOT)}:{pattern}"
            for pattern in patterns
            if pattern in text
        )
    assert not offenders, offenders



def test_model_loader_consumes_canonical_distance():
    from src.distance_sensitivity_models import DATA_PATH, load_modeling_data

    assert DATA_PATH.resolve() == CANONICAL.resolve()
    model_frame, _ = load_modeling_data()
    canonical = pd.read_csv(CANONICAL)
    expected = _complete_case_frame(canonical)
    merged = model_frame[["id", "distance_to_lithology_contact"]].merge(
        canonical[["id", "distance_to_lithology_contact"]],
        on="id",
        suffixes=("_model", "_canonical"),
        validate="one_to_one",
    )
    assert len(merged) == len(expected)
    assert model_frame["id"].tolist() == expected["id"].tolist()
    assert np.array_equal(
        merged["distance_to_lithology_contact_model"].to_numpy(),
        merged["distance_to_lithology_contact_canonical"].to_numpy(),
    )
    assert int(model_frame["deposit_present"].sum()) == int(
        expected["deposit_present"].sum()
    )
    counts = model_frame.groupby("spatial_block")["deposit_present"].agg(["size", "sum"])
    assert int(counts["size"].sum()) == len(model_frame)
    assert int(counts["sum"].sum()) == int(model_frame["deposit_present"].sum())
    assert (counts["size"] > 0).all()


def test_no_active_pipeline_reads_historical_archive():
    exempt = {
        "build_corrected_training_table.py",
        "validate_litho_contact_pipeline.py",
    }
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        if path.name in exempt:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "litho_contact_repair_2026-10-18" in text:
            offenders.append(path.relative_to(ROOT).as_posix())
    assert not offenders, offenders


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")
