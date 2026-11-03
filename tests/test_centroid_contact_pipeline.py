"""Regression tests for the V14 centroid-to-internal-contact feature."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

ROOT = Path(__file__).resolve().parents[1]
V13_TRAINING = ROOT / "data" / "copperbelt_training_v13_litho_contact.csv"
V13_DISTANCE = ROOT / "data" / "derived" / "litho_contact_distance_v13.csv"
V14_TRAINING = ROOT / "data" / "copperbelt_training_v14_centroid_contact.csv"
V14_DISTANCE_CSV = ROOT / "data" / "derived" / "litho_contact_centroid_distance_v14.csv"
V14_DISTANCE_GPKG = ROOT / "data" / "derived" / "litho_contact_centroid_distance_v14.gpkg"
V14_MANIFEST = ROOT / "data" / "derived" / "canonical_training_manifest_v14_centroid_contact.json"
DISTANCE_COLUMN = "distance_to_lithology_contact"

V13_TRAINING_SHA256 = "ccf8f4c256aebf6039d4ef87a6516ba50a41eeb4ee4100ac5c4aca74d9005af3"
V13_DISTANCE_SHA256 = "dda749f927762363c343021b5b9402c551076736f071ab51f35cf29621cb2e08"
TRACE_SHA256 = {
    "v11_fold_1_trace.pkl": "bbe7b0bc27b292389282c0d28bb9fa5947f0eda4db199c3651423fbeee9f3c08",
    "v11_fold_2_trace.pkl": "38007aacb85101b122e09d06340f48bfc85f98cd148f7d0de25a10de5ae2e016",
    "v11_fold_3_trace.pkl": "622c5e775f421182a5a47d08271ed6d339e443f8303cfee42623bc7462b494a6",
    "v11_fold_4_trace.pkl": "5a2e8797222b104aec5003ee4baca156c228734cd7d0406bb6e5c0b5d60e7161",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def artifacts():
    paths = [V13_TRAINING, V13_DISTANCE, V14_TRAINING, V14_DISTANCE_CSV,
             V14_DISTANCE_GPKG, V14_MANIFEST]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        pytest.fail(f"Missing V14 centroid-contact artifacts: {missing}")
    return {
        "v13": pd.read_csv(V13_TRAINING, encoding="utf-8-sig"),
        "training": pd.read_csv(V14_TRAINING),
        "distance": pd.read_csv(V14_DISTANCE_CSV),
        "gpkg": gpd.read_file(V14_DISTANCE_GPKG),
        "manifest": json.loads(V14_MANIFEST.read_text(encoding="utf-8")),
    }


def test_frozen_v13_and_trace_hashes(artifacts):
    assert sha256(V13_TRAINING) == V13_TRAINING_SHA256
    assert sha256(V13_DISTANCE) == V13_DISTANCE_SHA256
    for filename, expected in TRACE_SHA256.items():
        assert sha256(ROOT / "figures" / filename) == expected, filename


def test_manifest_hashes_and_status(artifacts):
    manifest = artifacts["manifest"]
    assert manifest["dataset_version"] == "V14_CENTROID_CONTACT"
    assert manifest["validation"]["status"] == "PASS"
    assert manifest["source_geometry"]["sha256"] == sha256(
        Path(manifest["source_geometry"]["path"])
    )
    assert manifest["outputs"]["training_table"]["sha256"] == sha256(V14_TRAINING)
    assert manifest["outputs"]["distance_csv"]["sha256"] == sha256(V14_DISTANCE_CSV)
    assert manifest["outputs"]["distance_gpkg"]["sha256"] == sha256(V14_DISTANCE_GPKG)


def test_one_row_per_grid_cell_and_numeric_invariants(artifacts):
    training = artifacts["training"]
    distance = artifacts["distance"]
    assert len(training) == len(distance) == 1899
    assert training["id"].is_unique and training["id"].notna().all()
    assert distance["id"].is_unique and distance["id"].notna().all()
    assert set(training["id"]) == set(distance["id"]) == set(artifacts["v13"]["id"])
    values = distance[DISTANCE_COLUMN].to_numpy(dtype=float)
    assert np.isfinite(values).all()
    assert (values >= 0.0).all()


def test_only_distance_column_changes(artifacts):
    base = artifacts["v13"]
    training = artifacts["training"]
    assert training["id"].equals(base["id"])
    unchanged = [column for column in base.columns if column != DISTANCE_COLUMN]
    pd.testing.assert_frame_equal(
        training[unchanged], base[unchanged], check_dtype=False, check_names=True
    )
    assert (training[DISTANCE_COLUMN] != base[DISTANCE_COLUMN]).all()


def test_csv_and_geopackage_agree(artifacts):
    distance = artifacts["distance"].set_index("id").sort_index()
    gpkg = artifacts["gpkg"].set_index("id").sort_index()
    assert distance.index.equals(gpkg.index)
    assert gpkg.crs.to_epsg() == 32735
    assert gpkg.geometry.geom_type.eq("Point").all()
    assert np.max(np.abs(gpkg["distance_m"] - distance[DISTANCE_COLUMN])) <= 1e-9
    offset = np.hypot(
        gpkg.geometry.x - distance["centroid_x"],
        gpkg.geometry.y - distance["centroid_y"],
    )
    assert float(offset.max()) <= 1e-8


def test_complete_case_frame_and_folds_are_unchanged(artifacts):
    from src.validation_strategies import get_along_belt_folds

    def frame(table):
        selected = table.dropna(
            subset=["id", "centroid_x", "centroid_y", "domain",
                    "litho_contact_litho_class", "distance_to_fault",
                    DISTANCE_COLUMN, "bouguer"]
        ).copy()
        selected["daly_domain"] = selected["domain"].map(
            lambda value: next(
                (code for token, code in (
                    ("3a", "NRB_3a"), ("3b", "NRB_3b"), ("crz", "CRZ"),
                    ("srb", "SRB"), ("nkb", "NKB"), ("mmsb", "MMSB")
                ) if token in str(value).lower()),
                "Unknown",
            )
        )
        selected = selected[selected["daly_domain"] != "Unknown"].reset_index(drop=True)
        selected["fold"] = get_along_belt_folds(selected, n_folds=4)
        return selected

    v13 = frame(artifacts["v13"])
    v14 = frame(artifacts["training"])
    assert len(v14) == 1872
    assert int(v14["deposit_present"].sum()) == 138
    assert int((v14["deposit_present"] == 0).sum()) == 1734
    assert v13["id"].equals(v14["id"])
    assert np.array_equal(v13["fold"].to_numpy(), v14["fold"].to_numpy())
    assert v14.groupby("fold").size().tolist() == [468, 468, 468, 468]
    assert v14.groupby("fold")["deposit_present"].sum().tolist() == [9, 31, 37, 61]


def test_independent_geometry_spot_checks(artifacts):
    manifest = artifacts["manifest"]
    source = Path(manifest["source_geometry"]["path"])
    if not source.is_file():
        pytest.skip(f"Authoritative contact geometry unavailable at {source}")
    contacts = gpd.read_file(source, fid_as_index=True).reset_index()
    training = artifacts["training"]
    distance = artifacts["distance"].set_index("id")
    points = shapely.points(training[["centroid_x", "centroid_y"]].to_numpy(dtype=float))
    line_array = contacts.geometry.to_numpy()
    matrix = shapely.distance(points[:, None], line_array[None, :])
    minimum = matrix.min(axis=1)
    deltas = np.abs(minimum - training[DISTANCE_COLUMN].to_numpy(dtype=float))
    assert float(deltas.max()) <= 1e-8

    candidate_indices = {
        int(np.argmin(minimum)),
        int(np.argmax(minimum)),
        int(np.argmin(training["centroid_x"])),
        int(np.argmax(training["centroid_x"])),
        int(np.argmin(training["centroid_y"])),
        int(np.argmax(training["centroid_y"])),
    }
    tie_row = training.index[artifacts["distance"]["tie_count"] > 1]
    if len(tie_row):
        candidate_indices.add(int(tie_row[0]))
    for row in candidate_indices:
        cell_id = int(training.iloc[row]["id"])
        stored = float(distance.loc[cell_id, DISTANCE_COLUMN])
        assert abs(float(minimum[row]) - stored) <= 1e-8
        assert int(distance.loc[cell_id, "tie_count"]) == int(
            (matrix[row] <= minimum[row] + 1e-9).sum()
        )


def test_no_class_proxy_fields_and_centroid_semantics(artifacts):
    distance = artifacts["distance"]
    assert "nearest_contact_litho_class" not in distance.columns
    assert "area_majority_class" not in distance.columns
    assert set(distance.columns) == {
        "id", DISTANCE_COLUMN, "representative_contact_fid", "tie_count",
        "tied_contact_fids", "tied_contact_classes", "representative_x",
        "representative_y", "feature_x", "feature_y", "centroid_x",
        "centroid_y", "n_contacts_evaluated",
    }
    assert np.allclose(distance["feature_x"], distance["centroid_x"], atol=1e-12)
    assert np.allclose(distance["feature_y"], distance["centroid_y"], atol=1e-12)
    assert (distance["tie_count"] >= 1).all()
    assert distance["tied_contact_classes"].str.len().gt(0).all()


def test_v14_runner_changes_paths_only():
    from src import v14_centroid_spatial_stability as runner

    assert sha256(runner.FROZEN_RUNNER) == runner.FROZEN_RUNNER_SHA256
    source = runner.FROZEN_RUNNER.read_text(encoding="utf-8-sig")
    transformed = source
    for old, new in runner.PAIRS:
        assert transformed.count(old) == 1, old
        transformed = transformed.replace(old, new, 1)
    assert "copperbelt_training_v13_litho_contact.csv" not in transformed
    assert "v11_fold_" not in transformed
    assert "v11_geological_stability_matrix" not in transformed
    assert "copperbelt_training_v14_centroid_contact.csv" in transformed
    assert "v14_centroid_fold_{fold+1}_trace.nc" in transformed
