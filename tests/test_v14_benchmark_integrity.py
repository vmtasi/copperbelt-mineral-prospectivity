"""Byte-level integrity of the Paper 1 V14 benchmark artefacts.

``paper_1/benchmark/SHA256SUMS`` lists the artefacts tracked in Git; every
one must be present and unchanged. ``SHA256SUMS_external`` lists the
distance-representation sensitivity traces that are distributed only through
the external archive; they are checked when present and skipped otherwise.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "paper_1" / "benchmark"


def _entries(name: str) -> list[tuple[str, str]]:
    rows = []
    for line in (BENCHMARK / name).read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, relative = line.split(maxsplit=1)
            rows.append((digest.lower(), relative.strip().lstrip("*")))
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


TRACKED = _entries("SHA256SUMS")
EXTERNAL = _entries("SHA256SUMS_external")


def test_manifest_covers_core_benchmark():
    paths = {relative for _, relative in TRACKED}
    required = {
        "data/copperbelt_training_v14_centroid_contact.csv",
        "figures/v14_centroid_contact/v14_centroid_oof_predictions.csv",
        "figures/v14_centroid_contact/v14_centroid_fold_metrics.csv",
        "figures/audit/phase4_oof_predictions_v14_centroid.csv",
        *(f"figures/v14_centroid_contact/v14_centroid_fold_{k}_trace.nc" for k in range(1, 5)),
    }
    assert required <= paths, sorted(required - paths)


@pytest.mark.parametrize("digest,relative", TRACKED, ids=[r for _, r in TRACKED])
def test_tracked_artefact_unchanged(digest, relative):
    path = ROOT / relative
    assert path.is_file(), f"missing tracked benchmark artefact: {relative}"
    assert _sha256(path) == digest, f"benchmark artefact changed: {relative}"


@pytest.mark.parametrize("digest,relative", EXTERNAL, ids=[r for _, r in EXTERNAL])
def test_external_artefact_unchanged_when_restored(digest, relative):
    path = ROOT / relative
    if not path.is_file():
        pytest.skip("external-archive artefact not restored")
    assert _sha256(path) == digest, f"external artefact changed: {relative}"
