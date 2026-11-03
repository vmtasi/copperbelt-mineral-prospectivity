"""Document the operational definition of the lithology-contact distance.

The manuscript and several earlier notes described this predictor as a
centroid-to-contact distance.  The validated rebuild implementation, however,
computes ``cell_polygon.distance(contact_line)``.  This audit records the
evidence and quantifies how far a centroid interpretation would be from the
stored values.  It does not alter the frozen feature or the fitted model.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.runtime_shim  # noqa: F401,E402

import numpy as np
import pandas as pd

from src.oof_provenance import sha256_file

DISTANCE_CSV = ROOT / "data" / "derived" / "litho_contact_distance_v13.csv"
CANONICAL = ROOT / "data" / "copperbelt_training_v13_litho_contact.csv"
REBUILD = ROOT / "src" / "rebuild_litho_contact_features.py"
OUT = ROOT / "figures" / "audit" / "contact_distance_definition_check.json"


def main():
    distance = pd.read_csv(DISTANCE_CSV)
    canonical = pd.read_csv(CANONICAL, usecols=["id", "centroid_x", "centroid_y"])
    merged = distance.merge(canonical, on="id", validate="one_to_one")

    attribute = merged["distance_to_lithology_contact"]
    zero = merged[attribute == 0.0].copy()
    positive = merged[attribute > 0.0].copy()

    centroid_to_representative_zero = np.hypot(
        zero["centroid_x"] - zero["representative_x"],
        zero["centroid_y"] - zero["representative_y"],
    )
    centroid_to_representative_positive = np.hypot(
        positive["centroid_x"] - positive["representative_x"],
        positive["centroid_y"] - positive["representative_y"],
    )
    excess = centroid_to_representative_positive - positive["distance_to_lithology_contact"]

    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "generator": "src/audit_contact_distance_definition.py",
        "status": "VERIFIED_POLYGON_TO_CONTACT",
        "files": {
            str(DISTANCE_CSV.relative_to(ROOT)): sha256_file(DISTANCE_CSV),
            str(CANONICAL.relative_to(ROOT)): sha256_file(CANONICAL),
            str(REBUILD.relative_to(ROOT)): sha256_file(REBUILD),
        },
        "operational_definition": (
            "Minimum Euclidean distance from each clipped cell polygon to the "
            "nearest mapped internal lithological contact line in EPSG:32735; "
            "zero where a contact intersects or touches the cell; the outer "
            "study-area perimeter is excluded."
        ),
        "code_evidence": (
            "src/rebuild_litho_contact_features.py computes "
            "geometry.distance(contact_geometry) for each cell polygon feature "
            "and stores point_on_cell as feature_x/feature_y (not the centroid)."
        ),
        "rows": int(len(merged)),
        "cells_with_zero_distance": int(len(zero)),
        "zero_distance_cells_with_off_contact_centroid": int(
            (centroid_to_representative_zero > 1.0).sum()
        ),
        "zero_distance_centroid_offset_m": {
            "min": float(centroid_to_representative_zero.min()),
            "median": float(np.median(centroid_to_representative_zero)),
            "max": float(centroid_to_representative_zero.max()),
        },
        "feature_point_equals_centroid_rows": {
            "x": int(np.isclose(merged["feature_x"], merged["centroid_x"], atol=1e-6).sum()),
            "y": int(np.isclose(merged["feature_y"], merged["centroid_y"], atol=1e-6).sum()),
        },
        "centroid_minus_stored_distance_m": {
            "median": float(np.median(excess)),
            "max": float(np.max(excess)),
        },
        "implication": (
            "The frozen predictor is polygon-to-contact, not centroid-to-contact. "
            "A centroid-based definition would change every distance and require "
            "refitting the frozen V11 traces; it was therefore not substituted. "
            "Manuscript wording has been corrected to the operational definition."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
