"""Compare active spatial folds with the frozen HEAD fold implementation.

The frozen baseline hash was recorded from the CRLF working-tree file.  Git
stores the same source with LF endings, so this check restores CRLF before
hashing and executing the baseline implementation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
V14 = ROOT / "data" / "copperbelt_training_v14_centroid_contact.csv"
BASELINE_SHA256 = "435aeb6ddc08ca018c2bfa009c78f0f2154cb2ae5a08e6c23d7eaf15dd9e9ed7"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def domain_code(value):
    text = str(value).lower()
    if "3a" in text:
        return "NRB_3a"
    if "3b" in text:
        return "NRB_3b"
    if "crz" in text:
        return "CRZ"
    if "srb" in text:
        return "SRB"
    if "nkb" in text:
        return "NKB"
    if "mmsb" in text:
        return "MMSB"
    return "Unknown"


def model_frame() -> pd.DataFrame:
    frame = pd.read_csv(V14)
    columns = [
        "id", "centroid_x", "centroid_y", "domain",
        "litho_contact_litho_class", "distance_to_fault",
        "distance_to_lithology_contact", "bouguer",
    ]
    frame = frame.dropna(subset=columns).copy()
    frame["daly_domain"] = frame["domain"].map(domain_code)
    return frame[frame["daly_domain"] != "Unknown"].reset_index(drop=True)


def load_baseline():
    raw = subprocess.check_output(
        ["git", "show", "HEAD:src/validation_strategies.py"], cwd=ROOT
    )
    source = raw.replace(b"\n", b"\r\n")
    if sha256(source) != BASELINE_SHA256:
        raise RuntimeError("HEAD source does not reproduce the frozen baseline hash")
    with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as directory:
        path = Path(directory) / "validation_strategies_baseline.py"
        path.write_bytes(source)
        spec = importlib.util.spec_from_file_location("validation_strategies_baseline", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, sha256(source)


def main(output_path: Path) -> None:
    from src.validation_strategies import get_along_belt_folds

    baseline, baseline_hash = load_baseline()
    frame = model_frame()
    active_folds = get_along_belt_folds(frame, n_folds=4)
    baseline_folds = baseline.get_along_belt_folds(frame, n_folds=4)
    mismatch = np.flatnonzero(active_folds != baseline_folds)
    active_sizes = np.bincount(active_folds, minlength=4).tolist()
    active_positives = frame.groupby(pd.Series(active_folds, name="fold"))[
        "deposit_present"
    ].sum().tolist()
    report = {
        "schema_version": 1,
        "status": "PASS" if mismatch.size == 0 else "FAIL",
        "baseline": {
            "source": "git show HEAD:src/validation_strategies.py",
            "sha256_crlf": baseline_hash,
        },
        "active_source_sha256": sha256(
            (ROOT / "src" / "validation_strategies.py").read_bytes()
        ),
        "rows": int(len(frame)),
        "positive_cells": int(frame["deposit_present"].sum()),
        "fold_sizes": active_sizes,
        "fold_positives": [int(value) for value in active_positives],
        "id_level_fold_mismatches": int(mismatch.size),
        "mismatched_ids": frame.iloc[mismatch]["id"].astype(int).tolist(),
        "comparison": "all 1,872 stable grid-cell ids in the model frame",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if mismatch.size:
        raise SystemExit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "figures" / "audit" / "fold_compatibility_pre_refit.json",
    )
    main(parser.parse_args().output)
