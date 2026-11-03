"""Run the frozen V11 model against the V14 centroid-contact dataset.

This is an execution wrapper, not a new model. It loads the byte-verified frozen
V11 runner and substitutes only the dataset path and output filenames. The
model equation, predictors, priors, hierarchy, parameterization, sampler
settings, seed, folds, and evaluation logic are not changed.

The default output directory is versioned and cannot contain the historical
``figures/v11_fold_*.pkl`` traces. A successful run writes:

    figures/v14_centroid_contact/v14_centroid_fold_{1..4}_trace.nc (or .pkl)
    figures/v14_centroid_contact/v14_centroid_fold_metrics.csv
    figures/v14_centroid_contact/v14_centroid_geological_stability_matrix.csv
    figures/v14_centroid_contact/run_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import types
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.runtime_shim  # noqa: F401,E402  (must precede PyMC/ArviZ imports)
from src import numerics

FROZEN_RUNNER = ROOT / "src" / "v11_spatial_stability_final.py"
FROZEN_RUNNER_SHA256 = (
    "587b64963153793570e84e0a4a5f2b8f586fd811659f1b30c17997bcf49a3c0b"
)
V14_TRAINING = ROOT / "data" / "copperbelt_training_v14_centroid_contact.csv"
V14_MANIFEST = (
    ROOT / "data" / "derived" / "canonical_training_manifest_v14_centroid_contact.json"
)
DEFAULT_OUTPUT = ROOT / "figures" / "v14_centroid_contact"
PAIRS = [
    (
        "data_path = ROOT / 'data' / 'copperbelt_training_v13_litho_contact.csv'",
        "data_path = ROOT / 'data' / 'copperbelt_training_v14_centroid_contact.csv'",
    ),
    (
        "trace_path = output_dir / f'v11_fold_{fold+1}_trace.nc'",
        "trace_path = output_dir / f'v14_centroid_fold_{fold+1}_trace.nc'",
    ),
    (
        "stability_csv = output_dir / 'v11_geological_stability_matrix.csv'",
        "stability_csv = output_dir / 'v14_centroid_geological_stability_matrix.csv'",
    ),
    (
        'pd.DataFrame(fold_rows).to_csv(output_dir / "v11_fold_metrics.csv", index=False)',
        'pd.DataFrame(fold_rows).to_csv(output_dir / "v14_centroid_fold_metrics.csv", index=False)',
    ),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def install_numerics_compat() -> None:
    """Expose project fallbacks for sklearn APIs that the frozen runner imports.

    The QGIS Python 3.12 interpreter cannot load the CPython 3.10 sklearn binary
    extension in ``.venv``. ``src.numerics`` supplies drop-in implementations of
    the two imports used by the frozen model (``StandardScaler`` and
    ``roc_auc_score``); their numerical agreement is covered by
    ``tests/test_numerics_compat.py``. This changes import resolution only, not
    the model specification or computation.
    """
    if numerics.SKLEARN_AVAILABLE:
        return
    sklearn_module = types.ModuleType("sklearn")
    preprocessing_module = types.ModuleType("sklearn.preprocessing")
    metrics_module = types.ModuleType("sklearn.metrics")
    preprocessing_module.StandardScaler = numerics.StandardScaler
    metrics_module.roc_auc_score = numerics.roc_auc_score
    sklearn_module.preprocessing = preprocessing_module
    sklearn_module.metrics = metrics_module
    sys.modules["sklearn"] = sklearn_module
    sys.modules["sklearn.preprocessing"] = preprocessing_module
    sys.modules["sklearn.metrics"] = metrics_module


def verify_inputs() -> dict:
    if not V14_TRAINING.is_file() or not V14_MANIFEST.is_file():
        raise FileNotFoundError("V14 training table or manifest is missing")
    manifest = json.loads(V14_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("dataset_version") != "V14_CENTROID_CONTACT":
        raise ValueError("V14 manifest has the wrong dataset version")
    if manifest.get("validation", {}).get("status") != "PASS":
        raise ValueError("V14 dataset does not have a passing independent validation")
    expected = manifest["outputs"]["training_table"]["sha256"]
    actual = sha256(V14_TRAINING)
    if actual != expected:
        raise ValueError(f"V14 training hash differs: {actual} != {expected}")
    if sha256(FROZEN_RUNNER) != FROZEN_RUNNER_SHA256:
        raise ValueError("Frozen V11 runner hash changed; refusing model execution")
    return manifest


def build_runner() -> object:
    if sha256(FROZEN_RUNNER) != FROZEN_RUNNER_SHA256:
        raise ValueError("Frozen V11 runner hash changed; refusing model execution")
    install_numerics_compat()
    source = FROZEN_RUNNER.read_text(encoding="utf-8-sig")
    transformed = source
    for old, new in PAIRS:
        count = transformed.count(old)
        if count != 1:
            raise ValueError(
                f"Expected exactly one frozen-runner match for {old!r}; found {count}"
            )
        transformed = transformed.replace(old, new, 1)
    if "copperbelt_training_v13_litho_contact.csv" in transformed:
        raise ValueError("V13 path remains in transformed V14 runner")
    if "v11_fold_" in transformed or "v11_geological_stability_matrix" in transformed:
        raise ValueError("Historical V11 output names remain in transformed V14 runner")
    namespace = {
        "__name__": "v14_centroid_model",
        "__file__": str(ROOT / "src" / "v14_centroid_spatial_stability.py"),
        "__package__": "src",
    }
    exec(compile(transformed, str(FROZEN_RUNNER), "exec"), namespace)
    return namespace["run_v11_spatial_cv"]


def output_files(output_dir: Path) -> list[Path]:
    files = sorted(output_dir.rglob("*")) if output_dir.exists() else []
    return [path for path in files if path.is_file()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--draws", type=int, default=1500)
    parser.add_argument("--tune", type=int, default=2500)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    dataset_manifest = verify_inputs()
    run_manifest_path = args.output_dir / "run_manifest.json"
    if run_manifest_path.exists():
        raise FileExistsError(
            f"Versioned output directory already contains a run: {run_manifest_path}"
        )
    existing = [str(path) for path in output_files(args.output_dir) if path.suffix in {".nc", ".pkl"}]
    if existing:
        raise FileExistsError(f"Versioned output directory already contains traces: {existing}")

    run_manifest = {
        "schema_version": 1,
        "status": "DRY_RUN" if not args.execute else "RUNNING",
        "model_identity": "frozen V11 specification executed through an I/O-only wrapper",
        "started_utc": utc_now(),
        "runtime": {
            "executable": sys.executable,
            "version": sys.version,
            "pytensor_flags": os.environ.get("PYTENSOR_FLAGS", ""),
        },
        "frozen_model": {
            "path": FROZEN_RUNNER.relative_to(ROOT).as_posix(),
            "sha256": sha256(FROZEN_RUNNER),
        },
        "execution_wrapper": {
            "path": "src/v14_centroid_spatial_stability.py",
            "sha256": sha256(Path(__file__)),
        },
        "dataset": {
            "path": V14_TRAINING.relative_to(ROOT).as_posix(),
            "sha256": sha256(V14_TRAINING),
            "manifest": V14_MANIFEST.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256(V14_MANIFEST),
            "validation_status": dataset_manifest["validation"]["status"],
        },
        "configuration": {
            "folds": args.folds,
            "draws": args.draws,
            "tune": args.tune,
            "chains": 2,
            "cores": 2,
            "target_accept": 0.99,
            "random_seed": 42,
        },
        "output_directory": str(args.output_dir),
        "expected_trace_names": [
            f"v14_centroid_fold_{fold}_trace.nc" for fold in range(1, args.folds + 1)
        ],
        "changes_to_frozen_runner": [
            {"from": old, "to": new} for old, new in PAIRS
        ],
        "compatibility_shim": {
            "applied": not numerics.SKLEARN_AVAILABLE,
            "reason": "QGIS Python 3.12 cannot load the CPython 3.10 sklearn extension",
            "standard_scaler": "src.numerics.StandardScaler",
            "roc_auc_score": "src.numerics.roc_auc_score",
            "model_specification_changed": False,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    if not args.execute:
        print(json.dumps(run_manifest, indent=2))
        print("Dry run complete; pass --execute to launch the four-fold refit.")
        return

    try:
        run = build_runner()
        rows = run(args.folds, args.draws, args.tune, args.output_dir)
    except Exception as error:
        run_manifest["status"] = "FAILED"
        run_manifest["finished_utc"] = utc_now()
        run_manifest["error"] = repr(error)
        run_manifest["traceback"] = traceback.format_exc()
        run_manifest_path.write_text(
            json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
        )
        raise

    produced = output_files(args.output_dir)
    run_manifest["status"] = "PASS"
    run_manifest["finished_utc"] = utc_now()
    run_manifest["fold_metrics"] = rows
    run_manifest["outputs"] = [
        {
            "path": path.relative_to(args.output_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in produced
        if path.resolve() != run_manifest_path.resolve()
    ]
    run_manifest_path.write_text(
        json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_manifest, indent=2))


if __name__ == "__main__":
    main()
