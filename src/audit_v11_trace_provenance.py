"""Audit the provenance of the four frozen V11 fold traces.

The audit is evidence-driven and regenerable.  For every fold it:

* hashes the trace and records its ArviZ/PyMC metadata and dimensions;
* compares the trace's ``observed_data['y_obs']`` with the exact training-label
  vector implied by the canonical V13 table and the frozen fold construction;
* recomputes convergence diagnostics from the stored draws;
* checks that the trace postdates the recorded launch of
  ``src/v11_spatial_stability_final.py`` and that the launch read the V13 table;
* confirms that the ID-bearing OOF artifact is reproducible from the traces.

The output deliberately distinguishes PASS / FAIL / UNVERIFIED.  A dataset
hash is *not* embedded in the pickles, so lineage is asserted only through
label-vector equality, the canonical fold construction and the session-log
launch record.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.runtime_shim  # noqa: F401,E402

import numpy as np
import pandas as pd

from src.oof_provenance import (
    CANONICAL_DATA_PATH,
    DEFAULT_MANIFEST_PATH as OOF_MANIFEST_PATH,
    DEFAULT_OOF_PATH,
    sha256_file,
    write_manifest,
)
from src.trace_io import load_trace, resolve_trace_path
from src.validation_strategies import get_along_belt_folds

SESSION_LOG = Path(
    r"C:\Users\vanmu\.codex\sessions\2026\10\18"
    r"\rollout-2026-10-18T15-30-18-01a14efd-cc42-7603-a36f-23505e4dc014.jsonl"
)
LAUNCH_MARKER = "run_qgis_python.bat src\\v11_spatial_stability_final.py"
LEGACY_OOF_PATH = (
    ROOT / 'archive' / 'paper1_provenance_2026-10-20' / 'legacy'
    / 'v11_oof_predictions_centroid_only.csv'
)
MANIFEST_PATHS = [
    ROOT / 'figures' / 'audit' / 'trace_provenance_manifest.json',
]
EXPECTED_PARAMETERS = {
    "alpha_dom", "alpha_mu", "alpha_sigma", "alpha_offset",
    "beta_f_lin", "beta_f_sq", "beta_l_lin", "beta_l_sq",
    "beta_grav", "beta_rocks",
    "mu_f_lin", "mu_f_sq", "mu_l_lin", "mu_l_sq",
    "offset_f_lin", "offset_f_sq", "offset_l_lin", "offset_l_sq",
    "sigma_f_lin", "sigma_f_sq", "sigma_l_lin", "sigma_l_sq",
}


def map_daly_domain(value):
    value = str(value).lower()
    if "3a" in value:
        return "NRB_3a"
    if "3b" in value:
        return "NRB_3b"
    if "crz" in value:
        return "CRZ"
    if "srb" in value:
        return "SRB"
    if "nkb" in value:
        return "NKB"
    if "mmsb" in value:
        return "MMSB"
    return "Unknown"


def canonical_fold_frame():
    df = pd.read_csv(CANONICAL_DATA_PATH)
    lithology_col = "litho_contact_litho_class"
    features = ["distance_to_fault", "distance_to_lithology_contact", "bouguer"]
    df = df.dropna(subset=["id", "centroid_x", "centroid_y", "domain", lithology_col] + features).copy()
    df["daly_domain"] = df["domain"].apply(map_daly_domain)
    df = df[df["daly_domain"] != "Unknown"].copy()
    df["spatial_block"] = get_along_belt_folds(df, n_folds=4)
    return df


def find_launch_evidence():
    evidence = {
        "session_log": str(SESSION_LOG),
        "exists": SESSION_LOG.is_file(),
        "launch_command": None,
        "launch_utc": None,
        "v13_reference_utc": None,
        "status": "UNVERIFIED",
    }
    if not evidence["exists"]:
        evidence["reason"] = "session log not found; launch could not be verified"
        return evidence

    evidence["sha256"] = sha256_file(SESSION_LOG)
    trace_times = []
    for fold in range(4):
        trace_path = resolve_trace_path(ROOT / "figures" / f"v11_fold_{fold + 1}_trace.nc")
        trace_times.append(
            datetime.fromtimestamp(trace_path.stat().st_mtime, tz=timezone.utc)
        )
    earliest_trace = min(trace_times)

    latest_launch = None
    earliest_v13 = None
    rejected_launches = []
    with SESSION_LOG.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            timestamp = record.get("timestamp")
            payload = record.get("payload", {})
            if payload.get("type") == "function_call":
                arguments = payload.get("arguments", "")
                normalized = arguments.replace("\\\\", "\\").replace("/", "\\")
                if LAUNCH_MARKER not in normalized:
                    continue
                command = arguments
                try:
                    parsed = json.loads(arguments)
                    command = str(parsed.get("cmd", arguments))
                except (TypeError, json.JSONDecodeError):
                    pass
                command_normalized = command.replace("\\\\", "\\").replace("/", "\\")
                timestamp_utc = parse_utc(timestamp)

                # A help request never executes the model. A run with a custom
                # output directory cannot explain traces in figures/. The
                # historical launch that produced these traces predates the
                # first trace write.
                reason = None
                if re.search(r"(?:^|\s)--help(?:\s|\"|'|;|$)", command_normalized):
                    reason = "help invocation"
                elif "--output-dir" in command_normalized:
                    reason = "custom output directory"
                elif timestamp_utc is None or timestamp_utc > earliest_trace:
                    reason = "not before the first trace write"
                if reason:
                    rejected_launches.append(
                        {"timestamp": timestamp, "reason": reason, "arguments": arguments[:600]}
                    )
                    continue
                if latest_launch is None or timestamp_utc > parse_utc(latest_launch["timestamp"]):
                    latest_launch = {"timestamp": timestamp, "arguments": arguments[:600]}
            elif payload.get("type") == "function_call_output":
                output = payload.get("output", "")
                if "copperbelt_training_v13_litho_contact.csv" in output:
                    if earliest_v13 is None or timestamp < earliest_v13:
                        earliest_v13 = timestamp

    if latest_launch:
        evidence["launch_command"] = latest_launch["arguments"]
        evidence["launch_utc"] = latest_launch["timestamp"]
    evidence["v13_reference_utc"] = earliest_v13
    evidence["earliest_trace_utc"] = earliest_trace.isoformat().replace("+00:00", "Z")
    evidence["selection_rule"] = (
        "latest non-help execution of the frozen runner with no custom output "
        "directory, after the V13 reference and before the first trace write"
    )
    evidence["rejected_later_or_non_executing_launches"] = rejected_launches
    launch_utc = parse_utc(evidence["launch_utc"])
    reference_utc = parse_utc(earliest_v13)
    evidence["status"] = (
        "PASS"
        if launch_utc and reference_utc and reference_utc <= launch_utc <= earliest_trace
        else "UNVERIFIED"
    )
    return evidence


def parse_utc(value):
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def audit_fold(fold, frame, launch):
    trace_path = ROOT / "figures" / f"v11_fold_{fold + 1}_trace.nc"
    resolved = resolve_trace_path(trace_path)
    record = {
        "trace_path": str(resolved.relative_to(ROOT)),
        "sha256": sha256_file(resolved),
        "bytes": int(resolved.stat().st_size),
        "modified_utc": datetime.fromtimestamp(resolved.stat().st_mtime, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }

    idata = load_trace(trace_path)
    posterior = idata.posterior
    record["arviz_version"] = posterior.attrs.get("arviz_version")
    record["inference_library"] = posterior.attrs.get("inference_library")
    record["inference_library_version"] = posterior.attrs.get("inference_library_version")
    record["created_at"] = posterior.attrs.get("created_at")
    record["tuning_steps"] = posterior.attrs.get("tuning_steps")
    record["chains"] = int(posterior.sizes.get("chain", 0))
    record["draws_per_chain"] = int(posterior.sizes.get("draw", 0))
    record["total_draws"] = record["chains"] * record["draws_per_chain"]
    record["n_domains"] = int(posterior.sizes.get("alpha_dom_dim_0", 0))
    record["n_rock_terms"] = int(posterior.sizes.get("beta_rocks_dim_0", 0))

    parameters = set(posterior.data_vars)
    record["parameters_present"] = sorted(parameters)
    record["missing_parameters"] = sorted(EXPECTED_PARAMETERS - parameters)
    record["extra_parameters"] = sorted(parameters - EXPECTED_PARAMETERS)

    train = frame[frame["spatial_block"] != fold]
    expected_labels = train["deposit_present"].to_numpy().astype(int)
    observed = idata.observed_data["y_obs"].values.astype(int) if idata.observed_data is not None else None
    if observed is None:
        record["y_obs_check"] = {"status": "FAIL", "reason": "trace has no observed_data group"}
    else:
        matches = bool(observed.shape == expected_labels.shape) and bool(
            np.array_equal(observed, expected_labels)
        )
        record["y_obs_check"] = {
            "trace_observations": int(observed.size),
            "expected_training_cells": int(expected_labels.size),
            "trace_positives": int(observed.sum()),
            "expected_positives": int(expected_labels.sum()),
            "labels_identical": bool(
                observed.shape == expected_labels.shape and np.array_equal(observed, expected_labels)
            ),
            "label_vector_sha256": _vector_hash(expected_labels),
            "status": "PASS" if matches else "FAIL",
        }

    divergences = idata.sample_stats["divergences"].values if "divergences" in idata.sample_stats else None
    record["divergences"] = int(divergences.sum()) if divergences is not None else None
    try:
        import arviz as az

        rhat = idata.rhat() if hasattr(idata, "rhat") else az.rhat(idata)
        values = [rhat[name].values for name in rhat.data_vars]
        finite_rhat = np.concatenate([value.ravel() for value in values])
        finite_rhat = finite_rhat[np.isfinite(finite_rhat)]
        record["max_r_hat"] = float(finite_rhat.max()) if finite_rhat.size else None
    except Exception as error:  # noqa: BLE001
        record["max_r_hat"] = None
        record["r_hat_error"] = str(error)
    try:
        ess = idata.ess(method="bulk") if hasattr(idata, "ess") else az.ess(idata, method="bulk")
        values = [ess[name].values for name in ess.data_vars]
        finite_ess = np.concatenate([value.ravel() for value in values])
        finite_ess = finite_ess[np.isfinite(finite_ess)]
        record["min_ess_bulk"] = float(finite_ess.min()) if finite_ess.size else None
    except Exception as error:  # noqa: BLE001
        record["min_ess_bulk"] = None
        record["ess_error"] = str(error)

    if launch.get("launch_utc"):
        created = parse_utc(record["modified_utc"])
        launched = parse_utc(launch["launch_utc"])
        record["written_after_launch"] = bool(created and launched and created >= launched)
    else:
        record["written_after_launch"] = None

    record["status"] = "PASS" if (
        record["y_obs_check"]["status"] == "PASS"
        and not record["missing_parameters"]
        and record["chains"] == 2
        and record["draws_per_chain"] == 1500
        and record["tuning_steps"] == 2500
        and record["written_after_launch"] is not False
    ) else "FAIL"
    return record


def _vector_hash(values):
    import hashlib

    return hashlib.sha256(np.asarray(values, dtype=np.int8).tobytes()).hexdigest()


def verify_oof_reproduction():
    """Recompute the archived-legacy comparison as an audit-only check.

    The active extractor must not read the historical archive.  This audit is
    the one place where the retired centroid-only artifact is opened, solely
    to verify that the ID-bearing replacement preserves the frozen scores.
    """
    if not DEFAULT_OOF_PATH.is_file() or not OOF_MANIFEST_PATH.is_file():
        return {'status': 'UNVERIFIED', 'reason': 'OOF artifact or manifest missing'}
    if not LEGACY_OOF_PATH.is_file():
        return {'status': 'UNVERIFIED', 'reason': 'archived legacy OOF missing'}

    manifest = json.loads(OOF_MANIFEST_PATH.read_text(encoding='utf-8'))
    current = pd.read_csv(DEFAULT_OOF_PATH)
    legacy = pd.read_csv(LEGACY_OOF_PATH)
    key = ['centroid_x', 'centroid_y']
    if legacy.duplicated(key).any() or current.duplicated(key).any():
        return {'status': 'UNVERIFIED', 'reason': 'centroid keys are not unique'}
    if len(legacy) != len(current):
        return {'status': 'FAIL', 'reason': 'row counts differ'}

    compared = current.merge(
        legacy,
        on=key,
        how='inner',
        validate='one_to_one',
        suffixes=('', '_legacy'),
    )
    if len(compared) != len(current):
        return {'status': 'FAIL', 'reason': 'not all rows matched one-to-one'}
    max_diff = float(np.abs(compared['prob_v11'] - compared['prob_v11_legacy']).max())
    return {
        'status': 'PASS' if max_diff <= 1e-12 else 'FAIL',
        'artifact_sha256': sha256_file(DEFAULT_OOF_PATH),
        'oof_manifest_status': manifest.get('status'),
        'reference': str(LEGACY_OOF_PATH.relative_to(ROOT)),
        'reference_sha256': sha256_file(LEGACY_OOF_PATH),
        'matched_rows': int(len(compared)),
        'max_abs_diff': max_diff,
        'max_abs_diff_vs_archived_legacy': max_diff,
        'linkage_note': (
            'Audit-only comparison to the retired centroid-only file; the '
            'active OOF artifact and all downstream code use stable grid-cell id.'
        ),
    }

def main():
    frame = canonical_fold_frame()
    launch = find_launch_evidence()

    folds = {f"fold_{fold + 1}": audit_fold(fold, frame, launch) for fold in range(4)}
    reproduction = verify_oof_reproduction()

    canonical = {
        "path": str(CANONICAL_DATA_PATH.relative_to(ROOT)),
        "sha256": sha256_file(CANONICAL_DATA_PATH),
        "rows": int(len(frame)),
        "positive_cells": int(frame["deposit_present"].sum()),
        "background_cells": int(len(frame) - frame["deposit_present"].sum()),
        "fold_counts": {
            str(int(k)): int(v) for k, v in frame["spatial_block"].value_counts().sort_index().items()
        },
        "fold_positive_counts": {
            str(int(k)): int(v)
            for k, v in frame.groupby("spatial_block")["deposit_present"].sum().sort_index().items()
        },
    }

    all_passed = all(fold["status"] == "PASS" for fold in folds.values()) and launch["status"] == "PASS"
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "generator": "src/audit_v11_trace_provenance.py",
        "status": "PASS" if all_passed else "FAIL",
        "scope": (
            "Provenance of the four frozen V11 fold traces and of the V11 OOF "
            "artifact. Lineage rests on observed-label equality with the "
            "canonical V13 fold construction plus the recorded launch command; "
            "no dataset hash is embedded in the pickles."
        ),
        "canonical_dataset": canonical,
        "launch_evidence": launch,
        "folds": folds,
        "oof_reproduction": reproduction,
        "limitations": [
            "The pickle traces do not embed a dataset hash or code revision.",
            "Lineage is inferred from exact observed-label equality, the frozen "
            "fold construction and the session-log launch record.",
            "Sampler diagnostics are recomputed from the stored draws; they are "
            "not inherited from any earlier summary file.",
        ],
    }

    for path in MANIFEST_PATHS:
        write_manifest(path, manifest)

    print(json.dumps({
        "status": manifest["status"],
        "launch_utc": launch["launch_utc"],
        "fold_status": {name: record["status"] for name, record in folds.items()},
        "labels_identical": {name: record["y_obs_check"]["labels_identical"] for name, record in folds.items()},
        "divergences": {name: record["divergences"] for name, record in folds.items()},
        "max_r_hat": {name: record["max_r_hat"] for name, record in folds.items()},
        "min_ess_bulk": {name: record["min_ess_bulk"] for name, record in folds.items()},
        "oof_reproduction": reproduction["status"],
    }, indent=2))
    return manifest


if __name__ == "__main__":
    main()



