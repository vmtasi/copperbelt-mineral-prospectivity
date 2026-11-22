"""Fit the V15 Bayesian model ladder under the canonical four spatial folds."""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTENSOR_FLAGS", f"compiledir={ROOT / '.runtime_cache' / 'pytensor_v15'}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import arviz as az  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.trace_io import save_trace  # noqa: E402
from src.v15 import config  # noqa: E402
from src.v15.common import (  # noqa: E402
    build_model_frame,
    ensure_v15_path,
    predict_probabilities,
    prepare_fold,
    sha256,
    summarize_draws,
    write_json,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_frame() -> pd.DataFrame:
    path = config.V15 / "data" / "v15_model_frame.csv"
    if path.is_file():
        return pd.read_csv(path)
    frame, audit = build_model_frame()
    path = ensure_v15_path(path)
    frame.to_csv(path, index=False)
    write_json(config.V15 / "diagnostics" / "data_build_audit.json", audit)
    return frame


def model_manifest(args) -> dict:
    return {
        "experiment": "V15_PAPER1",
        "started_utc": utc_now(),
        "python": sys.version,
        "executable": sys.executable,
        "git_head": _git_head(),
        "data": {
            "path": "paper_1/v15/data/v15_model_frame.csv",
            "sha256": sha256(config.V15 / "data" / "v15_model_frame.csv"),
        },
        "sampler": {
            "draws": args.draws,
            "tune": args.tune,
            "chains": args.chains,
            "cores": args.cores,
            "target_accept": args.target_accept,
            "random_seed": args.seed,
        },
        "models": args.models,
        "folds": args.folds,
        "tag": args.tag,
        "prior_scale": args.prior_scale,
    }


def _git_head() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unavailable"


def diagnostics(trace, model_name: str, fold: int) -> tuple[pd.DataFrame, dict]:
    variables = [
        name
        for name in trace.posterior.data_vars
        if name not in {"alpha_offset", "offset_f_lin", "offset_f_sq", "offset_l_lin", "offset_l_sq", "offset_beta_grav"}
    ]
    summary = az.summary(trace, var_names=variables, round_to=8)
    summary.insert(0, "fold", fold + 1)
    summary.insert(0, "model", model_name)
    diverging = int(np.asarray(trace.sample_stats["diverging"]).sum()) if "diverging" in trace.sample_stats else 0
    max_rhat = float(np.nanmax(summary["r_hat"].to_numpy())) if "r_hat" in summary else float("nan")
    min_ess = float(np.nanmin(summary["ess_bulk"].to_numpy())) if "ess_bulk" in summary else float("nan")
    min_tail = float(np.nanmin(summary["ess_tail"].to_numpy())) if "ess_tail" in summary else float("nan")
    info = {
        "model": model_name,
        "fold": fold + 1,
        "diverging": diverging,
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess,
        "min_ess_tail": min_tail,
        "converged": bool(diverging == 0 and np.isfinite(max_rhat) and max_rhat < 1.05 and min_ess >= 200),
    }
    return summary.reset_index(names="parameter"), info


def fit_one(model_name: str, fold: int, frame: pd.DataFrame, args) -> dict:
    spec = config.MODELS[model_name]
    arrays = prepare_fold(
        frame,
        fold,
        spec,
        litho_source=spec.get("litho_source", "v14"),
    )
    stem = f"{args.tag}_{model_name}" if args.tag else model_name
    trace_path = config.V15 / "traces" / stem / f"fold_{fold + 1}.nc"
    prediction_path = config.V15 / "predictions" / f"{stem}_fold_{fold + 1}.csv"
    if trace_path.exists() and prediction_path.exists() and not args.overwrite:
        return {"model": model_name, "fold": fold + 1, "status": "SKIPPED_EXISTING"}

    from src.v15.models import build_model

    model = build_model(model_name, arrays, spec, prior_scale=args.prior_scale)
    started = time.perf_counter()
    with model:
        trace = __import__("pymc").sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            target_accept=args.target_accept,
            random_seed=args.seed,
            progressbar=True,
            idata_kwargs={"log_likelihood": False},
        )
    elapsed = time.perf_counter() - started
    save_trace(trace, ensure_v15_path(trace_path))
    probability = predict_probabilities(trace, spec, arrays, "test")
    summary = summarize_draws(probability)
    prediction = pd.DataFrame(
        {
            "id": arrays["test_ids"],
            "fold": fold + 1,
            "y": arrays["y_test"],
            "prob": summary["prob"],
            "prob_sd": summary["prob_sd"],
            "prob_lower": summary["prob_lower"],
            "prob_upper": summary["prob_upper"],
            "p_ref_train": arrays["p_ref_train"],
        }
    )
    prediction.to_csv(ensure_v15_path(prediction_path), index=False)
    table, info = diagnostics(trace, model_name, fold)
    table.to_csv(
        ensure_v15_path(config.V15 / "diagnostics" / f"{stem}_fold_{fold + 1}_summary.csv"),
        index=False,
    )
    info.update(
        {
            "status": "PASS" if info["converged"] else "FAIL_CONVERGENCE",
            "elapsed_seconds": elapsed,
            "trace": str(trace_path.relative_to(config.V15)),
            "prediction": str(prediction_path.relative_to(config.V15)),
            "n_train": arrays["train_size"],
            "n_test": arrays["test_size"],
            "positive_train": int(arrays["y_train"].sum()),
            "positive_test": int(arrays["y_test"].sum()),
            "n_rocks": arrays["n_rocks"],
            "area_train_sum_km2": arrays["area_train_sum"],
            "link": spec["link"],
            "area_treatment": spec["area"],
            "family": spec["family"],
        }
    )
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=config.PRIMARY_MODELS)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--draws", type=int, default=config.SAMPLER["draws"])
    parser.add_argument("--tune", type=int, default=config.SAMPLER["tune"])
    parser.add_argument("--chains", type=int, default=config.SAMPLER["chains"])
    parser.add_argument("--cores", type=int, default=config.SAMPLER["cores"])
    parser.add_argument("--target-accept", type=float, default=config.SAMPLER["target_accept"])
    parser.add_argument("--seed", type=int, default=config.SAMPLER["random_seed"])
    parser.add_argument("--prior-scale", type=float, default=1.0)
    parser.add_argument("--tag", default="", help="Prefix for sensitivity-run output names")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.pilot:
        args.draws, args.tune = 200, 300
    for name in args.models:
        if name not in config.MODELS:
            raise KeyError(f"Unknown model {name}")
    frame = load_frame()
    manifest = model_manifest(args)
    tag_suffix = f"_{args.tag}" if args.tag else ""
    manifest_path = ensure_v15_path(config.V15 / "models" / f"fit_manifest{tag_suffix}.json")
    write_json(manifest_path, manifest)
    results = []
    for name in args.models:
        for fold in args.folds:
            print(f"[V15] fitting {name} fold {fold + 1}", flush=True)
            result = fit_one(name, fold, frame, args)
            print(f"[V15] {result}", flush=True)
            results.append(result)
            write_json(config.V15 / "models" / f"fit_status{tag_suffix}.json", results)
    failed = [row for row in results if row.get("status") not in {"PASS", "SKIPPED_EXISTING"}]
    if failed:
        raise SystemExit(f"Failed fits: {failed}")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
