"""Buffered spatial validation for the global and domain-distance models."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTENSOR_FLAGS", f"compiledir={ROOT / '.runtime_cache' / 'pytensor_v15'}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import pymc as pm  # noqa: E402
import shapely  # noqa: E402

from src.trace_io import save_trace  # noqa: E402
from src.v15 import config  # noqa: E402
from src.v15.common import (  # noqa: E402
    build_area_table,
    ensure_v15_path,
    predict_probabilities,
    prepare_fold,
    summarize_draws,
    write_json,
)
from src.v15.fit import diagnostics  # noqa: E402
from src.v15.metrics import metric_row  # noqa: E402
from src.v15.models import build_model  # noqa: E402


def subset_training(frame: pd.DataFrame, cells, fold: int, buffer_m: float) -> tuple[pd.DataFrame, int, int]:
    test = frame[frame["fold"] == fold].copy()
    train = frame[frame["fold"] != fold].copy()
    held_out = cells.loc[cells["id"].isin(test["id"]), "geometry"]
    union = shapely.union_all(held_out.to_numpy())
    points = shapely.points(train["centroid_x"].to_numpy(float), train["centroid_y"].to_numpy(float))
    distance = shapely.distance(union, points)
    keep = distance >= buffer_m
    removed = train.loc[~keep]
    return pd.concat([train.loc[keep], test], ignore_index=True), int(len(removed)), int(removed[config.Y_COL].sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buffers-km", nargs="+", type=float, default=[5.0, 10.0])
    parser.add_argument("--models", nargs="+", default=["M0_global", "M2_domain_distance"])
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--draws", type=int, default=config.SAMPLER["draws"])
    parser.add_argument("--tune", type=int, default=config.SAMPLER["tune"])
    parser.add_argument("--chains", type=int, default=config.SAMPLER["chains"])
    parser.add_argument("--cores", type=int, default=config.SAMPLER["cores"])
    parser.add_argument("--target-accept", type=float, default=config.SAMPLER["target_accept"])
    parser.add_argument("--seed", type=int, default=config.SAMPLER["random_seed"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    frame = pd.read_csv(config.V15 / "data" / "v15_model_frame.csv")
    cells, _ = build_area_table()
    status = []
    support_rows = []
    predictions = []
    for buffer_km in args.buffers_km:
        buffer_m = buffer_km * 1000.0
        for model_name in args.models:
            spec = config.MODELS[model_name]
            for fold in args.folds:
                tag = f"buffered_{buffer_km:g}km"
                trace_path = config.V15 / "traces" / tag / model_name / f"fold_{fold + 1}.nc"
                prediction_path = config.V15 / "predictions" / f"{tag}_{model_name}_fold_{fold + 1}.csv"
                train_frame, removed_cells, removed_positives = subset_training(frame, cells, fold, buffer_m)
                arrays = prepare_fold(train_frame, fold, spec, litho_source=spec.get("litho_source", "v14"))
                support = arrays["train"]
                for domain_index, domain in enumerate(config.DOMAINS):
                    mask = support["domain_idx"] == domain_index
                    support_rows.append(
                        {
                            "buffer_km": buffer_km,
                            "fold": fold + 1,
                            "model": model_name,
                            "domain": domain,
                            "train_cells": int(mask.sum()),
                            "train_positives": int(support["y"][mask].sum()),
                            "removed_cells": removed_cells,
                            "removed_positives": removed_positives,
                        }
                    )
                if trace_path.exists() and prediction_path.exists() and not args.overwrite:
                    result = {"buffer_km": buffer_km, "model": model_name, "fold": fold + 1, "status": "SKIPPED_EXISTING"}
                    status.append(result)
                    table = pd.read_csv(prediction_path)
                    table["buffer_km"] = buffer_km
                    table["model"] = model_name
                    predictions.append(table)
                    continue
                model = build_model(model_name, arrays, spec)
                started = time.perf_counter()
                with model:
                    trace = pm.sample(
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
                        "buffer_km": buffer_km,
                        "model": model_name,
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
                predictions.append(prediction)
                _, info = diagnostics(trace, model_name, fold)
                result = {
                    "buffer_km": buffer_km,
                    "model": model_name,
                    "fold": fold + 1,
                    "status": "PASS" if info["converged"] else "FAIL_CONVERGENCE",
                    "elapsed_seconds": elapsed,
                    "removed_training_cells": removed_cells,
                    "removed_training_positives": removed_positives,
                    "remaining_training_cells": arrays["train_size"],
                    "remaining_training_positives": int(arrays["y_train"].sum()),
                    **{key: info[key] for key in ("diverging", "max_rhat", "min_ess_bulk", "min_ess_tail")},
                }
                status.append(result)
                write_json(config.V15 / "models" / "buffered_fit_status.json", status)
                print(f"[buffered] {result}", flush=True)
    pd.DataFrame(support_rows).to_csv(
        ensure_v15_path(config.V15 / "diagnostics" / "buffered_training_support.csv"), index=False
    )
    metric_rows = []
    if predictions:
        combined = pd.concat(predictions, ignore_index=True)
        for (buffer_km, model_name), group in combined.groupby(["buffer_km", "model"]):
            metric_rows.append({"buffer_km": buffer_km, "model": model_name, **metric_row(group, "pooled")})
            for fold, fold_group in group.groupby("fold"):
                metric_rows.append(
                    {"buffer_km": buffer_km, "model": model_name, **metric_row(fold_group, f"fold_{int(fold)}")}
                )
    pd.DataFrame(metric_rows).to_csv(
        ensure_v15_path(config.V15 / "metrics" / "buffered_validation_metrics.csv"), index=False
    )
    write_json(config.V15 / "models" / "buffered_fit_status.json", status)
    if any(row.get("status") == "FAIL_CONVERGENCE" for row in status):
        raise SystemExit("At least one buffered fit failed convergence checks")


if __name__ == "__main__":
    main()
