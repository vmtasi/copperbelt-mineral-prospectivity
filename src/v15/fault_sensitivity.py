"""Audit the stored fault-distance predictor against the surviving structure layer."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402

from src.v15 import config  # noqa: E402
from src.v15.common import ensure_v15_path, write_json  # noqa: E402
from src.v15.metrics import metric_row  # noqa: E402

STRUCTURES = Path(r"C:\Project\data for projects\CACB_Structures.gpkg")
TOLERANCE_M = 1.0


def main() -> None:
    full = pd.read_csv(config.V14_TABLE)
    model = pd.read_csv(config.V15 / "data" / "v15_model_frame.csv")
    structures = gpd.read_file(STRUCTURES).to_crs(32735)
    merged = shapely.union_all(structures.geometry.to_numpy())
    points = shapely.points(full["centroid_x"].to_numpy(float), full["centroid_y"].to_numpy(float))
    full["fault_distance_recomputed_m"] = shapely.distance(merged, points)
    full["fault_distance_abs_diff_m"] = (
        full["fault_distance_recomputed_m"] - full["distance_to_fault"]
    ).abs()
    full["fault_distance_reproduced"] = full["fault_distance_abs_diff_m"] < TOLERANCE_M
    annotation = model[["id", "fold", "daly_domain", config.Y_COL]].rename(
        columns={config.Y_COL: "complete_case_label"}
    )
    audit = full.merge(annotation, on="id", how="left", validate="one_to_one")
    audit["in_complete_case_frame"] = audit["fold"].notna()
    audit.to_csv(
        ensure_v15_path(config.V15 / "diagnostics" / "fault_distance_cell_audit.csv"), index=False
    )

    def group_summary(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
        return frame.groupby(by, dropna=False).agg(
            cells=("id", "size"),
            positive_cells=("complete_case_label", "sum"),
            stored_distance_median_m=("distance_to_fault", "median"),
            recomputed_distance_median_m=("fault_distance_recomputed_m", "median"),
            abs_difference_median_m=("fault_distance_abs_diff_m", "median"),
            abs_difference_max_m=("fault_distance_abs_diff_m", "max"),
        ).reset_index()

    group_summary(audit, ["fault_distance_reproduced"]).to_csv(
        ensure_v15_path(config.V15 / "diagnostics" / "fault_distance_reproducibility.csv"), index=False
    )
    nonreproduced = audit[~audit["fault_distance_reproduced"]].copy()
    group_summary(nonreproduced, ["in_complete_case_frame", "fold"]).to_csv(
        ensure_v15_path(config.V15 / "diagnostics" / "fault_distance_nonreproduced_by_fold.csv"), index=False
    )
    group_summary(nonreproduced, ["daly_domain"]).to_csv(
        ensure_v15_path(config.V15 / "diagnostics" / "fault_distance_nonreproduced_by_domain.csv"), index=False
    )

    rows = []
    for path in sorted((config.V15 / "predictions").glob("oof_*.csv")):
        name = path.stem.removeprefix("oof_")
        frame = pd.read_csv(path).merge(
            audit[["id", "fault_distance_reproduced"]], on="id", how="left", validate="one_to_one"
        )
        for subset, group in (
            ("all_complete_cases", frame),
            ("reproduced_1672", frame[frame["fault_distance_reproduced"]]),
            ("nonreproduced_227", frame[~frame["fault_distance_reproduced"]]),
        ):
            if group.empty:
                continue
            rows.append({"model": name, "subset": subset, **metric_row(group, subset)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(
        ensure_v15_path(config.V15 / "metrics" / "fault_distance_sensitivity_metrics.csv"), index=False
    )
    summary = {
        "structure_source": str(STRUCTURES),
        "structure_lines": int(len(structures)),
        "cells": int(len(audit)),
        "reproduced_within_1m": int(audit["fault_distance_reproduced"].sum()),
        "nonreproduced": int((~audit["fault_distance_reproduced"]).sum()),
        "nonreproduced_positive_cells": int(nonreproduced["complete_case_label"].sum()),
        "nonreproduced_not_in_complete_case_frame": int(nonreproduced["fold"].isna().sum()),
        "max_abs_difference_m": float(audit["fault_distance_abs_diff_m"].max()),
        "interpretation": (
            "The stored predictor is not reproducible from the surviving 126-line structure layer for 227 cells. "
            "The original fault geometry is not preserved, so this is a cartographic provenance limitation, "
            "not a missing-value problem and not a reason to alter the canonical complete-case frame."
        ),
    }
    write_json(config.V15 / "diagnostics" / "fault_distance_sensitivity_summary.json", summary)
    if not metrics.empty:
        print(metrics[["model", "subset", "n", "positives", "auc", "mean_log_score"]].to_string(index=False))
    print(nonreproduced.groupby(["fold", "daly_domain"], dropna=False).size().to_string())


if __name__ == "__main__":
    main()
