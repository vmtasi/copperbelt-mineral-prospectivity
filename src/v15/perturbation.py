"""Positional-uncertainty label perturbation for V15 (no model refit)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402

from src.v15 import config  # noqa: E402
from src.v15.common import build_area_table, ensure_v15_path, write_json  # noqa: E402
from src.v15.metrics import log_score, roc_auc  # noqa: E402

POINTS = ROOT / "data" / "raw" / "usgs_sir2010_5090_t" / "CACB_deposits_utm.shp"


def load_points_and_cells():
    model = pd.read_csv(config.V15 / "data" / "v15_model_frame.csv")
    all_cells, _ = build_area_table()
    cells = model[["id"]].merge(all_cells, on="id", how="left", validate="one_to_one")
    cells["geometry"] = gpd.GeoSeries(
        [
            shapely.box(xmin, ymin, xmax, ymax)
            for xmin, ymin, xmax, ymax in zip(
                cells["xmin"], cells["ymin"], cells["xmax"], cells["ymax"]
            )
        ],
        crs="EPSG:32735",
    )
    cells = gpd.GeoDataFrame(cells, geometry="geometry", crs="EPSG:32735")
    id_to_cell = {int(cell_id): index for index, cell_id in enumerate(cells["id"])}
    points = gpd.read_file(POINTS).to_crs(32735)
    joined = gpd.sjoin(points, cells[["id", "geometry"]], how="left", predicate="within")
    joined = joined.rename(columns={"id": "grid_cell_id"}).drop(columns=["index_right"], errors="ignore")
    joined = joined.merge(
        model[["id", "deposit_present"]].rename(columns={"id": "grid_cell_id"}),
        on="grid_cell_id",
        how="left",
        validate="many_to_one",
    )
    joined["cell_index"] = joined["grid_cell_id"].map(id_to_cell)
    joined["assignment_status"] = np.select(
        [
            joined["grid_cell_id"].isna(),
            joined["deposit_present"].isna(),
            joined["deposit_present"].eq(1),
        ],
        ["outside modelled grid", "excluded complete-case cell", "used: model-positive cell"],
        default="not used: model-negative cell",
    )
    used = joined["assignment_status"].eq("used: model-positive cell")
    return points, cells, joined, used, joined.loc[used, "cell_index"].to_numpy(dtype=int)


def assign_new_cells(points: gpd.GeoSeries, cells: gpd.GeoDataFrame) -> np.ndarray:
    points_array = shapely.points(points.x.to_numpy(), points.y.to_numpy())
    contained = shapely.contains(cells.geometry.to_numpy()[None, :], points_array[:, None])
    assignment = np.empty(len(points_array), dtype=int)
    needs = ~contained.any(axis=1)
    assignment[~needs] = np.argmax(contained[~needs], axis=1)
    if needs.any():
        tree = shapely.STRtree(cells.geometry.to_numpy())
        assignment[needs] = tree.nearest(points_array[needs])
    return assignment


def main() -> None:
    points, cells, joined, used, base_cell = load_points_and_cells()
    joined.drop(columns=["geometry"], errors="ignore").to_csv(
        ensure_v15_path(config.V15 / "diagnostics" / "perturbation_point_assignment.csv"), index=False
    )
    moved_points = points.loc[used].reset_index(drop=True)
    base_positive = np.zeros(len(cells), dtype=int)
    base_positive[base_cell] = 1
    if int(base_positive.sum()) != 138:
        raise ValueError(f"Base point assignment gives {base_positive.sum()} positive cells, expected 138")

    rng = np.random.default_rng(config.PERTURB_SEED)
    records = []
    labels_by_simulation = []
    crossing = np.zeros(len(moved_points), dtype=float)
    for simulation in range(500):
        radius = 1000.0 * np.sqrt(rng.random(len(moved_points)))
        angle = rng.uniform(0.0, 2.0 * np.pi, len(moved_points))
        moved = gpd.GeoSeries(
            shapely.points(
                moved_points.geometry.x + radius * np.cos(angle),
                moved_points.geometry.y + radius * np.sin(angle),
            ),
            crs=moved_points.crs,
        )
        assigned = assign_new_cells(moved, cells)
        crossing += assigned != base_cell
        labels = np.zeros(len(cells), dtype=int)
        labels[assigned] = 1
        labels_by_simulation.append(labels)
        records.append(
            {
                "simulation": simulation + 1,
                "changed_cells": int((labels != base_positive).sum()),
                "positives": int(labels.sum()),
            }
        )

    perturbation = pd.DataFrame(records)
    perturbation.to_csv(
        ensure_v15_path(config.V15 / "diagnostics" / "label_perturbation_simulations_v1.csv"), index=False
    )
    model_rows = []
    model_summary = {}
    for path in sorted((config.V15 / "predictions").glob("oof_*.csv")):
        model_name = path.stem.removeprefix("oof_")
        predictions = pd.read_csv(path)
        aligned = pd.DataFrame({"id": cells["id"].astype(int)}).merge(
            predictions[["id", "prob"]], on="id", how="left", validate="one_to_one"
        )
        if aligned["prob"].isna().any():
            continue
        probabilities = aligned["prob"].to_numpy(float)
        for simulation, labels in enumerate(labels_by_simulation, start=1):
            model_rows.append(
                {
                    "model": model_name,
                    "simulation": simulation,
                    "positives": int(labels.sum()),
                    "auc": roc_auc(labels, probabilities),
                    "mean_log_score": float(log_score(labels, probabilities).mean()),
                }
            )
    model_table = pd.DataFrame(model_rows)
    if not model_table.empty:
        model_table.to_csv(
            ensure_v15_path(config.V15 / "diagnostics" / "label_perturbation_model_metrics_v1.csv"),
            index=False,
        )
        for model_name, group in model_table.groupby("model"):
            model_summary[model_name] = {
                "auc_mean": float(group["auc"].mean()),
                "auc_sd": float(group["auc"].std(ddof=1)),
                "auc_p05": float(group["auc"].quantile(0.05)),
                "auc_p95": float(group["auc"].quantile(0.95)),
                "mean_log_score_mean": float(group["mean_log_score"].mean()),
                "mean_log_score_sd": float(group["mean_log_score"].std(ddof=1)),
            }

    summary = {
        "source_points": int(len(points)),
        "used_points_in_positive_cells": int(used.sum()),
        "not_used_model_negative": int((joined["assignment_status"] == "not used: model-negative cell").sum()),
        "not_used_outside_modelled_grid": int((joined["assignment_status"] == "outside modelled grid").sum()),
        "base_positive_cells": int(base_positive.sum()),
        "simulations": 500,
        "radius_km": 1.0,
        "assignment": "full 5 km cell polygon, nearest-cell fallback for points outside the grid",
        "mean_changed_cells": float(perturbation["changed_cells"].mean()),
        "median_changed_cells": float(perturbation["changed_cells"].median()),
        "p95_changed_cells": float(perturbation["changed_cells"].quantile(0.95)),
        "mean_positives": float(perturbation["positives"].mean()),
        "point_edge_crossing_probability_mean": float(crossing.mean() / 500.0),
        "point_edge_crossing_probability_max": float(crossing.max() / 500.0),
        "uncertainty_assumption": (
            "The 1 km radius is the project's working positional-uncertainty assumption; the USGS "
            "table has no per-record uncertainty field. The experiment is label-churn sensitivity, "
            "not survey-grade relocation. It uses only the 184 records that lie in the 138 positive cells."
        ),
        "model_summary": model_summary,
    }
    write_json(config.V15 / "diagnostics" / "label_perturbation_summary_v1.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
