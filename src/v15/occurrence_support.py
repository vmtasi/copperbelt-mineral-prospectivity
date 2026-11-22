"""Audit how the 187 USGS occurrence records relate to the 138 model positives."""

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
from src.v15.common import build_area_table, ensure_v15_path, write_json  # noqa: E402

POINTS = ROOT / "data" / "raw" / "usgs_sir2010_5090_t" / "CACB_deposits_utm.shp"


def main() -> None:
    points = gpd.read_file(POINTS).to_crs(32735)
    cells, _ = build_area_table()
    cells = cells.copy()
    cells["geometry"] = gpd.GeoSeries(
        [
            shapely.box(xmin, ymin, xmax, ymax)
            for xmin, ymin, xmax, ymax in zip(
                cells["xmin"], cells["ymin"], cells["xmax"], cells["ymax"]
            )
        ],
        crs=cells.crs,
    )
    joined = gpd.sjoin(points, cells[["id", "geometry"]], how="left", predicate="within")
    joined = joined.rename(columns={"id": "grid_cell_id"}).drop(columns=["index_right"], errors="ignore")
    model = pd.read_csv(
        config.V15 / "data" / "v15_model_frame.csv",
        usecols=["id", "deposit_present", "fold", "daly_domain"],
    ).rename(
        columns={
            "id": "grid_cell_id",
            "deposit_present": "model_label",
            "fold": "model_fold",
            "daly_domain": "model_domain",
        }
    )
    joined = joined.merge(model, on="grid_cell_id", how="left", validate="many_to_one")
    joined["in_complete_case_frame"] = joined["model_label"].notna()
    joined["in_model_positive_cell"] = joined["model_label"].eq(1)
    joined.drop(columns=["geometry"], errors="ignore").to_csv(
        ensure_v15_path(config.V15 / "diagnostics" / "occurrence_point_cell_support.csv"), index=False
    )

    complete = joined[joined["in_complete_case_frame"]].copy()
    positive = complete[complete["in_model_positive_cell"]].copy()
    negative = complete[~complete["in_model_positive_cell"]].copy()
    excluded = joined[~joined["in_complete_case_frame"]].copy()
    per_positive_cell = positive.groupby("grid_cell_id").size()
    labels = pd.read_csv(config.V14_TABLE)
    summary = {
        "source_points": int(len(points)),
        "points_in_complete_case_cells": int(len(complete)),
        "points_in_model_positive_cells": int(len(positive)),
        "points_in_model_negative_cells": int(len(negative)),
        "points_in_excluded_cells": int(len(excluded)),
        "unique_complete_case_cells_with_points": int(complete["grid_cell_id"].nunique()),
        "unique_model_positive_cells_with_points": int(positive["grid_cell_id"].nunique()),
        "model_positive_cells": int(pd.read_csv(config.V15 / "data" / "v15_model_frame.csv")["deposit_present"].sum()),
        "max_points_in_one_positive_cell": int(per_positive_cell.max()),
        "positive_cells_with_multiple_points": int((per_positive_cell > 1).sum()),
        "points_in_multi_point_positive_cells": int(per_positive_cell[per_positive_cell > 1].sum()),
        "v14_positive_rows_before_complete_case": int(labels["deposit_present"].sum()),
        "v14_positive_rows_after_complete_case": int(
            pd.read_csv(config.V15 / "data" / "v15_model_frame.csv")["deposit_present"].sum()
        ),
        "negative_cell_points": negative[["Name", "grid_cell_id", "model_fold"]].to_dict("records"),
        "excluded_cell_points": excluded[["Name", "grid_cell_id"]].to_dict("records"),
        "target_interpretation": (
            "The binary label is at-least-one compiled occurrence in a cell. USGS records are not "
            "one-to-one with positive cells: 184 records lie in 138 positive cells, two records lie in "
            "model-negative cells, and one lies in an excluded complete-case cell. Multiple records in "
            "one cell are not independent observations."
        ),
    }
    write_json(config.V15 / "diagnostics" / "occurrence_support_summary.json", summary)
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
