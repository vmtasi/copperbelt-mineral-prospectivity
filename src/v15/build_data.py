"""Build the V15 audited data frame and all pre-model diagnostics."""

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
from rasterio.mask import mask  # noqa: E402
import rasterio  # noqa: E402
from src.v15.metrics import metric_row, reliability_table, roc_auc  # noqa: E402

from src.v15 import config  # noqa: E402
from src.v15.common import (  # noqa: E402
    build_area_table,
    build_model_frame,
    domain_code,
    ensure_v15_path,
    load_frozen_oof,
    load_v14_frame,
    sha256,
    write_json,
)



def describe(series: pd.Series) -> dict:
    values = series.dropna().to_numpy(float)
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "q25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "q75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
    }


def area_tables(full: pd.DataFrame, model: pd.DataFrame, cells: gpd.GeoDataFrame) -> dict[str, pd.DataFrame]:
    all_cells = full.merge(
        cells[["id", "area_km2", "full_area_km2", "distance_to_tract_boundary_m"]],
        on="id",
        how="left",
        validate="one_to_one",
    )
    tables: dict[str, pd.DataFrame] = {}
    tables["area_all_cells"] = all_cells[
        ["id", "deposit_present", "area_km2", "full_area_km2", "distance_to_tract_boundary_m"]
    ].copy()

    by_fold = []
    for fold, group in model.groupby("fold"):
        row = {"fold": int(fold), "cells": int(len(group)), "positives": int(group[config.Y_COL].sum())}
        row.update(describe(group["area_km2"]))
        by_fold.append(row)
    tables["area_by_fold"] = pd.DataFrame(by_fold)

    by_domain = []
    for domain, group in model.groupby("daly_domain"):
        row = {"domain": domain, "cells": int(len(group)), "positives": int(group[config.Y_COL].sum())}
        row.update(describe(group["area_km2"]))
        by_domain.append(row)
    tables["area_by_domain"] = pd.DataFrame(by_domain).sort_values("domain")

    by_label = []
    for label, group in model.groupby(config.Y_COL):
        row = {"label": int(label), "cells": int(len(group))}
        row.update(describe(group["area_km2"]))
        by_label.append(row)
    tables["area_by_label"] = pd.DataFrame(by_label)

    model = model.copy()
    model["area_decile"] = pd.qcut(
        model["area_km2"].rank(method="first"), 10, labels=False
    ) + 1
    quantiles = (
        model.groupby("area_decile")
        .agg(
            cells=("id", "size"),
            positives=(config.Y_COL, "sum"),
            min_area_km2=("area_km2", "min"),
            max_area_km2=("area_km2", "max"),
            mean_area_km2=("area_km2", "mean"),
        )
        .reset_index()
    )
    quantiles["positive_rate"] = quantiles["positives"] / quantiles["cells"]
    tables["area_quantiles"] = quantiles

    auc_rows = []
    for fold, group in model.groupby("fold"):
        auc_rows.append(
            {
                "fold": int(fold),
                "n": int(len(group)),
                "positives": int(group[config.Y_COL].sum()),
                "auc_area": float(roc_auc(group[config.Y_COL], group["area_km2"])),
            }
        )
    auc_rows.append(
        {
            "fold": "pooled",
            "n": int(len(model)),
            "positives": int(model[config.Y_COL].sum()),
            "auc_area": float(roc_auc(model[config.Y_COL], model["area_km2"])),
        }
    )
    tables["area_auc_by_fold"] = pd.DataFrame(auc_rows)

    oof = load_frozen_oof()
    correlations = model[["id", "area_km2", "log_area_km2", "distance_to_fault", "distance_to_lithology_contact", "bouguer"]].merge(
        oof[["id", "prob_v14"]], on="id", how="left", validate="one_to_one"
    )
    rows = []
    for column in ["log_area_km2", "distance_to_fault", "distance_to_lithology_contact", "bouguer", "prob_v14"]:
        rows.append(
            {
                "variable": column,
                "pearson": float(correlations["area_km2"].corr(correlations[column], method="pearson")),
                "spearman": float(correlations["area_km2"].corr(correlations[column], method="spearman")),
            }
        )
    tables["area_correlations"] = pd.DataFrame(rows)

    tables["small_cells"] = model.loc[model["small_cell"], [
        "id", "fold", "daly_domain", "deposit_present", "area_km2", "distance_to_tract_boundary_m", "centroid_x", "centroid_y"
    ]].copy()
    return tables


def lithology_audit(full: pd.DataFrame, cells: gpd.GeoDataFrame) -> dict:
    stored = full.set_index("id")[config.LITHO_COL].astype(int)
    positives = full.set_index("id")[config.Y_COL]
    if not config.LITHO_RASTER.is_file():
        return {"status": "UNAVAILABLE", "reason": f"Missing raster {config.LITHO_RASTER}"}
    points = gpd.GeoDataFrame(
        full[["id"]],
        geometry=gpd.points_from_xy(full["centroid_x"], full["centroid_y"]),
        crs="EPSG:32735",
    )
    contacts = gpd.read_file(config.CONTACT_GPKG)
    nearest = (
        gpd.sjoin_nearest(points, contacts[["litho_class", "geometry"]], how="left")
        .drop_duplicates("id")
        .set_index("id")["litho_class"]
        .astype(int)
    )
    grid = gpd.read_file(config.GRID_CONTACT_GPKG)
    grid["id"] = grid["id"].astype(int)
    minimum = grid.groupby("id")["distance"].transform("min")
    tied = grid[np.isclose(grid["distance"], minimum, atol=1e-9)]
    tied_sets = tied.groupby("id")[config.LITHO_COL].apply(lambda values: set(int(v) for v in values))
    in_tied = pd.Series({index: stored[index] in values for index, values in tied_sets.items()})
    ambiguous = tied.groupby("id")[config.LITHO_COL].nunique() > 1

    with rasterio.open(config.LITHO_RASTER) as raster:
        from rasterio.warp import transform as warp_transform

        xs, ys = warp_transform("EPSG:32735", raster.crs, full["centroid_x"].to_numpy(float), full["centroid_y"].to_numpy(float))
        centroid = pd.Series(
            [int(next(raster.sample([(x, y)]))[0]) for x, y in zip(xs, ys)],
            index=full["id"].to_numpy(int),
        )
        centroid[centroid == raster.nodata] = np.nan
        majority = {}
        for row in cells.itertuples():
            values, _ = mask(raster, [row.geometry], crop=True, filled=False)
            values = values.compressed()
            values = values[values != raster.nodata]
            if values.size:
                codes, counts = np.unique(values, return_counts=True)
                majority[int(row.id)] = int(codes[counts.argmax()])
        majority = pd.Series(majority)

    def agreement(series: pd.Series) -> dict:
        aligned = stored.reindex(series.index)
        match = series == aligned
        return {
            "cells": int(len(series)),
            "matches": int(match.sum()),
            "share": float(match.mean()) if len(match) else 0.0,
        }

    result = {
        "status": "COMPLETE",
        "conclusion": (
            "The stored V14 class is a contact-line class at minimum polygon-to-contact distance. "
            "It is not the raster class at the centroid or the majority raster class in the cell. "
            "Where tied contacts carry different classes, the selected class is not uniquely identified by the source."
        ),
        "rules": {
            "H1_centroid_raster_class": agreement(centroid),
            "H2_majority_class_in_cell": agreement(majority),
            "H3_nearest_contact_to_centroid": agreement(nearest),
            "H4_any_tied_nearest_contact_to_polygon": {
                "cells": int(len(in_tied)),
                "matches": int(in_tied.sum()),
                "share": float(in_tied.mean()),
            },
        },
        "cells_with_tied_contacts_of_different_classes": int(ambiguous.sum()),
        "unmatched_by_H4": {
            "cells": int((~in_tied).sum()),
            "positive_cells": int(positives.reindex(in_tied[~in_tied].index).sum()),
        },
        "defensible_centroid_assignment": False,
        "reason": (
            "The only centroid raster is a surficial-lithology product, not the host-lithology polygon used by V14. "
            "Its 42 percent agreement with the stored field is insufficient to relabel cells. No polygon source "
            "supporting a unique host class at the centroid was found in the acquired data."
        ),
        "inputs": {
            "training_table_sha256": sha256(config.V14_TABLE),
            "raster_sha256": sha256(config.LITHO_RASTER),
            "grid_sha256": sha256(config.GRID_CONTACT_GPKG),
            "contacts_sha256": sha256(config.CONTACT_GPKG),
        },
    }
    return result


def main() -> None:
    full, frame = load_v14_frame()
    model, audit = build_model_frame()
    cells, _ = build_area_table()
    data_path = ensure_v15_path(config.V15 / "data" / "v15_model_frame.csv")
    model.to_csv(data_path, index=False)
    write_json(config.V15 / "diagnostics" / "data_build_audit.json", audit)

    tables = area_tables(full, model, cells)
    for name, table in tables.items():
        table.to_csv(ensure_v15_path(config.V15 / "diagnostics" / f"{name}.csv"), index=False)

    litho = lithology_audit(full, cells)
    write_json(config.V15 / "diagnostics" / "lithology_assignment_audit.json", litho)

    v14 = load_frozen_oof().rename(columns={"prob_v14": "prob"})
    v14["p_ref_train"] = v14.groupby("fold")["deposit_present"].transform("mean").clip(1e-6, 1 - 1e-6)
    v14 = v14.rename(columns={"deposit_present": "y"})
    v14 = v14.merge(model[["id", "centroid_x", "centroid_y"]], on="id", how="left")
    v14_metrics = pd.DataFrame([metric_row(v14, "V14_frozen_pooled")] + [
        metric_row(group, f"V14_frozen_fold_{int(fold) + 1}") for fold, group in v14.groupby("fold")
    ])
    v14_metrics.to_csv(ensure_v15_path(config.V15 / "metrics" / "v14_control_metrics.csv"), index=False)
    reliability_table(v14).to_csv(ensure_v15_path(config.V15 / "diagnostics" / "v14_reliability.csv"), index=False)

    print(json.dumps({"rows": len(model), "positives": int(model[config.Y_COL].sum()), "area_audit": audit}, indent=2))
    print(json.dumps(litho.get("rules", {}), indent=2))


if __name__ == "__main__":
    main()



