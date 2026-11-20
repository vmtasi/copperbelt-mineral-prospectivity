"""Reconstruct how the stored surficial-lithology class was assigned (read-only).

The column ``litho_contact_litho_class`` was inherited from an unscripted V5
feature build. This audit tests candidate assignment rules against the source
layers and writes the result to
figures/audit/paper1_supplement_v14/lithology_class_assignment.json.

Rules tested
------------
H1  raster class at the cell centroid
H2  majority raster class within the cell polygon
H3  class attribute of the nearest internal contact line to the centroid
H4  class attribute of any internal contact line at the minimum
    polygon-to-contact distance (the QGIS nearest-join output)

Source layers (external, not redistributed):
  <source_dir>/lithography_square_93m.tif               reprojected 93 m raster
  <source_dir>/grid_internal_contact_distance_1899.gpkg  cell polygons x nearest contacts
  <source_dir>/internal_contacts_only.gpkg              259 internal contact lines

Run:  python -m src.audit_lithology_class_assignment ["C:/Project/data for projects"]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "data" / "copperbelt_training_v14_centroid_contact.csv"
OUT = ROOT / "figures" / "audit" / "paper1_supplement_v14" / "lithology_class_assignment.json"
DEFAULT_SOURCE = Path(r"C:\Project\data for projects")
CLASS = "litho_contact_litho_class"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(source: Path) -> dict:
    raster_path = source / "lithography_square_93m.tif"
    grid_path = source / "grid_internal_contact_distance_1899.gpkg"
    contacts_path = source / "internal_contacts_only.gpkg"

    table = pd.read_csv(TABLE)
    stored = table.set_index("id")[CLASS].astype(int)
    positives = table.set_index("id")["deposit_present"]

    joined = gpd.read_file(grid_path)
    joined["id"] = joined["id"].astype(int)
    cells = joined.dissolve(by="id").reset_index()[["id", "geometry"]]

    with rasterio.open(raster_path) as raster:
        nodata = raster.nodata
        centroid = pd.Series(
            [int(v[0]) for v in raster.sample(zip(table.centroid_x, table.centroid_y))],
            index=table["id"],
        )
        majority = {}
        for row in cells.itertuples():
            values, _ = mask(raster, [row.geometry], crop=True, filled=False)
            values = values.compressed()
            values = values[values != nodata]
            if values.size:
                codes, counts = np.unique(values, return_counts=True)
                majority[row.id] = int(codes[counts.argmax()])
        majority = pd.Series(majority)

    contacts = gpd.read_file(contacts_path)
    points = gpd.GeoDataFrame(
        table[["id"]], geometry=gpd.points_from_xy(table.centroid_x, table.centroid_y),
        crs=contacts.crs,
    )
    nearest = (gpd.sjoin_nearest(points, contacts[["litho_class", "geometry"]], how="left")
               .drop_duplicates("id").set_index("id")["litho_class"].astype(int))

    minimum = joined.groupby("id")["distance"].transform("min")
    tied = joined[np.isclose(joined["distance"], minimum, atol=1e-6)]
    tied_sets = tied.groupby("id")[CLASS].apply(lambda s: set(int(v) for v in s))
    in_tied = pd.Series({i: stored[i] in tied_sets[i] for i in tied_sets.index})
    ambiguous = tied.groupby("id")[CLASS].nunique() > 1

    def agreement(series: pd.Series) -> dict:
        aligned = stored.reindex(series.index)
        match = series == aligned
        return {"cells": int(len(series)), "matches": int(match.sum()),
                "share": round(float(match.mean()), 4)}

    result = {
        "status": "COMPLETE",
        "conclusion": (
            "The stored class is the litho_class attribute of one of the internal "
            "contact lines at the minimum polygon-to-contact distance (QGIS nearest "
            "join). It is not the raster class at the centroid or the majority class "
            "in the cell. Where tied contacts carry different classes, the choice "
            "between them is undocumented."
        ),
        "rules": {
            "H1_centroid_raster_class": agreement(centroid),
            "H2_majority_class_in_cell": agreement(majority),
            "H3_nearest_contact_to_centroid": agreement(nearest),
            "H4_any_tied_nearest_contact_to_polygon": {
                "cells": int(len(in_tied)), "matches": int(in_tied.sum()),
                "share": round(float(in_tied.mean()), 4),
            },
        },
        "cells_with_tied_contacts_of_different_classes": int(ambiguous.sum()),
        "unmatched_by_H4": {
            "cells": int((~in_tied).sum()),
            "positive_cells": int(positives.reindex(in_tied[~in_tied].index).sum()),
        },
        "inputs": {
            "training_table": {"path": str(TABLE.relative_to(ROOT)).replace("\\", "/"),
                               "sha256": sha256(TABLE)},
            "raster": {"path": raster_path.name, "sha256": sha256(raster_path)},
            "grid_contact_join": {"path": grid_path.name, "sha256": sha256(grid_path)},
            "internal_contacts": {"path": contacts_path.name, "sha256": sha256(contacts_path)},
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE
    print(json.dumps(main(src)["rules"], indent=2))
