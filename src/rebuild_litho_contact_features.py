"""Rebuild the lithology-contact distance and tie diagnostics in QGIS.

Implemented quantity
--------------------
For each clipped modelling cell (EPSG:32735), compute the minimum Euclidean
polygon-to-line distance to every eligible internal lithological contact. The
outer study-area boundary is excluded by using the ``internal_contacts_only``
layer. Distance is zero when a cell touches or intersects a contact.

The class predictor is deliberately NOT assigned here. The source contact layer
represents each internal boundary once per adjoining polygon, so the nearest
distance is shared by two or more contact features carrying different
lithology classes. Choosing one side is a separate scientific decision. This
script exports the tied contact IDs and classes so that decision can be applied
without re-running the geometry.

Run inside QGIS (Python console or ``python-qgis``), for example:

    python rebuild_litho_contact_features.py \
        --cells "C:/path/to/Clipped.shp" \
        --contacts "C:/path/to/internal_contacts_only.gpkg" \
        --out-csv "data/derived/litho_contact_distance_v13.csv" \
        --out-gpkg "data/derived/litho_contact_distance_v13.gpkg"
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

from qgis.core import (
    QgsApplication,
    QgsCoordinateTransformContext,
    QgsFeature,
    QgsField,
    QgsVectorFileWriter,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QVariant


CRS_AUTHID = "EPSG:32735"
TIE_TOL_M = 1e-6


def load_layer(uri: str, name: str) -> QgsVectorLayer:
    layer = QgsVectorLayer(uri, name, "ogr")
    if not layer.isValid():
        raise RuntimeError(f"Could not load layer: {uri}")
    if layer.crs().authid() != CRS_AUTHID:
        raise RuntimeError(
            f"{name} has CRS {layer.crs().authid()}, expected {CRS_AUTHID}"
        )
    return layer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", required=True, help="Clipped cell polygon layer")
    parser.add_argument(
        "--contacts", required=True, help="Internal lithological contact line layer"
    )
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-gpkg", required=True)
    args = parser.parse_args()

    owns_app = QgsApplication.instance() is None
    app = QgsApplication([], False) if owns_app else QgsApplication.instance()
    if owns_app:
        app.initQgis()

    try:
        cells = load_layer(args.cells, "cells")
        contacts_layer = load_layer(args.contacts, "contacts")

        contacts = sorted(
            (
                int(feature["fid"]),
                int(feature["litho_class"]),
                feature.geometry(),
            )
            for feature in contacts_layer.getFeatures()
        )
        if not contacts:
            raise RuntimeError("No contact features found")

        started = time.time()
        rows = []
        for feature in cells.getFeatures():
            cell_id = int(feature["id"])
            geometry = feature.geometry()
            if geometry is None or geometry.isEmpty():
                raise RuntimeError(f"Empty cell geometry for id={cell_id}")
            if not geometry.isGeosValid():
                raise RuntimeError(f"Invalid cell geometry for id={cell_id}")

            candidates = [
                (geometry.distance(contact_geometry), fid, litho_class, contact_geometry)
                for fid, litho_class, contact_geometry in contacts
            ]
            minimum = min(candidate[0] for candidate in candidates)
            tied = sorted(
                (c for c in candidates if c[0] <= minimum + TIE_TOL_M),
                key=lambda c: (c[2], c[1]),
            )
            nearest = tied[0]
            point_on_cell = geometry.nearestPoint(nearest[3]).asPoint()
            point_on_contact = nearest[3].nearestPoint(geometry).asPoint()

            rows.append(
                {
                    "id": cell_id,
                    "distance_to_lithology_contact": minimum,
                    "representative_contact_fid": nearest[1],
                    "tie_count": len(tied),
                    "tied_contact_fids": ";".join(str(c[1]) for c in tied),
                    "tied_contact_classes": ";".join(
                        str(c[2]) for c in sorted(tied, key=lambda c: (c[2], c[1]))
                    ),
                    "representative_x": point_on_contact.x(),
                    "representative_y": point_on_contact.y(),
                    "feature_x": point_on_cell.x(),
                    "feature_y": point_on_cell.y(),
                    "cell_area_m2": geometry.area(),
                    "n_contacts_evaluated": len(contacts),
                }
            )

        rows.sort(key=lambda row: row["id"])
        if len({row["id"] for row in rows}) != len(rows):
            raise RuntimeError("Duplicate cell ids in output")

        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        memory = QgsVectorLayer(
            f"MultiPolygon?crs={CRS_AUTHID}", "litho_contact_distance", "memory"
        )
        provider = memory.dataProvider()
        provider.addAttributes(
            [
                QgsField("id", QVariant.Int),
                QgsField("distance_m", QVariant.Double),
                QgsField("representative_fid", QVariant.Int),
                QgsField("tie_count", QVariant.Int),
                QgsField("representative_x", QVariant.Double),
                QgsField("representative_y", QVariant.Double),
                QgsField("feature_x", QVariant.Double),
                QgsField("feature_y", QVariant.Double),
            ]
        )
        memory.updateFields()

        geometries = {int(f["id"]): f.geometry() for f in cells.getFeatures()}
        output_features = []
        for row in rows:
            output_feature = QgsFeature(memory.fields())
            output_feature.setGeometry(geometries[row["id"]])
            output_feature.setAttributes(
                [
                    row["id"],
                    row["distance_to_lithology_contact"],
                    row["representative_contact_fid"],
                    row["tie_count"],
                    row["representative_x"],
                    row["representative_y"],
                    row["feature_x"],
                    row["feature_y"],
                ]
            )
            output_features.append(output_feature)
        provider.addFeatures(output_features)
        memory.updateExtents()

        out_gpkg = Path(args.out_gpkg)
        out_gpkg.parent.mkdir(parents=True, exist_ok=True)
        if out_gpkg.exists():
            os.remove(out_gpkg)
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "GPKG"
        options.layerName = "litho_contact_distance"
        result = QgsVectorFileWriter.writeAsVectorFormatV3(
            memory,
            str(out_gpkg),
            QgsCoordinateTransformContext(),
            options,
        )
        if result[0] != QgsVectorFileWriter.NoError:
            raise RuntimeError(f"GeoPackage write failed: {result[1]}")

        print(
            f"Wrote {len(rows)} cells, {len(contacts)} contacts, "
            f"{time.time() - started:.1f}s"
        )
        print(f"CSV:  {out_csv}")
        print(f"GPKG: {out_gpkg}")
    finally:
        if owns_app:
            app.exitQgis()


if __name__ == "__main__":
    main()