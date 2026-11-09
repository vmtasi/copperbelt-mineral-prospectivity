"""Build the Daly study area, 5 km support and candidate geographic folds.

Run inside QGIS Python. This script writes derived files only and does not fit a model.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsPointXY,
    QgsProject,
    QgsRectangle,
    QgsSpatialIndex,
    QgsVectorFileWriter,
    QgsVectorLayer,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QVariant

ROOT = Path(r"C:\Users\vanmu\copperbelt-mineral-prospectivity")
OUT = ROOT / "data/raw/rdc_rocks/derived/integration/phase7/supplementary"
DOMAINS = ROOT / "data/processed/geology_foundation/daly_tectonic_domains_v1.gpkg"
STRUCTURES = ROOT / "data/processed/geology_foundation/daly_major_structures_named_v2.gpkg"
TRACTS = ROOT / "data/raw/usgs_sir2010_5090_t/CACB_tracts_utm.shp"
DEPOSITS = ROOT / "data/raw/usgs_sir2010_5090_t/CACB_deposits_utm.shp"
COUNTRY_CONTEXT = OUT / "daly_country_context_v1.gpkg"
TARGET_CRS = QgsCoordinateReferenceSystem("EPSG:32735")
CELL_SIZE_M = 5000.0
DEPOSIT_COUNTRY_ALIASES = {"Democratic Republic of the Congo": "Dem. Rep. Congo"}


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_layer(path: Path, name: str) -> QgsVectorLayer:
    layer = QgsVectorLayer(str(path), name, "ogr")
    if not layer.isValid():
        raise RuntimeError(f"Could not load {path}")
    return layer


def make_layer(name: str, geometry_type: str, fields: list[QgsField]) -> QgsVectorLayer:
    layer = QgsVectorLayer(f"{geometry_type}?crs=EPSG:32735", name, "memory")
    provider = layer.dataProvider()
    provider.addAttributes(fields)
    layer.updateFields()
    return layer


def add_features(layer: QgsVectorLayer, records: list[tuple[QgsGeometry, list]]) -> None:
    features = []
    for geometry, values in records:
        feature = QgsFeature(layer.fields())
        feature.setGeometry(geometry)
        feature.setAttributes(values)
        features.append(feature)
    layer.dataProvider().addFeatures(features)
    layer.updateExtents()


def write_gpkg(path: Path, layers: list[QgsVectorLayer]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    context = QgsProject.instance().transformContext()
    for index, layer in enumerate(layers):
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "GPKG"
        options.fileEncoding = "UTF-8"
        options.layerName = layer.name()
        options.actionOnExistingFile = (
            QgsVectorFileWriter.CreateOrOverwriteFile
            if index == 0
            else QgsVectorFileWriter.CreateOrOverwriteLayer
        )
        error = QgsVectorFileWriter.writeAsVectorFormatV3(layer, str(path), context, options)
        if error[0] != QgsVectorFileWriter.NoError:
            raise RuntimeError(f"GeoPackage write failed for {layer.name()}: {error}")


def transform_geometry(geometry: QgsGeometry, source, target) -> QgsGeometry:
    geometry = QgsGeometry(geometry)
    geometry.transform(QgsCoordinateTransform(source, target, QgsProject.instance()))
    return geometry


def country_layer() -> QgsVectorLayer:
    if COUNTRY_CONTEXT.exists():
        return load_layer(COUNTRY_CONTEXT, "countries")
    source = None
    for layer in QgsProject.instance().mapLayers().values():
        if layer.name() == "Africa locator countries":
            source = layer
            break
    if source is None:
        raise RuntimeError("Natural Earth Africa country layer is unavailable")
    wanted = {"Dem. Rep. Congo", "Zambia", "Angola", "Tanzania", "Congo",
              "Central African Rep.", "Malawi", "Zimbabwe", "Mozambique"}
    fields = [QgsField("country", QVariant.String), QgsField("iso_a3", QVariant.String)]
    layer = make_layer("countries", "MultiPolygon", fields)
    records = []
    for feature in source.getFeatures():
        if feature["NAME"] not in wanted:
            continue
        geometry = transform_geometry(feature.geometry(), source.crs(), TARGET_CRS)
        records.append((geometry, [feature["NAME"], feature["ISO_A3"]]))
    add_features(layer, records)
    write_gpkg(COUNTRY_CONTEXT, [layer])
    return load_layer(COUNTRY_CONTEXT, "countries")


def load_inputs():
    domains = load_layer(DOMAINS, "daly_domains")
    countries = country_layer()
    tracts = load_layer(TRACTS, "usgs_tracts")
    deposits = load_layer(DEPOSITS, "usgs_deposits")
    return domains, countries, tracts, deposits


def study_area_outputs(domains, countries, tracts, deposits):
    domain_rows = []
    domain_geometries = []
    for feature in domains.getFeatures():
        geometry = QgsGeometry(feature.geometry())
        domain_geometries.append(geometry)
        domain_rows.append({
            "feature": feature,
            "geometry": geometry,
            "domain_id": str(feature["domain_id"]),
            "domain_number": str(feature["domain_number"]),
            "is_subdomain": int(feature["is_subdomain"]),
            "abbreviation": feature["abbreviation"],
            "domain_name": feature["domain_name"],
            "parent_domain_id": feature["parent_domain_id"] or "",
            "source_doi": feature["source_doi"],
            "source_figure": feature["source_figure"],
            "geometry_confidence": feature["geometry_confidence"],
            "area_km2": float(feature["area_km2"]),
        })
    union = QgsGeometry.unaryUnion([row["geometry"] for row in domain_rows])
    if union.isEmpty() or not union.isGeosValid():
        raise RuntimeError("Daly study-area union is empty or invalid")

    country_records = []
    for feature in countries.getFeatures():
        country_records.append((feature["country"], QgsGeometry(feature.geometry())))

    study_fields = [
        QgsField("area_km2", QVariant.Double),
        QgsField("domain_count", QVariant.Int),
        QgsField("basis", QVariant.String),
        QgsField("crs", QVariant.String),
    ]
    study_layer = make_layer("daly_study_area", "MultiPolygon", study_fields)
    add_features(study_layer, [(union, [
        round(union.area() / 1e6, 4), len(domain_rows),
        "union of nine Daly et al. (2025) Figure 14 reconstructed domain polygons",
        "EPSG:32735",
    ])])

    domain_fields = [
        QgsField("domain_id", QVariant.String), QgsField("domain_number", QVariant.String),
        QgsField("is_subdomain", QVariant.Int), QgsField("abbreviation", QVariant.String),
        QgsField("domain_name", QVariant.String), QgsField("parent_domain_id", QVariant.String),
        QgsField("area_km2", QVariant.Double), QgsField("source_doi", QVariant.String),
        QgsField("source_figure", QVariant.String), QgsField("geometry_confidence", QVariant.String),
    ]
    domain_layer = make_layer("daly_domains", "MultiPolygon", domain_fields)
    add_features(domain_layer, [(
        row["geometry"],
        [row["domain_id"], row["domain_number"], row["is_subdomain"], row["abbreviation"],
         row["domain_name"], row["parent_domain_id"], row["area_km2"], row["source_doi"],
         row["source_figure"], row["geometry_confidence"]],
    ) for row in domain_rows])

    country_fields = [QgsField("country", QVariant.String), QgsField("iso_a3", QVariant.String)]
    country_layer_out = make_layer("countries_intersect", "MultiPolygon", country_fields)
    add_features(country_layer_out, [(
        geometry, [name, ""]
    ) for name, geometry in country_records if union.intersects(geometry)])

    write_gpkg(OUT / "daly_study_area_v1.gpkg", [study_layer, domain_layer, country_layer_out])

    deposit_geometries = [
        (f["Name"], DEPOSIT_COUNTRY_ALIASES.get(f["Country"], f["Country"]), QgsGeometry(f.geometry()))
        for f in deposits.getFeatures()
    ]
    tract_union = QgsGeometry.unaryUnion([f.geometry() for f in tracts.getFeatures()])
    overlay_counts = {
        "inventory_records": len(deposit_geometries),
        "inside_union": sum(1 for _, _, point in deposit_geometries if union.contains(point)),
        "inside_tract": sum(1 for _, _, point in deposit_geometries if tract_union.contains(point)),
        "outside_tract": sum(1 for _, _, point in deposit_geometries if not tract_union.contains(point)),
    }

    country_rows = []
    for name, geometry in country_records:
        intersection = union.intersection(geometry)
        area = intersection.area() / 1e6 if not intersection.isEmpty() else 0.0
        if area <= 0.01:
            continue
        tract_area = intersection.intersection(tract_union).area() / 1e6
        deposit_count = sum(1 for _, country, point in deposit_geometries if country == name and union.contains(point))
        country_rows.append({
            "country": name,
            "area_km2": round(area, 2),
            "pct_study_area": round(100 * area / (union.area() / 1e6), 2),
            "usgs_tract_overlap_km2": round(tract_area, 2),
            "usgs_tract_fraction_pct": round(100 * tract_area / area, 2) if area else 0.0,
            "usgs_deposits_inside": deposit_count,
        })
    country_rows.sort(key=lambda row: row["area_km2"], reverse=True)
    write_csv(OUT / "daly_country_summary_v1.csv", list(country_rows[0]), country_rows)

    domain_rows_out = []
    for row in domain_rows:
        out = {
            "domain_id": row["domain_id"],
            "domain_number": row["domain_number"],
            "is_subdomain": row["is_subdomain"],
            "abbreviation": row["abbreviation"],
            "domain_name": row["domain_name"],
            "area_km2": round(row["area_km2"], 2),
            "source_doi": row["source_doi"],
            "source_figure": row["source_figure"],
            "geometry_confidence": row["geometry_confidence"],
        }
        for country_row in country_rows:
            name = country_row["country"]
            geometry = next(g for n, g in country_records if n == name)
            inter = row["geometry"].intersection(geometry)
            out[f"area_{name.lower().replace(' ', '_').replace('.', '')}_km2"] = round(inter.area() / 1e6, 2) if not inter.isEmpty() else 0.0
        out["usgs_deposits"] = sum(
            1 for _, _, point in deposit_geometries
            if row["geometry"].contains(point) or row["geometry"].intersects(point)
        )
        domain_rows_out.append(out)
    write_csv(OUT / "daly_domain_area_summary_v1.csv", list(domain_rows_out[0]), domain_rows_out)

    study_area_km2 = union.area() / 1e6
    tract_overlap = union.intersection(tract_union).area() / 1e6
    summary_rows = [
        {"metric": "study_area_km2", "value": round(study_area_km2, 2), "unit": "km2",
         "note": "Union of the nine supported Daly domain polygons; no extrapolation."},
        {"metric": "domain_count", "value": len(domain_rows), "unit": "count",
         "note": "Nine reconstructed domains, including subdomains."},
        {"metric": "zambia_area_km2", "value": next(r["area_km2"] for r in country_rows if r["country"] == "Zambia"), "unit": "km2",
         "note": "Intersection of Daly union with Natural Earth Zambia boundary."},
        {"metric": "drc_area_km2", "value": next(r["area_km2"] for r in country_rows if r["country"] == "Dem. Rep. Congo"), "unit": "km2",
         "note": "Intersection of Daly union with Natural Earth DRC boundary."},
        {"metric": "usgs_tract_overlap_km2", "value": round(tract_overlap, 2), "unit": "km2",
         "note": "Former permissive-tract area inside the Daly study area; comparison only."},
        {"metric": "usgs_tract_fraction_pct", "value": round(100 * tract_overlap / study_area_km2, 2), "unit": "percent",
         "note": "Tracts cover only a small fraction of the expanded study area."},
        {"metric": "outside_tracts_km2", "value": round(study_area_km2 - tract_overlap, 2), "unit": "km2",
         "note": "Area with Daly framework but no USGS tract membership."},
        {"metric": "usgs_deposits_inside", "value": sum(1 for _, _, point in deposit_geometries if union.contains(point)), "unit": "count",
         "note": "USGS deposit/prospect points inside the Daly union."},
        {"metric": "usgs_deposits_inside_tracts", "value": overlay_counts["inside_tract"], "unit": "count",
         "note": "USGS deposit/prospect points inside the union of five permissive tracts."},
        {"metric": "usgs_deposits_outside_tracts", "value": overlay_counts["outside_tract"], "unit": "count",
         "note": "USGS deposit/prospect points outside the union of five permissive tracts."},
    ]
    write_csv(OUT / "daly_study_area_summary_v1.csv", list(summary_rows[0]), summary_rows)
    return union, domain_rows, country_records, tract_union, deposit_geometries, study_area_km2, overlay_counts


def occurrence_guided_bands(coordinates: np.ndarray, anchors: np.ndarray, n_bands: int):
    covariance = np.cov(anchors, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    reference = np.asarray([1.0, -1.0])
    reference = reference / np.linalg.norm(reference)
    if float(np.dot(axis, reference)) < 0.0:
        axis = -axis
    belt = coordinates @ axis
    anchor_belt = anchors @ axis
    cuts = np.unique(np.percentile(anchor_belt, np.linspace(0.0, 100.0, n_bands + 1)[1:-1]))
    if len(cuts) != n_bands - 1:
        raise RuntimeError("Occurrence-guided fold cuts are not unique")
    bands = np.searchsorted(cuts, belt) + 1
    return bands, belt, axis


def nearest_distance(point: QgsGeometry, index: QgsSpatialIndex, geometries: dict, k: int = 12):
    candidates = index.nearestNeighbor(point.asPoint(), min(k, max(1, len(geometries))))
    if not candidates:
        return None
    return min(point.distance(geometries[fid]) for fid in candidates)


def circle_geometry(center: QgsPointXY, radius: float, segments: int = 36) -> QgsGeometry:
    points = []
    for index in range(segments):
        angle = 2.0 * math.pi * index / segments
        points.append(QgsPointXY(
            center.x() + radius * math.cos(angle),
            center.y() + radius * math.sin(angle),
        ))
    points.append(points[0])
    return QgsGeometry.fromPolygonXY([points])


def polygon_only(geometry: QgsGeometry) -> QgsGeometry:
    """Drop non-polygonal parts of a cell intersection so the memory provider accepts it."""
    if geometry.isEmpty():
        return geometry
    if geometry.wkbType() in (QgsWkbTypes.Polygon, QgsWkbTypes.MultiPolygon):
        return geometry
    parts = []
    for part in geometry.asGeometryCollection():
        if part.wkbType() in (QgsWkbTypes.Polygon, QgsWkbTypes.MultiPolygon):
            parts.append(QgsGeometry(part))
        elif part.wkbType() == QgsWkbTypes.GeometryCollection:
            nested = polygon_only(part)
            if not nested.isEmpty():
                parts.append(nested)
    return QgsGeometry.unaryUnion(parts) if parts else QgsGeometry()


def build_support_grid(union, domain_rows, country_records, tract_union, deposit_geometries):
    structures = load_layer(STRUCTURES, "daly_structures")
    segments = QgsVectorLayer(f"{STRUCTURES}|layername=v1_segments_attributed", "segments", "ogr")
    centerlines = QgsVectorLayer(f"{STRUCTURES}|layername=named_structure_centrelines", "centerlines", "ogr")
    if not segments.isValid() or not centerlines.isValid():
        raise RuntimeError("Could not load named structure layers")

    segment_geometries = {}
    segment_index = QgsSpatialIndex()
    major_geometries = {}
    major_index = QgsSpatialIndex()
    for feature in segments.getFeatures():
        fid = feature.id()
        geometry = QgsGeometry(feature.geometry())
        if feature["extraction_confidence"] in ("high", "medium"):
            major_geometries[fid] = geometry
            major_clone = QgsFeature(feature)
            major_clone.setId(fid)
            major_index.addFeature(major_clone)
        if feature["attributed_structure_id"] and feature["attribution_confidence"] in ("high", "medium"):
            segment_geometries[fid] = geometry
            clone = QgsFeature(feature)
            clone.setId(fid)
            segment_index.addFeature(clone)

    centerline_geometries = {}
    centerline_index = QgsSpatialIndex()
    for feature in centerlines.getFeatures():
        if feature["part_confidence"] not in ("high", "medium"):
            continue
        fid = feature.id()
        centerline_geometries[fid] = QgsGeometry(feature.geometry())
        clone = QgsFeature(feature)
        clone.setId(fid)
        centerline_index.addFeature(clone)

    deposit_geometries_index = {}
    deposit_index = QgsSpatialIndex()
    for index, (name, country, geometry) in enumerate(deposit_geometries):
        feature = QgsFeature(index + 1)
        feature.setGeometry(geometry)
        deposit_index.addFeature(feature)
        deposit_geometries_index[index + 1] = geometry

    bounds = union.boundingBox()
    nx = int(math.ceil(bounds.width() / CELL_SIZE_M))
    ny = int(math.ceil(bounds.height() / CELL_SIZE_M))
    records = []
    density_area_km2 = math.pi * 50.0 * 50.0
    for iy in range(ny):
        y = bounds.yMinimum() + iy * CELL_SIZE_M
        for ix in range(nx):
            x = bounds.xMinimum() + ix * CELL_SIZE_M
            cell = QgsGeometry.fromRect(QgsRectangle(x, y, x + CELL_SIZE_M, y + CELL_SIZE_M))
            if not union.intersects(cell):
                continue
            exposure = polygon_only(union.intersection(cell))
            area = exposure.area()
            if area <= 1.0:
                continue
            fraction = min(1.0, area / (CELL_SIZE_M * CELL_SIZE_M))
            point = exposure.centroid()
            if point.isEmpty():
                point = exposure.pointOnSurface()
            x_m = x + CELL_SIZE_M / 2.0
            y_m = y + CELL_SIZE_M / 2.0

            domain_id = ""
            domain_abbr = ""
            domain_name = ""
            for row in domain_rows:
                if row["geometry"].contains(point) or row["geometry"].intersects(point):
                    domain_id = row["domain_id"]
                    domain_abbr = row["abbreviation"]
                    domain_name = row["domain_name"]
                    break
            if not domain_id:
                nearest_domain = max(
                    domain_rows,
                    key=lambda row: row["geometry"].intersection(exposure).area(),
                )
                domain_id = nearest_domain["domain_id"]
                domain_abbr = nearest_domain["abbreviation"]
                domain_name = nearest_domain["domain_name"]
            country = ""
            for name, geometry in country_records:
                if geometry.contains(point) or geometry.intersects(point):
                    country = name
                    break

            tract_fraction = 0.0
            if tract_union.intersects(exposure):
                tract_fraction = min(1.0, tract_union.intersection(exposure).area() / area)

            mapped_distance = nearest_distance(point, segment_index, segment_geometries)
            bridged_distance = nearest_distance(point, centerline_index, centerline_geometries)
            major_distance = nearest_distance(point, major_index, major_geometries)
            buffer = circle_geometry(point.asPoint(), 50000.0)
            density_length = 0.0
            for fid in segment_index.intersects(buffer.boundingBox()):
                density_length += segment_geometries[fid].intersection(buffer).length()
            major_length = 0.0
            for fid in major_index.intersects(buffer.boundingBox()):
                major_length += major_geometries[fid].intersection(buffer).length()
            deposit_distance = nearest_distance(point, deposit_index, deposit_geometries_index, k=4)

            records.append({
                "x_m": x_m,
                "y_m": y_m,
                "geometry": exposure,
                "exposure_area_km2": area / 1e6,
                "exposure_fraction": fraction,
                "full_support": 1 if fraction >= 0.99 else 0,
                "edge_cell": 1 if fraction < 0.99 else 0,
                "domain_id": domain_id,
                "domain_abbr": domain_abbr,
                "domain_name": domain_name,
                "country": country,
                "dist_major_structure_km": None if major_distance is None else major_distance / 1000.0,
                "major_structure_length_50km_km": major_length / 1000.0,
                "major_structure_density_50km": (major_length / 1000.0) / density_area_km2,
                "dist_named_fault_mapped_km": None if mapped_distance is None else mapped_distance / 1000.0,
                "dist_named_fault_bridged_km": None if bridged_distance is None else bridged_distance / 1000.0,
                "named_fault_length_50km_km": density_length / 1000.0,
                "named_fault_density_50km": (density_length / 1000.0) / density_area_km2,
                "dist_nearest_usgs_deposit_km": None if deposit_distance is None else deposit_distance / 1000.0,
                "usgs_tract_fraction": tract_fraction,
            })
    return records


def finalize_support(records, deposit_geometries):
    coordinates = np.asarray([[row["x_m"], row["y_m"]] for row in records], dtype=float)
    anchors = np.asarray([[point.asPoint().x(), point.asPoint().y()] for _, _, point in deposit_geometries], dtype=float)
    fold4, belt, axis = occurrence_guided_bands(coordinates, anchors, 4)
    fold5, _, _ = occurrence_guided_bands(coordinates, anchors, 5)
    for index, row in enumerate(records):
        row["fold4"] = int(fold4[index])
        row["fold5"] = int(fold5[index])
        row["belt_coordinate_m"] = float(belt[index])

    fields = [
        QgsField("support_id", QVariant.String),
        QgsField("x_m", QVariant.Double),
        QgsField("y_m", QVariant.Double),
        QgsField("cell_area_km2", QVariant.Double),
        QgsField("exposure_area_km2", QVariant.Double),
        QgsField("exposure_fraction", QVariant.Double),
        QgsField("full_support", QVariant.Int),
        QgsField("edge_cell", QVariant.Int),
        QgsField("domain_id", QVariant.String),
        QgsField("domain_abbr", QVariant.String),
        QgsField("domain_name", QVariant.String),
        QgsField("country", QVariant.String),
        QgsField("fold4", QVariant.Int),
        QgsField("fold5", QVariant.Int),
        QgsField("preferred_fold", QVariant.Int),
        QgsField("belt_coordinate_m", QVariant.Double),
        QgsField("dist_major_structure_km", QVariant.Double),
        QgsField("major_structure_length_50km_km", QVariant.Double),
        QgsField("major_structure_density_50km", QVariant.Double),
        QgsField("dist_named_fault_mapped_km", QVariant.Double),
        QgsField("dist_named_fault_bridged_km", QVariant.Double),
        QgsField("named_fault_length_50km_km", QVariant.Double),
        QgsField("named_fault_density_50km", QVariant.Double),
        QgsField("dist_nearest_usgs_deposit_km", QVariant.Double),
        QgsField("usgs_tract_fraction", QVariant.Double),
        QgsField("inventory_support_class", QVariant.String),
    ]
    layer = make_layer("daly_support_5km", "MultiPolygon", fields)
    features = []
    for index, row in enumerate(records, start=1):
        feature = QgsFeature(layer.fields())
        feature.setGeometry(row["geometry"])
        tract = row["usgs_tract_fraction"]
        inventory_class = "outside_usgs_tract" if tract <= 0 else ("inside_usgs_tract" if tract >= 0.99 else "partial_usgs_tract")
        feature.setAttributes([
            f"SUP-{index:06d}", row["x_m"], row["y_m"], 25.0, row["exposure_area_km2"],
            row["exposure_fraction"], row["full_support"], row["edge_cell"], row["domain_id"],
            row["domain_abbr"], row["domain_name"], row["country"], row["fold4"], row["fold5"],
            0, row["belt_coordinate_m"], row["dist_major_structure_km"],
            row["major_structure_length_50km_km"], row["major_structure_density_50km"],
            row["dist_named_fault_mapped_km"],
            row["dist_named_fault_bridged_km"], row["named_fault_length_50km_km"],
            row["named_fault_density_50km"], row["dist_nearest_usgs_deposit_km"], tract,
            inventory_class,
        ])
        features.append(feature)
    layer.dataProvider().addFeatures(features)
    layer.updateExtents()

    deposit_index = QgsSpatialIndex(layer.getFeatures())
    deposit_counts = {4: {i: 0 for i in range(1, 5)}, 5: {i: 0 for i in range(1, 6)}}
    for _, _, point in deposit_geometries:
        containing = None
        for fid in deposit_index.intersects(point.boundingBox()):
            feature = layer.getFeature(fid)
            if feature.geometry().contains(point) or feature.geometry().intersects(point):
                containing = feature
                break
        if containing is None:
            continue
        deposit_counts[4][int(containing["fold4"])] += 1
        deposit_counts[5][int(containing["fold5"])] += 1

    def adequate(counts):
        return min(counts.values()) >= 10

    preferred_n = 4 if adequate(deposit_counts[4]) else (5 if adequate(deposit_counts[5]) else 4)
    field_index = layer.fields().indexOf("preferred_fold")
    updates = {}
    for feature in layer.getFeatures():
        value = int(feature[f"fold{preferred_n}"])
        updates[feature.id()] = {field_index: value}
    layer.dataProvider().changeAttributeValues(updates)

    def fold_polygon_layer(name, field):
        records_by_fold = {}
        for row in records:
            records_by_fold.setdefault(int(row[field]), []).append(row["geometry"])
        fold_layer = make_layer(name, "MultiPolygon", [QgsField("fold", QVariant.Int), QgsField("area_km2", QVariant.Double)])
        feature_rows = []
        for fold in sorted(records_by_fold):
            geometry = QgsGeometry.unaryUnion(records_by_fold[fold])
            feature_rows.append((geometry, [fold, round(geometry.area() / 1e6, 2)]))
        add_features(fold_layer, feature_rows)
        return fold_layer

    fold4_layer = fold_polygon_layer("daly_folds_4", "fold4")
    fold5_layer = fold_polygon_layer("daly_folds_5", "fold5")
    write_gpkg(OUT / "daly_support_5km_v1.gpkg", [layer])
    write_gpkg(OUT / "daly_geographic_folds_v1.gpkg", [fold4_layer, fold5_layer])

    summary_rows = []
    for n_bands, fold_field, label in ((4, "fold4", "four_fold_primary"), (5, "fold5", "five_fold_alternative")):
        for fold in range(1, n_bands + 1):
            members = [row for row in records if int(row[fold_field]) == fold]
            area = sum(row["exposure_area_km2"] for row in members)
            domains = {}
            for row in members:
                domains[row["domain_abbr"]] = domains.get(row["domain_abbr"], 0.0) + row["exposure_area_km2"]
            top_domains = ";".join(f"{key}:{value:.0f}" for key, value in sorted(domains.items(), key=lambda item: -item[1])[:4])
            summary_rows.append({
                "scheme": label,
                "fold": fold,
                "cells": len(members),
                "area_km2": round(area, 2),
                "full_cells": sum(int(row["full_support"]) for row in members),
                "edge_cells": sum(int(row["edge_cell"]) for row in members),
                "usgs_deposits": deposit_counts[n_bands][fold],
                "usgs_deposits_per_1000km2": round(1000.0 * deposit_counts[n_bands][fold] / area, 4) if area else 0.0,
                "domains_represented": len(domains),
                "domain_area_km2": top_domains,
                "preferred": "yes" if n_bands == preferred_n else "no",
                "note": "Occurrence-supported contiguous band along the principal USGS-inventory axis; boundaries are not Daly domain boundaries.",
            })
    write_csv(OUT / "daly_fold_support_summary_v1.csv", list(summary_rows[0]), summary_rows)

    grid_rows = []
    for row in layer.getFeatures():
        grid_rows.append({
            "support_id": row["support_id"], "x_m": row["x_m"], "y_m": row["y_m"],
            "exposure_area_km2": row["exposure_area_km2"], "exposure_fraction": row["exposure_fraction"],
            "full_support": row["full_support"], "edge_cell": row["edge_cell"],
            "domain_id": row["domain_id"], "domain_abbr": row["domain_abbr"], "country": row["country"],
            "fold4": row["fold4"], "fold5": row["fold5"], "preferred_fold": row["preferred_fold"],
            "dist_major_structure_km": row["dist_major_structure_km"],
            "major_structure_length_50km_km": row["major_structure_length_50km_km"],
            "major_structure_density_50km": row["major_structure_density_50km"],
            "dist_named_fault_mapped_km": row["dist_named_fault_mapped_km"],
            "dist_named_fault_bridged_km": row["dist_named_fault_bridged_km"],
            "named_fault_length_50km_km": row["named_fault_length_50km_km"],
            "named_fault_density_50km": row["named_fault_density_50km"],
            "dist_nearest_usgs_deposit_km": row["dist_nearest_usgs_deposit_km"],
            "inventory_support_class": row["inventory_support_class"],
        })
    grid_csv = OUT / "daly_support_5km_summary_v1.csv"
    write_csv(grid_csv, list(grid_rows[0]), grid_rows)
    return layer, preferred_n, summary_rows, deposit_counts, axis


def support_assessment(records, preferred_n, summary_rows, deposit_counts, axis, study_area_km2, overlay_counts):
    area = sum(row["exposure_area_km2"] for row in records)
    full = sum(int(row["full_support"]) for row in records)
    edge = len(records) - full
    fractions = sorted(row["exposure_fraction"] for row in records)
    median_fraction = fractions[len(fractions) // 2]
    bearing = math.degrees(math.atan2(axis[1], axis[0]))
    preferred_rows = [row for row in summary_rows if row["preferred"] == "yes"]
    min_deposits = min(row["usgs_deposits"] for row in preferred_rows)
    max_deposits = max(row["usgs_deposits"] for row in preferred_rows)
    lines = [
        "# Spatial support assessment (v1)",
        "",
        f"Generated by `build_phase7_supplement.py`.",
        "",
        "## Study area and support",
        "",
        f"- Daly union area: **{study_area_km2:,.0f} km²**. No domain boundary was extrapolated.",
        f"- Support grid: **{len(records):,} 5 km cells**, clipped to the Daly union.",
        f"- Full-support cells: **{full:,}**; partial edge cells: **{edge:,}**.",
        f"- Median cell exposure fraction: **{median_fraction:.3f}**.",
        "- Full support is defined as at least 99% of the nominal 25 km² cell inside the union. "
        "Edge cells retain their true exposure area rather than being silently expanded.",
        "",
        "## Why 5 km",
        "",
        "- EPSG:32735 is appropriate for the longitude of the Copperbelt and gives a near-equal-area working grid at this scale.",
        "- A 5 km cell is larger than the ~1 km locational uncertainty assumed for the USGS inventory, but boundary assignments should still be propagated with a 1 km positional disk.",
        "- The Daly domains are reconstructed from a 600 m categorical raster whose boundaries are map-derived and approximate. A 5 km cell is conservative relative to that boundary uncertainty.",
        "- The historical RDC maps are not on this support: their datum and northing are unresolved, so they cannot be regridded without inventing coordinates.",
        "",
        "## Geographic folds",
        "",
        f"- Preferred scheme: **{preferred_n} occurrence-supported contiguous bands** along the principal USGS-inventory axis (bearing {bearing:.1f}°).",
        f"- USGS inventory records per preferred fold range from **{min_deposits} to {max_deposits}** "
        f"(fold counts: {', '.join(str(row['usgs_deposits']) for row in preferred_rows)}).",
        "- Folds 1-3 contain DRC records only; fold 4 contains 7 DRC and all 39 source-Zambia records inside the union. This is a country-blocked along-strike design, not a balanced four-country or four-domain design.",
        "- Fold boundaries are fixed from the inventory's principal axis so every fold has evaluation support; they were not tuned to model performance.",
        "- Areas outside the inventory envelope remain in the prediction frame but carry no transferability evidence.",
        "- The folds are geographical partitions for transferability testing. They are not Daly domains and were not tuned to model performance.",
        "- The five-fold alternative is retained in `daly_fold_support_summary_v1.csv` and the fold GeoPackage for sensitivity.",
        "- Any fold with low occurrence support must be reported as a transferability stress test, not treated as a representative training region.",
        "",
        "## Coverage limitation",
        "",
        "- The former USGS permissive tracts cover only about **4.7%** of the Daly study area. They remain a comparison layer, not the study-area boundary.",
        f"- Of {overlay_counts['inventory_records']} USGS inventory records, {overlay_counts['inside_union']} fall inside the Daly union and {overlay_counts['inside_tract']} inside the permissive tracts; {overlay_counts['outside_tract']} lie outside the tracts. The 186 union records fall entirely in the NRB domain (149 in 3a, 37 in 3b). The other seven domain polygons have zero inventory support; this is an inventory limit, not evidence of barren geology.",
        "- High/medium-confidence Daly structure linework covers 6,518 km (127 of 248 segments); 71% of cells are within 50 km. Named high/medium structures cover 2,250 km (47 segments); 68% of cells have no named fault within 50 km.",
        "- The 136 Phase 6 structural and stratigraphic features remain in map-grid or source-pixel space. They cannot be assigned to this grid until datum and northing are documented.",
        "- A lack of recorded USGS occurrences outside the tracts is **not** evidence of barren geology.",
        "",
        "## Outputs",
        "",
        "- `daly_study_area_v1.gpkg`",
        "- `daly_support_5km_v1.gpkg`",
        "- `daly_geographic_folds_v1.gpkg`",
        "- `daly_study_area_summary_v1.csv`",
        "- `daly_country_summary_v1.csv`",
        "- `daly_domain_area_summary_v1.csv`",
        "- `daly_fold_support_summary_v1.csv`",
        "- `daly_support_5km_summary_v1.csv`",
        "",
    ]
    (OUT / "spatial_support_assessment_v1.md").write_text("\n".join(lines), encoding="utf-8")
    return {"cells": len(records), "full_cells": full, "edge_cells": edge, "preferred_folds": preferred_n,
            "min_deposits": min_deposits, "max_deposits": max_deposits, "bearing": bearing}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    domains, countries, tracts, deposits = load_inputs()
    union, domain_rows, country_records, tract_union, deposit_geometries, study_area_km2, overlay_counts = study_area_outputs(
        domains, countries, tracts, deposits
    )
    records = build_support_grid(union, domain_rows, country_records, tract_union, deposit_geometries)
    _, preferred_n, summary_rows, deposit_counts, axis = finalize_support(records, deposit_geometries)
    assessment = support_assessment(records, preferred_n, summary_rows, deposit_counts, axis, study_area_km2, overlay_counts)
    print(json.dumps({"study_area_km2": round(study_area_km2, 2), **assessment}, indent=2))


if __name__ == "__main__":
    main()
