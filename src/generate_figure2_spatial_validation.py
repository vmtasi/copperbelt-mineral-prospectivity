"""Generate Paper 1 Figure 2 from the frozen V11 spatial-fold artifact.

Run this script inside an open QGIS session.  It never calls the fold-building
code and never writes to the V11 model, traces, or frozen validation artifact.
Instead it copies the recorded ``spatial_block`` values from
``figures/v11_oof_posterior_uncertainty.gpkg`` to a derived display layer and
uses the corresponding pre-existing grid polygons for the cartographic cells.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from PyQt5.QtCore import QVariant
from PyQt5.QtGui import QColor, QFont
from qgis.core import (
    QgsCategorizedSymbolRenderer,
    QgsCoordinateReferenceSystem,
    QgsFeature,
    QgsField,
    QgsFields,
    QgsFillSymbol,
    QgsGeometry,
    QgsLayoutExporter,
    QgsLayoutItemLabel,
    QgsLayoutItemMap,
    QgsLayoutItemScaleBar,
    QgsLayoutPoint,
    QgsLayoutSize,
    QgsLineSymbol,
    QgsMarkerSymbol,
    QgsPalLayerSettings,
    QgsPrintLayout,
    QgsProject,
    QgsRendererCategory,
    QgsSingleSymbolRenderer,
    QgsSpatialIndex,
    QgsTextFormat,
    QgsTextBufferSettings,
    QgsUnitTypes,
    QgsVectorFileWriter,
    QgsVectorLayer,
    QgsVectorLayerSimpleLabeling,
    QgsWkbTypes,
)


ROOT = Path(__file__).resolve().parents[1]
FROZEN_FOLDS = ROOT / "figures" / "v11_oof_posterior_uncertainty.gpkg"
TRAINING_CSV = ROOT / "data" / "copperbelt_training_v5_with_tectonic_domain.csv"
OUTPUT_GPKG = ROOT / "figures" / "figure2_spatial_validation_display_cells.gpkg"
OUTPUT_PNG = ROOT / "paper_1" / "manuscript" / "images" / "fig2_spatial_validation_design.png"
OUTPUT_PDF = ROOT / "paper_1" / "manuscript" / "images" / "fig2_spatial_validation_design.pdf"
VERIFY_JSON = ROOT / "figures" / "audit" / "figure2_spatial_validation_verification.json"
LAYOUT_NAME = "Paper 1 - Along-Belt Spatial Validation Design (Figure 2)"
EXPECTED_CELLS = 1872
EXPECTED_FOLDS = {0, 1, 2, 3}
FOLD_COLORS = {
    0: "#3B75AF",  # blue
    1: "#DD8A28",  # orange
    2: "#459E7A",  # green
    3: "#8E5AA8",  # purple
}


def _key(x: float, y: float) -> tuple[float, float]:
    """Coordinate key matching the frozen artifact to the training table."""
    return (round(float(x), 3), round(float(y), 3))


def _layer(project: QgsProject, name: str) -> QgsVectorLayer:
    matches = project.mapLayersByName(name)
    if not matches:
        raise RuntimeError(f"Required project layer is unavailable: {name}")
    return matches[0]


def _clone(project: QgsProject, source: QgsVectorLayer, name: str) -> QgsVectorLayer:
    """Add a layout-only copy so Figure 1 styling is never changed."""
    for existing in project.mapLayersByName(name):
        project.removeMapLayer(existing.id())
    clone = QgsVectorLayer(source.source(), name, source.providerType())
    if not clone.isValid():
        raise RuntimeError(f"Could not clone context layer: {source.name()}")
    project.addMapLayer(clone)
    project.layerTreeRoot().findLayer(clone.id()).setItemVisibilityChecked(False)
    return clone


def _add_label(layout: QgsPrintLayout, text: str, x: float, y: float,
               width: float, height: float, point_size: float = 8,
               bold: bool = False, color: str = "#222222") -> QgsLayoutItemLabel:
    label = QgsLayoutItemLabel(layout)
    label.setText(text)
    fmt = QgsTextFormat()
    # QFont's constructor accepts an integer point size; QgsTextFormat below
    # retains the requested fractional size for final layout rendering.
    font = QFont("Arial", int(round(point_size)))
    font.setBold(bold)
    fmt.setFont(font)
    fmt.setSize(point_size)
    fmt.setColor(QColor(color))
    label.setTextFormat(fmt)
    label.attemptMove(QgsLayoutPoint(x, y, QgsUnitTypes.LayoutMillimeters))
    label.attemptResize(QgsLayoutSize(width, height, QgsUnitTypes.LayoutMillimeters))
    layout.addLayoutItem(label)
    return label


def _create_cell_layer(project: QgsProject) -> tuple[QgsVectorLayer, dict]:
    frozen = QgsVectorLayer(str(FROZEN_FOLDS), "frozen_v11_spatial_folds", "ogr")
    if not frozen.isValid():
        raise RuntimeError(f"Frozen fold artifact could not be read: {FROZEN_FOLDS}")
    if frozen.crs().authid() != "EPSG:32735":
        raise RuntimeError(f"Unexpected frozen-fold CRS: {frozen.crs().authid()}")
    if frozen.featureCount() != EXPECTED_CELLS:
        raise RuntimeError(f"Expected {EXPECTED_CELLS} frozen cells; found {frozen.featureCount()}")

    grid = _layer(project, "Clipped")
    if grid.crs().authid() != "EPSG:32735" or grid.geometryType() != QgsWkbTypes.PolygonGeometry:
        raise RuntimeError("The pre-existing Clipped grid is not polygonal EPSG:32735 geometry")
    grid_index = QgsSpatialIndex(grid.getFeatures())

    deposits_by_coord = {}
    with TRAINING_CSV.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            deposits_by_coord[_key(row["centroid_x"], row["centroid_y"])] = int(float(row["deposit_present"]))

    fields = QgsFields()
    fields.append(QgsField("cell_id", QVariant.Int))
    fields.append(QgsField("spatial_block", QVariant.Int))
    fields.append(QgsField("fold_label", QVariant.String, len=32))
    fields.append(QgsField("daly_domain", QVariant.String, len=16))
    fields.append(QgsField("deposit_present", QVariant.Int))
    fields.append(QgsField("source_artifact", QVariant.String, len=56))

    if OUTPUT_GPKG.exists():
        raise RuntimeError(
            f"Refusing to overwrite an existing derived Figure 2 artifact: {OUTPUT_GPKG}. "
            "Archive it explicitly before regenerating."
        )
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    options.layerName = "spatial_validation_cells"
    writer = QgsVectorFileWriter.create(
        str(OUTPUT_GPKG), fields, QgsWkbTypes.MultiPolygon,
        QgsCoordinateReferenceSystem("EPSG:32735"), project.transformContext(), options
    )
    if writer.hasError() != QgsVectorFileWriter.NoError:
        raise RuntimeError(f"Could not create Figure 2 cell layer: {writer.errorMessage()}")

    counts = Counter()
    unmatched_training = 0
    invalid_geometry = 0
    max_match_distance = 0.0
    for source_feature in frozen.getFeatures():
        point = source_feature.geometry().asPoint()
        nearest = grid_index.nearestNeighbor(point, 1)
        if not nearest:
            raise RuntimeError(f"No grid polygon found for frozen cell {source_feature['fid']}")
        grid_feature = grid.getFeature(nearest[0])
        distance = source_feature.geometry().distance(grid_feature.geometry().centroid())
        max_match_distance = max(max_match_distance, distance)
        if distance > 0.01:
            raise RuntimeError(f"Grid matching tolerance exceeded for frozen cell {source_feature['fid']}: {distance} m")
        geom = grid_feature.geometry()
        if geom.isEmpty() or not geom.isGeosValid():
            invalid_geometry += 1
            raise RuntimeError(f"Invalid grid geometry for frozen cell {source_feature['fid']}")
        block = int(source_feature["spatial_block"])
        if block not in EXPECTED_FOLDS:
            raise RuntimeError(f"Unexpected frozen fold identifier {block}")
        dep = deposits_by_coord.get(_key(point.x(), point.y()))
        if dep is None:
            unmatched_training += 1
            raise RuntimeError(f"No training-row match for frozen cell {source_feature['fid']}")
        feature = QgsFeature(fields)
        feature.setGeometry(geom)
        feature.setAttributes([
            int(source_feature["fid"]), block, f"Fold {block + 1} (held out)",
            str(source_feature["daly_domain"]), dep, "v11_oof_posterior_uncertainty.gpkg",
        ])
        if not writer.addFeature(feature):
            raise RuntimeError(f"Could not write derived cell {source_feature['fid']}")
        counts[block] += 1
    del writer

    if set(counts) != EXPECTED_FOLDS or any(counts[b] != 468 for b in EXPECTED_FOLDS):
        raise RuntimeError(f"Frozen fold count check failed: {dict(counts)}")

    output = QgsVectorLayer(f"{OUTPUT_GPKG}|layername=spatial_validation_cells", "Figure 2 — frozen spatial-validation cells", "ogr")
    if not output.isValid() or output.featureCount() != EXPECTED_CELLS:
        raise RuntimeError("Derived Figure 2 cells could not be reopened and verified")
    project.addMapLayer(output)
    project.layerTreeRoot().findLayer(output.id()).setItemVisibilityChecked(False)

    categories = []
    for block in sorted(EXPECTED_FOLDS):
        symbol = QgsFillSymbol.createSimple({
            "color": FOLD_COLORS[block],
            "outline_color": "#FFFFFF",
            "outline_width": "0.08",
            "outline_width_unit": "MM",
        })
        symbol.setOpacity(0.86)
        categories.append(QgsRendererCategory(block, symbol, f"Fold {block + 1} (n = {counts[block]})"))
    output.setRenderer(QgsCategorizedSymbolRenderer("spatial_block", categories))

    verification = {
        "figure": "Paper 1 Figure 2 — Along-Belt Spatial Validation Design",
        "source_artifact": str(FROZEN_FOLDS.relative_to(ROOT)).replace("\\", "/"),
        "source_artifact_crs": frozen.crs().authid(),
        "cell_geometry_source": "active QGIS project layer: Clipped",
        "derived_cells": str(OUTPUT_GPKG.relative_to(ROOT)).replace("\\", "/"),
        "expected_cells": EXPECTED_CELLS,
        "derived_cells_count": output.featureCount(),
        "fold_counts": {str(k): counts[k] for k in sorted(counts)},
        "fold_ids": sorted(counts),
        "max_centroid_to_grid_match_m": max_match_distance,
        "invalid_geometries": invalid_geometry,
        "unmatched_training_rows": unmatched_training,
        "crs": output.crs().authid(),
        "frozen_assignments_recomputed": False,
    }
    return output, verification


def _style_context(project: QgsProject):
    boundaries = _clone(project, _layer(project, "Daly_Domain_Boundaries"), "Figure 2 — Daly domain boundaries")
    boundaries.setRenderer(QgsSingleSymbolRenderer(QgsLineSymbol.createSimple({
        "line_color": "#363636", "line_width": "0.42", "line_width_unit": "MM",
    })))

    tracts = _clone(project, _layer(project, "tract_boundary"), "Figure 2 — assessment tracts")
    tracts.setRenderer(QgsSingleSymbolRenderer(QgsLineSymbol.createSimple({
        "line_color": "#B2182B", "line_width": "0.30", "line_width_unit": "MM",
        "penstyle": "dash",
    })))

    deposits = _clone(project, _layer(project, "deposits_utm"), "Figure 2 — known deposits and prospects")
    deposits.setRenderer(QgsSingleSymbolRenderer(QgsMarkerSymbol.createSimple({
        "name": "circle", "color": "#FFD166", "outline_color": "#1F1F1F",
        "outline_width": "0.32", "size": "1.65", "size_unit": "MM",
    })))
    return boundaries, tracts, deposits


def _fold_labels(project: QgsProject, cells: QgsVectorLayer) -> QgsVectorLayer:
    """Create four unobtrusive map labels; one for each already-recorded fold."""
    name = "Figure 2 — fold labels"
    for existing in project.mapLayersByName(name):
        project.removeMapLayer(existing.id())
    labels = QgsVectorLayer("Point?crs=EPSG:32735", name, "memory")
    provider = labels.dataProvider()
    provider.addAttributes([QgsField("label", QVariant.String, len=12)])
    labels.updateFields()
    features = []
    for block in sorted(EXPECTED_FOLDS):
        union = QgsGeometry.unaryUnion([
            feature.geometry() for feature in cells.getFeatures()
            if int(feature["spatial_block"]) == block
        ])
        feature = QgsFeature(labels.fields())
        feature.setGeometry(union.pointOnSurface())
        feature["label"] = f"Fold {block + 1}"
        features.append(feature)
    provider.addFeatures(features)
    text = QgsTextFormat()
    text.setFont(QFont("Arial", 9))
    text.setSize(9)
    text.setColor(QColor("#202020"))
    buffer = QgsTextBufferSettings()
    buffer.setEnabled(True)
    buffer.setColor(QColor("#FFFFFF"))
    buffer.setSize(1.1)
    buffer.setSizeUnit(QgsUnitTypes.RenderMillimeters)
    text.setBuffer(buffer)
    settings = QgsPalLayerSettings()
    settings.fieldName = "label"
    settings.setFormat(text)
    labels.setLabeling(QgsVectorLayerSimpleLabeling(settings))
    labels.setLabelsEnabled(True)
    project.addMapLayer(labels)
    project.layerTreeRoot().findLayer(labels.id()).setItemVisibilityChecked(False)
    return labels


def _build_layout(project: QgsProject, cells: QgsVectorLayer, verification: dict) -> QgsPrintLayout:
    manager = project.layoutManager()
    prior = manager.layoutByName(LAYOUT_NAME)
    if prior:
        manager.removeLayout(prior)
    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.setName(LAYOUT_NAME)
    page = layout.pageCollection().page(0)
    page.setPageSize(QgsLayoutSize(210, 210, QgsUnitTypes.LayoutMillimeters))
    manager.addLayout(layout)

    boundaries, tracts, deposits = _style_context(project)
    labels = _fold_labels(project, cells)
    map_item = QgsLayoutItemMap(layout)
    map_item.attemptMove(QgsLayoutPoint(10, 29, QgsUnitTypes.LayoutMillimeters))
    map_item.attemptResize(QgsLayoutSize(150, 165, QgsUnitTypes.LayoutMillimeters))
    extent = cells.extent()
    extent.grow(12000)
    map_item.setExtent(extent)
    map_item.setLayers([labels, deposits, tracts, boundaries, cells])
    map_item.setKeepLayerSet(True)
    map_item.setFrameEnabled(True)
    layout.addLayoutItem(map_item)

    _add_label(layout, "Four-fold along-belt spatial validation design", 10, 7, 190, 8, 13, True)
    _add_label(layout, "Frozen V11 assignments: 1,872 modelling cells (5 km × 5 km); each fold held out in turn", 10, 16, 190, 7, 7.2)
    _add_label(layout, "MAP KEY", 166, 32, 38, 6, 8, True)
    _add_label(layout, "Held-out fold", 166, 40, 38, 5, 7.5, True)

    # Text swatches create a stable, publication-scale legend without relying on a live legend tree.
    y = 47
    for block in sorted(EXPECTED_FOLDS):
        _add_label(layout, "■", 166, y, 6, 5, 11, False, FOLD_COLORS[block])
        _add_label(layout, f"Fold {block + 1} (n = 468)", 172, y + 0.4, 32, 5, 7.2)
        y += 7
    _add_label(layout, "Context", 166, y + 2, 38, 5, 7.5, True)
    _add_label(layout, "━", 166, y + 9, 7, 5, 10, False, "#363636")
    _add_label(layout, "Daly domain boundary", 172, y + 9.7, 32, 5, 7.0)
    _add_label(layout, "╌", 166, y + 16, 7, 5, 11, False, "#B2182B")
    _add_label(layout, "Assessment tracts", 172, y + 16.7, 32, 5, 7.0)
    _add_label(layout, "●", 166, y + 23, 7, 5, 10, False, "#1F1F1F")
    _add_label(layout, "Known deposit/prospect", 172, y + 23.7, 32, 5, 7.0)
    _add_label(layout, "North ↑", 176, y + 35, 25, 5, 8, True)

    scale = QgsLayoutItemScaleBar(layout)
    scale.setStyle("Single Box")
    scale.setLinkedMap(map_item)
    scale.setUnits(QgsUnitTypes.DistanceKilometers)
    scale.setUnitLabel("km")
    scale.setNumberOfSegments(2)
    scale.setNumberOfSegmentsLeft(0)
    scale.setUnitsPerSegment(50)
    scale.setHeight(2.2)
    scale.applyDefaultSize()
    # Applying the default size can reset unit display in some QGIS builds.
    scale.setUnits(QgsUnitTypes.DistanceKilometers)
    scale.setUnitLabel("km")
    scale.setUnitsPerSegment(50)
    scale.attemptMove(QgsLayoutPoint(166, 151, QgsUnitTypes.LayoutMillimeters))
    layout.addLayoutItem(scale)

    _add_label(layout, "Projection: WGS 84 / UTM zone 35S (EPSG:32735)", 166, 164, 38, 11, 6.4)
    _add_label(layout, "Fold source: frozen V11 OOF posterior-uncertainty artifact. Geological context shown only for orientation.", 166, 177, 38, 17, 5.8)
    return layout


def main() -> None:
    project = QgsProject.instance()
    cells, verification = _create_cell_layer(project)
    layout = _build_layout(project, cells, verification)
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    exporter = QgsLayoutExporter(layout)
    image_settings = QgsLayoutExporter.ImageExportSettings()
    image_settings.dpi = 600
    image_result = exporter.exportToImage(str(OUTPUT_PNG), image_settings)
    pdf_settings = QgsLayoutExporter.PdfExportSettings()
    pdf_settings.dpi = 600
    pdf_result = exporter.exportToPdf(str(OUTPUT_PDF), pdf_settings)
    if image_result != QgsLayoutExporter.Success or pdf_result != QgsLayoutExporter.Success:
        raise RuntimeError(f"Figure 2 export failed: PNG={image_result}, PDF={pdf_result}")
    verification["output_png"] = str(OUTPUT_PNG.relative_to(ROOT)).replace("\\", "/")
    verification["output_pdf"] = str(OUTPUT_PDF.relative_to(ROOT)).replace("\\", "/")
    verification["layout_name"] = LAYOUT_NAME
    VERIFY_JSON.parent.mkdir(parents=True, exist_ok=True)
    VERIFY_JSON.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
