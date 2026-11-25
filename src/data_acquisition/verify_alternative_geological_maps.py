"""Assert the alternative geological map deliverables are internally consistent."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from osgeo import gdal, osr


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/raw/zambia_acquisition/derived/alternative_geological_maps"
LAYERS = {
    "dewaele_2006": ["geological_units", "geological_contacts", "structural_features"],
    "hanson_1994": ["structural_features", "geological_units", "geological_contacts"],
    "zambia_overview": ["geological_units", "geological_contacts", "structural_features"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((OUT / "processing_manifest_v1.json").read_text(encoding="utf-8"))
    metrics = {row["source_id"]: row for row in json.loads(
        (OUT / "reports/integration_metrics_v1.json").read_text(encoding="utf-8"))}

    for source_id, layers in LAYERS.items():
        package = OUT / "source_layers" / f"{source_id}.gpkg"
        assert package.is_file(), f"missing {package}"
        for layer in layers:
            frame = gpd.read_file(package, layer=layer)
            assert frame.crs is not None and frame.crs.to_epsg() == 32735, f"{source_id}/{layer}: CRS"
            assert bool(frame.geometry.is_valid.all()), f"{source_id}/{layer}: invalid geometry"
        units = gpd.read_file(package, layer="geological_units")
        assert len(units) == metrics[source_id]["unit_features"], f"{source_id}: unit count"
        contacts = gpd.read_file(package, layer="geological_contacts")
        assert len(contacts) == metrics[source_id]["contact_features"], f"{source_id}: contact count"
        assert units["source_class"].nunique() == units["legend_term"].nunique()

    for raster in sorted((OUT / "georeferenced").glob("*_epsg32735_250m.tif")):
        dataset = gdal.Open(str(raster))
        assert dataset is not None, f"unreadable {raster}"
        assert dataset.GetGeoTransform()[1] == 250 and abs(dataset.GetGeoTransform()[5]) == 250
        srs = osr.SpatialReference(wkt=dataset.GetProjection())
        assert srs.GetAuthorityCode(None) == "32735", f"{raster.name}: CRS"

    for source_id, source in manifest["sources"].items():
        assert sha256(ROOT / source["pdf_path"]) == source["pdf_sha256"], f"{source_id}: PDF hash"

    crosswalk = pd.read_csv(OUT / "crosswalk/classification_crosswalk_v1.csv")
    assert not crosswalk.duplicated(["source_id", "source_legend_term", "source_geometry_role"]).any()
    assert (crosswalk["existing_phase6_class"] == "not_present").all()
    assert (crosswalk["schema_extension_required"] == "yes").all()
    assert set(crosswalk["predictor_eligibility"]) == {
        "no_5km_predictor; regional_context_only",
        "no_5km_predictor; national_context_only",
        "not_digitised; source_only",
        "no",
    }
    for source_id, source in manifest["sources"].items():
        terms = set(crosswalk.loc[crosswalk["source_id"] == source_id, "source_legend_term"])
        units = gpd.read_file(OUT / "source_layers" / f"{source_id}.gpkg", layer="geological_units")
        assert set(units["legend_term"]) <= terms, f"{source_id}: legend terms missing from crosswalk"

    assert (OUT / "reports/integration_validation_report_v1.md").is_file()
    print(json.dumps({"status": "ok", "sources": len(LAYERS), "crosswalk_rows": len(crosswalk)}))


if __name__ == "__main__":
    main()
