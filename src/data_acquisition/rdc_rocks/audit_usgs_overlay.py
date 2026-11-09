"""Exact per-record overlay audit for the USGS Copperbelt inventory.

Run with QGIS Python. Writes audit tables into the Phase 7 supplementary folder.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from qgis.core import QgsGeometry, QgsSpatialIndex, QgsVectorLayer

ROOT = Path(r"C:\Users\vanmu\copperbelt-mineral-prospectivity")
SUP = ROOT / "data/raw/rdc_rocks/derived/integration/phase7/supplementary"
DEPOSITS = ROOT / "data/raw/usgs_sir2010_5090_t/CACB_deposits_utm.shp"
TRACTS = ROOT / "data/raw/usgs_sir2010_5090_t/CACB_tracts_utm.shp"
DOMAINS = ROOT / "data/processed/geology_foundation/daly_tectonic_domains_v1.gpkg"
COUNTRIES = SUP / "daly_country_context_v1.gpkg"
SUPPORT = SUP / "daly_support_5km_v1.gpkg"


def load(path: Path, name: str) -> QgsVectorLayer:
    layer = QgsVectorLayer(str(path), name, "ogr")
    if not layer.isValid():
        raise RuntimeError(f"Could not load {path}")
    return layer


def relation(polygon: QgsGeometry, point: QgsGeometry) -> str:
    if polygon.contains(point):
        return "inside"
    if polygon.intersects(point):
        return "boundary"
    return "outside"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    deposits = load(DEPOSITS, "deposits")
    tracts = load(TRACTS, "tracts")
    domains = load(DOMAINS, "domains")
    countries = load(COUNTRIES, "countries")
    support = load(SUPPORT, "support")

    tract_geoms = {f.id(): QgsGeometry(f.geometry()) for f in tracts.getFeatures()}
    tract_names = {f.id(): str(f["Tract_ID"]) for f in tracts.getFeatures()}
    tract_union = QgsGeometry.unaryUnion(list(tract_geoms.values()))
    domain_rows = [
        (str(f["domain_id"]), str(f["abbreviation"]), QgsGeometry(f.geometry()))
        for f in domains.getFeatures()
    ]
    union = QgsGeometry.unaryUnion([row[2] for row in domain_rows])
    country_rows = [(str(f["country"]), QgsGeometry(f.geometry())) for f in countries.getFeatures()]

    support_index = QgsSpatialIndex()
    support_by_fid = {}
    for f in support.getFeatures():
        support_index.addFeature(f)
        support_by_fid[f.id()] = f

    rows = []
    for f in deposits.getFeatures():
        point = QgsGeometry(f.geometry())
        xy = point.asPoint()
        containing_tracts = [
            tract_names[fid] for fid, geom in tract_geoms.items()
            if geom.contains(point) or geom.intersects(point)
        ]
        containing_domains = [
            (domain_id, abbr) for domain_id, abbr, geom in domain_rows
            if geom.contains(point) or geom.intersects(point)
        ]
        country_hits = [
            name for name, geom in country_rows
            if geom.contains(point) or geom.intersects(point)
        ]
        cell_hits = support_index.intersects(point.boundingBox())
        cells = [support_by_fid[fid] for fid in cell_hits if support_by_fid[fid].geometry().intersects(point)]
        cell = cells[0] if cells else None
        rows.append({
            "deposit_fid": f.id(),
            "name": f["Name"] or "",
            "source_country": f["Country"] or "",
            "latitude": f["Latitude"],
            "longitude": f["Longitude"],
            "x_m": round(xy.x(), 3),
            "y_m": round(xy.y(), 3),
            "union_relation": relation(union, point),
            "tract_relation": relation(tract_union, point),
            "tract_ids": ";".join(sorted(containing_tracts)),
            "domain_ids": ";".join(domain_id for domain_id, _ in containing_domains),
            "domain_abbreviations": ";".join(abbr for _, abbr in containing_domains),
            "natural_earth_country": ";".join(country_hits),
            "support_id": cell["support_id"] if cell else "",
            "support_country": cell["country"] if cell else "",
            "fold4": cell["fold4"] if cell else "",
            "fold5": cell["fold5"] if cell else "",
        })

    rows.sort(key=lambda row: (row["source_country"], row["name"], row["deposit_fid"]))
    fields = list(rows[0])
    out = SUP / "usgs_deposit_overlay_audit_v1.csv"
    write_csv(out, fields, rows)

    def count(predicate):
        return sum(1 for row in rows if predicate(row))

    combinations = Counter(
        (row["source_country"], row["natural_earth_country"], row["union_relation"], row["tract_relation"])
        for row in rows
    )
    summary = [
        {"metric": "inventory_records", "value": len(rows), "detail": "All USGS deposit/prospect points."},
        {"metric": "inside_daly_union", "value": count(lambda r: r["union_relation"] == "inside"),
         "detail": "Strict point-in-polygon containment; boundary cases counted separately."},
        {"metric": "boundary_daly_union", "value": count(lambda r: r["union_relation"] == "boundary"),
         "detail": "Point intersects a domain boundary but is not strictly contained."},
        {"metric": "inside_tracts", "value": count(lambda r: r["tract_relation"] == "inside"),
         "detail": "Strict point-in-polygon containment in the union of five permissive tracts."},
        {"metric": "outside_tracts", "value": count(lambda r: r["tract_relation"] == "outside"),
         "detail": "Points outside the union of five permissive tracts."},
        {"metric": "source_country_drc", "value": count(lambda r: r["source_country"] == "Democratic Republic of the Congo"),
         "detail": "USGS source Country attribute."},
        {"metric": "source_country_zambia", "value": count(lambda r: r["source_country"] == "Zambia"),
         "detail": "USGS source Country attribute."},
        {"metric": "natural_earth_country_drc", "value": count(lambda r: r["natural_earth_country"] == "Dem. Rep. Congo"),
         "detail": "Natural Earth point containment; not authoritative at the border."},
        {"metric": "natural_earth_country_zambia", "value": count(lambda r: r["natural_earth_country"] == "Zambia"),
         "detail": "Natural Earth point containment; not authoritative at the border."},
        {"metric": "country_assignment_mismatches",
         "value": count(lambda r: (r["source_country"] == "Democratic Republic of the Congo" and r["natural_earth_country"] != "Dem. Rep. Congo")
                        or (r["source_country"] == "Zambia" and r["natural_earth_country"] != "Zambia")),
         "detail": "Source Country versus Natural Earth polygon assignment."},
    ]
    summary_out = SUP / "usgs_deposit_overlay_summary_v1.csv"
    write_csv(summary_out, ["metric", "value", "detail"], summary)

    print("wrote", out)
    for metric in summary:
        print(metric["metric"], metric["value"])
    print("combinations")
    for key, value in sorted(combinations.items()):
        print(key, value)


if __name__ == "__main__":
    main()
