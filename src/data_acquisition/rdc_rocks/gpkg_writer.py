"""Minimal GeoPackage writer (no GDAL) with an undefined local CRS.

Layers are written with srs_id 0 so no datum or projection is implied.
"""
from __future__ import annotations

import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path

WKB_POINT = 1
WKB_LINESTRING = 2
WKB_POLYGON = 3

WKT_WGS84 = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
)


def _header(srs_id: int = 0) -> bytes:
    return b"GP" + bytes([0, 0b00000001]) + struct.pack("<i", srs_id)


def point_wkb(x: float, y: float, srs_id: int = 0) -> bytes:
    return _header(srs_id) + struct.pack("<BIdd", 1, WKB_POINT, x, y)


def linestring_wkb(points: list[tuple[float, float]], srs_id: int = 0) -> bytes:
    body = struct.pack("<BII", 1, WKB_LINESTRING, len(points))
    body += b"".join(struct.pack("<dd", x, y) for x, y in points)
    return _header(srs_id) + body


def polygon_wkb(rings: list[list[tuple[float, float]]], srs_id: int = 0) -> bytes:
    body = struct.pack("<BII", 1, WKB_POLYGON, len(rings))
    for ring in rings:
        body += struct.pack("<I", len(ring))
        body += b"".join(struct.pack("<dd", x, y) for x, y in ring)
    return _header(srs_id) + body


SQL_TYPES = {"text": "TEXT", "integer": "INTEGER", "real": "REAL"}


def write_gpkg(path: Path, layers: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    cursor = connection.cursor()
    cursor.execute("PRAGMA application_id = 1196444487")
    cursor.execute("PRAGMA user_version = 10200")
    cursor.execute(
        "CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT NOT NULL, srs_id INTEGER PRIMARY KEY, "
        "organization TEXT NOT NULL, organization_coordsys_id INTEGER NOT NULL, definition TEXT NOT NULL, description TEXT)"
    )
    cursor.executemany(
        "INSERT INTO gpkg_spatial_ref_sys VALUES (?,?,?,?,?,?)",
        [
            ("Undefined Cartesian", 0, "NONE", 0, "undefined", "Local map grid; datum and projection unresolved"),
            ("Undefined geographic", -1, "NONE", -1, "undefined", "Undefined"),
            ("WGS 84", 4326, "EPSG", 4326, WKT_WGS84, "WGS 84 geographic coordinates"),
        ],
    )
    cursor.execute(
        "CREATE TABLE gpkg_contents (table_name TEXT PRIMARY KEY, data_type TEXT NOT NULL, identifier TEXT UNIQUE, "
        "description TEXT DEFAULT '', last_change DATETIME NOT NULL, min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, "
        "max_y DOUBLE, srs_id INTEGER)"
    )
    cursor.execute(
        "CREATE TABLE gpkg_geometry_columns (table_name TEXT NOT NULL, column_name TEXT NOT NULL, "
        "geometry_type_name TEXT NOT NULL, srs_id INTEGER NOT NULL, z TINYINT NOT NULL, m TINYINT NOT NULL, "
        "PRIMARY KEY (table_name, column_name))"
    )
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    summary = {}
    for layer in layers:
        name = layer["name"]
        fields = layer.get("fields", [])
        srs_id = int(layer.get("srs_id", 0))
        columns = ", ".join(f'"{field}" {SQL_TYPES[kind]}' for field, kind in fields)
        cursor.execute(f'CREATE TABLE "{name}" (fid INTEGER PRIMARY KEY AUTOINCREMENT, geom BLOB NOT NULL{", " + columns if columns else ""})')
        xs, ys = [], []
        for feature in layer["features"]:
            blob = feature["geom"]
            attributes = [feature["attributes"].get(field) for field, _ in fields]
            placeholders = ", ".join("?" for _ in range(len(attributes) + 1))
            cursor.execute(f'INSERT INTO "{name}" (geom{", " + ", ".join(chr(34)+f+chr(34) for f,_ in fields) if fields else ""}) VALUES ({placeholders})', [blob, *attributes])
            for x, y in feature["points"]:
                xs.append(x)
                ys.append(y)
        cursor.execute(
            "INSERT INTO gpkg_contents VALUES (?,?,?,?,?,?,?,?,?,?)",
            (name, "features", name, layer.get("description", ""), now, min(xs), min(ys), max(xs), max(ys), srs_id),
        )
        cursor.execute(
            "INSERT INTO gpkg_geometry_columns VALUES (?,?,?,?,?,?)",
            (name, "geom", layer["geometry_type"], srs_id, 0, 0),
        )
        summary[name] = len(layer["features"])
    connection.commit()
    connection.close()
    return summary
