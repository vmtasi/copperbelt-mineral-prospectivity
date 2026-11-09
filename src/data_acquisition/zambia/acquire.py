from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from discover import DATA_ROOT, USER_AGENT, fetch, read_csv, utc_now, write_csv_atomic

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rdc_rocks"))
from gpkg_writer import polygon_wkb, write_gpkg


ROOT = Path(__file__).resolve().parents[3]
REGISTER = DATA_ROOT / "metadata" / "acquisition_register_v1.csv"
REFERENCE = DATA_ROOT / "metadata" / "reference"
MANIFEST = DATA_ROOT / "metadata" / "download_manifest_v1.csv"
INDEX_URL = (
    "https://services1.arcgis.com/AhXvNWFdL7hH4TjJ/ArcGIS/rest/services/"
    "Zambia_250k_geology/FeatureServer/0/query"
    "?where=1%3D1&outFields=*&outSR=4326&f=geojson"
)
GSD_PREVIEWS = {
    "NW": ("bbbf0814-67ad-41f5-891b-c7aa58e5e6db", "North West Quarter"),
    "NE": ("76badacb-5242-4f23-8640-ce9db87764e7", "North East Quarter"),
    "SE": ("d5ecd642-cd5f-47b4-9111-0d0ee7ce8599", "South East Quarter"),
    "SW": ("d6d54808-591e-4edd-82e6-0b7d31b52de3", "South West Quarter"),
}
MAGIC = {
    "application/pdf": (b"%PDF-",),
    "image/tiff": (b"II*\x00", b"MM\x00*"),
    "image/png": (b"\x89PNG\r\n\x1a\n",),
}
EXTENSIONS = {"application/pdf": ".pdf", "image/tiff": ".tif", "image/png": ".png"}
MANIFEST_FIELDS = [
    "record_id", "platform", "source_url", "title", "institution", "publication_year",
    "content_type", "bytes", "sha256", "page_count", "local_path", "downloaded_at_utc",
    "status", "error",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request(url: str, byte_range: str = "") -> Request:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/pdf,image/tiff,image/png,application/json,*/*;q=0.8",
    }
    if byte_range:
        headers["Range"] = f"bytes={byte_range}"
    return Request(url, headers=headers)


def probe(url: str, timeout: int) -> dict[str, object]:
    with urlopen(request(url, "0-15"), timeout=timeout) as response:
        content_range = response.headers.get("Content-Range", "")
        total = response.headers.get("Content-Length", "")
        if "/" in content_range:
            total = content_range.rsplit("/", 1)[-1]
        return {
            "status": response.status,
            "content_type": response.headers.get_content_type(),
            "bytes": int(total) if str(total).isdigit() else 0,
            "head": response.read(16),
        }


def download(url: str, destination: Path, content_type: str, expected: int, timeout: int) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, 3):
        try:
            digest = hashlib.sha256()
            size = 0
            with urlopen(request(url), timeout=timeout) as response:
                actual_type = response.headers.get_content_type()
                if actual_type != content_type:
                    raise RuntimeError(f"content type {actual_type!r}, expected {content_type!r}")
                with temporary.open("wb") as handle:
                    first = True
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        if first:
                            if not any(block.startswith(signature) for signature in MAGIC[content_type]):
                                raise RuntimeError("file signature does not match content type")
                            first = False
                        handle.write(block)
                        digest.update(block)
                        size += len(block)
            if expected and size != expected:
                raise RuntimeError(f"received {size} bytes, expected {expected}")
            os.replace(temporary, destination)
            return size, digest.hexdigest()
        except (HTTPError, URLError, TimeoutError, OSError, RuntimeError) as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt == 1:
                time.sleep(3)
    raise RuntimeError(f"download failed: {last_error}")


def pdf_pages(path: Path) -> int:
    pdfinfo = shutil.which("pdfinfo") or r"C:\Users\vanmu\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdfinfo.exe"
    result = subprocess.run(
        [pdfinfo, str(path)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120
    )
    match = re.search(r"^Pages:\s+(\d+)", result.stdout, re.MULTILINE)
    return int(match.group(1)) if match else 0


def png_size(path: Path) -> tuple[int, int]:
    payload = path.read_bytes()[:24]
    return struct.unpack(">II", payload[16:24])


def load_manifest() -> list[dict[str, object]]:
    return read_csv(MANIFEST) if MANIFEST.is_file() else []


def save_result(rows: list[dict[str, object]], row: dict[str, object]) -> None:
    rows[:] = [existing for existing in rows if existing["record_id"] != row["record_id"]]
    rows.append(row)
    write_csv_atomic(MANIFEST, rows, MANIFEST_FIELDS)


def result_row(record: dict[str, str], **values: object) -> dict[str, object]:
    row = {field: "" for field in MANIFEST_FIELDS}
    row.update({
        "record_id": record.get("record_id", ""),
        "platform": record.get("platform", ""),
        "source_url": record.get("source_url", ""),
        "title": record.get("title", ""),
        "institution": record.get("institution", ""),
        "publication_year": record.get("publication_year", ""),
        "downloaded_at_utc": utc_now(),
    })
    row.update(values)
    return row


def command_index(refresh: bool) -> int:
    cache = REFERENCE / "zambia_250k_geology.geojson"
    payload = fetch(INDEX_URL, cache, refresh, "application/geo+json,application/json")
    document = json.loads(payload.decode("utf-8"))
    features = document.get("features") or []
    fields = [
        "sheet_no", "sheet_name", "library_owns", "library_availability", "call",
        "library_catalog", "min_lon", "min_lat", "max_lon", "max_lat", "area_deg2",
    ]
    rows, layer_features, xs, ys = [], [], [], []
    for feature in features:
        properties = feature.get("properties") or {}
        geometry = feature.get("geometry") or {}
        rings = geometry.get("coordinates") or []
        if geometry.get("type") != "Polygon" or not rings:
            continue
        flat = [point for ring in rings for point in ring]
        min_lon = min(point[0] for point in flat)
        min_lat = min(point[1] for point in flat)
        max_lon = max(point[0] for point in flat)
        max_lat = max(point[1] for point in flat)
        area = abs(sum(
            ring[i][0] * ring[(i + 1) % len(ring)][1] - ring[(i + 1) % len(ring)][0] * ring[i][1]
            for ring in rings for i in range(len(ring))
        )) / 2
        row = {
            "sheet_no": properties.get("Sheet_no", ""),
            "sheet_name": properties.get("Sheet_name", ""),
            "library_owns": properties.get("Library_owns", ""),
            "library_availability": properties.get("Library_availability", ""),
            "call": properties.get("Call", ""),
            "library_catalog": properties.get("Library_catalog", ""),
            "min_lon": round(min_lon, 6),
            "min_lat": round(min_lat, 6),
            "max_lon": round(max_lon, 6),
            "max_lat": round(max_lat, 6),
            "area_deg2": round(area, 8),
        }
        rows.append(row)
        xs.extend(point[0] for point in flat)
        ys.extend(point[1] for point in flat)
        layer_features.append({
            "geom": polygon_wkb(rings, srs_id=4326),
            "attributes": row,
            "points": flat,
        })
    write_csv_atomic(REFERENCE / "zambia_250k_map_sheets_v1.csv", rows, fields)
    layers = [{
        "name": "map_sheets",
        "geometry_type": "POLYGON",
        "srs_id": 4326,
        "description": "Index of Zambia Geological Survey 1:250,000 map sheets; footprint, not geological evidence",
        "fields": [(field, "real" if field in ("min_lon", "min_lat", "max_lon", "max_lat", "area_deg2") else "text") for field in fields],
        "features": layer_features,
    }]
    summary = write_gpkg(REFERENCE / "zambia_250k_map_sheets_v1.gpkg", layers)
    metadata = {
        "source_url": INDEX_URL,
        "retrieved_at_utc": utc_now(),
        "feature_count": len(rows),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "bounds_wgs84": [min(xs), min(ys), max(xs), max(ys)] if xs else [],
        "layer_counts": summary,
        "note": "Map-sheet footprint index only. The index is not geological evidence and does not contain map content.",
    }
    (REFERENCE / "zambia_250k_map_sheets_v1.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    owned = sum(str(row["library_owns"]).strip().lower() in ("yes", "true", "1") for row in rows)
    print(f"index: {len(rows)} sheets, {owned} marked owned, bounds {metadata['bounds_wgs84']}")
    return 0


def command_previews(timeout: int) -> int:
    output = DATA_ROOT / "gsd" / "previews"
    manifest_path = DATA_ROOT / "metadata" / "gsd_preview_manifest_v1.csv"
    fields = ["code", "uuid", "title", "preview_url", "width", "height", "bytes", "sha256", "local_path", "retrieved_at_utc", "status"]
    rows = []
    for code, (uuid, title) in GSD_PREVIEWS.items():
        url = f"https://gsd.gov.zm/api/v1/File/view_thumbnail/{uuid}"
        destination = output / f"gsd_1m_{code}_{uuid}.png"
        status = "already_present"
        try:
            if not destination.is_file() or destination.read_bytes()[:8] != MAGIC["image/png"][0]:
                size, _ = download(url, destination, "image/png", 0, timeout)
                status = "downloaded"
            else:
                size = destination.stat().st_size
            width, height = png_size(destination)
            rows.append({
                "code": code, "uuid": uuid, "title": title, "preview_url": url,
                "width": width, "height": height, "bytes": size,
                "sha256": sha256_file(destination),
                "local_path": str(destination.relative_to(ROOT)),
                "retrieved_at_utc": utc_now(), "status": status,
            })
            print(f"preview {code}: {width}x{height} {status}")
        except Exception as exc:
            rows.append({
                "code": code, "uuid": uuid, "title": title, "preview_url": url,
                "width": "", "height": "", "bytes": "", "sha256": "", "local_path": "",
                "retrieved_at_utc": utc_now(), "status": f"error: {type(exc).__name__}: {exc}",
            })
            print(f"preview {code}: ERROR {exc}")
    write_csv_atomic(manifest_path, rows, fields)
    return 0


def command_netd(max_priority: int, max_files: int, timeout: int) -> int:
    records = [
        row for row in read_csv(REGISTER)
        if row["platform"] == "Zambia National ETD Portal" and int(row["priority"]) <= max_priority
    ]
    records.sort(key=lambda row: (int(row["priority"]), row.get("publication_year", "")))
    records = records[:max_files]
    output = DATA_ROOT / "netd" / "files"
    manifest = load_manifest()
    errors = 0
    for index, record in enumerate(records, start=1):
        destination = output / f"{record['record_id']}.pdf"
        try:
            if destination.is_file() and destination.stat().st_size > 1000 and destination.read_bytes()[:5] == b"%PDF-":
                pages = pdf_pages(destination)
                row = result_row(
                    record, content_type="application/pdf", bytes=destination.stat().st_size,
                    sha256=sha256_file(destination), page_count=pages,
                    local_path=str(destination.relative_to(ROOT)), status="already_present",
                )
            else:
                details = probe(record["file_url"], timeout)
                if details["content_type"] != "application/pdf":
                    raise RuntimeError(f"unexpected content type {details['content_type']}")
                size, digest = download(record["file_url"], destination, "application/pdf", int(details["bytes"]), timeout)
                row = result_row(
                    record, content_type="application/pdf", bytes=size, sha256=digest,
                    page_count=pdf_pages(destination),
                    local_path=str(destination.relative_to(ROOT)), status="verified",
                )
            save_result(manifest, row)
            print(f"[{index}/{len(records)}] {record['record_id']}: {row['page_count']} pp, {int(row['bytes'])/1024/1024:.1f} MiB, {row['status']}")
        except Exception as exc:
            errors += 1
            row = result_row(record, status="failed", error=f"{type(exc).__name__}: {exc}")
            save_result(manifest, row)
            print(f"[{index}/{len(records)}] {record['record_id']}: ERROR {exc}")
    print(f"netd: {len(records) - errors} ok, {errors} failed, manifest {MANIFEST}")
    return 1 if errors else 0


def command_rdc(max_priority: int, max_files: int, timeout: int, ids: str = "") -> int:
    records = [
        row for row in read_csv(REGISTER)
        if row["platform"] == "RDC.ROCKS" and int(row["priority"]) <= max_priority
    ]
    records.sort(key=lambda row: (int(row["priority"]), row["record_id"]))
    if ids:
        allowed = {value.strip() for value in ids.split(",") if value.strip()}
        records = [row for row in records if row["record_id"] in allowed]
    records = records[:max_files]
    output = DATA_ROOT / "rdc" / "files"
    manifest = load_manifest()
    errors = 0
    for index, record in enumerate(records, start=1):
        document_id = int(record["record_id"].removeprefix("RDC-"))
        try:
            existing = list(output.glob(f"RDC_{document_id:04d}.*"))
            existing = [path for path in existing if path.suffix in (".pdf", ".tif") and path.stat().st_size > 1000]
            if existing:
                path = existing[0]
                content_type = "application/pdf" if path.suffix == ".pdf" else "image/tiff"
                row = result_row(
                    record, content_type=content_type, bytes=path.stat().st_size,
                    sha256=sha256_file(path), page_count=pdf_pages(path) if content_type == "application/pdf" else "",
                    local_path=str(path.relative_to(ROOT)), status="already_present",
                )
            else:
                details = probe(record["file_url"], timeout)
                content_type = str(details["content_type"])
                if content_type not in EXTENSIONS:
                    raise RuntimeError(f"unsupported content type {content_type}")
                destination = output / f"RDC_{document_id:04d}{EXTENSIONS[content_type]}"
                size, digest = download(record["file_url"], destination, content_type, int(details["bytes"]), timeout)
                row = result_row(
                    record, content_type=content_type, bytes=size, sha256=digest,
                    page_count=pdf_pages(destination) if content_type == "application/pdf" else "",
                    local_path=str(destination.relative_to(ROOT)), status="verified",
                )
            save_result(manifest, row)
            print(f"[{index}/{len(records)}] {record['record_id']}: {row['page_count']} pp, {int(row['bytes'])/1024/1024:.1f} MiB, {row['status']}")
        except Exception as exc:
            errors += 1
            row = result_row(record, status="failed", error=f"{type(exc).__name__}: {exc}")
            save_result(manifest, row)
            print(f"[{index}/{len(records)}] {record['record_id']}: ERROR {exc}")
    print(f"rdc: {len(records) - errors} ok, {errors} failed, manifest {MANIFEST}")
    return 1 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Acquire Zambian geological reference layers and public source documents")
    parser.add_argument("source", choices=["index", "previews", "netd", "rdc"])
    parser.add_argument("--max-priority", type=int, default=1)
    parser.add_argument("--max-files", type=int, default=50)
    parser.add_argument("--ids", default="", help="comma-separated record IDs")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.source == "index":
        return command_index(args.refresh)
    if args.source == "previews":
        return command_previews(args.timeout)
    if args.source == "netd":
        return command_netd(args.max_priority, args.max_files, args.timeout)
    return command_rdc(args.max_priority, args.max_files, args.timeout, args.ids)


if __name__ == "__main__":
    raise SystemExit(main())
