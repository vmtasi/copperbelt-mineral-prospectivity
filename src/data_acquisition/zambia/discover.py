from __future__ import annotations

import csv
import json
import os
import re
import ssl
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from lxml import html


ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT / "data" / "raw" / "zambia_acquisition"
RAW_ROOT = DATA_ROOT / "metadata" / "raw"

GSD_API = "https://gsd.gov.zm/api/v1"
NETD_BASE = "https://netd.ac.zm"
RDC_API = "https://rdc.rocks/api/v1/catalog/documents"
RDC_LOCAL_CATALOGUE = ROOT / "data" / "raw" / "rdc_rocks" / "metadata" / "catalog_all_documents.csv"
RDC_LOCAL_MANIFEST = ROOT / "data" / "raw" / "rdc_rocks" / "metadata" / "acquisition_manifest_v1.csv"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
)

QUERY_TERMS = [
    "Copperbelt", "Zambian Copperbelt", "Lufilian", "Konkola",
    "Chililabombwe", "Chingola", "Nchanga", "Kitwe", "Nkana",
    "Mufulira", "Luanshya", "Chambishi", "Chibuluma", "Roan Antelope",
    "Kansanshi", "Lumwana", "Northern Rhodesia", "Katanga Supergroup",
    "Roan Group", "Muva", "Kundelungu", "geological memoir",
    "degree sheet", "1:250,000", "1:100,000", "Zambia geological map",
    "Zambia stratigraphy", "Zambia structural geology",
    "Zambia mineralisation", "Zambia mineral occurrence",
    "Zambia geophysics", "Zambia aeromagnetic", "Zambia geochemistry",
    "Zambia drilling", "Thieme",
]

CORE_PLACES = [
    "copperbelt", "lufilian", "konkola", "chililabombwe", "chingola",
    "nchanga", "kitwe", "nkana", "mufulira", "luanshya", "chambishi",
    "chibuluma", "roan antelope", "kansanshi", "lumwana", "zambia",
    "northern rhodesia",
]

STRONG_GEOLOGY_TERMS = [
    "geolog", "géolog", "stratigraph", "structural geology", "tecton",
    "mineralisation", "mineralization", "minéralisation", "geophys", "géophys",
    "geochem", "géochim", "aeromagnetic", "gravity survey", "drill",
    "borehole", "sondage", "ore deposit", "gisement", "litholog",
    "lithostratigraph", "facies", "sediment", "metamorph",
    "roan group", "kundelungu", "katanga supergroup",
    "prospection", "exploration geology",
]

SPECIFIC_PLACES = [
    "copperbelt", "lufilian", "konkola", "chililabombwe", "chingola",
    "nchanga", "kitwe", "nkana", "mufulira", "luanshya", "chambishi",
    "chibuluma", "roan antelope", "kansanshi", "lumwana",
]

CATALOGUE_FIELDS = [
    "record_id", "platform", "institution", "catalogue_id", "source_url",
    "title", "authors", "organisation_authors", "publication_year",
    "document_type", "data_types", "locations", "provinces", "map_scale",
    "page_count", "file_size_bytes", "access_status", "preview_url",
    "file_url", "local_path", "sha256", "acquisition_status",
    "extraction_status", "priority", "priority_reason", "matched_queries",
    "independent_geological_evidence", "notes", "discovered_at_utc",
]

REGISTER_FIELDS = [
    "register_id", "record_id", "platform", "priority", "title", "authors",
    "publication_year", "document_type", "locations", "access_status",
    "acquisition_action", "planned_file_path", "source_url", "file_url",
    "local_path", "sha256", "page_count", "status", "reason",
    "last_attempt_utc",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")[:80]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_atomic(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def fetch(url: str, cache: Path, refresh: bool, accept: str) -> bytes:
    if cache.exists() and not refresh:
        return cache.read_bytes()
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with urlopen(request, timeout=70, context=ssl.create_default_context()) as response:
                payload = response.read()
            write_bytes_atomic(cache, payload)
            time.sleep(0.15)
            return payload
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(attempt * 2)
    raise RuntimeError(f"Could not retrieve {url}: {last_error}")


def fetch_json(url: str, cache: Path, refresh: bool) -> dict:
    payload = fetch(url, cache, refresh, "application/json,text/plain,*/*")
    return json.loads(payload.decode("utf-8"))


def publication_year(text: str, fallback: str = "") -> str:
    match = re.search(r"\b(18\d{2}|19\d{2}|20\d{2})\b", text or "")
    return match.group(1) if match else fallback


def priority_for(record: dict[str, object]) -> tuple[int, str]:
    title = str(record.get("title") or "").lower()
    text = " ".join(str(record.get(field) or "") for field in (
        "title", "locations", "data_types", "organisation_authors", "notes"
    )).lower()
    place_hits = [term for term in SPECIFIC_PLACES if term in text]
    title_place = [term for term in SPECIFIC_PLACES if term in title]
    strong_geo_title = [term for term in STRONG_GEOLOGY_TERMS if term in title]
    strong_geo_text = [term for term in STRONG_GEOLOGY_TERMS if term in text]
    zambia_context = any(term in text for term in ("zambia", "northern rhodesia", "katanga"))
    document_type = str(record.get("document_type") or "").lower()
    platform = str(record.get("platform") or "")
    if platform == "Zambia National ETD Portal":
        if title_place and strong_geo_title:
            return 1, "Copperbelt/Lufilian thesis with geological term in title"
        if strong_geo_title and zambia_context:
            return 2, "Zambian geological thesis"
        if title_place and any(
            term in title for term in ("mine", "mining", "metallurg", "geotechnical", "ore", "copper", "cobalt")
        ):
            return 3, "mining, metallurgy or geotechnical thesis"
        return 4, "context only"
    if place_hits and strong_geo_title:
        return 1, "named Copperbelt/Lufilian place plus geological term in title"
    if document_type in ("map", "cross_section") and zambia_context and strong_geo_text:
        return 1, "national or regional geological map"
    if place_hits and strong_geo_text:
        return 2, "single core place with geological evidence"
    if strong_geo_text and zambia_context:
        return 2, "Zambian or cross-border geological evidence"
    if strong_geo_title:
        return 2, "geological title without a specific Copperbelt locality"
    if title_place and any(term in title for term in ("mine", "mining", "metallurg", "ore", "copper", "cobalt")):
        return 3, "mining context without primary geological evidence"
    return 4, "context only"


def finalise(record: dict[str, object]) -> dict[str, object]:
    priority, reason = priority_for(record)
    record["priority"] = priority
    record["priority_reason"] = reason
    record.setdefault("discovered_at_utc", utc_now())
    return record


def gsd_records(refresh: bool) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    records: dict[str, dict[str, object]] = {}
    log: list[dict[str, object]] = []
    for term in QUERY_TERMS:
        term_slug = slug(term)
        first_url = f"{GSD_API}/File?search={quote(term)}&page=1&limit=100"
        first_path = RAW_ROOT / "gsd" / f"{term_slug}_p001.json"
        try:
            first = fetch_json(first_url, first_path, refresh)
        except RuntimeError as exc:
            log.append({"platform": "gsd", "query": term, "status": "error", "error": str(exc)})
            continue
        documents = list(first.get("documents") or [])
        total = int(first.get("total") or len(documents))
        pages = int(first.get("totalPages") or 1)
        for page in range(2, min(pages, 20) + 1):
            url = f"{GSD_API}/File?search={quote(term)}&page={page}&limit=100"
            cache = RAW_ROOT / "gsd" / f"{term_slug}_p{page:03d}.json"
            try:
                payload = fetch_json(url, cache, refresh)
            except RuntimeError:
                break
            documents.extend(payload.get("documents") or [])
        log.append({
            "platform": "gsd", "query": term, "status": "ok", "total": total,
            "returned": len(documents), "pages_requested": min(pages, 20),
            "url": first_url,
        })
        for document in documents:
            document_id = str(document.get("id") or "")
            if not document_id:
                continue
            record_id = f"GSD-{document_id}"
            location_links = document.get("DocumentLocation") or []
            locations = [str((item.get("location") or {}).get("name") or "") for item in location_links]
            provinces = [str((item.get("location") or {}).get("province") or "") for item in location_links]
            organisations = [
                str((item.get("organisation") or {}).get("name") or "")
                for item in (document.get("DocumentOrganisationAuthor") or [])
            ]
            title = str(document.get("title") or "")
            record = records.get(record_id)
            if record is None:
                record = {
                    "record_id": record_id,
                    "platform": "Geological Survey of Zambia store",
                    "institution": "Geological Survey of Zambia",
                    "catalogue_id": document_id,
                    "source_url": f"{GSD_API}/File/{document_id}",
                    "title": title,
                    "authors": "; ".join(str(item) for item in (document.get("authors") or [])),
                    "organisation_authors": "; ".join(sorted(set(filter(None, organisations)))),
                    "publication_year": publication_year(title),
                    "document_type": str(document.get("documentType") or ""),
                    "data_types": "; ".join(document.get("dataTypes") or []),
                    "locations": "; ".join(sorted(set(filter(None, locations)))),
                    "provinces": "; ".join(sorted(set(filter(None, provinces)))),
                    "map_scale": str(document.get("scale") or ""),
                    "page_count": str(document.get("pageCount") or ""),
                    "file_size_bytes": "",
                    "access_status": "metadata_and_preview_public; full_file_requires_account_or_purchase",
                    "preview_url": f"{GSD_API}/File/view_thumbnail/{document_id}",
                    "file_url": f"{GSD_API}/File/download/{document_id}",
                    "local_path": "",
                    "sha256": "",
                    "acquisition_status": "catalogue_only",
                    "extraction_status": "not_started",
                    "notes": "GSD year field is uniform archive metadata and is not treated as publication year.",
                    "matched_queries": "",
                }
                records[record_id] = record
            previous = str(record.get("matched_queries") or "")
            record["matched_queries"] = "; ".join(sorted(set(filter(None, previous.split("; ") + [term]))))
    return [finalise(record) for record in records.values()], log


def netd_detail(record_id: str, item_url: str, refresh: bool) -> dict[str, object]:
    cache = RAW_ROOT / "netd_items" / f"{slug(record_id)}.html"
    payload = fetch(item_url, cache, refresh, "text/html,application/xhtml+xml")
    document = html.fromstring(payload.decode("utf-8", "replace"))
    meta: dict[str, list[str]] = {}
    for node in document.xpath("//meta[@content]"):
        key = node.get("name") or node.get("property")
        if key:
            meta.setdefault(key, []).append(node.get("content") or "")
    first = lambda key: (meta.get(key) or [""])[0]
    title = first("citation_title") or " ".join(document.xpath("//title//text()")).strip()
    abstract = first("citation_abstract") or first("description")
    authors = "; ".join(dict.fromkeys(filter(None, meta.get("citation_author") or [])))
    source_url = ""
    for link in document.xpath("//a[@href]"):
        href = link.get("href") or ""
        label = " ".join(link.xpath(".//text()")).strip()
        if "Source Record" in label or "rims.unza.ac.zm" in href:
            source_url = href
            break
    pdf_url = first("citation_pdf_url") or f"{item_url}/pdf"
    record = {
        "record_id": record_id,
        "platform": "Zambia National ETD Portal",
        "institution": first("citation_dissertation_institution") or "Zambian university",
        "catalogue_id": record_id.removeprefix("NETD-"),
        "source_url": item_url,
        "title": title,
        "authors": authors,
        "organisation_authors": "",
        "publication_year": publication_year(first("citation_publication_date"), first("citation_publication_date")),
        "document_type": "thesis_or_dissertation",
        "data_types": "",
        "locations": "",
        "provinces": "",
        "map_scale": "",
        "page_count": "",
        "file_size_bytes": "",
        "access_status": "public_pdf" if pdf_url else "metadata_only",
        "preview_url": "",
        "file_url": pdf_url,
        "local_path": "",
        "sha256": "",
        "acquisition_status": "pending_download" if pdf_url else "metadata_only",
        "extraction_status": "not_started",
        "notes": (f"Repository record: {source_url}; " if source_url else "") + "abstract: " + abstract[:700],
        "matched_queries": "",
    }
    return finalise(record)


def netd_records(refresh: bool) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ids: dict[str, str] = {}
    matched: dict[str, set[str]] = {}
    log: list[dict[str, object]] = []
    for term in QUERY_TERMS:
        term_slug = slug(term)
        page = 1
        total_pages = 1
        total = 0
        returned = 0
        while page <= min(total_pages, 20):
            url = f"{NETD_BASE}/catalog.json?q={quote(term)}&search_field=all_fields&per_page=100&page={page}"
            cache = RAW_ROOT / "netd" / f"{term_slug}_p{page:03d}.json"
            try:
                payload = fetch_json(url, cache, refresh)
            except RuntimeError as exc:
                log.append({"platform": "netd", "query": term, "status": "error", "error": str(exc)})
                break
            pages = payload.get("meta", {}).get("pages", {})
            total = int(pages.get("total_count") or 0)
            total_pages = int(pages.get("total_pages") or 1)
            for item in payload.get("data") or []:
                item_id = str(item.get("id") or "")
                self_url = str((item.get("links") or {}).get("self") or "")
                if item_id and self_url:
                    ids[item_id] = self_url
                    matched.setdefault(item_id, set()).add(term)
                    returned += 1
            page += 1
        log.append({
            "platform": "netd", "query": term, "status": "ok", "total": total,
            "returned": returned, "pages_requested": min(total_pages, 20),
            "url": f"{NETD_BASE}/catalog.json?q={quote(term)}&search_field=all_fields&per_page=100",
        })
    records: list[dict[str, object]] = []
    for item_id, item_url in sorted(ids.items()):
        try:
            record = netd_detail(item_id, item_url, refresh)
        except RuntimeError as exc:
            log.append({"platform": "netd", "query": "detail", "status": "error", "error": f"{item_id}: {exc}"})
            continue
        record["matched_queries"] = "; ".join(sorted(matched[item_id]))
        records.append(finalise(record))
    return records, log


def rdc_records(refresh: bool) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    catalogue = read_csv(RDC_LOCAL_CATALOGUE)
    manifest_rows = read_csv(RDC_LOCAL_MANIFEST) if RDC_LOCAL_MANIFEST.exists() else []
    manifest = {row["id"]: row for row in manifest_rows}
    matched: dict[str, set[str]] = {}
    for row in catalogue:
        haystack = " ".join(
            str(row.get(key) or "") for key in ("title", "location", "document_type", "data_type")
        ).lower()
        for term in QUERY_TERMS:
            if term.lower() in haystack:
                matched.setdefault(row["id"], set()).add(term)
    records: list[dict[str, object]] = []
    log: list[dict[str, object]] = []
    for document_id in sorted(matched, key=int):
        url = f"{RDC_API}/{document_id}/"
        cache = RAW_ROOT / "rdc_rocks" / f"document_{document_id}.json"
        try:
            payload = fetch_json(url, cache, refresh)
        except RuntimeError as exc:
            log.append({"platform": "rdc_rocks", "query": "detail", "status": "error", "error": f"{document_id}: {exc}"})
            continue
        metadata = {item.get("label"): item.get("value") for item in payload.get("metadata") or []}
        title = str(payload.get("title") or "")
        prior = manifest.get(document_id, {})
        local_path = str(prior.get("local_path") or "")
        record = {
            "record_id": f"RDC-{document_id}",
            "platform": "RDC.ROCKS",
            "institution": "University of Lubumbashi",
            "catalogue_id": document_id,
            "source_url": f"https://rdc.rocks/api/v1/catalog/documents/{document_id}/",
            "title": title,
            "authors": str(metadata.get("authors") or ""),
            "organisation_authors": str(metadata.get("organization_authors") or ""),
            "publication_year": str(metadata.get("date_published") or publication_year(title)),
            "document_type": str(metadata.get("document_type") or ""),
            "data_types": str(metadata.get("data_types") or ""),
            "locations": str(metadata.get("locations") or ""),
            "provinces": "",
            "map_scale": str(metadata.get("scale") or ""),
            "page_count": str(metadata.get("page_count") or ""),
            "file_size_bytes": str(prior.get("bytes") or ""),
            "access_status": "public_download",
            "preview_url": f"https://rdc.rocks{payload.get('hero_image_url') or ''}" if payload.get("hero_image_url") else "",
            "file_url": f"https://rdc.rocks/data-url/{document_id}/",
            "local_path": local_path,
            "sha256": str(prior.get("sha256") or ""),
            "acquisition_status": "acquired_verified" if local_path and Path(ROOT / local_path).exists() else "pending_download",
            "extraction_status": "ocr_available" if local_path and Path(ROOT / local_path).suffix.lower() == ".pdf" else "not_started",
            "notes": str(payload.get("summary") or "")[:700],
            "matched_queries": "; ".join(sorted(matched[document_id])),
        }
        records.append(finalise(record))
    return records, log


def acquisition_register(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ordered = sorted(records, key=lambda item: (int(item["priority"]), str(item["platform"]), str(item["title"])))
    for record in ordered:
        if int(record["priority"]) > 2:
            continue
        platform = str(record["platform"])
        if platform == "RDC.ROCKS":
            planned = record["local_path"] or f"data/raw/rdc_rocks/files/pdf/RDC_{int(record['catalogue_id']):04d}.pdf"
            action = "reuse_or_download_public_record"
            status = record["acquisition_status"]
        elif platform == "Zambia National ETD Portal":
            planned = f"data/raw/zambia_acquisition/files/netd/{record['record_id']}.pdf"
            action = "download_public_pdf"
            status = "ready_to_download" if record["access_status"] == "public_pdf" else "needs_manual_resolution"
        else:
            planned = f"data/raw/zambia_acquisition/files/gsd_previews/{record['record_id']}.png"
            action = "download_public_preview_and_request_full_file"
            status = "preview_ready_full_file_restricted"
        rows.append({
            "register_id": f"ZM-REG-{len(rows) + 1:04d}",
            "record_id": record["record_id"],
            "platform": platform,
            "priority": record["priority"],
            "title": record["title"],
            "authors": record["authors"],
            "publication_year": record["publication_year"],
            "document_type": record["document_type"],
            "locations": record["locations"],
            "access_status": record["access_status"],
            "acquisition_action": action,
            "planned_file_path": planned,
            "source_url": record["source_url"],
            "file_url": record["file_url"],
            "local_path": record["local_path"],
            "sha256": record["sha256"],
            "page_count": record["page_count"],
            "status": status,
            "reason": record["priority_reason"],
            "last_attempt_utc": "",
        })
    return rows


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build the Zambian acquisition catalogue.")
    parser.add_argument("--refresh", action="store_true", help="Refetch cached platform responses.")
    args = parser.parse_args()

    gsd, gsd_log = gsd_records(args.refresh)
    netd, netd_log = netd_records(args.refresh)
    rdc, rdc_log = rdc_records(args.refresh)
    records = gsd + netd + rdc
    records.sort(key=lambda item: (int(item["priority"]), str(item["platform"]), str(item["title"])))
    log = gsd_log + netd_log + rdc_log

    write_csv_atomic(DATA_ROOT / "metadata" / "catalogue_records_v1.csv", records, CATALOGUE_FIELDS)
    write_csv_atomic(
        DATA_ROOT / "metadata" / "query_log_v1.csv",
        log,
        list(log[0]) if log else ["platform", "query", "status"],
    )
    register = acquisition_register(records)
    write_csv_atomic(DATA_ROOT / "metadata" / "acquisition_register_v1.csv", register, REGISTER_FIELDS)

    counts: dict[str, int] = {}
    for record in records:
        key = f"{record['platform']} P{record['priority']}"
        counts[key] = counts.get(key, 0) + 1
    summary = {
        "generated_at_utc": utc_now(),
        "records": len(records),
        "register_rows": len(register),
        "counts": counts,
        "queries": len(QUERY_TERMS),
        "errors": [row for row in log if row.get("status") == "error"],
    }
    write_bytes_atomic(
        DATA_ROOT / "metadata" / "discovery_summary_v1.json",
        json.dumps(summary, indent=2, ensure_ascii=False).encode("utf-8"),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
