from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


BASE_URL = "https://rdc.rocks"
CONTENT_EXTENSIONS = {
    "application/pdf": ".pdf",
    "image/tiff": ".tif",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def latest_state(path: Path) -> dict[str, dict[str, object]]:
    records: dict[str, list[dict[str, object]]] = {}
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records.setdefault(str(record["id"]), []).append(record)
    selected = {}
    for document_id, candidates in records.items():
        successful = [item for item in candidates if item.get("status") != "failed"]
        pool = successful or candidates
        selected[document_id] = max(pool, key=lambda item: str(item.get("completed_at_utc", "")))
    return selected


def source_role(detail: dict[str, str]) -> str:
    document_type = detail.get("document_type", "")
    title = detail.get("title", "").casefold()
    data_types = detail.get("data_types", "").casefold()
    if any(token in title for token in ("inventaire", "indices", "occurrences", "gisements")):
        return "occurrence_inventory"
    if "structures" in title or "tectonique" in title or "structurale" in title:
        return "structural_map" if document_type == "map" else "structural_study"
    if "gravimetr" in title or "magnet" in title:
        return "geophysical_map"
    if document_type == "cross-section":
        return "stratigraphic_or_mine_section"
    if document_type == "map":
        return "geological_map"
    if "drilling" in data_types:
        return "drilling_record"
    if "stratigraph" in title or any(token in title for token in ("roan", "nguba", "kundelungu", "mines")):
        return "stratigraphic_report"
    if "exploration" in data_types:
        return "exploration_report"
    return "site_or_geological_report"


def main() -> int:
    parser = argparse.ArgumentParser(description="Join RDC.ROCKS metadata and acquisition evidence")
    parser.add_argument("--probe", required=True)
    parser.add_argument("--details", required=True)
    parser.add_argument("--pdf-audit", required=True)
    parser.add_argument("--state", action="append", default=[])
    parser.add_argument("--files-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    probe_rows = read_csv(Path(args.probe))
    details = {row["id"]: row for row in read_csv(Path(args.details))}
    audit = {row["id"]: row for row in read_csv(Path(args.pdf_audit))}
    states: dict[str, dict[str, object]] = {}
    for state_path in args.state:
        states.update(latest_state(Path(state_path)))
    files_root = Path(args.files_root)

    title_counts: dict[str, int] = {}
    for row in probe_rows:
        key = row["title"].strip().casefold()
        title_counts[key] = title_counts.get(key, 0) + 1

    output_rows = []
    for probe in sorted(probe_rows, key=lambda row: int(row["id"])):
        document_id = probe["id"]
        detail = details.get(document_id, {})
        content_type = probe["content_type"]
        extension = CONTENT_EXTENSIONS[content_type]
        local_path = files_root / ("pdf" if extension == ".pdf" else "tiff") / f"RDC_{int(document_id):04d}{extension}"
        state = states.get(document_id, {})
        present = local_path.is_file() and local_path.stat().st_size == int(probe["bytes_total"])
        audit_row = audit.get(document_id, {})
        title_collision = title_counts.get(probe["title"].strip().casefold(), 0) > 1
        output_rows.append(
            {
                "id": document_id,
                "title": probe["title"],
                "document_type": detail.get("document_type", ""),
                "data_types": detail.get("data_types", ""),
                "authors": detail.get("authors", ""),
                "organization_authors": detail.get("organization_authors", ""),
                "date_published": detail.get("date_published", ""),
                "scale": detail.get("scale", ""),
                "locations": detail.get("locations", ""),
                "page_count_reported": detail.get("page_count", ""),
                "page_count_actual": audit_row.get("actual_page_count", ""),
                "page_count_match": audit_row.get("page_count_match", ""),
                "text_status": audit_row.get("text_status", ""),
                "source_role": source_role(detail),
                "content_type": content_type,
                "bytes": probe["bytes_total"],
                "sha256": state.get("sha256", ""),
                "acquisition_status": "acquired_verified" if present else "not_acquired",
                "local_path": str(local_path) if present else "",
                "object_filename_untrusted": probe["object_filename"],
                "catalogue_endpoint": f"{BASE_URL}/api/v1/catalog/documents/{document_id}/",
                "download_endpoint": f"{BASE_URL}/data-url/{document_id}/",
                "title_collision": str(title_collision).lower(),
                "content_identity_note": "catalogue title checked visually against first page; object filename is not an identity field",
                "license_status": "publicly accessible; formal reuse licence not located",
                "review_note": audit_row.get("inspection_error", ""),
            }
        )

    fields = list(output_rows[0].keys())
    write_csv(Path(args.output), output_rows, fields)
    acquired = sum(row["acquisition_status"] == "acquired_verified" for row in output_rows)
    print(f"Wrote {len(output_rows)} records to {args.output}; acquired_verified={acquired}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
