from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT / "data" / "raw" / "zambia_acquisition"
MANIFEST = DATA_ROOT / "metadata" / "download_manifest_v1.csv"
OUT = DATA_ROOT / "derived" / "text"
SUMMARY = DATA_ROOT / "metadata" / "text_extraction_summary_v1.csv"
FIELDS = [
    "record_id", "platform", "title", "page_count", "pages_extracted",
    "characters", "chars_per_page", "text_status", "local_path", "extracted_at_utc",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def extract(record: dict[str, str], page_limit: int) -> dict[str, object]:
    pdf = ROOT / record["local_path"]
    target = OUT / record["record_id"]
    reader = PdfReader(str(pdf))
    pages = len(reader.pages)
    last_page = min(pages, page_limit) if page_limit else pages
    target.mkdir(parents=True, exist_ok=True)
    characters = 0
    extracted = 0
    for number in range(1, last_page + 1):
        path = target / f"p{number:03d}.txt"
        if not path.is_file():
            text = reader.pages[number - 1].extract_text() or ""
            path.write_text(text + ("\n" if text else ""), encoding="utf-8")
        text = path.read_text(encoding="utf-8")
        characters += len(text)
        extracted += 1
    per_page = characters / extracted if extracted else 0
    status = "digital_text" if per_page >= 400 else "sparse" if per_page >= 40 else "image_only_or_sparse"
    return {
        "record_id": record["record_id"],
        "platform": record["platform"],
        "title": record["title"],
        "page_count": pages,
        "pages_extracted": extracted,
        "characters": characters,
        "chars_per_page": round(per_page, 1),
        "text_status": status,
        "local_path": str(target.relative_to(ROOT)),
        "extracted_at_utc": utc_now(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract embedded PDF text locally, one document at a time")
    parser.add_argument("--platform", choices=["NETD", "RDC", "all"], default="all")
    parser.add_argument("--ids", default="")
    parser.add_argument("--max-docs", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=0)
    args = parser.parse_args()

    records = [
        row for row in read_csv(MANIFEST)
        if row["status"] in ("verified", "already_present")
        and row["content_type"] == "application/pdf"
    ]
    if args.platform == "NETD":
        records = [row for row in records if row["platform"] == "Zambia National ETD Portal"]
    elif args.platform == "RDC":
        records = [row for row in records if row["platform"] == "RDC.ROCKS"]
    if args.ids:
        allowed = {value.strip() for value in args.ids.split(",") if value.strip()}
        records = [row for row in records if row["record_id"] in allowed]
    records = records[: args.max_docs]

    results = []
    errors = 0
    for index, record in enumerate(records, start=1):
        try:
            result = extract(record, args.max_pages)
            results.append(result)
            print(
                f"[{index}/{len(records)}] {record['record_id']}: "
                f"{result['pages_extracted']}/{result['page_count']} pp, "
                f"{result['chars_per_page']} chars/page, {result['text_status']}",
                flush=True,
            )
        except Exception as exc:
            errors += 1
            print(f"[{index}/{len(records)}] {record['record_id']}: ERROR {type(exc).__name__}: {exc}", flush=True)

    merged = {row["record_id"]: row for row in read_csv(SUMMARY)} if SUMMARY.is_file() else {}
    merged.update({row["record_id"]: row for row in results})
    write_csv_atomic(SUMMARY, list(merged.values()))
    assert all(int(row["pages_extracted"]) <= int(row["page_count"]) for row in merged.values())
    digital = sum(row["text_status"] == "digital_text" for row in merged.values())
    print(f"text: {len(merged)} documents, {digital} with usable embedded text, {errors} errors, summary {SUMMARY}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
