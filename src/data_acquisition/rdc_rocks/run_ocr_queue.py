"""Run the local OCR queue unattended in bounded batches.

Shallow mode covers every document once (first N pages); deep mode returns to
documents with pages left. Python does the bulk work and writes a small status
digest, so Codex never needs document pages or full OCR text in a request.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = PROJECT_ROOT / "data/raw/rdc_rocks/metadata/acquisition_manifest_v1.csv"
OCR_DIR = PROJECT_ROOT / "data/raw/rdc_rocks/derived/ocr"
STATE = OCR_DIR / "state_v1.json"
OCR_JSONL = OCR_DIR / "ocr_manifest_v1.jsonl"
STATUS = OCR_DIR / "status_v1.json"
OCR_SCRIPT = Path(__file__).with_name("ocr_pages.py")

ROLE_ORDER = (
    "stratigraphic_report", "structural_study", "occurrence_inventory",
    "structural_map", "stratigraphic_or_mine_section", "geological_map",
    "exploration_report", "site_or_geological_report", "drilling_record", "geophysical_map",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_rows() -> list[dict[str, str]]:
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_documents() -> dict:
    if not STATE.is_file():
        return {}
    return json.loads(STATE.read_text(encoding="utf-8")).get("documents", {})


def remaining_rows(rows: list[dict[str, str]], documents: dict, deep: bool, skip: set[str]) -> list[dict[str, str]]:
    rank = {role: index for index, role in enumerate(ROLE_ORDER)}
    todo = []
    for row in rows:
        doc_id = row["id"]
        if row["text_status"] == "text_layer" or row["local_path"].lower().endswith(".tif") or doc_id in skip:
            continue
        done = len(documents.get(doc_id, {}).get("pages", {}))
        expected = int(row["page_count_actual"] or 0)
        if deep:
            if done and done < expected:
                todo.append(row)
        elif doc_id not in documents:
            todo.append(row)
    return sorted(todo, key=lambda row: (rank.get(row["source_role"], 99), int(row["id"])))


def write_status(rows: list[dict[str, str]], remaining: list[dict[str, str]], last_batch: list[str]) -> dict:
    documents = load_documents()
    pages = sum(len(doc.get("pages", {})) for doc in documents.values())
    chars = sum(p.get("chars", 0) for doc in documents.values() for p in doc.get("pages", {}).values())
    errors = 0
    if OCR_JSONL.is_file():
        with OCR_JSONL.open("r", encoding="utf-8") as handle:
            errors = sum(1 for line in handle if '"status": "error"' in line)
    by_role: dict[str, int] = {}
    for row in remaining:
        by_role[row["source_role"]] = by_role.get(row["source_role"], 0) + 1
    status = {
        "generated_at_utc": utc_now(),
        "documents_ocrd": len(documents),
        "pages_ocrd": pages,
        "characters_ocrd": chars,
        "page_errors": errors,
        "remaining_documents": len(remaining),
        "remaining_pages": sum(
            int(r["page_count_actual"] or 0) - len(documents.get(r["id"], {}).get("pages", {})) for r in remaining
        ),
        "remaining_by_role": dict(sorted(by_role.items(), key=lambda item: -item[1])),
        "last_batch_ids": last_batch,
        "state_path": str(STATE.relative_to(PROJECT_ROOT)),
    }
    tmp = STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status, indent=2), encoding="utf-8")
    os.replace(tmp, STATUS)
    return status


def run_document(row: dict[str, str], documents: dict, batch_pages: int, deep: bool, log: Path) -> int:
    done = len(documents.get(row["id"], {}).get("pages", {}))
    target = done + batch_pages if deep else batch_pages
    command = [
        sys.executable, str(OCR_SCRIPT), "--ids", row["id"],
        "--max-docs", "1", "--max-pages-per-doc", str(target),
    ]
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n=== {utc_now()} {row['id']} target={target} ===\n")
        handle.flush()
        return subprocess.call(command, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("shallow", "deep"), default="shallow")
    parser.add_argument("--batch-docs", type=int, default=5)
    parser.add_argument("--batch-pages", type=int, default=10)
    parser.add_argument("--max-batches", type=int, default=0, help="0 = until queue is empty")
    parser.add_argument("--max-minutes", type=float, default=0, help="0 = no limit")
    parser.add_argument("--dry-run", action="store_true", help="print the next batch and exit")
    args = parser.parse_args()

    log = PROJECT_ROOT / "work/ocr_runs.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    assert len(rows) == 118, f"expected 118 manifest rows, got {len(rows)}"
    deep = args.mode == "deep"
    skip: set[str] = set()
    documents = load_documents()
    remaining = remaining_rows(rows, documents, deep, skip)
    if args.dry_run:
        print(json.dumps({"mode": args.mode, "remaining": len(remaining), "next": [r["id"] for r in remaining[: args.batch_docs]]}))
        return 0

    started = time.monotonic()
    batches = 0
    while remaining and (args.max_batches == 0 or batches < args.max_batches):
        if args.max_minutes and (time.monotonic() - started) / 60 >= args.max_minutes:
            break
        batch = remaining[: args.batch_docs]
        failed = []
        for row in batch:
            if run_document(row, documents, args.batch_pages, deep, log) != 0:
                failed.append(row["id"])
        skip.update(failed)
        documents = load_documents()
        batches += 1
        remaining = remaining_rows(rows, documents, deep, skip)
        status = write_status(rows, remaining, [r["id"] for r in batch])
        print(json.dumps({"batch": batches, "ids": [r["id"] for r in batch], "failed": failed, "remaining": status["remaining_documents"]}), flush=True)
    write_status(rows, remaining_rows(rows, load_documents(), deep, skip), [])
    print(f"done batches={batches} remaining={len(remaining_rows(rows, load_documents(), deep, skip))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
