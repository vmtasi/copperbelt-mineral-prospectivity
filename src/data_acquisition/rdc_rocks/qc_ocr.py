"""Check the local OCR output and write a small quality digest."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = PROJECT_ROOT / "data/raw/rdc_rocks/metadata/acquisition_manifest_v1.csv"
OCR_DIR = PROJECT_ROOT / "data/raw/rdc_rocks/derived/ocr"
STATE = OCR_DIR / "state_v1.json"
OUTPUT = OCR_DIR / "qc_v1.json"


def main() -> int:
    rows = {row["id"]: row for row in csv.DictReader(MANIFEST.open(encoding="utf-8-sig", newline=""))}
    documents = json.loads(STATE.read_text(encoding="utf-8"))["documents"] if STATE.is_file() else {}

    documents_out = []
    flagged = []
    for doc_id, doc in documents.items():
        row = rows.get(doc_id, {})
        pages = doc.get("pages", {})
        if not pages:
            continue
        chars = [page.get("chars", 0) for page in pages.values()]
        empty = sum(1 for value in chars if value == 0)
        expected = int(row.get("page_count_actual") or 0)
        entry = {
            "id": doc_id,
            "source_role": row.get("source_role", ""),
            "pages_ocrd": len(pages),
            "pages_expected": expected,
            "empty_pages": empty,
            "empty_ratio": round(empty / len(pages), 3),
            "chars": sum(chars),
            "median_chars": sorted(chars)[len(chars) // 2],
        }
        documents_out.append(entry)
        if entry["empty_ratio"] > 0.5 and len(pages) >= 8:
            flagged.append({"id": doc_id, "reason": "mostly_empty"})
        elif entry["median_chars"] < 20 and len(pages) >= 8:
            flagged.append({"id": doc_id, "reason": "low_text_density"})
        elif expected > len(pages) and not row.get("local_path", "").lower().endswith(".tif"):
            flagged.append({"id": doc_id, "reason": "first_pass_only", "pages_missing": expected - len(pages)})

    digest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "documents": len(documents_out),
        "pages": sum(entry["pages_ocrd"] for entry in documents_out),
        "empty_pages": sum(entry["empty_pages"] for entry in documents_out),
        "flagged": flagged,
    }
    OUTPUT.write_text(json.dumps(digest, indent=2), encoding="utf-8")
    print(json.dumps({k: digest[k] for k in ("documents", "pages", "empty_pages")}))
    print(json.dumps({"flagged": len(flagged), "first_run": flagged[:8]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
