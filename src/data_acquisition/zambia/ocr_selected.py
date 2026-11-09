from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "data_acquisition" / "rdc_rocks"))

from ocr_pages import (  # noqa: E402
    DEFAULT_PDFTOPPM,
    DEFAULT_POWERSHELL,
    DEFAULT_POWERSCRIPT,
    append_jsonl,
    load_state,
    ocr_best,
    render_page,
    save_json,
    shrink_if_needed,
    utc_now,
)

DATA_ROOT = ROOT / "data" / "raw" / "zambia_acquisition"
MANIFEST = DATA_ROOT / "metadata" / "download_manifest_v1.csv"
OUT = DATA_ROOT / "derived" / "ocr"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="OCR selected pages of newly acquired PDFs locally")
    parser.add_argument("--ids", default="")
    parser.add_argument("--first-pages", type=int, default=5)
    parser.add_argument("--max-docs", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--lang", default="en-US")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    records = [
        row for row in read_csv(MANIFEST)
        if row["status"] in ("verified", "already_present") and row["content_type"] == "application/pdf"
    ]
    if args.ids:
        allowed = {value.strip() for value in args.ids.split(",") if value.strip()}
        records = [row for row in records if row["record_id"] in allowed]
    records = records[: args.max_docs]

    state_path = OUT / "state_selected_v1.json"
    manifest_path = OUT / "ocr_selected_manifest_v1.jsonl"
    state = load_state(state_path)
    processed = 0
    errors = 0
    for index, record in enumerate(records, start=1):
        pdf = ROOT / record["local_path"]
        page_count = int(record["page_count"] or 0)
        last_page = min(page_count, args.first_pages) if args.first_pages else page_count
        document = state.setdefault("documents", {}).setdefault(record["record_id"], {"pages": {}, "status": "in_progress"})
        lang = "fr-FR" if record["platform"] == "RDC.ROCKS" else args.lang
        dpi = 300 if page_count == 1 else args.dpi
        for page in range(1, last_page + 1):
            text_path = OUT / record["record_id"] / "text" / f"p{page:03d}.txt"
            if not args.force and str(page) in document["pages"] and text_path.is_file():
                continue
            image_path = OUT / record["record_id"] / "pages" / f"p{page:03d}.png"
            entry = {
                "record_id": record["record_id"],
                "platform": record["platform"],
                "title": record["title"],
                "page": page,
                "page_count": page_count,
                "dpi": dpi,
                "lang": lang,
                "ocr_at_utc": utc_now(),
            }
            try:
                if not image_path.is_file():
                    render_page(DEFAULT_PDFTOPPM, pdf, page, image_path, dpi)
                    shrink_if_needed(image_path)
                text, rotation = ocr_best(DEFAULT_POWERSHELL, DEFAULT_POWERSCRIPT, image_path, lang, args.timeout)
                text_path.parent.mkdir(parents=True, exist_ok=True)
                text_path.write_text(text + ("\n" if text else ""), encoding="utf-8")
                document["pages"][str(page)] = {"chars": len(text), "text_path": str(text_path.relative_to(ROOT))}
                entry.update({"status": "ok" if text else "empty", "chars": len(text), "rotation_cw": rotation, "text_path": str(text_path.relative_to(ROOT))})
            except Exception as exc:
                errors += 1
                entry.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                print(f"[{index}/{len(records)}] {record['record_id']} p{page}: ERROR {exc}", flush=True)
            append_jsonl(manifest_path, entry)
            save_json(state_path, state)
            processed += 1
            if entry["status"] != "error":
                print(f"[{index}/{len(records)}] {record['record_id']} p{page}: {entry['status']} ({entry.get('chars', 0)} chars)", flush=True)
        document["status"] = "complete" if len(document["pages"]) >= last_page else "partial"
        save_json(state_path, state)
    print(f"ocr selected: {processed} pages, {errors} errors, state {state_path}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
