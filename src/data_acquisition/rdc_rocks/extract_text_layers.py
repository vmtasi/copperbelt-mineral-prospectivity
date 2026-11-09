from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = PROJECT_ROOT / "data/raw/rdc_rocks/metadata/acquisition_manifest_v1.csv"
DEFAULT_OUT = PROJECT_ROOT / "data/raw/rdc_rocks/derived/text"


def load_state(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"documents": {}}


def save_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract embedded text from RDC.ROCKS PDFs with a text layer")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows = list(csv.DictReader(args.manifest.open("r", encoding="utf-8-sig", newline="")))
    state_path = args.out / "text_layer_state_v1.json"
    manifest_path = args.out / "text_layer_manifest_v1.jsonl"
    state = load_state(state_path)
    done = 0
    errors = 0
    for row in sorted((item for item in rows if item["text_status"] == "text_layer"), key=lambda item: int(item["id"])):
        document_id = row["id"]
        output_path = args.out / f"RDC_{int(document_id):04d}.txt"
        if not args.force and document_id in state["documents"] and output_path.is_file():
            continue
        pdf_path = PROJECT_ROOT / row["local_path"]
        record = {
            "id": document_id,
            "title": row["title"],
            "document_type": row["document_type"],
            "pdf_path": row["local_path"],
            "pdf_sha256": row["sha256"],
            "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            reader = PdfReader(pdf_path)
            parts = []
            for index, page in enumerate(reader.pages, start=1):
                parts.append(f"\n\n=== PAGE {index} ===\n\n{page.extract_text() or ''}")
            text = "".join(parts).strip()
            args.out.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text + ("\n" if text else ""), encoding="utf-8")
            state["documents"][document_id] = {
                "pages": len(reader.pages),
                "chars": len(text),
                "text_path": str(output_path.relative_to(PROJECT_ROOT)),
                "source": "embedded_pdf_text",
            }
            record.update({"pages": len(reader.pages), "chars": len(text), "status": "ok", "text_path": str(output_path.relative_to(PROJECT_ROOT))})
            done += 1
        except Exception as exc:
            record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            errors += 1
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        save_state(state_path, state)
        print(f"{document_id}: {record['status']} ({record.get('pages', 0)} pages, {record.get('chars', 0)} chars)", flush=True)
    print(f"done docs={done} errors={errors} state={state_path}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
