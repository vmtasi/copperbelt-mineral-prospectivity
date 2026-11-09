from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = PROJECT_ROOT / "data/raw/rdc_rocks/metadata/acquisition_manifest_v1.csv"
DEFAULT_OUT = PROJECT_ROOT / "data/raw/rdc_rocks/derived/ocr"
DEFAULT_POWERSCRIPT = Path(__file__).with_name("ocr_one.ps1")
DEFAULT_PDFTOPPM = Path(r"C:\Users\vanmu\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe")
DEFAULT_POWERSHELL = Path(r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe")
PRIORITY_IDS = "3863,4055,3848,3854,3856,1300,3129,1043,1317,2376,1919,2755,3399,3731,3461"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def load_state(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"documents": {}}


def save_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def render_page(pdftoppm: Path, pdf: Path, page: int, target: Path, dpi: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    prefix = target.with_suffix("")
    subprocess.run(
        [str(pdftoppm), "-f", str(page), "-l", str(page), "-r", str(dpi), "-png", "-singlefile", str(pdf), str(prefix)],
        check=True,
        capture_output=True,
        timeout=300,
    )
    if not target.is_file():
        raise RuntimeError(f"pdftoppm did not create {target}")


def shrink_if_needed(image_path: Path, max_dimension: int = 9000) -> None:
    from PIL import Image

    with Image.open(image_path) as image:
        if max(image.size) <= max_dimension:
            return
        image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        image.convert("RGB").save(image_path)


def ocr_page(powershell: Path, script: Path, image: Path, lang: str, timeout: int) -> str:
    result = subprocess.run(
        [str(powershell), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), "-Path", str(image), "-Lang", lang],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip()[:500])
    return result.stdout.replace("\x00", "").strip()


OCR_TERMS = (
    "ECHELLE", "LEGENDE", "STRUCTURES", "TECTONIQUES", "REGION", "KATANGA",
    "GROUPE", "CARTE", "GEOLOGIQUE", "ROAN", "KOLWEZI", "STRATIGRAPHIE",
)


FRENCH_WORDS = ("DE", "LA", "DU", "DES", "ET", "LES", "DANS", "POUR", "SUR", "AVEC", "EST", "SONT", "PAR", "AU", "AUX", "CE", "CETTE", "FIGURE", "TABLEAU")


def score_text(text: str) -> int:
    words = re.findall(r"[A-Za-zÀ-ÿ]{3,}", text)
    upper = text.upper()
    term_hits = sum(upper.count(term) for term in OCR_TERMS)
    word_hits = sum(len(re.findall(rf"\b{word}\b", upper)) for word in FRENCH_WORDS)
    return min(len(words), 50) + 20 * term_hits + 2 * word_hits


def ocr_best(powershell: Path, script: Path, image: Path, lang: str, timeout: int) -> tuple[str, int]:
    best_text = ocr_page(powershell, script, image, lang, timeout)
    best_score = score_text(best_text)
    best_rotation = 0
    if best_score >= 80:
        return best_text, best_rotation
    from PIL import Image

    temporary = image.with_name(f".rotate_{image.stem}.png")
    try:
        for rotation in (90, 180, 270):
            with Image.open(image) as source:
                source.rotate(-rotation, expand=True).save(temporary)
            candidate = ocr_page(powershell, script, temporary, lang, timeout)
            candidate_score = score_text(candidate)
            if candidate_score > best_score:
                best_text, best_score, best_rotation = candidate, candidate_score, rotation
    finally:
        temporary.unlink(missing_ok=True)
    return best_text, best_rotation


def main() -> int:
    parser = argparse.ArgumentParser(description="OCR selected RDC.ROCKS PDF pages locally, one page at a time")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ids", default=PRIORITY_IDS)
    parser.add_argument("--max-docs", type=int, default=10)
    parser.add_argument("--max-pages-per-doc", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--lang", default="fr-FR")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pdftoppm", type=Path, default=DEFAULT_PDFTOPPM)
    parser.add_argument("--powershell", type=Path, default=DEFAULT_POWERSHELL)
    parser.add_argument("--ocr-script", type=Path, default=DEFAULT_POWERSCRIPT)
    args = parser.parse_args()

    rows = load_rows(args.manifest)
    ids = [value.strip() for value in args.ids.split(",") if value.strip()]
    state_path = args.out / "state_v1.json"
    manifest_path = args.out / "ocr_manifest_v1.jsonl"
    text_dir = args.out / "text"
    page_dir = args.out / "pages"
    state = load_state(state_path)
    processed = 0
    errors = 0

    for document_id in ids[: args.max_docs]:
        row = rows.get(document_id)
        if row is None:
            print(f"{document_id}: not in acquisition manifest", flush=True)
            errors += 1
            continue
        pdf = PROJECT_ROOT / row["local_path"]
        if not pdf.is_file():
            print(f"{document_id}: missing PDF {pdf}", flush=True)
            errors += 1
            continue
        page_count = int(row["page_count_actual"] or 0) or 1
        last_page = min(page_count, args.max_pages_per_doc) if args.max_pages_per_doc > 0 else page_count
        doc_state = state["documents"].setdefault(document_id, {"pages": {}, "status": "in_progress"})
        for page in range(1, last_page + 1):
            text_path = text_dir / f"RDC_{int(document_id):04d}_p{page:03d}.txt"
            if not args.force and str(page) in doc_state["pages"] and text_path.is_file():
                continue
            image_path = page_dir / f"RDC_{int(document_id):04d}_p{page:03d}.png"
            record = {
                "id": document_id,
                "title": row["title"],
                "document_type": row["document_type"],
                "pdf_path": row["local_path"],
                "pdf_sha256": row["sha256"],
                "page": page,
                "page_count": page_count,
                "dpi": args.dpi,
                "lang": args.lang,
                "ocr_at_utc": utc_now(),
            }
            try:
                if not image_path.is_file():
                    render_page(args.pdftoppm, pdf, page, image_path, args.dpi)
                    shrink_if_needed(image_path)
                text, rotation = ocr_best(args.powershell, args.ocr_script, image_path, args.lang, args.timeout)
                text_dir.mkdir(parents=True, exist_ok=True)
                text_path.write_text(text + ("\n" if text else ""), encoding="utf-8")
                doc_state["pages"][str(page)] = {"chars": len(text), "text_path": str(text_path.relative_to(PROJECT_ROOT))}
                record.update({"text_path": str(text_path.relative_to(PROJECT_ROOT)), "chars": len(text), "status": "ok" if text else "empty", "rotation_cw": rotation})
            except Exception as exc:
                record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                errors += 1
                print(f"{document_id} p{page}: ERROR {exc}", flush=True)
            append_jsonl(manifest_path, record)
            save_json(state_path, state)
            processed += 1
            print(f"{document_id} p{page}: {record['status']} ({record.get('chars', 0)} chars)", flush=True)
        doc_state["status"] = "complete" if len(doc_state["pages"]) >= last_page else "partial"
        save_json(state_path, state)

    print(f"done pages={processed} errors={errors} state={state_path}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
