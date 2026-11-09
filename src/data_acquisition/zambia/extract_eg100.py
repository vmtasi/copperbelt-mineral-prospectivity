from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[3]
PDF = next((ROOT / "literature").glob("Economic geology * hundredth anniversary volume *.pdf"))
OUT = ROOT / "data" / "derived" / "eg100_2005"
START_PDF_PAGE = 981
END_PDF_PAGE = 1016
START_PRINTED_PAGE = 965


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    pages_dir = OUT / "chapter_pages"
    layout_dir = OUT / "chapter_pages_layout"
    pages_dir.mkdir(parents=True, exist_ok=True)
    layout_dir.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(PDF))
    combined = []
    characters = 0
    for pdf_page in range(START_PDF_PAGE, END_PDF_PAGE + 1):
        printed_page = START_PRINTED_PAGE + (pdf_page - START_PDF_PAGE)
        page = reader.pages[pdf_page - 1]
        text = page.extract_text() or ""
        layout = page.extract_text(extraction_mode="layout") or ""
        characters += len(text)
        path = pages_dir / f"p{printed_page:04d}.txt"
        path.write_text(text + ("\n" if text else ""), encoding="utf-8")
        (layout_dir / f"p{printed_page:04d}.txt").write_text(layout + ("\n" if layout else ""), encoding="utf-8")
        combined.append(f"\n===== PRINTED PAGE {printed_page} / PDF PAGE {pdf_page} =====\n{text}")
    (OUT / "chapter_text_v1.txt").write_text("".join(combined), encoding="utf-8")
    manifest = {
        "source_pdf": str(PDF.relative_to(ROOT)),
        "source_sha256": sha256(PDF),
        "pdf_pages": len(reader.pages),
        "chapter_pdf_pages": [START_PDF_PAGE, END_PDF_PAGE],
        "chapter_printed_pages": [START_PRINTED_PAGE, START_PRINTED_PAGE + (END_PDF_PAGE - START_PDF_PAGE)],
        "characters": characters,
        "page_text_directory": str(pages_dir.relative_to(ROOT)),
    }
    (OUT / "chapter_extraction_manifest_v1.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"extracted {END_PDF_PAGE - START_PDF_PAGE + 1} pages, {characters} chars -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
