from __future__ import annotations

import csv
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT / "data" / "raw" / "zambia_acquisition"
SUMMARY = DATA_ROOT / "metadata" / "text_extraction_summary_v1.csv"
OUT = DATA_ROOT / "derived" / "evidence_index" / "page_hits_v1.csv"
TERMS = {
    "deposit": 3, "occurrence": 3, "mineralisation": 3, "mineralization": 3,
    "copper": 2, "cobalt": 2, "mine": 2, "prospect": 2, "orebody": 3,
    "assay": 2, "grade": 2, "reserve": 1, "host": 1, "drill": 1,
    "latitude": 3, "longitude": 3, "grid reference": 4, "easting": 3, "northing": 3,
}
FIELDS = ["record_id", "platform", "text_source", "page", "chars", "score", "matched_terms", "snippet"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def page_files(record: dict[str, str]) -> list[tuple[str, Path]]:
    files = []
    text_dir = DATA_ROOT / "derived" / "text" / record["record_id"]
    if text_dir.is_dir():
        files.extend(("embedded_text", path) for path in sorted(text_dir.glob("p*.txt")))
    ocr_dir = DATA_ROOT / "derived" / "ocr" / record["record_id"] / "text"
    if ocr_dir.is_dir():
        files.extend(("selected_ocr", path) for path in sorted(ocr_dir.glob("p*.txt")))
    return files


def score_page(text: str) -> tuple[int, list[str], str]:
    lower = text.lower()
    hits = []
    score = 0
    first_position = len(text)
    for term, weight in TERMS.items():
        matches = list(re.finditer(rf"\b{re.escape(term)}\b", lower))
        if matches:
            hits.append(term)
            score += weight * len(matches)
            first_position = min(first_position, matches[0].start())
    start = max(0, first_position - 140)
    snippet = " ".join(text[start:start + 280].split())
    return score, hits, snippet


def main() -> int:
    records = {row["record_id"]: row for row in read_csv(SUMMARY)}
    for folder in (DATA_ROOT / "derived" / "ocr").iterdir():
        if folder.is_dir() and (folder / "text").is_dir() and folder.name not in records:
            records[folder.name] = {"record_id": folder.name, "platform": "RDC.ROCKS"}
    rows = []
    for record in records.values():
        for source, path in page_files(record):
            text = path.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                continue
            score, hits, snippet = score_page(text)
            if score == 0:
                continue
            rows.append({
                "record_id": record["record_id"],
                "platform": record.get("platform", ""),
                "text_source": source,
                "page": int(re.sub(r"\D", "", path.stem) or 0),
                "chars": len(text),
                "score": score,
                "matched_terms": "; ".join(hits),
                "snippet": snippet,
            })
    rows.sort(key=lambda row: (-row["score"], row["record_id"], row["page"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(OUT.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, OUT)
    assert all(0 <= row["score"] for row in rows)
    print(f"evidence index: {len(rows)} pages from {len(records)} documents, top score {rows[0]['score'] if rows else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
