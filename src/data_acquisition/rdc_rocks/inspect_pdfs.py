from __future__ import annotations

import argparse
import csv
import hashlib
import math
import shutil
import subprocess
import textwrap
from itertools import combinations
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_text(value: str | None, limit: int = 1200) -> str:
    if not value:
        return ""
    text = " ".join(value.split())
    return text[:limit]


def inspect_pdf(path: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "actual_page_count": "",
        "encrypted": "",
        "text_chars_first_2_pages": 0,
        "text_status": "unreadable",
        "first_page_text": "",
        "inspection_error": "",
    }
    try:
        reader = PdfReader(path)
        result["encrypted"] = bool(reader.is_encrypted)
        page_count = len(reader.pages)
        result["actual_page_count"] = page_count
        text_parts = []
        for page in reader.pages[:2]:
            text_parts.append(page.extract_text() or "")
        text = "\n".join(text_parts)
        result["text_chars_first_2_pages"] = len(text.strip())
        result["text_status"] = "text_layer" if len(text.strip()) >= 50 else "image_only_or_sparse"
        result["first_page_text"] = clean_text(text_parts[0] if text_parts else "")
    except Exception as exc:
        result["inspection_error"] = f"{type(exc).__name__}: {exc}"
    return result


def command_audit(args: argparse.Namespace) -> int:
    probe_rows = read_csv(Path(args.manifest))
    probe_rows = [row for row in probe_rows if row["content_type"] == "application/pdf"]
    details = {row["id"]: row for row in read_csv(Path(args.details))}
    pdf_dir = Path(args.pdf_dir)
    rows = []
    for probe in sorted(probe_rows, key=lambda row: int(row["id"])):
        document_id = probe["id"]
        path = pdf_dir / f"RDC_{int(document_id):04d}.pdf"
        detail = details.get(document_id, {})
        record: dict[str, object] = {
            "id": document_id,
            "title": probe["title"],
            "document_type": detail.get("document_type", ""),
            "scale": detail.get("scale", ""),
            "expected_page_count": detail.get("page_count", ""),
            "file_bytes": path.stat().st_size if path.is_file() else "",
            "sha256": sha256_file(path) if path.is_file() else "",
            "stored_path": str(path),
        }
        if path.is_file():
            record.update(inspect_pdf(path))
        else:
            record.update(
                {
                    "actual_page_count": "",
                    "encrypted": "",
                    "text_chars_first_2_pages": 0,
                    "text_status": "missing",
                    "first_page_text": "",
                    "inspection_error": "file missing",
                }
            )
        expected = str(record["expected_page_count"]).strip()
        actual = str(record["actual_page_count"]).strip()
        record["page_count_match"] = (
            "" if not expected or not actual else str(expected == actual).lower()
        )
        rows.append(record)

    fields = [
        "id",
        "title",
        "document_type",
        "scale",
        "expected_page_count",
        "actual_page_count",
        "page_count_match",
        "file_bytes",
        "sha256",
        "encrypted",
        "text_chars_first_2_pages",
        "text_status",
        "first_page_text",
        "stored_path",
        "inspection_error",
    ]
    write_csv(Path(args.output), rows, fields)
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row["text_status"])
        counts[key] = counts.get(key, 0) + 1
    print(f"Wrote {len(rows)} PDF audit records to {args.output}")
    print("text_status=" + ", ".join(f"{key}:{counts[key]}" for key in sorted(counts)))
    return 0


def fitted_image(path: Path, width: int, height: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    left = (width - image.width) // 2
    top = (height - image.height) // 2
    canvas.paste(image, (left, top))
    return canvas


def average_hash(path: Path, size: int = 32) -> list[bool]:
    image = Image.open(path).convert("L").resize((size, size), Image.Resampling.LANCZOS)
    pixels = list(image.getdata())
    average = sum(pixels) / len(pixels)
    return [value >= average for value in pixels]


def hamming(left: list[bool], right: list[bool]) -> int:
    return sum(a != b for a, b in zip(left, right))


def command_duplicate_review(args: argparse.Namespace) -> int:
    details = {row["id"]: row for row in read_csv(Path(args.details))}
    thumbnail_dir = Path(args.thumbnail_dir)
    thumbnails = sorted(thumbnail_dir.glob("RDC_*.jpg"))
    hashes = {path: average_hash(path) for path in thumbnails}
    rows = []
    for left, right in combinations(thumbnails, 2):
        left_id = str(int(left.stem.removeprefix("RDC_")))
        right_id = str(int(right.stem.removeprefix("RDC_")))
        left_title = details.get(left_id, {}).get("title", "")
        right_title = details.get(right_id, {}).get("title", "")
        distance = hamming(hashes[left], hashes[right])
        same_title = left_title.casefold() == right_title.casefold()
        if distance <= args.max_distance or same_title:
            rows.append(
                {
                    "id_a": left_id,
                    "id_b": right_id,
                    "title_a": left_title,
                    "title_b": right_title,
                    "same_title": str(same_title).lower(),
                    "perceptual_hash_distance": distance,
                    "thumbnail_a": str(left),
                    "thumbnail_b": str(right),
                    "review_status": "candidate",
                }
            )
    rows.sort(key=lambda row: (int(row["perceptual_hash_distance"]), int(row["id_a"])))
    fields = [
        "id_a",
        "id_b",
        "title_a",
        "title_b",
        "same_title",
        "perceptual_hash_distance",
        "thumbnail_a",
        "thumbnail_b",
        "review_status",
    ]
    write_csv(Path(args.output), rows, fields)
    print(f"Wrote {len(rows)} duplicate-review candidates to {args.output}")
    return 0


def command_page_sheets(args: argparse.Namespace) -> int:
    if shutil.which("pdftoppm") is None:
        raise SystemExit("pdftoppm is required but was not found on PATH")
    pdf_path = Path(args.pdf)
    thumbnail_dir = Path(args.thumbnail_dir)
    output_dir = Path(args.output_dir)
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = thumbnail_dir / pdf_path.stem
    subprocess.run(
        [
            "pdftoppm",
            "-f",
            str(args.first_page),
            "-l",
            str(args.last_page),
            "-scale-to",
            str(args.thumbnail_width),
            "-jpeg",
            "-jpegopt",
            "quality=75",
            str(pdf_path),
            str(prefix),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.timeout,
    )
    pages = []
    for candidate in thumbnail_dir.glob(pdf_path.stem + "-*.jpg"):
        try:
            page_number = int(candidate.stem.rsplit("-", 1)[-1])
        except ValueError:
            continue
        if args.first_page <= page_number <= args.last_page:
            pages.append(candidate)
    pages.sort()
    if not pages:
        print("No pages rendered")
        return 1

    font_path = Path(r"C:\Windows\Fonts\arial.ttf")
    font = ImageFont.truetype(str(font_path), 16) if font_path.is_file() else ImageFont.load_default()
    cell_width = args.thumbnail_width + 20
    cell_height = int(args.thumbnail_width * 0.72) + 42
    per_sheet = args.columns * args.rows
    sheet_count = math.ceil(len(pages) / per_sheet)
    for sheet_index in range(sheet_count):
        group = pages[sheet_index * per_sheet : (sheet_index + 1) * per_sheet]
        sheet = Image.new("RGB", (args.columns * cell_width, args.rows * cell_height), "white")
        draw = ImageDraw.Draw(sheet)
        for position, page_path in enumerate(group):
            column = position % args.columns
            row = position // args.columns
            x = column * cell_width
            y = row * cell_height
            image = fitted_image(page_path, args.thumbnail_width, int(args.thumbnail_width * 0.72))
            sheet.paste(image, (x + 10, y + 8))
            page_label = page_path.stem.rsplit("-", 1)[-1]
            draw.text((x + 10, y + image.height + 14), f"p. {page_label}", fill="black", font=font)
        output_path = output_dir / (
            f"{pdf_path.stem}_pages_{args.first_page:03d}-{args.last_page:03d}_{sheet_index + 1:02d}.jpg"
        )
        sheet.save(output_path, quality=88, subsampling=0)
        print(f"Wrote {output_path}", flush=True)
    return 0


def command_contact_sheets(args: argparse.Namespace) -> int:
    if shutil.which("pdftoppm") is None:
        raise SystemExit("pdftoppm is required but was not found on PATH")
    details = {row["id"]: row for row in read_csv(Path(args.details))}
    pdf_dir = Path(args.pdf_dir)
    thumbnail_dir = Path(args.thumbnail_dir)
    output_dir = Path(args.output_dir)
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_paths = sorted(pdf_dir.glob("RDC_*.pdf"))
    if args.ids:
        allowed = {item.strip().lstrip("0") for item in args.ids.split(",") if item.strip()}
        pdf_paths = [path for path in pdf_paths if path.stem.removeprefix("RDC_").lstrip("0") in allowed]

    thumbnails = []
    for index, pdf_path in enumerate(pdf_paths, start=1):
        padded_id = pdf_path.stem.removeprefix("RDC_")
        document_id = str(int(padded_id))
        thumbnail = thumbnail_dir / f"RDC_{padded_id}.jpg"
        if not thumbnail.is_file():
            subprocess.run(
                [
                    "pdftoppm",
                    "-f",
                    "1",
                    "-l",
                    "1",
                    "-singlefile",
                    "-scale-to",
                    str(args.thumbnail_width),
                    "-jpeg",
                    "-jpegopt",
                    "quality=75",
                    str(pdf_path),
                    str(thumbnail.with_suffix("")),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=180,
            )
        detail = details.get(document_id, {})
        thumbnails.append(
            (
                padded_id,
                detail.get("title", pdf_path.stem),
                detail.get("document_type", ""),
                thumbnail,
            )
        )
        if index % 10 == 0 or index == len(pdf_paths):
            print(f"Rendered {index}/{len(pdf_paths)} first pages", flush=True)

    if not thumbnails:
        print("No PDFs selected")
        return 1

    font_path = Path(r"C:\Windows\Fonts\arial.ttf")
    font = ImageFont.truetype(str(font_path), 16) if font_path.is_file() else ImageFont.load_default()
    label_font = ImageFont.truetype(str(font_path), 15) if font_path.is_file() else ImageFont.load_default()
    cell_width = args.thumbnail_width + 20
    cell_height = int(args.thumbnail_width * 0.72) + 92
    per_sheet = args.columns * args.rows
    sheet_count = math.ceil(len(thumbnails) / per_sheet)
    for sheet_index in range(sheet_count):
        group = thumbnails[sheet_index * per_sheet : (sheet_index + 1) * per_sheet]
        sheet = Image.new(
            "RGB",
            (args.columns * cell_width, args.rows * cell_height),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for position, (document_id, title, document_type, thumbnail_path) in enumerate(group):
            column = position % args.columns
            row = position // args.columns
            x = column * cell_width
            y = row * cell_height
            image = fitted_image(thumbnail_path, args.thumbnail_width, int(args.thumbnail_width * 0.72))
            sheet.paste(image, (x + 10, y + 8))
            label = textwrap.fill(f"RDC {document_id} | {document_type or 'record'}", width=34)
            draw.multiline_text((x + 10, y + image.height + 14), label, fill="black", font=font)
            title_text = textwrap.fill(title, width=40)[:160]
            draw.multiline_text(
                (x + 10, y + image.height + 40),
                title_text,
                fill="#333333",
                font=label_font,
                spacing=2,
            )
        output_path = output_dir / f"first_page_contact_sheet_{sheet_index + 1:02d}.jpg"
        sheet.save(output_path, quality=88, subsampling=0)
        print(f"Wrote {output_path}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and review acquired RDC.ROCKS PDFs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--details", required=True)
    audit.add_argument("--pdf-dir", required=True)
    audit.add_argument("--output", required=True)
    audit.set_defaults(func=command_audit)

    sheets = subparsers.add_parser("contact-sheets")
    sheets.add_argument("--details", required=True)
    sheets.add_argument("--pdf-dir", required=True)
    sheets.add_argument("--thumbnail-dir", required=True)
    sheets.add_argument("--output-dir", required=True)
    sheets.add_argument("--ids", default="")
    sheets.add_argument("--thumbnail-width", type=int, default=360)
    sheets.add_argument("--columns", type=int, default=5)
    sheets.add_argument("--rows", type=int, default=6)
    sheets.set_defaults(func=command_contact_sheets)

    duplicates = subparsers.add_parser("duplicate-review")
    duplicates.add_argument("--details", required=True)
    duplicates.add_argument("--thumbnail-dir", required=True)
    duplicates.add_argument("--output", required=True)
    duplicates.add_argument("--max-distance", type=int, default=120)
    duplicates.set_defaults(func=command_duplicate_review)

    pages = subparsers.add_parser("page-sheets")
    pages.add_argument("--pdf", required=True)
    pages.add_argument("--thumbnail-dir", required=True)
    pages.add_argument("--output-dir", required=True)
    pages.add_argument("--first-page", type=int, default=1)
    pages.add_argument("--last-page", type=int, required=True)
    pages.add_argument("--thumbnail-width", type=int, default=300)
    pages.add_argument("--columns", type=int, default=5)
    pages.add_argument("--rows", type=int, default=4)
    pages.add_argument("--timeout", type=int, default=600)
    pages.set_defaults(func=command_page_sheets)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
