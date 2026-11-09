from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = "https://rdc.rocks"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/142.0.0.0 Safari/537.36"
)
MAGIC = {
    "application/pdf": (b"%PDF-",),
    "image/tiff": (b"II*\x00", b"MM\x00*"),
}
EXTENSIONS = {
    "application/pdf": ".pdf",
    "image/tiff": ".tif",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_atomic(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def sanitize_probe(source: Path, output: Path) -> list[dict[str, object]]:
    fields = [
        "id",
        "title",
        "status",
        "content_type",
        "content_range",
        "bytes_total",
        "magic_hex",
        "object_filename",
        "download_endpoint",
        "catalogue_endpoint",
        "probe_at_utc",
    ]
    rows = []
    for row in read_csv(source):
        document_id = row["id"]
        rows.append(
            {
                "id": document_id,
                "title": row["title"],
                "status": row["status"],
                "content_type": row["content_type"],
                "content_range": row["content_range"],
                "bytes_total": row["bytes_total"],
                "magic_hex": row["magic_hex"],
                "object_filename": row["object_filename"],
                "download_endpoint": f"{BASE_URL}/data-url/{document_id}/",
                "catalogue_endpoint": f"{BASE_URL}/api/v1/catalog/documents/{document_id}/",
                "probe_at_utc": row["probe_at_utc"],
            }
        )
    write_csv_atomic(output, rows, fields)
    return rows


def stream_response(
    url: str,
    timeout: int,
    temporary: Path,
    expected: int,
    content_type: str,
) -> tuple[int, str]:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/pdf,image/tiff,application/octet-stream,*/*;q=0.8",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        response_type = response.headers.get_content_type()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
        if response_type != content_type:
            raise RuntimeError(f"content type {response_type!r}, expected {content_type!r}")
        digest = hashlib.sha256()
        size = 0
        first_block = True
        with temporary.open("wb") as handle:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                if first_block:
                    if not any(block.startswith(signature) for signature in MAGIC[content_type]):
                        raise RuntimeError("file signature does not match content type")
                    first_block = False
                handle.write(block)
                digest.update(block)
                size += len(block)
        if size != expected:
            raise RuntimeError(f"received {size} bytes, expected {expected}")
        return size, digest.hexdigest()


def download_one(
    row: dict[str, str],
    output_dir: Path,
    timeout: int,
    retries: int,
) -> dict[str, object]:
    document_id = row["id"]
    content_type = row["content_type"]
    extension = EXTENSIONS.get(content_type)
    if extension is None:
        raise ValueError(f"Unsupported content type for record {document_id}: {content_type}")
    expected = int(row["bytes_total"])
    endpoint = f"{BASE_URL}/data-url/{document_id}/"
    destination = output_dir / f"RDC_{int(document_id):04d}{extension}"
    error = ""
    temporary = destination.with_suffix(destination.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            actual, digest = stream_response(
                endpoint, timeout, temporary, expected, content_type
            )
            os.replace(temporary, destination)
            return {
                "id": document_id,
                "title": row["title"],
                "content_type": content_type,
                "expected_bytes": expected,
                "actual_bytes": actual,
                "sha256": digest,
                "signature_valid": True,
                "download_endpoint": endpoint,
                "stored_path": str(destination),
                "completed_at_utc": utc_now(),
                "status": "downloaded",
                "error": "",
            }
        except (HTTPError, URLError, TimeoutError, OSError, RuntimeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            temporary.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2 ** attempt, 30))

    return {
        "id": document_id,
        "title": row["title"],
        "content_type": content_type,
        "expected_bytes": expected,
        "actual_bytes": "",
        "sha256": "",
        "signature_valid": False,
        "download_endpoint": endpoint,
        "stored_path": str(destination),
        "completed_at_utc": utc_now(),
        "status": "failed",
        "error": error,
    }


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def existing_result(row: dict[str, str], output_dir: Path) -> dict[str, object] | None:
    extension = EXTENSIONS.get(row["content_type"])
    if extension is None:
        return None
    path = output_dir / f"RDC_{int(row['id']):04d}{extension}"
    expected = int(row["bytes_total"])
    if not path.is_file() or path.stat().st_size != expected:
        return None
    with path.open("rb") as handle:
        payload_head = handle.read(8)
    if not any(payload_head.startswith(signature) for signature in MAGIC[row["content_type"]]):
        return None
    return {
        "id": row["id"],
        "title": row["title"],
        "content_type": row["content_type"],
        "expected_bytes": expected,
        "actual_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "signature_valid": True,
        "download_endpoint": f"{BASE_URL}/data-url/{row['id']}/",
        "stored_path": str(path),
        "completed_at_utc": utc_now(),
        "status": "already_present",
        "error": "",
    }


def command_sanitize(args: argparse.Namespace) -> int:
    rows = sanitize_probe(Path(args.source), Path(args.output))
    print(f"Wrote {len(rows)} sanitized records to {args.output}")
    return 0


def command_download(args: argparse.Namespace) -> int:
    source = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state)
    rows = read_csv(source)
    allowed_ids = {item.strip().lstrip("0") for item in args.ids.split(",") if item.strip()}
    if allowed_ids:
        rows = [row for row in rows if row["id"].lstrip("0") in allowed_ids]
    if args.content_type:
        allowed_types = {item.strip() for item in args.content_type.split(",") if item.strip()}
        rows = [row for row in rows if row["content_type"] in allowed_types]
    rows.sort(key=lambda row: int(row["bytes_total"]))
    if args.max_files:
        rows = rows[: args.max_files]

    failures = 0
    for index, row in enumerate(rows, start=1):
        prior = existing_result(row, output_dir)
        if prior is not None and not args.force:
            result = prior
        else:
            print(
                f"[{index}/{len(rows)}] {row['id']} {row['title'][:80]} "
                f"({int(row['bytes_total']) / 1024 / 1024:.1f} MiB)",
                flush=True,
            )
            result = download_one(row, output_dir, args.timeout, args.retries)
            time.sleep(args.delay)
        append_jsonl(state_path, result)
        if result["status"] == "failed":
            failures += 1
            print(f"  FAILED: {result['error']}", file=sys.stderr, flush=True)
        else:
            print(
                f"  {result['status']}: {result['sha256'][:16]}... "
                f"{result['actual_bytes']} bytes",
                flush=True,
            )
    return 1 if failures else 0


def command_verify(args: argparse.Namespace) -> int:
    rows = read_csv(Path(args.manifest))
    output_dir = Path(args.output_dir)
    if args.content_type:
        allowed_types = {item.strip() for item in args.content_type.split(",") if item.strip()}
        rows = [row for row in rows if row["content_type"] in allowed_types]
    missing = []
    invalid = []
    valid = 0
    for row in rows:
        extension = EXTENSIONS.get(row["content_type"])
        path = output_dir / f"RDC_{int(row['id']):04d}{extension}"
        if not path.is_file():
            missing.append(row["id"])
            continue
        expected = int(row["bytes_total"])
        with path.open("rb") as handle:
            head = handle.read(8)
        if path.stat().st_size != expected or not any(
            head.startswith(signature) for signature in MAGIC[row["content_type"]]
        ):
            invalid.append(row["id"])
        else:
            valid += 1
    print(f"valid={valid} missing={len(missing)} invalid={len(invalid)}")
    if missing:
        print("missing=" + ",".join(missing))
    if invalid:
        print("invalid=" + ",".join(invalid))
    return 1 if missing or invalid else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Acquire RDC.ROCKS source files")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sanitize = subparsers.add_parser("sanitize-probe")
    sanitize.add_argument("--source", required=True)
    sanitize.add_argument("--output", required=True)
    sanitize.set_defaults(func=command_sanitize)

    download = subparsers.add_parser("download")
    download.add_argument("--manifest", required=True)
    download.add_argument("--output-dir", required=True)
    download.add_argument("--state", required=True)
    download.add_argument("--content-type", default="application/pdf")
    download.add_argument("--ids", default="")
    download.add_argument("--max-files", type=int, default=0)
    download.add_argument("--retries", type=int, default=4)
    download.add_argument("--timeout", type=int, default=180)
    download.add_argument("--delay", type=float, default=0.75)
    download.add_argument("--force", action="store_true")
    download.set_defaults(func=command_download)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--output-dir", required=True)
    verify.add_argument("--content-type", default="application/pdf")
    verify.set_defaults(func=command_verify)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
