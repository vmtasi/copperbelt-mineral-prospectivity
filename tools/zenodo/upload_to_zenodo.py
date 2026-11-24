"""Upload the prepared deposit to Zenodo as a DRAFT and reserve its DOI.

Run this yourself; it needs your personal Zenodo access token.

1. On zenodo.org, open Account > Applications > Personal access tokens and
   create a token with the scopes ``deposit:write`` and ``deposit:actions``.
2. Put it in an environment variable in your own terminal, never in a file:
       $env:ZENODO_TOKEN = "..."          # PowerShell
3. Prepare the files, then upload:
       python tools/zenodo/prepare_deposit.py
       python tools/zenodo/upload_to_zenodo.py                     # new draft
       python tools/zenodo/upload_to_zenodo.py --deposition ID     # resume a draft

Every file is retried up to five times. Resuming compares MD5 checksums with
the draft: intact files are skipped, changed files are replaced, and files
that are no longer in the deposit folder are removed from the draft. The
metadata is then set and the reserved DOI printed.

Nothing is published. Review the draft on the website and publish it there,
or run ``--publish --deposition ID``. Publishing is permanent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOSTS = {False: "https://zenodo.org", True: "https://sandbox.zenodo.org"}
ATTEMPTS = 5


class TransferError(Exception):
    """A network failure that is worth retrying."""


def _request(method: str, url: str, token: str, body: bytes | None = None,
             stream=None, length: int | None = None):
    headers = {"Authorization": f"Bearer {token}"}
    data = body
    if stream is not None:
        data = stream
        headers["Content-Type"] = "application/octet-stream"
        headers["Content-Length"] = str(length)
    elif body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            payload = response.read()
            return json.loads(payload) if payload else {}
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        if error.code >= 500 or error.code in (408, 429):
            raise TransferError(f"HTTP {error.code}: {detail[:200]}") from error
        raise SystemExit(f"Zenodo returned HTTP {error.code} for {method} {url}:\n{detail}")
    except (urllib.error.URLError, OSError) as error:
        raise TransferError(str(error)) from error


def _with_retries(action, description: str):
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return action()
        except TransferError as error:
            if attempt == ATTEMPTS:
                raise SystemExit(
                    f"{description} failed {ATTEMPTS} times ({error}). Rerun with "
                    "--deposition ID to resume; completed files are kept."
                )
            wait = 15 * 2 ** (attempt - 1)
            print(f"  attempt {attempt} failed ({error}); retrying in {wait} s", flush=True)
            time.sleep(wait)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _upload(bucket: str, path: Path, token: str) -> None:
    size = path.stat().st_size

    def send():
        with path.open("rb") as handle:
            _request("PUT", f"{bucket}/{urllib.parse.quote(path.name)}", token,
                     stream=handle, length=size)

    print(f"Uploading {path.name} ({size / 1e6:.1f} MB) ...", flush=True)
    _with_retries(send, f"Uploading {path.name}")


def _sync_files(api: str, deposition: int, files: list[Path], token: str) -> None:
    draft = _with_retries(lambda: _request("GET", f"{api}/{deposition}", token),
                          "Reading the draft")
    if draft.get("submitted"):
        raise SystemExit(f"Deposition {deposition} is already published; create a new version.")
    remote = {f["filename"]: f for f in draft.get("files", [])}
    local = {p.name: p for p in files}
    for name, entry in remote.items():
        if name not in local:
            print(f"Removing {name} from the draft (no longer in the deposit folder)")
            _with_retries(lambda: _request("DELETE", f"{api}/{deposition}/files/{entry['id']}",
                                           token), f"Removing {name}")
    for name, path in local.items():
        entry = remote.get(name)
        if entry and entry.get("checksum", "").removeprefix("md5:") == _md5(path):
            print(f"Already uploaded: {name}")
            continue
        if entry:
            print(f"Replacing {name} (contents changed)")
            _with_retries(lambda: _request("DELETE", f"{api}/{deposition}/files/{entry['id']}",
                                           token), f"Removing {name}")
        _upload(draft["links"]["bucket"], path, token)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", type=Path, default=ROOT / "work" / "zenodo_deposit")
    parser.add_argument("--metadata", type=Path, default=ROOT / "tools" / "zenodo" / "metadata.json")
    parser.add_argument("--sandbox", action="store_true", help="use sandbox.zenodo.org")
    parser.add_argument("--deposition", type=int, help="existing draft ID to resume")
    parser.add_argument("--add", type=Path, nargs="+",
                        help="with --deposition: upload only these extra files")
    parser.add_argument("--publish", action="store_true",
                        help="with --deposition: publish the draft (permanent)")
    args = parser.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("Set ZENODO_TOKEN in this terminal first (see the module docstring).")
    api = f"{HOSTS[args.sandbox]}/api/deposit/depositions"

    if args.deposition and args.publish:
        result = _request("POST", f"{api}/{args.deposition}/actions/publish", token, b"")
        print(f"Published: {result.get('doi')}  {result.get('links', {}).get('html')}")
        return

    if args.deposition and args.add:
        draft = _with_retries(lambda: _request("GET", f"{api}/{args.deposition}", token),
                              "Reading the draft")
        for path in args.add:
            _upload(draft["links"]["bucket"], path, token)
        print(f"Added to draft {args.deposition}; still not published.")
        return

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if any("FAMILY NAME" in creator["name"] for creator in metadata["creators"]):
        sys.exit("Fill in the creators in tools/zenodo/metadata.json before uploading.")
    files = sorted(p for p in args.folder.iterdir() if p.is_file())
    if not files:
        sys.exit(f"No files in {args.folder}; run prepare_deposit.py first.")

    deposition = args.deposition
    if deposition is None:
        draft = _with_retries(lambda: _request("POST", api, token, b"{}"), "Creating the draft")
        deposition = draft["id"]
        print(f"Draft created: ID {deposition} (if the upload stops, rerun with "
              f"--deposition {deposition})")
    else:
        print(f"Resuming draft {deposition}")
    _sync_files(api, deposition, files, token)

    updated = _with_retries(
        lambda: _request("PUT", f"{api}/{deposition}", token,
                         json.dumps({"metadata": metadata}).encode("utf-8")),
        "Setting the metadata",
    )
    doi = updated.get("metadata", {}).get("prereserve_doi", {}).get("doi")
    print(f"All {len(files)} files are in draft {deposition}.")
    print(f"Reserved DOI: {doi}")
    print(f"Review the draft at: {updated.get('links', {}).get('html')}")
    print("Not published. Publish on the website after review, or rerun with "
          f"--publish --deposition {deposition}.")


if __name__ == "__main__":
    main()
