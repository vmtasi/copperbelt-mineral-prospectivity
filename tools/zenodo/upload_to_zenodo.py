"""Upload the prepared deposit to Zenodo as a DRAFT and reserve its DOI.

Run this yourself; it needs your personal Zenodo access token.

1. On zenodo.org (or sandbox.zenodo.org for a trial run), open
   Account > Applications > Personal access tokens and create a token with
   the scopes ``deposit:write`` and ``deposit:actions``.
2. Put it in an environment variable in your own terminal, never in a file:
       $env:ZENODO_TOKEN = "..."          # PowerShell
3. Fill in the creators in tools/zenodo/metadata.json.
4. Prepare the files, then upload:
       python tools/zenodo/prepare_deposit.py --with-snapshot
       python tools/zenodo/upload_to_zenodo.py --sandbox      # optional trial
       python tools/zenodo/upload_to_zenodo.py

The script creates a draft, uploads every file in the deposit folder, sets the
metadata and prints the reserved DOI. It does not publish. Review the draft
on the website and publish it there, or rerun with ``--publish --deposition ID``.
Publishing is permanent: files cannot be changed afterwards, only versioned.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOSTS = {False: "https://zenodo.org", True: "https://sandbox.zenodo.org"}


def _request(method: str, url: str, token: str, body: bytes | None = None,
             content_type: str = "application/json", stream=None, length: int | None = None):
    headers = {"Authorization": f"Bearer {token}"}
    data = body
    if stream is not None:
        data = stream
        headers["Content-Type"] = "application/octet-stream"
        headers["Content-Length"] = str(length)
    elif body is not None:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            payload = response.read()
            return json.loads(payload) if payload else {}
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Zenodo returned HTTP {error.code} for {method} {url}:\n{detail}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", type=Path, default=ROOT / "work" / "zenodo_deposit")
    parser.add_argument("--metadata", type=Path, default=ROOT / "tools" / "zenodo" / "metadata.json")
    parser.add_argument("--sandbox", action="store_true", help="use sandbox.zenodo.org")
    parser.add_argument("--deposition", type=int, help="existing draft ID (skip creation and upload)")
    parser.add_argument("--add", type=Path, nargs="+",
                        help="with --deposition: upload these files to the existing draft")
    parser.add_argument("--publish", action="store_true", help="publish the draft (permanent)")
    args = parser.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("Set ZENODO_TOKEN in this terminal first (see the module docstring).")
    api = f"{HOSTS[args.sandbox]}/api/deposit/depositions"

    if args.deposition and args.add:
        draft = _request("GET", f"{api}/{args.deposition}", token)
        for path in args.add:
            size = path.stat().st_size
            print(f"Uploading {path.name} ({size / 1e6:.1f} MB) ...", flush=True)
            with path.open("rb") as handle:
                _request("PUT", f"{draft['links']['bucket']}/{urllib.parse.quote(path.name)}",
                         token, stream=handle, length=size)
        print(f"Added to draft {args.deposition}; still not published.")
        return

    if args.deposition and args.publish:
        result = _request("POST", f"{api}/{args.deposition}/actions/publish", token, b"")
        print(f"Published: {result.get('doi')}  {result.get('links', {}).get('html')}")
        return

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if any("FAMILY NAME" in creator["name"] for creator in metadata["creators"]):
        sys.exit("Fill in the creators in tools/zenodo/metadata.json before uploading.")
    files = sorted(p for p in args.folder.iterdir() if p.is_file())
    if not files:
        sys.exit(f"No files in {args.folder}; run prepare_deposit.py first.")

    draft = _request("POST", api, token, b"{}")
    bucket = draft["links"]["bucket"]
    print(f"Draft created: ID {draft['id']}")
    for path in files:
        size = path.stat().st_size
        print(f"Uploading {path.name} ({size / 1e6:.1f} MB) ...", flush=True)
        with path.open("rb") as handle:
            _request("PUT", f"{bucket}/{urllib.parse.quote(path.name)}", token,
                     stream=handle, length=size)
    updated = _request("PUT", f"{api}/{draft['id']}", token,
                       json.dumps({"metadata": metadata}).encode("utf-8"))
    doi = updated.get("metadata", {}).get("prereserve_doi", {}).get("doi")
    print(f"Reserved DOI: {doi}")
    print(f"Review the draft at: {updated.get('links', {}).get('html')}")
    print("Not published. Publish on the website after review, or rerun with "
          f"--publish --deposition {draft['id']}.")


if __name__ == "__main__":
    main()
