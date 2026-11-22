"""Create or verify SHA-256 checksums for the V15 output tree."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v15 import config  # noqa: E402
from src.v15.common import ensure_v15_path, sha256  # noqa: E402


def files() -> list[Path]:
    checksum_dir = config.V15 / "checksums"
    return sorted(
        path
        for path in config.V15.rglob("*")
        if path.is_file() and checksum_dir not in path.parents and "__pycache__" not in path.parts
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    sums_path = config.V15 / "checksums" / "SHA256SUMS_v15"
    if args.verify:
        expected = {}
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            digest, relative = line.split("  ", 1)
            expected[relative] = digest
        failures = []
        for relative, digest in expected.items():
            path = config.V15 / relative
            if not path.is_file() or sha256(path) != digest:
                failures.append(relative)
        extra = sorted(set(path.relative_to(config.V15).as_posix() for path in files()) - set(expected))
        if failures or extra:
            raise SystemExit(f"Checksum failures: {failures}; unlisted files: {extra}")
        print(f"Verified {len(expected)} V15 files.")
        return
    lines = [f"{sha256(path)}  {path.relative_to(config.V15).as_posix()}" for path in files()]
    ensure_v15_path(sums_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} checksums to {sums_path}.")


if __name__ == "__main__":
    main()
