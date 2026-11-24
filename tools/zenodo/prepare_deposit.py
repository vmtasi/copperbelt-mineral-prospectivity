"""Assemble the Zenodo deposit for Paper 1.

Collects the files that are too large for Git (posterior traces and the
concatenated transfer predictions), checks each one against the repository's
own checksum manifests, and writes zip archives that keep repository-relative
paths, so that unzipping at the repository root restores them in place.

    python tools/zenodo/prepare_deposit.py [--out work/zenodo_deposit] [--with-snapshot]

Nothing is uploaded. Upload with tools/zenodo/upload_to_zenodo.py.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# archive name -> (description, list of (manifest file, manifest root), glob patterns)
ARCHIVES = {
    "traces_transfer.zip": (
        "Posterior traces of the 106 geographic-transfer fits (the paper's analyses)",
        [("paper_1/transfer/checksums/SHA256SUMS", "paper_1/transfer")],
        ["paper_1/transfer/**/*.nc"],
    ),
    "predictions_transfer_concatenated.zip": (
        "Concatenation of all per-fit held-out predictions of the transfer analysis",
        [("paper_1/transfer/checksums/SHA256SUMS", "paper_1/transfer")],
        ["paper_1/transfer/predictions/all_amendment1.csv"],
    ),
    "traces_domain_models.zip": (
        "Posterior traces of the earlier area-aware domain models (Supplementary Section S7)",
        [
            ("paper_1/v15/checksums/SHA256SUMS_v15", "paper_1/v15"),
            ("paper_1/v15/phase2b_two_domain/SHA256SUMS", "paper_1/v15/phase2b_two_domain"),
            ("paper_1/v15/phase2c_stored_faults/SHA256SUMS", "paper_1/v15/phase2c_stored_faults"),
        ],
        ["paper_1/v15/**/*.nc"],
    ),
    "traces_v14_distance_sensitivity.zip": (
        "Posterior traces of the earlier distance-representation sensitivity models",
        [("paper_1/benchmark/SHA256SUMS_external", ".")],
        ["figures/audit/distance_representation_sensitivity_v14_centroid/traces/**/*.nc"],
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(manifest: str, base: str) -> dict[str, str]:
    """Map repository-relative paths to expected hashes."""
    expected = {}
    text = (ROOT / manifest).read_text(encoding="utf-8").replace("\r", "")
    for line in text.splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*").strip()
        expected[(Path(base) / name).as_posix().removeprefix("./")] = digest
    return expected


def build(out: Path, with_snapshot: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest_lines = []
    summary = []
    for archive, (description, manifests, patterns) in ARCHIVES.items():
        expected: dict[str, str] = {}
        for manifest, base in manifests:
            expected.update(load_manifest(manifest, base))
        files = sorted({p for pattern in patterns for p in ROOT.glob(pattern) if p.is_file()})
        if not files:
            raise FileNotFoundError(f"No files for {archive}")
        unlisted = []
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            actual = sha256(path)
            if relative in expected:
                if expected[relative] != actual:
                    raise ValueError(f"Checksum mismatch: {relative}")
            else:
                unlisted.append(relative)
            manifest_lines.append(f"{actual}  {relative}")
        target = out / archive
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in files:
                bundle.write(path, path.relative_to(ROOT).as_posix())
        size = target.stat().st_size / 1e6
        summary.append((archive, len(files), size, description, unlisted))
    if with_snapshot:
        snapshot = out / "repository_snapshot.zip"
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        subprocess.run(
            ["git", "archive", "--format=zip", f"--output={snapshot}", "HEAD"],
            cwd=ROOT, check=True,
        )
        summary.append(("repository_snapshot.zip", 0, snapshot.stat().st_size / 1e6,
                        f"git archive of commit {commit}", []))
    (out / "CONTENTS.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="ascii",
                                         newline="\n")
    archive_hashes = [f"{sha256(out / name)}  {name}" for name, *_ in summary]
    (out / "ARCHIVES.sha256").write_text("\n".join(archive_hashes) + "\n", encoding="ascii",
                                         newline="\n")
    for name, count, size, description, unlisted in summary:
        note = f"; {len(unlisted)} not in any manifest" if unlisted else ""
        print(f"{name}: {count} files, {size:.1f} MB{note} - {description}")
        for path in unlisted:
            print(f"    not in a manifest (included unverified): {path}")
    print(f"Written to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "work" / "zenodo_deposit")
    parser.add_argument("--with-snapshot", action="store_true",
                        help="also add a git archive of the current commit")
    args = parser.parse_args()
    build(args.out, args.with_snapshot)


if __name__ == "__main__":
    main()
