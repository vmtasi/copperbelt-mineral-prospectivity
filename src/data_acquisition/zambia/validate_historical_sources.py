from __future__ import annotations

import csv
import json
from collections import Counter

from build_historical_source_registers import OUT, ROOT, VOLUME, sha256


def read(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    manifest = json.loads((OUT / "historical_source_registers_manifest_v1.json").read_text(encoding="utf-8"))
    assert sha256(VOLUME) == manifest["source_volume_sha256"]
    strat = read("stratigraphic_correlations_v1.csv")
    struct = read("structural_interpretations_v1.csv")
    maps = read("geological_map_reproduction_register_v1.csv")
    assert (len(strat), len(struct), len(maps)) == (20, 18, 15)
    for rows in (strat, struct, maps):
        assert len({row["record_id"] for row in rows}) == len(rows)
    assert all(row["source_page"] for row in strat + struct)
    assert all(row["source_page_or_plate"] and row["source_document"] for row in maps)

    paths = [part for row in maps for part in row["local_paths"].split("|") if part.strip()]
    missing = [path for path in paths if not (ROOT / path).exists()]
    assert not missing, missing

    chapter = (OUT / "chapter_text_v1.txt").read_text(encoding="utf-8")
    for name in ("Thieme", "Thierne", "Theime", "Drysdall", "Drysdale"):
        assert name not in chapter

    for name, expected in manifest["output_sha256"].items():
        assert sha256(OUT / name) == expected

    print(json.dumps({
        "rows": {"stratigraphic": len(strat), "structural": len(struct), "maps": len(maps)},
        "map_local_paths_checked": len(paths),
        "map_status_counts": dict(Counter(row["reproduction_or_derivative_status"] for row in maps)),
        "source_volume_sha256_matches": True,
        "chapter_negative_result": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
