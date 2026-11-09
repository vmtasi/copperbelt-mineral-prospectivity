"""Write the Phase 6 status table from the delivered GeoPackages.

This used to regenerate placeholder reports that said 3854 was rejected and 4055
had no control points. Those reports are superseded; running this script must not
overwrite the delivered transformation reports or registers.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GEOREF = PROJECT_ROOT / "data/raw/rdc_rocks/derived/georef"
OUT = GEOREF / "phase6"
PACKAGES = {
    "RDC_3863": OUT / "3863/rdc_3863_structural_v1.gpkg",
    "RDC_3854": OUT / "3854/rdc_3854_stratigraphy_v1.gpkg",
    "RDC_4055": OUT / "4055/rdc_4055_structural_v1.gpkg",
}
TRANSFORMS = GEOREF / "transform_params_v1.json"


def layer_counts(path: Path) -> tuple[int, int, str]:
    connection = sqlite3.connect(path)
    rows = connection.execute("SELECT table_name FROM gpkg_contents ORDER BY table_name").fetchall()
    counts = []
    for (name,) in rows:
        count = connection.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
        counts.append(f"`{name}` {count}")
    total = sum(int(item.rsplit(" ", 1)[1]) for item in counts)
    connection.close()
    return len(counts), total, ", ".join(counts)


def main() -> int:
    transforms = json.loads(TRANSFORMS.read_text(encoding="utf-8"))
    rows = []
    for name, path in PACKAGES.items():
        assert path.is_file(), f"missing {path}"
        layers, total, detail = layer_counts(path)
        rows.append((name, layers, total, detail))

    lines = [
        "# Phase 6 status (v2)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Map | Coordinate basis | Layers / features | Package |",
        "|---|---|---|---|",
    ]
    bases = {
        "RDC_3863": "sheet metric grid; datum unresolved; transform RMSE 135.4 m",
        "RDC_3854": "sheet metric grid; datum unresolved; RMSE 83 m E / 131 m N; 260 km label excluded",
        "RDC_4055": "source pixels; easting-only calibration RMSE 87.9 m; northing unresolved",
    }
    for name, layers, total, detail in rows:
        lines.append(f"| {name} | {bases[name]} | {layers} layers / {total} features | `{PACKAGES[name].relative_to(PROJECT_ROOT)}` |")
    lines += [
        "",
        "## Layer counts",
        "",
    ]
    for name, _, _, detail in rows:
        lines.append(f"- {name}: {detail}.")
    lines += [
        "",
        "## Notes",
        "",
        "- No datum or projection is asserted anywhere. RDC_4055 has no northing control and remains in pixel space.",
        "- RDC_4055 prints ten numbered thrust sheets (8-17), not seventeen. The old count misread the last number.",
        "- Detailed limitations are in each package folder's transformation report.",
        "",
    ]
    status = OUT / "phase6_status_v1.md"
    status.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"packages": {name: total for name, _, total, _ in rows}, "status": str(status)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
