"""Structure the 21 exploration evidence rows without adding spatial or temporal claims."""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE = PROJECT_ROOT / "data/raw/rdc_rocks/derived/exploration"
SOURCE = BASE / "exploration_evidence_v1.csv"
OUT = BASE / "exploration_history_v1.csv"
REPORT = BASE / "exploration_history_report_v1.md"

CLASSES = {
    "historical_exploration_campaign": "historical_campaign",
    "airborne_radiometric": "airborne_survey",
    "airborne_geophysics": "airborne_survey",
    "integrated_exploration": "historical_campaign",
    "drilling": "drilling",
    "trenching_and_drilling": "drilling",
    "borehole_radiometry": "drilling",
    "tenure": "historical_campaign",
    "geological_mapping": "ground_survey",
    "sampling_and_petrography": "sampling_and_assay",
    "whole_rock_geochemistry": "sampling_and_assay",
    "geochemical_sample_location": "sampling_and_assay",
    "geochemical_assay": "sampling_and_assay",
    "underground_mapping": "ground_survey",
    "modern_operational_context": "modern_operational_context",
    "historical_map_compilation": "historical_campaign",
    "ground_radiometric_survey": "ground_survey",
    "absence": "documentary_absence",
}
COMMODITIES = [
    ("U", r"\buranium\b|\bU-Ni\b|\bU\b"),
    ("Cu", r"\bcopper\b|\bCu\b"),
    ("Co", r"\bcobalt\b|\bCo\b"),
    ("Au", r"\bgold\b|\bAu\b"),
    ("Nb", r"\bniobium\b|\bNb\b"),
    ("Ta", r"\btantalum\b|\bTa\b"),
    ("Fe", r"\biron\b|\bFe\b"),
    ("Sn", r"\btin\b|\bSn\b"),
    ("W", r"\btungsten\b|\bW\b"),
]
YEARS = re.compile(r"\b(1[89]\d{2}|20[0-2]\d)\b")


def main() -> int:
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 21

    output = []
    for row in rows:
        text = f"{row['direct_observation']} {row['observation_or_inference']}"
        years = [int(year) for year in YEARS.findall(text)]
        start = min(years) if years else ""
        end = max(years) if years else ""
        commodities = [symbol for symbol, pattern in COMMODITIES if re.search(pattern, text)]
        scope = row["geographic_scope"]
        lower = scope.lower()
        if "historical" in lower and len([part for part in scope.split(",") if part.strip()]) == 1:
            precision = "non_spatial"
        elif any(word in lower for word in ("katanga", "lufilian", "concession", "province", "region", "arc")):
            precision = "regional"
        else:
            precision = "local"
        output.append({
            "evidence_id": row["evidence_id"],
            "record_id": row["record_id"],
            "source_page": row["source_page"],
            "evidence_class": CLASSES[row["evidence_type"]],
            "evidence_type_original": row["evidence_type"],
            "date_start": start,
            "date_end": end,
            "date_text": f"{min(years)}-{max(years)}" if years else "not stated",
            "commodity": "; ".join(commodities),
            "geographic_scope": scope,
            "spatial_precision": precision,
            "direct_observation": row["direct_observation"],
            "observation_or_inference": row["observation_or_inference"],
            "confidence": row["confidence"],
            "spatial_geometry": "",
        })

    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0].keys()))
        writer.writeheader()
        writer.writerows(output)

    classes = Counter(row["evidence_class"] for row in output)
    commodities = Counter(c for row in output for c in row["commodity"].split("; ") if c)
    dated = sum(1 for row in output if row["date_start"] != "")
    report = [
        "# Exploration history database (v1)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- Evidence rows: **{len(output)}** (all preserved).",
        f"- Rows with a stated year: **{dated}**; rows without: **{len(output)-dated}**.",
        f"- Spatial geometry: **none created**. No campaign polygon or survey footprint was inferred.",
        "",
        "## Evidence classes",
        "",
    ]
    report += [f"- `{name}`: {count}" for name, count in classes.most_common()]
    report += ["", "## Commodity mentions", ""]
    report += [f"- {name}: {count}" for name, count in commodities.most_common()]
    report += [
        "",
        "## Rules and limitations",
        "",
        "- Dates are taken only from four-digit years present in the source text; no campaign dates were inferred.",
        "- Commodity is a keyword extraction from the evidence text, not a re-interpretation of the deposit.",
        "- `spatial_precision` is a coarse class derived from the stated scope (non_spatial / regional / local). It is not a mapped extent.",
        "- `absence` rows document what the acquisition set does not contain; they are not negative exploration observations.",
        "- Exploration activity is not treated as evidence of mineralisation.",
    ]
    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(output), "classes": dict(classes), "dated": dated}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
