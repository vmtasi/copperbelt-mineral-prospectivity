"""Build site entities, observations and a review crosswalk from the RDC occurrence table.

Conservative by design: entities are unique normalised place names; nothing is merged
across variants and no USGS record is assigned one-to-one automatically.
"""
from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE = PROJECT_ROOT / "data/raw/rdc_rocks/derived/occurrences"
OUT = BASE / "entities"

DIRECTION = re.compile(r"^(ouest|est|nord|sud|north|south|east|west)\s+(de|of|du|d')\s+", re.I)


def normalise(name: str) -> str:
    text = unicodedata.normalize("NFKD", name or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("'", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> int:
    with (BASE / "rdc_occurrences_v1.csv").open(encoding="utf-8-sig", newline="") as handle:
        observations = list(csv.DictReader(handle))
    with (BASE / "occurrence_usgs_reconciliation_v1.csv").open(encoding="utf-8-sig", newline="") as handle:
        candidates = {row["occurrence_id"]: row for row in csv.DictReader(handle)}

    assert len(observations) == len({row["occurrence_id"] for row in observations}) == 147

    entities: dict[str, dict] = {}
    for row in observations:
        key = normalise(row["original_name"])
        entity = entities.setdefault(key, {
            "site_entity_id": f"SITE-{len(entities)+1:04d}",
            "canonical_name": row["original_name"],
            "normalised_name": key,
            "observations": [],
        })
        entity["observations"].append(row)

    usgs_counts = Counter()
    for row in observations:
        name = candidates[row["occurrence_id"]]["usgs_name"]
        if name:
            usgs_counts[name] += 1

    usgs_entities = defaultdict(set)
    for entity in entities.values():
        for row in entity["observations"]:
            item = candidates[row["occurrence_id"]]
            if item["usgs_name"] and item["match_confidence"] != "none":
                usgs_entities[item["usgs_name"]].add(entity["site_entity_id"])

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "site_entities_v1.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["site_entity_id", "canonical_name", "normalised_name", "observation_count",
                         "commodities", "source_documents", "usgs_candidates", "entity_status", "note"])
        for entity in entities.values():
            rows = entity["observations"]
            commodities = sorted({row["commodity"] for row in rows if row["commodity"]})
            docs = sorted({row["source_record_id"] for row in rows})
            names = sorted({candidates[row["occurrence_id"]]["usgs_name"] for row in rows
                            if candidates[row["occurrence_id"]]["usgs_name"]
                            and candidates[row["occurrence_id"]]["match_confidence"] != "none"})
            status = "multiple_observations" if len(rows) > 1 else "single_observation"
            if len(names) > 1:
                status = "ambiguous_candidates"
            elif len(names) == 1 and len(usgs_entities[names[0]]) > 1:
                status = "ambiguous_many_to_one"
            elif len(names) == 1:
                status = "consistent_single_candidate"
            writer.writerow([entity["site_entity_id"], entity["canonical_name"], entity["normalised_name"],
                             len(rows), "; ".join(commodities), "; ".join(docs), "; ".join(names), status,
                             "Entity is a normalised place name, not a confirmed deposit."])

    with (OUT / "occurrence_observations_v1.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(observations[0].keys()) + ["site_entity_id"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in observations:
            key = normalise(row["original_name"])
            writer.writerow({**row, "site_entity_id": entities[key]["site_entity_id"]})

    rows_out = []
    for entity in entities.values():
        for row in entity["observations"]:
            candidate = candidates[row["occurrence_id"]]
            name = candidate["usgs_name"]
            site_names = {candidates[other["occurrence_id"]]["usgs_name"] for other in entity["observations"]
                          if candidates[other["occurrence_id"]]["usgs_name"]
                          and candidates[other["occurrence_id"]]["match_confidence"] != "none"}
            if not name or candidate["match_confidence"] == "none":
                review = "no_candidate"
                rationale = f"No USGS candidate passed the threshold ({candidate['match_basis']})."
            elif len(site_names) > 1:
                review = "ambiguous_multiple_candidates"
                rationale = "This site entity has more than one distinct USGS candidate; left unresolved."
            elif len(usgs_entities[name]) > 1:
                review = "ambiguous_many_to_one"
                rationale = (f"USGS '{name}' is claimed by {len(usgs_entities[name])} distinct site entities; "
                             "one-to-one identity is not established. Left unresolved.")
            else:
                review = "site_level_candidate_unverified"
                rationale = (f"All observations for this site map to USGS '{name}' "
                             f"({candidate['match_confidence']}, {candidate['match_basis']}); not field-verified.")
            rows_out.append({
                "crosswalk_id": f"XW-{len(rows_out)+1:04d}",
                "occurrence_id": row["occurrence_id"],
                "site_entity_id": entity["site_entity_id"],
                "original_name": row["original_name"],
                "commodity": row["commodity"],
                "usgs_name": name,
                "usgs_latitude": candidate["usgs_latitude"],
                "usgs_longitude": candidate["usgs_longitude"],
                "match_confidence": candidate["match_confidence"],
                "match_basis": candidate["match_basis"],
                "review_status": review,
                "match_rationale": rationale,
            })
    assert len(rows_out) == len(observations)
    with (OUT / "occurrence_entity_crosswalk_v1.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    status_counts = Counter(row["review_status"] for row in rows_out)
    report = [
        "# Occurrence entity resolution (v1)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- Source observations: **{len(observations)}** (none discarded).",
        f"- Site entities: **{len(entities)}** (unique normalised place names).",
        f"- Crosswalk rows: **{len(rows_out)}**, one per observation.",
        f"- Entity status: {json.dumps(dict(Counter('multiple' if len(e['observations'])>1 else 'single' for e in entities.values())))}",
        "",
        "## Review status",
        "",
    ]
    report += [f"- `{key}`: {value}" for key, value in status_counts.most_common()]
    report += [
        "",
        "## Rules",
        "",
        "- Repeated commodity-specific observations are kept as separate rows and are NOT counted as independent deposits.",
        "- A site entity is a normalised place name only; directional variants (e.g. 'Ouest de X') remain separate entities.",
        "- No USGS record is merged into the source data and no model positive is created.",
        "- Repeated USGS candidates are flagged `ambiguous_repeated_candidate` rather than resolved.",
    ]
    (OUT / "entity_resolution_report_v1.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"observations": len(observations), "entities": len(entities), "status": dict(status_counts)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
