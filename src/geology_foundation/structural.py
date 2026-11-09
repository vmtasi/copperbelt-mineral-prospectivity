from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd


def build_structural_baseline(
    source_path: str | Path,
    config: dict[str, Any],
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    source = gpd.read_file(
        str(source_path),
        layer=config["source_layer"],
        engine="pyogrio",
        fid_as_index=True,
    )
    if source.empty:
        raise RuntimeError("The structural source contains no features")
    if not source.index.is_unique:
        raise RuntimeError("Structural source fid values are not unique")
    source_fids = source.index.to_numpy(dtype="int64", copy=True)
    output = source.to_crs("EPSG:32735").copy()
    output = output.reset_index(drop=True)
    output["source_fid"] = source_fids
    output["structure_id"] = [f"SLY2005_{index:04d}" for index in range(1, len(output) + 1)]
    output["source_label"] = output.get("source", pd.Series(index=output.index, dtype="object"))
    output["source_notes"] = output.get("notes", pd.Series(index=output.index, dtype="object"))
    output["structure_name"] = None
    output["structure_type"] = None
    output["tier"] = int(config["tier"])
    output["attribution_status"] = config["attribution_status"]
    output["allowed_use"] = config["allowed_use"]
    output["prohibited_use"] = config["prohibited_use"]
    output["length_km"] = output.geometry.length / 1000.0
    output["source_reference"] = config["source_reference"]
    output["geometry_confidence"] = "unassessed; inherited compilation geometry"
    output = output[
        [
            "structure_id",
            "source_fid",
            "source_label",
            "source_notes",
            "structure_name",
            "structure_type",
            "tier",
            "attribution_status",
            "allowed_use",
            "prohibited_use",
            "length_km",
            "source_reference",
            "geometry_confidence",
            "geometry",
        ]
    ]
    stats = {
        "features": int(len(output)),
        "valid_geometries": int(output.geometry.is_valid.sum()),
        "crs": output.crs.to_string(),
        "total_length_km": float(output.geometry.length.sum() / 1000.0),
        "null_structure_names": int(output["structure_name"].isna().sum()),
        "null_structure_types": int(output["structure_type"].isna().sum()),
        "distinct_source_labels": sorted(
            str(value) for value in output["source_label"].dropna().unique()
        ),
    }
    return output, stats


def named_structure_reference() -> list[dict[str, Any]]:
    rows = [
        ("Ch", "Chilonga", "Major fault zone named in the Figure 12 caption."),
        ("Kf", "Kafue", "Major fault zone named in the Figure 12 caption."),
        ("Ks", "Kasempa", "Major fault zone named in the Figure 12 caption."),
        ("Kw", "Kawire", "Major fault zone named in the Figure 12 caption."),
        ("Ky", "Kanyama", "Major fault zone named in the Figure 12 caption."),
        ("LB", "Lwawa", "Caption spelling retained; body text also uses 'Lwawu'."),
        ("LL", "Lungu / Lamala", "Caption spelling retained; body text also uses 'Lungu / Luamala'."),
        ("M", "Mwinilunga", "Major fault zone named in the Figure 12 caption."),
        ("Mb", "Mubulashi", "Caption spelling retained; body text also uses 'Mubalashi'."),
        ("Mh", "Maheba", "Major fault zone named in the Figure 12 caption."),
        ("Mt", "Matebo", "Caption spelling retained; body text also uses 'Mateba'."),
        ("MbFZ", "Mwembeshi Fault Zone", "Major fault zone named in the Figure 12 caption."),
        ("Ny", "Nyama", "Major fault zone named in the Figure 12 caption."),
        ("Pd", "Pedicle", "Major fault zone named in the Figure 12 caption."),
        ("Si", "Sialinga", "Major fault zone named in the Figure 12 caption."),
    ]
    return [
        {
            "abbreviation_as_published": abbreviation,
            "structure_name": name,
            "source_doi": "10.1029/2025TC008955",
            "source_figure": "Figure 12 caption",
            "paper_section": "4.2 (regional structural framework)",
            "attribution_status": "literature reference only",
            "geometry_status": "not linked to the 126-line anonymous compilation",
            "notes": notes
            + " No geometry, fault type, displacement, age or mineralisation association is assigned here.",
        }
        for abbreviation, name, notes in rows
    ]


def bedrock_stratigraphy_status() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "status": "missing_source",
        "decision": "No bedrock or stratigraphic layer is admitted to the geological foundation in this build.",
        "reason": (
            "No authoritative, cross-border DRC-Zambia bedrock or stratigraphic polygon "
            "source was available and independently validated in this milestone."
        ),
        "target_units": [
            "Katanga Supergroup",
            "Roan Group",
            "Mines-Kitwe Subgroup (with country-specific subdivisions where available)",
            "Nguba Group",
            "Kundelungu Group",
            "Biano Group where relevant",
            "Basement units",
            "Intrusive bodies",
            "Stratigraphic contacts and major unconformities",
        ],
        "rejected_substitutes": [
            {
                "asset": "Africa_Surface_Lithology.tif",
                "reason": "Surficial lithology, not bedrock stratigraphy; cannot support Roan/Mines-Kitwe/Nguba/Kundelungu labels.",
                "decision": "not admitted",
            },
            {
                "asset": "internal_contacts_only.gpkg",
                "reason": "Derived from the surficial-lithology raster; it is not a bedrock contact layer.",
                "decision": "not admitted",
            },
        ],
        "required_evidence_before_use": [
            "Source identity, publication date and authoritative map scale",
            "Consistent DRC and Zambian stratigraphic nomenclature and crosswalk",
            "Polygon coverage of the study frame with explicit gaps",
            "Contact-type semantics rather than an undifferentiated contact mask",
            "Licence and redistribution status",
        ],
        "consequence": (
            "Stratigraphic predictors must remain absent until a source meeting these criteria is "
            "acquired and reviewed. The existing surficial classes must not be relabelled as bedrock units."
        ),
    }
