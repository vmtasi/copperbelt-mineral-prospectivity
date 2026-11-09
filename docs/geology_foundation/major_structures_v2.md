# Daly et al. (2025) named major-structure reconstruction v2

Status: built and independently verified on 2026-11-06.  
Build ID: `20261106T135434Z`  
Source DOI: [10.1029/2025TC008955](https://doi.org/10.1029/2025TC008955)

This document reports a cartographic naming and reconstruction pass over the
accepted v1 heavy-linework extraction. It is not a model fit, a kinematic
interpretation or a field-validated fault map. The v1 extraction remains
immutable. Mineral-occurrence locations, deposit proximity and exploration
density were not read or used at any stage of naming, joining or classification.

## 1. Scope and evidence hierarchy

The layer answers a narrow question: which parts of the emboldened linework
printed in Daly et al. (2025) can be associated with the 15 major fault-zone
names given in the Figure 12 caption, and what is the smallest defensible
cartographic reconstruction of each named parent system?

The evidence hierarchy is explicit:

| Role | Source | What it supports | What it does not support |
|---|---|---|---|
| Primary naming reference | Figure 12 raster | Published abbreviations, label placement and caption names | A unique geometry for every name |
| Geometry cross-check | Figure 14 raster | Continuity and topology of the emboldened heavy linework | Names, kinematics, displacement or fault-zone width |
| Original geometry | Accepted v1 extraction | 248 segment geometries and source identifiers | Names or parent-system membership |
| Published text | Daly et al. (2025) PDF and Figure 12 caption | The 15-name catalogue and geological context | Geometry where the printed evidence is ambiguous |

The source files are:

- `data/raw/daly_2025/Daly_Figure12_Katangan_Basin_Structural_Map.jpg`
- `data/raw/daly_2025/Daly_Figure14_Tectonic_Domains_FULL_modified.tif`
- `literature/Tectonics - 2025 - Daly - The Inter‐Cratonic Neoproterozoic Katangan Basin of Central Africa  Rift Basin Formation .pdf`
- `data/processed/geology_foundation/daly_major_structures_v1.gpkg`,
  layer `daly_major_structures`

The exact input fields used are:

| Input | Fields used |
|---|---|
| v1 GeoPackage | `structure_id`, all original attribute columns, `geometry` |
| `configs/major_structures_v2_attribution.json` | `label_anchors_px`; `structures[].abbreviation`, `name`, `aliases`, `label_anchor_key`, `attribution_confidence`, `geometry_status`; `centerline_parts[].part_id`, `ordered_segment_ids`, `bridge_max_km`, `role`, `confidence`; `supporting_segments[]`; `candidate_segments_not_assigned[]` |
| `configs/major_structures_v2.json` | `figure12_georeference`, `sources`, `protected_source_hashes`, `outputs` |
| `named_structure_reference.csv` | `abbreviation_as_published`, `structure_name` |
| Frozen V14 table | Hash check only; no occurrence field is read for naming or reconstruction |

The accepted v1 source has 248 LineStrings, 173 networks and 7,059.396 km of
extracted centreline. Its GeoPackage SHA-256 is
`8b816736b6571be68a2c6bec07307144d0f6fedef1cace9a2e2562220a921ab8`.
The v1 attribute table SHA-256 is
`58f131f69652493251ae1b07e3b25db664f81fb71e01babe77550d02a0436fde`.
The frozen V14 training table SHA-256 remains
`161ad50684950f7f106c079079eacbae9de09c1851dc15867b014c6b88bbd878`.

## 2. Attribution rules

The reconstruction follows these rules:

1. A published name is attached to geometry only when the label position, the
   heavy printed trace and the local trace topology support a unique
   association. Proximity alone is not evidence.
2. Every v1 segment retains its original `structure_id`. No source geometry is
   cut, snapped, dissolved or rewritten.
3. Ordered joins only reverse a segment if that puts its nearest endpoint next
   to the preceding segment. Joining does not move any segment endpoint.
4. A gap greater than 2 m between ordered components is represented by a
   separate straight bridge connector. Bridge geometry is stored in
   `bridge_connectors`; it is never presented as observed printed geometry.
5. Parent axes are reconstructed only from accepted printed components. No
   geometry is extrapolated to a label, an inferred continuation or a mineral
   occurrence.
6. The cartographic stroke width of a heavy printed line is not treated as
   physical fault-zone width. No fault-zone polygon was created.
7. Disconnected components remain separate parts when continuity is not
   mapped. The Kanyama south-west component is the explicit example.
8. Names with no defensible geometry remain in the catalogue as unresolved
   references. No nearest-line proxy is invented for them.

Confidence classes are:

| Class | Meaning |
|---|---|
| `high` | Label placement, unique heavy trace and local topology support the association. |
| `medium` | Association is supported but requires an interpreted offset or continuation across closely spaced linework. |
| `medium_low` | A plausible association exists, but continuity or uniqueness is not fully demonstrated. |
| `low` | A supporting or junction fragment is retained as mapped detail, not as a parent trace. |
| `unresolved` | The label cannot be uniquely associated with a continuous printed trace. |

The structure-level confidence for Kanyama is recorded as `mixed` because its
north-west main trace is high confidence, its north-west splay is medium, and
its south-west component is medium-low and explicitly disconnected.

## 3. Structure catalogue

All 15 published names are retained. Twelve have defensible geometry. Three are
reference-only cases.

| Abbr. | Published name | Aliases | Geometry status | Confidence | Attributed v1 segments | Centreline parts | Bridges | Mapped km | Centreline km |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Ch | Chilonga | - | attributed | high | 3 | 1 | 0 | 122.775 | 122.775 |
| Kf | Kafue | - | attributed | medium | 4 | 1 | 0 | 300.303 | 300.303 |
| Ks | Kasempa | - | unresolved | unresolved | 0 | 0 | 0 | 0.000 | 0.000 |
| Kw | Kawire | Kawiri | attributed | medium | 3 | 1 | 0 | 148.287 | 148.287 |
| Ky | Kanyama | - | partially attributed | mixed | 7 | 3 | 0 | 480.189 | 480.189 |
| LB | Lwawa | Lwawu | attributed | medium | 4 | 2 | 0 | 297.503 | 297.503 |
| LL | Lungu / Lamala | Lungu / Luamala; Lunga | attributed | high | 1 | 1 | 0 | 41.877 | 41.877 |
| M | Mwinilunga | - | attributed | medium | 3 | 1 | 0 | 180.566 | 180.566 |
| Mb | Mubulashi | Mubalashi | attributed | high | 1 | 1 | 0 | 71.125 | 71.125 |
| Mh | Maheba | - | attributed | high | 1 | 1 | 0 | 131.618 | 131.618 |
| Mt | Matebo | Mateba | unresolved | unresolved | 0 | 0 | 0 | 0.000 | 0.000 |
| MbFZ | Mwembeshi Fault Zone | Mwembeshi | attributed | high | 16 | 1 | 14 | 200.563 | 279.935 |
| Ny | Nyama | - | attributed | high | 10 | 1 | 7 | 85.217 | 114.702 |
| Pd | Pedicle | P | attributed | high | 1 | 1 | 0 | 190.387 | 190.387 |
| Si | Sialinga | Sailunga | unresolved | unresolved | 0 | 0 | 0 | 0.000 | 0.000 |

The attributed-segment count includes junction and supporting fragments.
`Mapped km` is the sum of printed components used in the named centreline.
`Centreline km` includes straight bridge connectors where recorded. The
supporting fragments add a further 24.960 km to the attributed-segment total;
that is why the sum of all 54 attributed v1 segments is 2,275.371 km, while the
named-centreline mapped-component total is 2,250.411 km.

## 4. Segment-to-structure attribution

Segment numbers below are the numeric suffixes of `DALY2025_MSC_*` identifiers.
The full row-level record, including label distance and evidence text, is in
`data/processed/geology_foundation/daly_structure_segment_attribution_v2.csv`.

| Structure | Centreline part | Ordered printed components | Supporting fragments | Candidate components not assigned |
|---|---|---|---|---|
| Ch | `CH_MAIN` | 0047 -> 0028 | 0199 | 0088 |
| Kf | `KF_MAIN` | 0012 -> 0033 -> 0034 -> 0062 | - | 0072, 0071 |
| Kw | `KW_MAIN` | 0038 -> 0024 | 0241 | - |
| Ky | `KY_NW_MAIN` | 0102 -> 0042 -> 0109 -> 0021 | 0217 | - |
| Ky | `KY_NW_SPLAY` | 0020 | - | - |
| Ky | `KY_SW_PROBABLE` | 0005 | - | - |
| LB | `LB_MAIN` | 0064 -> 0009 | - | - |
| LB | `LB_PARALLEL_MARGIN` | 0046 -> 0032 | - | - |
| LL | `LL_MAIN` | 0050 | - | - |
| M | `M_MAIN` | 0053 -> 0011 | 0112 | - |
| Mb | `MB_MAIN` | 0031 | - | 0117, 0172 |
| Mh | `MH_MAIN` | 0014 | - | 0001, 0037, 0016 |
| MbFZ | `MBFZ_MAIN` | 0206 -> 0059 -> 0243 -> 0124 -> 0098 -> 0242 -> 0077 -> 0156 -> 0125 -> 0084 -> 0155 -> 0153 -> 0131 -> 0204 -> 0127 -> 0041 | - | 0168 |
| Ny | `NY_MAIN` | 0063 -> 0145 -> 0120 -> 0248 -> 0115 -> 0137 -> 0119 -> 0141 -> 0212 | 0230 | - |
| Pd | `PD_MAIN` | 0004 | - | - |
| Ks | none | - | - | 0002, 0007, 0010, 0076 |
| Mt | none | - | - | 0092, 0087, 0136, 0007, 0001, 0174 |
| Si | none | - | - | 0105, 0208, 0222, 0176, 0101, 0134, 0121, 0126 |

Candidate components are retained in the evidence register but are not assigned
to a structure. The same candidate can be considered under more than one label
(for example 0007 under Ks and Mt) because the purpose is to document why it was
rejected, not to exhaustively partition the linework.

The v2 GeoPackage retains all 248 v1 segments in
`v1_segments_attributed`. Of these, 54 are attributed and 194 remain
`unnamed_no_supported_attribution`. The 194 unnamed segments total 4,784.025 km
and remain in the audit layer; they are not discarded or renamed by proximity.

Attribution-status counts are:

| Status | Segments |
|---|---:|
| `named_high` | 35 |
| `named_medium` | 12 |
| `named_low` | 4 |
| `named_medium_low` | 3 |
| `unnamed_no_supported_attribution` | 194 |
| **Total** | **248** |

## 5. Connectivity and bridge decisions

Only two structures required cartographic bridges:

| Structure | Bridges | Mapped component length | Bridge length | Maximum single gap | Reconstructed length |
|---|---:|---:|---:|---:|---:|
| MbFZ | 14 | 200.563 km | 79.371 km | 15.504 km | 279.935 km |
| Ny | 7 | 85.217 km | 29.484 km | 15.542 km | 114.702 km |
| **Total** | **21** | **285.780 km** | **108.855 km** | **15.542 km** | **394.637 km** |

Lengths in this table are rounded independently; totals are computed from the
unrounded geometry.

The bridges connect ordered components across printed dashes or label
interruptions. They are straight lines between the nearest component endpoints,
not traced fault geometry. Every bridge is stored with its source and target
segment identifiers, gap length and geometry.

The complete bridge record is:

| Bridge ID | Structure | Part | From | To | Gap (km) |
|---|---|---|---|---:|
| DALY2025_BRG_0001 | MbFZ | MBFZ_MAIN | 0206 | 0059 | 9.457 |
| DALY2025_BRG_0002 | MbFZ | MBFZ_MAIN | 0059 | 0243 | 1.291 |
| DALY2025_BRG_0003 | MbFZ | MBFZ_MAIN | 0243 | 0124 | 4.018 |
| DALY2025_BRG_0004 | MbFZ | MBFZ_MAIN | 0124 | 0098 | 2.371 |
| DALY2025_BRG_0005 | MbFZ | MBFZ_MAIN | 0242 | 0077 | 2.431 |
| DALY2025_BRG_0006 | MbFZ | MBFZ_MAIN | 0077 | 0156 | 15.504 |
| DALY2025_BRG_0007 | MbFZ | MBFZ_MAIN | 0156 | 0125 | 8.634 |
| DALY2025_BRG_0008 | MbFZ | MBFZ_MAIN | 0125 | 0084 | 2.372 |
| DALY2025_BRG_0009 | MbFZ | MBFZ_MAIN | 0084 | 0155 | 2.372 |
| DALY2025_BRG_0010 | MbFZ | MBFZ_MAIN | 0155 | 0153 | 6.427 |
| DALY2025_BRG_0011 | MbFZ | MBFZ_MAIN | 0153 | 0131 | 5.863 |
| DALY2025_BRG_0012 | MbFZ | MBFZ_MAIN | 0131 | 0204 | 1.292 |
| DALY2025_BRG_0013 | MbFZ | MBFZ_MAIN | 0204 | 0127 | 6.427 |
| DALY2025_BRG_0014 | MbFZ | MBFZ_MAIN | 0127 | 0041 | 10.913 |
| DALY2025_BRG_0015 | Ny | NY_MAIN | 0063 | 0145 | 2.375 |
| DALY2025_BRG_0016 | Ny | NY_MAIN | 0145 | 0120 | 5.310 |
| DALY2025_BRG_0017 | Ny | NY_MAIN | 0120 | 0248 | 1.293 |
| DALY2025_BRG_0018 | Ny | NY_MAIN | 0248 | 0115 | 15.542 |
| DALY2025_BRG_0019 | Ny | NY_MAIN | 0115 | 0137 | 1.293 |
| DALY2025_BRG_0020 | Ny | NY_MAIN | 0119 | 0141 | 1.294 |
| DALY2025_BRG_0021 | Ny | NY_MAIN | 0141 | 0212 | 2.377 |

No gap between 0098 and 0242 in the MbFZ chain required a bridge because the
nearest endpoints are within the 2 m contiguous tolerance. No bridge was added
between separate parent structures. The Kafue-Mubulashi and Ky-Lwawa contacts
are retained as distinct named axes even where their mapped linework touches.

## 6. Unresolved and rejected cases

Kasempa, Matebo and Sialinga remain in `named_structure_catalogue` with label
coordinates but no centreline. Their candidate components are recorded in the
attribution JSON, but none is promoted to a named parent axis.

| Name | Why no geometry is assigned | Candidate components |
|---|---|---|
| Ks (Kasempa) | Several heavy traces occur near the label, but none is uniquely abutted or continuous with it. | 0002, 0007, 0010, 0076 |
| Mt (Matebo) | A long heavy trace is nearby, but the label is offset and several components are similarly plausible. | 0092, 0087, 0136, 0007, 0001, 0174 |
| Si (Sialinga) | The label is surrounded by broken heavy components, with no preserved continuous parent axis. | 0105, 0208, 0222, 0176, 0101, 0134, 0121, 0126 |

The map marks these three names with orange triangles and shows their label
anchors without drawing a centreline. The absence of geometry is a result, not
a missing output.

## 7. Output layers and fields

### `data/processed/geology_foundation/daly_major_structures_named_v2.gpkg`

The GeoPackage has four layers, all in EPSG:32735:

| Layer | Geometry | Rows | Purpose |
|---|---|---:|---|
| `v1_segments_attributed` | LineString | 248 | Every original v1 segment, unchanged, with attribution fields appended |
| `named_structure_centrelines` | LineString | 15 | Approximate parent-axis parts reconstructed from accepted printed components |
| `bridge_connectors` | LineString | 21 | Explicit straight connectors across printed gaps |
| `named_structure_catalogue` | Point | 15 | One catalogue row per published name, including unresolved names |

Key columns in `v1_segments_attributed` are the original v1 fields plus:

| Field | Meaning |
|---|---|
| `attributed_structure_id` | Stable v2 structure identifier, null when unnamed |
| `attributed_structure_abbreviation` | Published abbreviation, null when unnamed |
| `attributed_structure_name` | Published name, null when unnamed |
| `attributed_status` | `named_high`, `named_medium`, `named_low`, `named_medium_low` or `unnamed_no_supported_attribution` |
| `attribution_confidence` | Same confidence scale as the catalogue |
| `mapping_role` | `main_trace`, `splay`, `parallel_fault_zone_margin`, `disconnected_southwest_component` or `junction_fragment` |
| `parent_part_id` | Centreline part to which the segment contributes |
| `label_anchor_key` | Figure 12 label anchor used in the attribution decision |
| `label_distance_px` | Cartographic distance from the segment to the label anchor, in source-image pixels |
| `attribution_evidence` | Reason text for the association |
| `source_name_figure` | `Daly et al. (2025) Figure 12` |
| `source_geometry_figure` | `Daly et al. (2025) Figure 14` |
| `occurrence_evidence_used` | Always `no` for this build |

The original v1 columns and geometries are not altered. The verifier compares
the 248 output rows against the v1 source column by column and geometry by
geometry, with zero tolerance.

### Tables and reports

| File | Contents |
|---|---|
| `data/processed/geology_foundation/daly_structure_segment_attribution_v2.csv` | One row per v1 segment with original attributes and v2 attribution fields |
| `data/processed/geology_foundation/daly_structure_summary_v2.csv` | One row per published name with geometry status, confidence, counts and lengths |
| `data/processed/geology_foundation/daly_structure_attribution_v2.json` | Machine-readable evidence, label anchors, accepted components, rejected candidates and unresolved register |
| `data/processed/geology_foundation/daly_structure_reconstruction_summary_v2.json` | Build counts, bridge totals and quality flags |
| `data/processed/geology_foundation/daly_structure_reconstruction_manifest_v2.json` | Source, configuration, output hashes, software versions and build command |
| `data/processed/geology_foundation/daly_structure_reconstruction_validation_v2.json` | Independent verifier output |

### Maps

- `outputs/geology_foundation/daly_named_structures_map_v2.png` shows named
  structures by confidence, the approximate parent axes, explicit bridges,
  unnamed heavy linework and unresolved published names.
- `outputs/geology_foundation/daly_structure_attribution_map_v2.png` shows the
  same evidence at segment level with source segment numbers. The disconnected
  Kanyama south-west part is labelled `Ky (SW)`.

## 8. Verification

The independent verifier is
`src/geology_foundation/verify_major_structures_v2.py`. It writes
`daly_structure_reconstruction_validation_v2.json`. The current build passes all
32 checks, including:

- all manifest source and output hashes match the files;
- v1, its attribute table and its manifest have unchanged SHA-256 values;
- the frozen V14 training table has its expected SHA-256 value;
- all 248 v1 segment IDs appear exactly once;
- every original v1 attribute and geometry is unchanged;
- all 15 published names are catalogued, with published names matched against
  `named_structure_reference.csv`;
- the three unresolved names have no centreline and no bridge;
- all 54 accepted segments exist in v1 and are uniquely assigned;
- all 15 centreline parts match the ordered configuration;
- centreline lengths match geometry lengths within 1.5 m;
- the 21 bridge rows match the declared count and bridge length;
- every bridge endpoint is a source-segment endpoint, within 1 mm;
- each bridge geometry length matches its stored gap length, within 1 mm;
- the segment CSV and structure summary CSV agree with the GeoPackage;
- both maps are readable PNG products at the expected scale; and
- no occurrence evidence is recorded in the layer or catalogue.

The validation file is the authoritative machine-readable audit. A failure in
any check makes the verifier exit non-zero.

An earlier build of this reconstruction produced an empty `bridge_connectors`
layer while the centreline metrics declared 21 bridges. The build now rejects
any mismatch between the stored bridge layer and the declared bridge count or
length, and the verifier independently reconciles every declared bridge with a
stored connector geometry. An empty or incomplete bridge layer therefore fails
the build instead of passing silently.

The build and verification commands are:

```powershell
& 'C:\Program Files\QGIS 3.40.15\bin\python-qgis-ltr.bat' -m py_compile `
    src\geology_foundation\reconstruct_major_structures_v2.py `
    src\geology_foundation\verify_major_structures_v2.py

& 'C:\Program Files\QGIS 3.40.15\bin\python-qgis-ltr.bat' `
    -m src.geology_foundation.reconstruct_major_structures_v2 `
    --config configs\major_structures_v2.json --overwrite

& 'C:\Program Files\QGIS 3.40.15\bin\python-qgis-ltr.bat' `
    -m src.geology_foundation.verify_major_structures_v2 `
    --manifest data\processed\geology_foundation\daly_structure_reconstruction_manifest_v2.json `
    --output data\processed\geology_foundation\daly_structure_reconstruction_validation_v2.json
```

The build is deterministic for the geometry, attribution and tables. The
GeoPackage and map file hashes can change when plotting code changes, so the
manifest is the authoritative record for a particular build.

## 9. Source and output hashes

Source hashes verified by the build and verifier:

| Input | SHA-256 |
|---|---|
| Figure 12 raster | `8960d8f6c71abc6208e2ad189019c54e1f752f68da9e562387cc8d17b2189912` |
| Figure 14 raster | `5b09b82d966e6b5fea7a48bb6083a8b3882ae375635a8b638cd3404b62fa8cf1` |
| Daly PDF | `43d9c94152987e56ce0a68cb2bf7988f5cf700e8803a3f10092927041c3a569a` |
| v1 GeoPackage | `8b816736b6571be68a2c6bec07307144d0f6fedef1cace9a2e2562220a921ab8` |
| v1 attribute CSV | `58f131f69652493251ae1b07e3b25db664f81fb71e01babe77550d02a0436fde` |
| v1 manifest | `5ef256f253553467f3fbcb22047dcd3e5d8eb333bb7e147b21a1927d8cdb6bcd` |
| V14 training CSV | `161ad50684950f7f106c079079eacbae9de09c1851dc15867b014c6b88bbd878` |

Current output hashes:

| Output | SHA-256 |
|---|---|
| v2 GeoPackage | `10ba59c7bbe771f3e74dfe0e6db3263718f292f95efe394d39a5fcd73413e330` |
| Segment attribution CSV | `4e270bb17fb623a0b16f62ee1809d6da6c8147756511e8c8a2a5cabfe0bbc269` |
| Structure summary CSV | `05bffb2db33172476e84f878acb29a595576c99d481a14f059dfbfe2c7f73f8c` |
| Attribution JSON | `a5c06a9b28dbe46bde437ff1b718a8a13b947d6c218b84b9e7520a5a0aeb83b3` |
| Reconstruction summary | `89ac39fc80b899ba62b23226bfe9b59278312b36723bdccb3dd566dcaa93485a` |
| Named-structure map | `ea371bc750e0c493e27b9c7d73f3f32efc62c456181eddfc31e6c48896962536` |
| Attribution map | `ed89e205459fdb91da332f591726489c70de73f4d10a336d0e7d5761e0683a55` |
| Validation report | `670868aa2f8a875dcd8a03caf3c1ec0af5b5dfbc125226220bce9e634562e635` |

The validation-report hash changes when the verifier writes a new timestamp;
the build manifest does not include it in the immutable output list.

## 10. Known limitations

1. **Cartographic source.** Figure 12 is a published raster with approximate
   coordinate ticks. Label anchors are accurate only to the pixel grid and the
   linear Plate Carree georeference; positional uncertainty is a few kilometres,
   not survey precision.
2. **Figure 14 is the geometry source.** Figure 12 names are primary, but its
   linework is not separately vectorised. Figure 14 provides the heavy trace
   that can be extracted consistently. A feature visible in one figure but not
   the other would require author-supplied vectors to resolve.
3. **Stroke width is symbolic.** The heavy printed line is a fault-zone symbol.
   Its width is not a displacement, damage-zone width or mappable polygon.
4. **Bridges are inferred.** Straight bridge connectors are reconstruction
   aids. They should not be used as observed fault geometry or as evidence of a
   continuous surface rupture.
5. **Parent axes are approximations.** `named_structure_centrelines` are ordered
   printed components plus bridges. They are suitable for candidate proximity
   and density analysis, not for structural-length or displacement statistics.
6. **Coverage is uneven.** Some labels and traces are close to the map frame or
   are interrupted by cartography. The extraction cannot establish whether a
   structure continues outside the published frame.
7. **Unresolved names are not zero-evidence names.** Kasempa, Matebo and
   Sialinga have literature names and nearby linework, but no unique printed
   association. They must not be converted to nearest-line proxies.
8. **No kinematics.** The layer carries no fault type, displacement, age,
   dip, slip sense or tectonic hierarchy. Any such attributes in later work must
   come from an independent source and remain separately identified.
9. **Old reference table.** `named_structure_reference.csv` is retained as the
   unmodified 15-name literature registry. Its `geometry_status` prose predates
   v2 and is not authoritative for the reconstructed layer; use
   `daly_structure_summary_v2.csv` and the v2 catalogue instead.

## 11. Interpretation boundary

The layer supports a structural proximity or structural-density predictor only
after the distinctions above are preserved. The primary geological geometry is
the printed heavy linework in `v1_segments_attributed`; the parent-axis layer is
a derived convenience. No result in this document is evidence that a named
structure controlled mineralisation, that a bridge is a real fault, or that any
unresolved name should be placed on the nearest trace.

The next-stage feature specification is in
`docs/geology_foundation/structural_predictor_specification.md`.
