# Geological Foundation v1

Status: built and independently verified on 2026-11-06. This is a reproducible
geological-data foundation, not a complete geological model and not a modelling run.
V14 remains the historical benchmark; no V14 data, model output or manuscript file
was modified.

## Scope

This milestone establishes four things:

1. A canonical reconstruction of the nine Daly et al. (2025) Figure 14 tectonic
   domains in EPSG:32735.
2. A provenance-controlled baseline of the existing 126 anonymous structural lines,
   explicitly restricted to Tier 3 use.
3. A machine-readable statement of what is still missing, especially bedrock and
   stratigraphy, plus an auditable coverage table for the frozen V14 grid.
4. A georeferenced vector extraction of the emboldened major structural linework
   printed in Figure 14, retained as candidate geometry without invented names or
   fault kinematics.
5. A separately versioned, evidence-based naming and parent-axis reconstruction
   of the Figure 12 major fault zones, with unresolved names left unresolved.

It does not establish a complete fault hierarchy, bedrock units, stratigraphic
contacts, occurrence-process intensity or observation bias.

## Sources

| ID | Source | Local copy | Upstream locator | Status |
|---|---|---|---|---|
| `DALY2025_FIG14` | Daly et al. (2025), Figure 14, DOI [10.1029/2025TC008955](https://doi.org/10.1029/2025TC008955) | `data/raw/daly_2025/Daly_Figure14_Tectonic_Domains_FULL_modified.tif` | `H:\My Drive\Research\GeoMining\Data\Daly_Figure14_Tectonic_Domains_FULL_modified.tif` | CC BY 4.0 article; attribution required |
| `SELLEY2005_FAULTS` | Selley et al. (2005), Figure 1, as held in the existing `CACB_Structures` compilation | `data/raw/selley_2005/Selley_2005_faults_shear_zones_compilation.gpkg` | `C:\Project\data for projects\CACB_Structures.gpkg#faults_shear_zones` | Licence/redistribution status unverified; local research use only |

The Daly source is a georeferenced RGB rendering of the published figure, not an
author-supplied GIS layer. It is 1,924 x 1,912 pixels in EPSG:4326, with approximately
586 m ground pixels. The output raster is resampled to 600 m in EPSG:32735 (UTM zone
35S). These are cartographic reconstructions, not surveyed polygon boundaries.

## Canonical domain reconstruction

The build classifies RGB pixels against the nine Figure 14 palette colours, removes
legend swatches and small text components, closes a small 2-pixel seed gap to bridge
black cartographic linework, closes and fills the map envelope, assigns envelope
pixels to the nearest retained class seed, applies a 5-pixel median filter, and
polygonises the result.

| Class | `domain_id` | Published meaning in Figure 14 | Area (km²) |
|---:|---|---|---:|
| 1 | `1` | NKB: Northeastern Katangan Basin | 126,249.48 |
| 2 | `2` | MMSB: Mwashia marginal sub-basins | 20,821.32 |
| 3 | `3a` | NRB: inverted salt province | 82,841.40 |
| 4 | `3b` | NRB: Kafue platform | 19,486.80 |
| 5 | `4a` | CRZ: HP/HT orogen | 39,184.20 |
| 6 | `4b` | CRZ: high-strain, LP/LT | 6,687.00 |
| 7 | `5` | SRB: inverted basin | 63,483.12 |
| 8 | `6` | MSB: Mumbwa Sub-Basin | 28,426.32 |
| 9 | `7` | Hook Granite Batholith and early/syn-tectonic granites | 11,248.56 |

The published Figure 14 legend groups 4a and 4b under CRZ. The paper also describes
Domain 4 as a high-strain and metamorphic zone with basement domes. Both usages are
retained in the crosswalk notes; they must not be conflated in later prose.

Domain boundaries are changes of published figure colour class. They are not mapped
faults, stratigraphic contacts or surveyed tectonic boundaries. The raster and vector
layers therefore carry only approximate boundary confidence.

The earlier 11-polygon tectonic layer is not used as the canonical source. Its raw
labels include repeated `2 MMSB` polygons, `Inverted Basid`, a missing Domain 7 and a
materially different Domain 7/sub-domain 4b geometry. The raw V14 domain strings are
preserved in the V14 training table and in `v14_domain_assignment_coverage.csv`; they
are evidence of the historical implementation, not the current classification.

## V14 coverage audit

The frozen V14 table has 1,899 cells, 1,872 complete cases, 139 positive cells and 138
complete-case positive cells. Its SHA-256 is
`161ad50684950f7f106c079079eacbae9de09c1851dc15867b014c6b88bbd878`, and it matches
the preservation bundle.

Cell centroids were assigned only when they fall inside a reconstructed domain polygon.
No extrapolation was applied outside the published map envelope.

| Measure | All cells | Complete cases |
|---|---:|---:|
| Cells | 1,899 | 1,872 |
| Assigned to a reconstructed domain | 1,709 | 1,709 |
| Outside reconstructed envelope | 190 | 163 |
| Positive cells | 139 | 138 |
| Assigned positive cells | 138 | 138 |
| Unassigned positive cells | 1 | 0 |

Assigned complete cases by reconstructed domain:

| Domain | Cells |
|---|---:|
| `1` NKB | 46 |
| `2` MMSB | 10 |
| `3a` NRB | 1,120 |
| `3b` NRB | 478 |
| `4a` CRZ | 2 |
| `4b` CRZ | 52 |
| `5` SRB | 1 |
| `6` MSB | 0 |
| `7` Granite | 0 |

The 190 unassigned cells comprise 3 with legacy label `2 MMSB`, 59 with
`Kafue NRB 3b`, 101 with `LP/LT CRZ 4b`, and 27 with no V14 domain label. The single
unassigned positive cell (`cell_id` 7074) is also one of the 27 cells without a V14
domain label and is not a complete case.

This is a material constraint on the redesign. The reconstructed domain layer does not
provide a complete categorical covariate over the V14 frame, and the usable V14 data
are overwhelmingly concentrated in domains 3a and 3b. Domains 4a and 5 contain 2 and 1
cells respectively; domains 6 and 7 contain none. A domain-varying model fitted to
this particular overlap would reproduce the old sparsity problem rather than resolve
it. The next modelling decision must explicitly choose among frame restriction,
additional geological evidence, continuous spatial structure, or a combination, and
must state which estimand each choice changes.

The 2-pixel seed connection is a deliberate compromise. A 5-pixel connection would add
one further positive and 155 cells, but increases the envelope by about 113,000 source
pixels; that was rejected as overfilling. The coverage table, not a narrative claim,
is the acceptance evidence.

## Structural baseline

`structural_linework_tier3_v1.gpkg` contains the 126 source lines in EPSG:32735 with
source FIDs 1-126 preserved. It contains no structure names, fault types,
displacements, ages, mineralisation associations or hierarchy. `structure_name` and
`structure_type` are deliberately null. These features may be used only as a baseline
or sensitivity feature.

Fifteen named structures from the Daly et al. Figure 12 caption are recorded in
`named_structure_reference.csv`. That table is the unmodified literature registry and
does not itself link names to geometry. The separately versioned v2 reconstruction
described below links a name only where the printed evidence supports it.

## Major structural linework

`daly_major_structures_v1.gpkg` contains 248 georeferenced centreline segments
extracted from the thick black ribbons in Figure 14. The Figure 12 caption states
that the emboldened linework marks the major fault zones of the basin. The
extraction rejects the 1-2 pixel bedding/form lines and masks map text and legends.
It retains dashed traces as separate short segments rather than inventing the gaps.

The layer totals 7,059 km of extracted centreline, with segment lengths from 2.3 km
to 216.2 km. All 248 geometries are valid LineStrings in EPSG:32735. Names, fault
types and structural hierarchy are deliberately null: no segment has been assigned
a name by proximity alone. The full method, attribute dictionary and limitations are
in `docs/geology_foundation/major_structures.md`.

### Named-structure v2

`daly_major_structures_named_v2.gpkg` adds an evidence-based attribution pass without
changing v1. Twelve of the 15 published names have defensible geometry. Kasempa,
Matebo and Sialinga remain unresolved and have no invented centreline. The layer
contains:

- `v1_segments_attributed` - all 248 original segments with attribution fields appended;
- `named_structure_centrelines` - 15 approximate parent-axis parts for 12 names;
- `bridge_connectors` - 21 explicit straight links across printed gaps;
- `named_structure_catalogue` - 15 rows, including unresolved names.

Of the 248 original segments, 54 are attributed and 194 remain unnamed. The bridge
total is 108.855 km, with 14 bridges for Mwembeshi and 7 for Nyama. The layer is not a
kinematic or displacement interpretation, and printed stroke width is not physical
fault-zone width. The full method, attribution evidence, unresolved cases and visual
products are in `docs/geology_foundation/major_structures_v2.md`. The next-stage
structural predictor definitions are in
`docs/geology_foundation/structural_predictor_specification.md`.

Visual products are:

- `outputs/geology_foundation/daly_major_structures_overlay.png` - extraction on the published figure.
- `outputs/geology_foundation/daly_major_structures_overlay_detail.png` - enlarged overlay.
- `outputs/geology_foundation/daly_major_structures_segments.png` - segment-identity audit.
- `outputs/geology_foundation/daly_major_structures_map.png` - clean cartographic map.

## Bedrock and stratigraphy gap

No bedrock or stratigraphic layer is admitted. `Africa_Surface_Lithology.tif` and
`internal_contacts_only.gpkg` were rejected because they are surficial or derived
representations, not Roan/Nguba/Kundelungu bedrock stratigraphy. See
`bedrock_stratigraphy_status.json` for the required acquisition evidence and the
target units.

This means the current foundation cannot support claims about host stratigraphy,
stratigraphic contacts or a Mines-Kitwe host-rock predictor. Those claims remain
explicitly out of scope until an authoritative cross-border source is acquired and
reviewed.

## Outputs

All canonical outputs are under `data/processed/geology_foundation/`:

- `daly_tectonic_domains_v1.gpkg` - nine reconstructed domain polygons (`daly_domains`).
- `daly_domain_boundaries_v1.gpkg` - internal and external derived boundaries.
- `daly_tectonic_domains_v1_600m.tif` - categorical domain raster; 0 is NoData.
- `structural_linework_tier3_v1.gpkg` - anonymous Tier 3 structural lines.
- `daly_major_structures_v1.gpkg` - 248 major-structure candidate centrelines (`daly_major_structures`).
- `daly_major_structures_attributes_v1.csv` - segment attributes without geometry.
- `daly_major_structures_summary_v1.json` - extraction counts, lengths and parameters.
- `daly_major_structures_manifest_v1.json` - source, configuration and output hashes.
- `daly_major_structures_named_v2.gpkg` - named segments, parent axes, bridges and catalogue.
- `daly_structure_segment_attribution_v2.csv` - segment-to-structure attribution evidence.
- `daly_structure_summary_v2.csv` - structure-level summary table.
- `daly_structure_attribution_v2.json` - machine-readable attribution and unresolved register.
- `daly_structure_reconstruction_summary_v2.json` - v2 build summary and quality flags.
- `daly_structure_reconstruction_manifest_v2.json` - v2 source and output provenance.
- `daly_structure_reconstruction_validation_v2.json` - independent v2 verification report.
- `domain_crosswalk.csv` - class, name, area and method metadata.
- `v14_domain_assignment_coverage.csv` - one row per V14 cell with the domain assignment.
- `source_registry.csv` - source provenance and hashes.
- `named_structure_reference.csv` - literature names without geometry.
- `bedrock_stratigraphy_status.json` - explicit stratigraphy gap.
- `validation_report.json` - build-time checks and coverage evidence.
- `manifest.json` - inputs, outputs, checksums, software versions and limitations.

Stage 3 structural predictors are documented separately in
`docs/geology_foundation/structural_predictors_v1.md`. Their canonical outputs are
under `data/processed/structural_predictors/`, including the support GeoPackage,
support-keyed feature table, feature dictionary, sensitivity tables, redundancy
shortlist, manifest and independent validation report. Figures are under
`outputs/structural_predictors/`. No model has been fitted on these features.

The visual reconstruction is `outputs/geology_foundation/daly_domain_reconstruction.png`.

## Reproduction and verification

Use the QGIS Python environment because it contains GeoPandas, Shapely, pyogrio and
GDAL. The build does not fit a model.

```powershell
& 'C:\Program Files\QGIS 3.40.15\bin\python-qgis-ltr.bat' -m py_compile src\geology_foundation\__init__.py src\geology_foundation\common.py src\geology_foundation\domains.py src\geology_foundation\structural.py src\geology_foundation\build.py src\geology_foundation\verify.py

& 'C:\Program Files\QGIS 3.40.15\bin\python-qgis-ltr.bat' -m src.geology_foundation.build --config configs/geology_foundation.json

& 'C:\Program Files\QGIS 3.40.15\bin\python-qgis-ltr.bat' -m src.geology_foundation.verify --manifest data/processed/geology_foundation/manifest.json

& 'C:\Program Files\QGIS 3.40.15\bin\python-qgis-ltr.bat' -m src.geology_foundation.major_structures --config configs/major_structures.json --overwrite

& 'C:\Program Files\QGIS 3.40.15\bin\python-qgis-ltr.bat' -m src.geology_foundation.verify_major_structures --manifest data/processed/geology_foundation/daly_major_structures_manifest_v1.json

& 'C:\Program Files\QGIS 3.40.15\bin\python-qgis-ltr.bat' -m src.geology_foundation.reconstruct_major_structures_v2 --config configs/major_structures_v2.json --overwrite

& 'C:\Program Files\QGIS 3.40.15\bin\python-qgis-ltr.bat' -m src.geology_foundation.verify_major_structures_v2 --manifest data/processed/geology_foundation/daly_structure_reconstruction_manifest_v2.json --output data/processed/geology_foundation/daly_structure_reconstruction_validation_v2.json

& 'C:\Program Files\QGIS 3.40.15\bin\python-qgis-ltr.bat' -m src.geology_foundation.build_structural_predictors_v1 --config configs/structural_predictors_v1.json --overwrite

& 'C:\Program Files\QGIS 3.40.15\bin\python-qgis-ltr.bat' -m src.geology_foundation.verify_structural_predictors_v1 --manifest data/processed/structural_predictors/structural_predictor_manifest_v1.json
```

Use `--overwrite` only after reviewing that the existing files are generated artifacts
from this build. The verifier is read-only and fails if an output hash, source hash,
geometry, CRS, structural FID, coverage count or V14 preservation check does not match.

## Acceptance limits

The foundation is accepted only with the following qualifications:

- domain polygons are a ~600 m map-derived approximation;
- the closed envelope removes some cartographic blanks and does not extrapolate beyond
  the published coloured map;
- 190 V14 cells, including one non-complete positive, lie outside the reconstructed
  envelope;
- domain 3a/3b dominate the V14 overlap, while domains 4a, 5, 6 and 7 are sparse or
  absent;
- the anonymous Selley structural layer has no usable attribution beyond its source linework;
- the v1 major-structure layer remains an unnamed figure-derived candidate network;
- v2 names 12 of 15 published structures with cartographic confidence, but three names
  remain unresolved and no kinematics or displacement is assigned;
- bedrock and stratigraphy remain unavailable;
- licence status for the structural compilation is unresolved.

These are limits on inference, not merely implementation details. They must be carried
into the next modelling stage rather than treated as resolved by the reconstruction.
