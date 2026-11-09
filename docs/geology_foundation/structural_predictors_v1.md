# Structural predictors v1

Status: built and independently verified on 2026-11-06.

Build ID: `20261106T151000Z`

This note documents Stage 3 of the geological-foundation work: construction of
structural predictors on a new equal-area hexagonal support. No model was fitted,
no predictor was selected using mineral occurrences, and no V14 input, V14 output
or manuscript file was modified.

## 1. Scope

The build answers a limited question: how should the accepted v2 named-structure
geometry be represented as candidate structural predictors on a defensible spatial
support?

The build does not establish that any predictor controls mineralisation. The primary
named-structure geometry remains the observed heavy linework printed in Daly et al.
(2025). Bridges and reconstructed parent axes are retained only as sensitivity
representations.

Source DOI: [10.1029/2025TC008955](https://doi.org/10.1029/2025TC008955)

## 2. Inputs and protected evidence

| Input | Role | SHA-256 |
|---|---|---|
| `data/processed/geology_foundation/daly_major_structures_named_v2.gpkg` | Named major-structure geometry | `10ba59c7bbe771f3e74dfe0e6db3263718f292f95efe394d39a5fcd73413e330` |
| `data/derived/litho_contact_distance_v13.gpkg` | Historical tract footprint; geometry only | `ee9c975f2bf1fad5738d65d256fc142c61f6622ab75a086f0e18591f2fa24a33` |
| `data/processed/geology_foundation/daly_tectonic_domains_v1.gpkg` | Domain assignment and context | `ed01d7e639203e01fdc472cd3a1797e63c7e99bb35df08dc7d960b380124c497` |
| `data/processed/geology_foundation/daly_domain_boundaries_v1.gpkg` | Internal Daly boundaries for context distance | `631608a1fcf3e43b63ed218c1b3b39babb4d604c0bf381790a36577088d6c9f9` |
| `data/copperbelt_training_v14_centroid_contact.csv` | V14 id and centroid crosswalk only | `161ad50684950f7f106c079079eacbae9de09c1851dc15867b014c6b88bbd878` |

The V14 table was read with `usecols = [id, centroid_x, centroid_y]`. No
`deposit_present` field or other occurrence field was read. The v1 structure
GeoPackage, v1 attribute table, v1 manifest and V14 table were hashed before the
build and again by the verifier.

## 3. Spatial support

The new support is a flat-top equal-area hexagonal quadrature grid with a nominal
cell area of 10 km2. Hexagon centres were retained when they fell inside the union
of the historical tract-footprint polygons.

| Support measure | Value |
|---|---:|
| Hexagon-centroid support points | 2,193 |
| Nominal cell area | 10 km2 |
| Nominal total support area | 21,930 km2 |
| Cells at least 99% inside the tract footprint | 1,019 |
| Cells at least 80% inside the tract footprint | 1,447 |
| Edge cells below 99% overlap | 1,174 |
| Minimum overlap fraction | 0.112 |
| Median overlap fraction | 0.960 |
| Sum of overlap area | 18,445.73 km2 |

Every support polygon in the GeoPackage is a full equal-area hexagon. The clipped
overlap is stored as `frame_overlap_area_km2` and `frame_overlap_fraction` and is
not used as the model support geometry. This distinction is intentional: clipped
slivers are diagnostics, not unequal-area sampling units.

The support is not yet frozen as a modelling subset. The `full_support` and
`edge_hex` fields allow a later experiment to compare the full set with the
>=99% or >=80% overlap subsets. Those thresholds and the support size must be
fixed before model evaluation.

## 4. Structural attribution sets

| Set | Selection | Segments | Use |
|---|---|---:|---|
| `HM` | `named_high` or `named_medium` | 47 | Primary observed-structure geometry |
| `HIGH` | `named_high` | 35 | Confidence sensitivity |
| `ALL_NAMED` | any non-null `attributed_structure_id` | 54 | Broad confidence sensitivity |
| `UNNAMED` | no attributed structure | 194 | Mapping-texture sensitivity |

There are 12 HM named systems with geometry and 21 bridge connectors across two
systems: Mwembeshi Fault Zone (14 bridges) and Nyama (7 bridges).

## 5. Generated feature set

The output table contains 27 structural features. All 2,193 support points have a
defined value for every feature; there are no missing values.

### 5.1 Global distance and geometry-sensitivity features

- `STR_DIST_NAMED_HM_KM`
- `STR_DIST_NAMED_HIGH_KM`
- `STR_DIST_NAMED_ALL_KM`
- `STR_DIST_NAMED_BRIDGED_KM`
- `STR_DIST_NAMED_PARENT_AXIS_KM`
- `STR_DIST_NAMED_BRIDGED_DELTA_KM`
- `STR_DIST_NAMED_PARENT_AXIS_DELTA_KM`

### 5.2 Per-system distances

Twelve exploratory distances are written as `STR_DIST_<ABBR>_KM` for Ch, Kf, Kw,
Ky, LB, LL, M, Mb, MbFZ, Mh, Ny and Pd. They must not all be entered in one model.

### 5.3 Density and structural-complexity features

- `STR_NAMED_LINE_DENSITY_10KM`
- `STR_NAMED_LINE_DENSITY_25KM`
- `STR_NAMED_LINE_DENSITY_50KM`
- `STR_UNNAMED_LINE_DENSITY_25KM`
- `STR_NAMED_SYSTEM_COUNT_25KM`

Densities are exact line-length intersections with circular windows, expressed as
kilometres of line per 1,000 km2. The 25 km scale was designated as the primary
density candidate before any model comparison; the 10 km and 50 km scales are
sensitivity variants. The three named-line scales are correlated by construction.

### 5.4 Candidate intersection and domain-context features

- `STR_DIST_MAJOR_INTERSECTION_KM`
- `STR_MAJOR_INTERSECTION_COUNT_25KM`
- `STR_DIST_DALY_DOMAIN_BOUNDARY_KM`

Only two exact contacts exist between dissolved named systems:

| Candidate | Pair | Easting (m) | Northing (m) |
|---|---|---:|---:|
| `XSEC_001` | Kafue - Mubulashi | 713442.921 | 8516152.365 |
| `XSEC_002` | Kanyama - Lwawa | 98615.502 | 8602739.618 |

These are cartographic contacts, not established through-going fault intersections
or geological nodes. The Kanyama-Lwawa contact lies west of the tract support and
is the nearest candidate for 367 support cells. The Kafue-Mubulashi contact is the
nearest for the other 1,826 cells. Only three support cells have either candidate
within 25 km, so the count feature is predominantly zero and is context-only.

## 6. Descriptive results

### 6.1 Distance distributions

| Feature | Median (km) | 95th percentile (km) | Maximum (km) |
|---|---:|---:|---:|
| HM observed named structures | 33.67 | 92.91 | 119.63 |
| HIGH-confidence named structures | 49.64 | 115.95 | 145.47 |
| ALL_NAMED structures | 33.67 | 92.91 | 119.63 |
| Bridged HM geometry | 33.67 | 92.91 | 119.63 |
| HM parent axes | 33.67 | 92.91 | 119.63 |
| Candidate named-system contacts | 160.16 | 388.51 | 423.51 |
| Internal Daly domain boundaries | 24.29 | 68.01 | 94.39 |

The HM and ALL_NAMED distributions are almost identical because the additional
low-confidence fragments rarely become the nearest feature. The HIGH set is
materially sparser: 1,018 support points are farther from HIGH than from HM, with
a maximum difference of 112.49 km. This is a substantive confidence-sensitivity
result, not a small implementation detail.

### 6.2 Bridge and parent-axis sensitivity

Bridges and parent axes change the nearest distance at only 18 support points.

| System | Bridges | Support cells where bridge is closer by >1 m | Maximum bridge effect |
|---|---:|---:|---:|
| Mwembeshi Fault Zone | 14 | 17 | 0.173 km |
| Nyama | 7 | 21 | 0.850 km |

For 2,176 MBFZ support cells and 2,172 Nyama support cells, the observed and
bridged distances are identical within 1 m. The bridged and parent-axis layers
therefore remain sensitivity representations; they do not replace observed
linework.

### 6.3 Density and system counts

| Feature | Median | 95th percentile | Maximum | Zero-valued support cells |
|---|---:|---:|---:|---:|
| HM density, 10 km | 0.00 | 65.34 | 126.35 | 1,741 |
| HM density, 25 km | 0.00 | 45.37 | 56.91 | 1,317 |
| HM density, 50 km | 4.49 | 26.80 | 28.41 | 711 |
| Unnamed linework density, 25 km | 1.35 | 29.52 | 43.42 | 921 |

Units are km of line per 1,000 km2. The zero medians show that most support cells
are beyond the local 10 km and 25 km windows of HM linework; the 50 km scale is
needed to describe regional structural intensity.

Named-system counts within 25 km are 0 for 1,317 cells, 1 for 627, 2 for 184 and
3 for 65. This counts only the 12 HM systems and is not a fault census.

### 6.4 Domain and fold composition

| Domain group | Support cells |
|---|---:|
| NRB | 1,797 |
| CRZ | 76 |
| NKB | 27 |
| MMSB | 8 |
| Unassigned | 285 |

The 285 unassigned cells lie outside the reconstructed Daly domain envelope. They
must remain explicitly unassigned; no nearest-domain extrapolation was applied.

The historical V14 nearest-cell crosswalk distributes the 2,193 support points as
Fold 1 = 783, Fold 2 = 563, Fold 3 = 416 and Fold 4 = 431. These are not a new
spatial validation design. They are a backward crosswalk to the frozen V14
partitions and must not be described as independent evidence of transferability.

### 6.5 Redundancy diagnostics

There are 63 feature pairs with absolute Pearson or Spearman correlation >= 0.8.
The shortlist assigns fixed roles without using occurrence performance:

| Role | Features |
|---|---:|
| Primary distance candidate | 1 |
| Primary density alternative | 1 |
| Sensitivity variants | 9 |
| Context-only features | 4 |
| Exploratory per-system distances | 12 |

The HM, ALL_NAMED, bridged and parent-axis distances, 25 km and 50 km densities,
and named-system count form one large redundancy cluster. They should not be
treated as independent geological evidence. The 10 km density is less redundant
with the regional distance fields but has many zeros.

## 7. Outputs

Canonical tables and the GeoPackage are under
`data/processed/structural_predictors/`:

- `structural_predictors_v1.gpkg` - support polygons, support points with features,
  and the two candidate intersections.
- `structural_predictors_v1.csv` - support-keyed modelling table with 27 features.
- `structural_feature_dictionary_v1.csv` - definitions, sources and roles.
- `structural_feature_support_v1.csv` - distribution summary for every feature.
- `structural_feature_by_domain_v1.csv` - feature summaries by domain.
- `structural_feature_by_v14_fold_v1.csv` - feature summaries by V14 fold.
- `structural_feature_correlations_v1.csv` - all pairwise correlations.
- `structural_bridge_sensitivity_by_system_v1.csv` - observed, bridged and
  parent-axis distances for the two bridged systems.
- `structural_confidence_sensitivity_v1.csv` - HM, HIGH and ALL_NAMED comparisons.
- `structural_predictor_shortlist_v1.csv` - fixed roles and redundancy clusters.
- `structural_candidate_intersections_v1.csv` - candidate contacts and qualifications.
- `v14_crosswalk_v1.csv` - support-to-V14 nearest-cell and fold crosswalk.
- `structural_predictor_manifest_v1.json` - source and output hashes, support
  summary, feature inventory and limitations.
- `structural_predictor_validation_v1.json` - independent verification report.

Figures are under `outputs/structural_predictors/`:

- `structural_support_coverage.png`
- `structural_distance_hm.png`
- `structural_density_named_25km.png`
- `structural_intersection_distance.png`
- `structural_domain_boundary_distance.png`
- `structural_unnamed_density_25km.png`
- `structural_feature_distributions.png`
- `structural_correlation_heatmap.png`

## 8. Verification

The independent verifier reports 23 of 23 checks passing. It verifies:

- source and output SHA-256 hashes;
- unchanged v1 and V14 protected evidence;
- all support polygons have a nominal area of 10 km2 and all points are centroids;
- all 2,193 support points have defined HM distances;
- independent nearest-distance calculations on a 50-point sample;
- zero distance on HM linework;
- synthetic distance monotonicity and symmetry;
- a known-line density calculation and a 10-point independent density audit;
- exact reproduction of the named-system count;
- reproduction of the two candidate contacts and their qualifications;
- consistency of domain/fold summaries and the V14 crosswalk;
- the absence of occurrence fields from the feature table and crosswalk;
- the presence and dimensions of all eight PNG products.

Current manifest SHA-256:
`0281ad5fd8b4e126c1f441d2ebade86b9caba3387b19174dba101c0b07138d9e`.

## 9. Interpretation boundaries

The following statements are not supported by this build:

- that named major structures control mineralisation;
- that bridge connectors are observed faults;
- that parent axes are continuous surface ruptures;
- that the two candidate contacts are geological nodes;
- that unnamed line density is a census of structural complexity rather than a
  mixture of geology and map compilation;
- that Daly domain membership is a complete coverage classification;
- that the support subset is final for modelling;
- that V14 fold performance transfers to the new hexagonal support.

The next modelling stage must choose a compact, pre-specified predictor set and
training-fold-aware transformations. It should compare HM distance with the 25 km
density representation, use HIGH and ALL_NAMED as sensitivity analyses, and keep
the unnamed-line density and Daly boundary distance as contextual checks rather
than primary geological evidence.
