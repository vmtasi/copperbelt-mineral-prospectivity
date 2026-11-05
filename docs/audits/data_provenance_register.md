# Data Provenance Register

**Audit date:** 2026-11-05
**Scope:** every dataset that materially supports the V14 benchmark or the manuscript.
**Method:** manifest and metadata inspection; schema and checksum verification where the file was readable. No dataset was modified.

Two evidence tiers are used:

- **Verified now** -- inspected in this session, or checksum matched a manifest in this session.
- **Carried forward** -- verified in an earlier session but not re-readable now because the source is on an inaccessible drive or locked by another process.

---

## 1. Lineage

```text
USGS CACB deposits/prospects (D01)
Selley et al. (2005) fault compilation (D12)
WGM2012 Bouguer / isostatic grids (D13, D14)
Africa surficial lithology (D11)
        |
        |  (no repository script creates this join)
        v
V5 base table (D02)
  labels, domains, centroids, fault distance,
  gravity, lithology class, legacy contact distance
        |
        +--> build_corrected_training_table.py + V13 contact distance (D07)
        |         -> V13 table (D03)
        |
        +--> rebuild_litho_contact_centroid_features.py
                  -> V14 distance CSV/GPKG (D05, D06)
                  -> V14 table (D04)
        |
        v
V14 traces (D18)
        |
        +--> V14 OOF predictions (D19)
        +--> M7 matched-global OOF predictions (D20)
        +--> phase outputs (D21, D23)
        +--> figures and tables
```

The critical provenance fact is that **V14 changes only the lithology-contact distance**. The canonical manifest asserts `non_distance_columns_unchanged: true` and `labels_domains_classes_folds_unchanged: true`. Labels, domains, centroids, fault distance, gravity and surficial-lithology class are inherited from the V13/V5 chain. Their original construction is not scripted in this repository.

---

## 2. Register

| ID | Dataset | Path | Type | Origin | CRS / resolution | Size / count | Checksum | Status |
|---|---|---|---|---|---|---|---|---|
| D01 | USGS CACB deposits/prospects | `H:\...\CACB_deposits_prospects.shp` | vector points | USGS SIR 2010-5090-T | geographic; positional accuracy 1,000 m | 187 records (65 deposits, 102 prospects, 20 groups); 184 in frame | not computed (drive inaccessible) | **Carried forward** |
| D02 | V5 base training table | `archive/litho_contact_repair_2026-10-18/data/copperbelt_training_v5_with_tectonic_domain.csv` | CSV | undocumented join of D01, D11, D12, D13 | EPSG:32735 centroids | 1,899 rows | `59FA57B1...` | **Verified now** |
| D03 | V13 training table | `data/copperbelt_training_v13_litho_contact.csv` | CSV | `build_corrected_training_table.py` from D02 + D07 | EPSG:32735 | 1,899 rows | `ccf8f4c2...` (snapshot) | **Carried forward** (file locked) |
| D04 | V14 training table | `data/copperbelt_training_v14_centroid_contact.csv` | CSV | `rebuild_litho_contact_centroid_features.py` from D03 + D08 | EPSG:32735 | 1,899 rows; 139 positives; 27 missing domain (1 positive); 1,872 complete cases; 138 complete-case positives | `161ad506...` (manifest) | **Carried forward** (file locked) |
| D05 | V14 centroid-contact distance | `data/derived/litho_contact_centroid_distance_v14.csv` | CSV | `rebuild_litho_contact_centroid_features.py` | metres | 1,899 rows; min 11.13 m; median 11,979.61 m; mean 16,306.40 m; max 70,130.56 m; zero 0 | `f982dba4...` (manifest) | **Verified via manifest** |
| D06 | V14 distance geometry | `data/derived/litho_contact_centroid_distance_v14.gpkg` | GPKG points | same | EPSG:32735 | 1,899 points; layer `litho_contact_centroid_distance` | `6f5a061a...` (manifest) | **Verified via manifest** |
| D07 | V13 polygon-to-contact distance | `data/derived/litho_contact_distance_v13.csv` | CSV | `rebuild_litho_contact_features.py` | metres | 1,899 rows; 270 zero-distance cells | `dda749f9...` | **Verified via manifest** |
| D08 | Internal contacts | `C:\Project\data for projects\internal_contacts_only.gpkg` | MultiLineString | polygonized D11, perimeter removed | EPSG:32735 | 259 features; 2,201,124 m total; classes 4:75, 6:3, 7:54, 14:5, 16:27, 20:95 | `0F444BF2...` | **Verified now** |
| D09 | All lithology lines | `C:\Project\data for projects\all_lithology_lines.gpkg` | MultiLineString | polygonized D11 | EPSG:32735 | 15,562,620 m total | `d7d4d091...` (manifest) | **Verified via manifest** |
| D10 | Outer boundary | `C:\Project\data for projects\outer_boundary_line.gpkg` | line | tract union perimeter | EPSG:32735 | 1 feature; 13,361,496 m | `8cc81093...` (manifest) | **Verified via manifest** |
| D11 | Africa surficial lithology | `C:\Project\data for projects\Africa_Surface_Lithology.tif` | Byte thematic raster | Sayre et al. / USGS Africa compilation | EPSG:4326; 3 arc-second; 106,636 x 101,430; NoData 255 | 19 classes + water; six in frame | not computed (181 MB; metadata verified) | **Carried forward** |
| D12 | Fault / shear-zone lines | `C:\Project\data for projects\CACB_Structures.gpkg`, layer `faults_shear_zones` | LineString | digitized from Selley et al. (2005) Fig. 1 | EPSG:32735 | 126 features; fields only `source`, `notes`, `geometry`; all source `Selley_et_al_2005_Fig1`; `notes` null | `C410FCB3...` | **Verified now** |
| D13 | Bouguer gravity | no raster in repository or `C:\Project\data for projects` | grid | inferred WGM2012 | 2 arc-minute | values in D02/D04: -55.08 to 17.44 mGal | not available | **Unresolved** |
| D14 | Isostatic gravity | no raster in repository or `C:\Project\data for projects` | grid | inferred companion grid | 2 arc-minute | values: -45.06 to 31.20 mGal; corr with Bouguer 0.8925 | not available | **Unresolved** |
| D15 | Magnetic (`emag`) | `C:\Project\data for projects\emag_clip.tif` | float raster | not documented | EPSG:32735; 119 x 133; approximately 3,680 m pixel; NoData -3.4028235e38 | 819 sentinel cells; 91 positive; valid 6.836% | `9249EFC9...` | **Unresolved; not used in V14** |
| D16 | Grid polygons | `C:\Project\data for projects\grid_internal_contact_distance_1899.gpkg` | polygons | 5 km grid clipped to USGS tract union | EPSG:32735 | 3,970 rows for 1,899 IDs (many-to-many contact join); must collapse to unique IDs | `877CFC16...` | **Verified now** |
| D17 | Mining footprints | `C:\Project\data for projects\mining_footprints_clipped.gpkg` | polygons | not documented | not inspected in this session | not used in V14 | not computed | **Unresolved; not used** |
| D18 | V14 traces | `figures/v14_centroid_contact/v14_centroid_fold_{1..4}_trace.nc` | NetCDF | `v14_centroid_spatial_stability.py` | 2 chains x 1,500 draws | 4 files, approximately 2.09 MB each | `05f72bf4...`, `13737735...`, `47be8419...`, `20833b82...` | **Verified now** |
| D19 | V14 OOF predictions | `figures/v14_centroid_contact/v14_centroid_oof_predictions.csv` | CSV | `extract_v14_oof.py` | EPSG:32735 centroids | 1,872 rows; 138 positives | `f1713d20...` | **Verified now** |
| D20 | M7 matched-global OOF | `figures/audit/phase4_oof_predictions_v14_centroid.csv` | CSV | `phase4_tectonic_heterogeneity.py` | EPSG:32735 centroids | 1,872 rows; 12 columns; **no stable `id` column** | `1A6283D7...` | **Verified now** |
| D21 | Phase 7 validation summary | `figures/audit/phase7_spatial_validation_v14_centroid.csv` | CSV | `phase7_spatial_validation.py` | fold-level | 4 rows | `D188CFF3...` | **Verified now** |
| D22 | Figure 2 cell geometry | `figures/figure2_spatial_validation_cells.gpkg` | GPKG polygons | `generate_figure2_spatial_validation.py` | EPSG:32735 | not re-counted in this session | not computed | **Historical-essential** |
| D23 | V14 stability matrix | `figures/v14_centroid_contact/v14_centroid_geological_stability_matrix.csv` | CSV | V14 runner | 4 folds x 6 domains x 2 distances | 1,103 bytes | `710d3520...` | **Verified now** |
| D24 | V14 run and validation manifests | `figures/v14_centroid_contact/run_manifest.json`; `figures/audit/v14_model_output_validation_post_refit_20261021.json` | JSON | V14 runner and validator | -- | `PASS` | manifest hashes recorded internally | **Verified now** |

---

## 3. Detailed entries

### D01 -- Occurrence inventory

The authoritative source is the USGS Central African Copperbelt assessment (SIR 2010-5090-T). The shapefile is on the `H:` drive and was **not accessible** in this session. The following figures are carried forward from the prior verified audit and are used consistently in the manuscript:

```text
187 records: 65 deposits, 102 prospects, 20 grouped records
184 records inside the 1,899-cell frame
139 unique positive cells
138 complete-case positive cells
source positional accuracy: 1,000 m
20 groups contain 71 components; 48 of 71 matched exactly
54 of 184 records within 250 m of a cell boundary
113 of 184 within 500 m
162 of 184 within 1 km
median distance to cell boundary: 398 m
```

The 1,000 m positional accuracy is one-fifth of the 5 km cell width, so records near cell edges can move between cells. Background cells are **not verified geological negatives**; they are cells without a compiled occurrence record.

**Licence:** USGS products are generally public domain in the United States; confirm the specific SIR data-release statement before redistribution.

### D02 -- V5 base table

The V5 table is the true root of the analytical chain:

```text
1,899 rows; 11 columns
id, litho_contact_litho_class, distance_to_lithology_contact,
deposit_present, bouguer, isostatic, centroid_x, centroid_y,
emag, distance_to_fault, domain
```

It contains the raw domain strings that the fitting code later collapses:

```text
Inverted Salt Province NRB 3a   1127
Kafue NRB 3b                     550
LP/LT CRZ 4b                     154
N Katangan Basin 1 NKB            24
2 MMSB                            16
Inverted Basid 5 SRB               1
```

No repository script creates this table. Its `emag` column already contains the `-3.4028235e+38` sentinel. Its `distance_to_fault` column is the stored distance that only 1,672 of 1,899 cells reproduce from the current fault layer.

### D03 -- V13 table

Built by `build_corrected_training_table.py` from D02 plus the V13 polygon-to-contact distance. The script preserves every other V5 column byte-for-byte at the text level. The V13 table is the base for the V14 rebuild.

### D04 -- V14 table

Built by `rebuild_litho_contact_centroid_features.py` from D03 plus D08. The only changed modelling column is `distance_to_lithology_contact`. The manifest asserts that labels, domains, classes and folds are unchanged.

Key counts:

```text
1,899 rows; 1,899 unique ids
139 positives in the full table
27 rows missing domain, including 1 positive
1,872 complete cases; 138 positives; 1,734 background
```

The V14 contact distance has no zero values:

```text
minimum 11.13 m; median 11,979.61 m; mean 16,306.40 m; maximum 70,130.56 m
```

The V13 polygon distance had 270 zero-distance cells. Every V14 value differs from V13 by at least 1.57 m; the median change is 2,213.58 m and the maximum change is 5,392.78 m.

### D08 -- Internal contacts

259 MultiLineString features in EPSG:32735, with fields only `fid` and `litho_class`. There are no class names, formation names, contact types, dips or stratigraphic attributes. The class counts are the numeric surficial-lithology codes, not bedrock units:

```text
4:75, 6:3, 7:54, 14:5, 16:27, 20:95
```

The outer perimeter is excluded; the contact total plus the perimeter total equals the all-lines total (partition check `PASS`).

### D11 -- Africa surficial lithology

Metadata verified:

```text
106,636 x 101,430 pixels
EPSG:4326
3 arc-second pixels
Byte thematic palette
NoData = 255
```

This is a **surficial lithology** compilation that mixes bedrock with unconsolidated parent material. It is not a bedrock host-rock map and contains no Roan, Mines--Kitwe, Nguba or Kundelungu formation labels. Six classes occur in the frame:

| Code | Class | Cells | Positives |
|---:|---|---:|---:|
| 4 | Metasedimentary | 891 | 70 |
| 20 | Water | 372 | 35 |
| 7 | Metaigneous | 239 | 14 |
| 16 | Alluvium--Saline | 229 | 14 |
| 6 | Silicic | 85 | 6 |
| 14 | Alluvium--Fluvial | 83 | 0 |

The class names are an external lookup, not stored in the raster attribute table used by the model. This answers the recurring question: **the data layer has codes; the names exist only in the legend/table mapping, and formation names do not exist at all in this dataset.**

### D12 -- Fault layer

```text
126 LineString features
EPSG:32735
fields: source, notes, geometry
source: Selley_et_al_2005_Fig1 (all features)
notes: null (all features)
```

There are no fault names, types, ages, dips, displacements, senses of movement or mineralisation associations. Length statistics:

```text
minimum 1.17 km; median 10.71 km; mean 17.32 km; maximum 112.70 km; total 2,182 km
```

A defensible fault hierarchy **cannot** be constructed from these attributes. Ranking by length or proximity to known deposits would be circular and is prohibited by the mandate.

Stored-distance reproduction:

```text
1,672 of 1,899 cells reproduce within 1 m
227 differ by more than 1 m
219 differ by more than 1 km
26 positive cells differ by more than 1 m
maximum difference: 33,076.98 m
```

### D13/D14 -- Gravity

The gravity predictors are a 2 arc-minute complete Bouguer grid and an isostatic companion. The Bouguer header names neither the product nor the datum; the identity as WGM2012 is **inferred** from companion free-air, disturbance and topography grids that share its limits and spacing. Each cell centroid was transformed to geographic coordinates, assumed WGS 84, and assigned the nearest node without interpolation:

```text
centroid-to-node offsets: 0.07--2.56 km; median 1.47 km
```

The two grids are highly redundant:

```text
corr(bouguer, isostatic) = 0.8924559, n = 1,899
```

They must not be treated as independent geological evidence without justification. The source rasters are absent from the repository and from `C:\Project\data for projects`; only the sampled values survive in D02/D04. No gravity-ingestion script exists.

### D15 -- Magnetic

`emag_clip.tif` is a 119 x 133 float raster in EPSG:32735 with approximately 3,680 m pixels. Its NoData value is the sentinel `-3.4028235e+38`. Valid statistics are:

```text
min -116.727; max 103.702; mean -18.711; SD 33.595; valid 6.836%
```

In the 1,899-cell table:

```text
819 cells contain the sentinel
91 positive cells affected
803 sentinel cells in NRB_3a
16 in MMSB
```

`emag` is **not used in V14**. It is classified **Unresolved** and must not be promoted into the redesigned model until provenance and NoData treatment are established.

### D16 -- Spatial support

The grid GeoPackage has **3,970 rows for 1,899 IDs** because the contact join is many-to-many. Any area calculation must first collapse to unique cell IDs. Independent measurements carried forward:

```text
1,899 cells
252 exactly 25 km2
254 at or above 24.999 km2
1,645 below 24.999 km2
total area 21,790.36 km2
mean cell area 11.4746 km2
mean clipped-cell area 9.9472 km2
minimum area approximately 1.11e-5 km2
median area 9.7673 km2
141 stored centroids lie outside their cell polygon
12 of those are positive cells
maximum centroid-to-cell distance 2,365.57 m
```

The manuscript reports 253 full-area cells and 1,646 clipped cells. The independent count differs by one cell depending on the area threshold. This is an unresolved definition/geometry discrepancy; neither value should be edited until the threshold is agreed.

Area-bin positive rates:

```text
<1 km2:       0 / 212
1--5 km2:    14 / 412
5--10 km2:   39 / 340
10--20 km2:  55 / 474
20--25.1 km2:31 / 461
```

The smallest cells contain no positives in this binning. This supports the area-weighting/spatial-support limitation.

---

## 4. Redundancy and consistency checks

| Check | Result | Implication |
|---|---|---|
| V14 vs V13 contact distance | all 1,899 values change; median +2,213.58 m | V14 is a genuine geometric change, not a relabel |
| V14 non-distance columns vs V13 | unchanged (`non_distance_columns_unchanged: true`) | labels/domains/gravity/faults are inherited |
| Bouguer vs isostatic | r = 0.8925 | do not use as independent evidence |
| Magnetic valid coverage | 6.836% | not usable as a regional predictor without treatment |
| Magnetic sentinel in positives | 91 of 138 | any naive promotion would corrupt the response |
| Domain missingness | 27 cells, 1 positive | complete-case exclusion is not random |
| Fold sizes | 468 each | balanced by cell count, not by positives or area |

---

## 5. Licensing and access

| Dataset | Access | Licence status |
|---|---|---|
| USGS CACB occurrence and tract data | public download / local `H:` copy | USGS public domain in the US; confirm data-release statement |
| Africa surficial lithology | local `C:\Project` copy | source terms not recorded in repo; verify before redistribution |
| WGM2012 gravity | local copy absent; values inherited | BGI terms not recorded; identity inferred, not header-confirmed |
| Selley et al. (2005) fault digitization | local GPKG | digitized from a published figure; attribution and permission not documented |
| Daly et al. (2025) domain digitization | local GPKG / QGIS project | digitized from Figure 14; attribution and permission not documented |
| Mining footprints | local GPKG | not used in V14; terms unresolved |
| Derived V14 tables and traces | local repository | repository code and manifests; no separate licence file exists |

No licence file (`LICENSE`) exists at the repository root. This must be resolved before public release.

---

## 6. Provenance gaps

1. **Occurrence ingestion is not reproducible in-repository.** The USGS shapefile is external and currently inaccessible; the V5 table is the effective source.
2. **Gravity ingestion is not reproducible.** No raster and no ingestion script survive.
3. **Fault distance is only partly reproducible.** The current fault layer reproduces 1,672 of 1,899 stored values; 227 differ.
4. **Domain assignment is digitized from a figure raster.** The 11-polygon layer contains gaps and overlaps; 49 complete-case background cells differ from a strict centroid-in-polygon join.
5. **V5-table construction is undocumented.** The chain begins at an archived artifact.
6. **Magnetic provenance is undocumented.** The raster is not usable as-is.
7. **The supplementary material cited by the manuscript is absent.** Supplementary S1 cannot be audited.
8. **No licence file exists.**

---

## 7. Status summary

| Status | Datasets |
|---|---|
| **Verified now** | D02, D08, D12, D16, D18, D19, D20, D21, D23, D24 |
| **Verified via manifest** | D05, D06, D07, D09, D10 |
| **Carried forward** | D01, D03, D04, D11, D22 |
| **Unresolved** | D13, D14, D15, D17 |

The V14 analytical table is internally consistent and its contact-distance rebuild is independently validated. The unresolved items are **upstream provenance**, not V14 arithmetic.
