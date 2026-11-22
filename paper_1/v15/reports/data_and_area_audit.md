# V15 data and area audit

## Scope

This audit reconstructs the modelling population, the binary outcome and the clipped-cell exposure used by the V15 Paper 1 comparison. It does not refit V14 or alter the frozen benchmark.

## Modelling population

The V14 table contains 1,899 grid rows and 1,899 unique grid identifiers. Of these, 139 rows carry `deposit_present = 1`.

The V15 complete-case frame contains 1,872 cells after excluding 27 rows with a missing tectonic domain. One positive cell, grid identifier 7074, is lost in that exclusion; the complete-case population therefore contains 138 positives and 1,734 negatives.

The four spatial folds each contain 468 cells. Their positive counts are:

| Fold | Cells | Positives |
|---|---:|---:|
| 1 | 468 | 9 |
| 2 | 468 | 31 |
| 3 | 468 | 37 |
| 4 | 468 | 61 |
| Total | 1,872 | 138 |

The outcome is binary: at least one compiled USGS occurrence in the cell. It is not a deposit count, a prospectivity score or a count of independent deposits.

## Occurrence accounting

The USGS source inventory contains 187 records. Their relationship to the modelling frame is:

- 186 records fall inside complete-case cells.
- 184 records fall inside the 138 model-positive cells.
- Two records fall in model-negative cells: Kanaronga (grid 986) and Luishia (grid 4288).
- One record, Seembere, falls in excluded positive cell 7074.
- 140 complete-case cells contain at least one point.
- 138 positive cells contain points, with a maximum of five points in one cell.
- 33 positive cells contain multiple records; 79 records lie in those cells.

The 184 occurrence records are therefore not 184 independent positive observations. The model has 138 positive cells. The 147 RDC.ROCKS observations are not part of this training inventory and are not model positives.

## Area definition

Cell area is the intersection of the full 5 km cell footprint with the union of the five USGS permissive tracts. It was calculated in EPSG:32735 and expressed in square kilometres. The complete 5 km cell footprint is 25 km2.

The 1,899-cell grid has a total clipped area of 21,790.3558 km2. The maximum difference between the stored polygon area and an independent cell-box/tract intersection is 4.3e-14 km2.

Area is therefore a genuine support measure, not a proxy derived from occurrence density. It is not, however, a measure of exploration effort, reporting intensity or geological favourability.

## Area results

There are 205 complete-case cells below 1 km2. None is positive. All are boundary cells against the permissive-tract edge.

| Group | Cells | Positives | Mean area (km2) | Median area (km2) |
|---|---:|---:|---:|---:|
| Negative | 1,734 | 0 | 11.38 | 9.43 |
| Positive | 138 | 138 | 13.34 | 12.02 |

Area alone has modest pooled but strongly non-stationary discrimination, rising from below chance in Fold 1 to 0.70 in Fold 4:

| Sample | Cells | Positives | ROC-AUC of area alone |
|---|---:|---:|---:|
| Fold 1 | 468 | 9 | 0.4103 |
| Fold 2 | 468 | 31 | 0.6095 |
| Fold 3 | 468 | 37 | 0.6745 |
| Fold 4 | 468 | 61 | 0.7031 |
| Pooled | 1,872 | 138 | 0.5834 |

**Correction (22 Nov 2026):** an earlier version of this table reported 0.5338/0.5538/0.5847/0.5663 (pooled 0.5591) and attributed the preliminary 0.583/0.703 values to a degree-area calculation. That diagnosis was wrong. `src/v15/metrics.py::roc_auc` clipped its input to the probability range [1e-12, 1], which ties every cell larger than 1 km2. Areas in degrees are all far below 1 and escaped the clipping, which is why they reproduced the preliminary pattern. The function now ranks raw scores (regression test `test_roc_auc_ranks_raw_covariates_without_clipping`). The values above are recomputed in EPSG:32735 and agree with the preliminary figures.

The area quantile relationship is non-monotone:

| Decile | Cells | Positives | Positive rate | Mean area (km2) |
|---|---:|---:|---:|---:|
| 1 | 188 | 0 | 0.000 | 0.33 |
| 2 | 187 | 3 | 0.016 | 1.65 |
| 3 | 187 | 9 | 0.048 | 3.32 |
| 4 | 187 | 15 | 0.080 | 5.59 |
| 5 | 187 | 23 | 0.123 | 8.29 |
| 6 | 187 | 27 | 0.144 | 11.58 |
| 7 | 187 | 19 | 0.102 | 15.29 |
| 8 | 187 | 19 | 0.102 | 19.80 |
| 9 | 187 | 15 | 0.080 | 24.39 |
| 10 | 188 | 8 | 0.043 | 25.00 |

Area is weakly negatively correlated with the frozen V14 probability (Pearson -0.119, Spearman -0.112). It is correlated with distance to fault (Pearson 0.172) and weakly with contact distance (-0.089) and Bouguer gravity (-0.051).

## Interpretation

The point-process interpretation is partially defensible. A cell's opportunity to contain a recorded occurrence does depend on how much of that cell lies inside the permissive tract. Using log clipped area as a fixed exposure offset is therefore geometrically justified.

It is not a correction for unequal exploration or reporting effort. The occurrence inventory is a compiled record of known deposits and prospects, not a complete spatial point pattern. Area-adjusting it does not create unbiased sampling.

The primary inference therefore treats the offset as spatial support, not as a claim that deposits arise from a homogeneous Poisson process across the Copperbelt.

## Reproducible outputs

- `paper_1/v15/data/v15_model_frame.csv`
- `paper_1/v15/diagnostics/data_build_audit.json`
- `paper_1/v15/diagnostics/area_all_cells.csv`
- `paper_1/v15/diagnostics/area_by_fold.csv`
- `paper_1/v15/diagnostics/area_by_domain.csv`
- `paper_1/v15/diagnostics/area_by_label.csv`
- `paper_1/v15/diagnostics/area_quantiles.csv`
- `paper_1/v15/diagnostics/area_auc_by_fold.csv`
- `paper_1/v15/diagnostics/area_correlations.csv`
- `paper_1/v15/diagnostics/small_cells.csv`
- `paper_1/v15/diagnostics/occurrence_support_summary.json`
