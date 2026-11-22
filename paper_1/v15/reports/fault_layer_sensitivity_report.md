# V15 fault-layer sensitivity report

## Problem

The stored `distance_to_fault` predictor is reproducible within 1 m for 1,672 of 1,899 grid cells from the surviving `CACB_Structures.gpkg` layer. It is not reproducible for 227 cells. The maximum absolute difference is 33.1 km.

The original fault geometry used to construct the V14 field is not preserved. This is a provenance limitation, not a missing-value problem. The canonical complete-case population was not altered.

## Distribution of the 227 non-reproduced cells

| Group | Cells | Positive cells |
|---|---:|---:|
| Fold 1 | 69 | 9 |
| Fold 2 | 158 | 17 |
| NKB | 15 | 0 |
| NRB_3a | 94 | 6 |
| NRB_3b | 118 | 20 |

## Sensitivity metrics

| Model | Full frame AUC | Reproduced-cell AUC | Non-reproduced-cell AUC | Full-frame log score | Reproduced log score | Non-reproduced log score |
|---|---:|---:|---:|---:|---:|---:|
| M0_global | 0.6882 | 0.6943 | 0.6695 | -0.2725 | -0.2545 | -0.4027 |
| M1_domain_intercept | 0.7087 | 0.7315 | 0.6014 | -0.2633 | -0.2420 | -0.4179 |
| M2_domain_distance | 0.7340 | 0.7608 | 0.6967 | -0.2571 | -0.2339 | -0.4250 |
| M3_domain_distance_gravity | 0.8020 | 0.8177 | 0.7354 | -0.2267 | -0.2118 | -0.3352 |
| RF comparator | 0.7528 | 0.7629 | 0.6967 | -0.2352 | -0.2205 | -0.3415 |
| V14 frozen | 0.7155 | 0.7430 | 0.5989 | -0.2510 | -0.2307 | -0.3986 |

Predictive performance is consistently weaker in the non-reproduced cells, especially for the hierarchical models. The degradation is largest on log score, meaning those cells are not merely ranked differently but receive materially poorer probability assignments.

## Interpretation

All fold 1 non-reproduced cells are in NRB_3b or NKB; the affected folds are 1 and 2 only. The central M1-M2 comparison is therefore exposed to an unverifiable fault-distance component in part of the test support.

No corrected fault-distance field can be produced from the surviving source without changing the predictor definition. The limitation should be stated with the Paper 1 results. A future reconstruction requires the original structure layer or a documented rebuilding and validation of the entire distance field.

## Outputs

- `paper_1/v15/diagnostics/fault_distance_reproducibility.csv`
- `paper_1/v15/diagnostics/fault_distance_cell_audit.csv`
- `paper_1/v15/diagnostics/fault_distance_nonreproduced_by_fold.csv`
- `paper_1/v15/diagnostics/fault_distance_nonreproduced_by_domain.csv`
- `paper_1/v15/diagnostics/fault_distance_sensitivity_summary.json`
