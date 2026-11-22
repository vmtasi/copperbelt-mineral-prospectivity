# V15 lithology sensitivity report

## Retained component

`distance_to_lithology_contact` is retained in every primary model. It is the distance to the nearest internal lithological contact, not the distance to a specific stratigraphic horizon.

## Categorical field

The V14 categorical field is retained in the primary M0-M3 ladder because no defensible host-lithology replacement is available. The provenance audit found only 42% agreement with the raster class at the cell centroid and 45% agreement with the cell-majority raster class. The stored field is instead consistent with one of the tied nearest contact classes in 95.4% of cells, with no documented tie-breaking rule.

## Removed-lithology sensitivity

`M2_cloglog_no_litho` removes only the categorical lithology indicators. It retains the validated contact-distance terms, domain intercepts, domain-varying distance coefficients, shared gravity and the log-area offset.

| Model | AUC | PR-AUC | Mean log score | Brier |
|---|---:|---:|---:|---:|
| M2 with V14 lithology | 0.7340 | 0.1633 | -0.25711 | 0.06922 |
| M2 without categorical lithology | 0.7347 | 0.1633 | -0.25709 | 0.06927 |

The paired mean log-score difference is 0.000021 per cell, with a block-bootstrap interval of [-0.00078, 0.00072]. The categorical field has no detectable predictive contribution under these folds.

## Centroid-lithology experiment

No defensible centroid assignment was implemented. The only centroid raster is a surficial-lithology product, not the host-lithology source from which the V14 class was derived. Relabelling cells from that raster would substitute a different geological quantity and would not be a sensitivity test of the existing model.

The raster-centroid specification remains listed in the configuration as exploratory, but it was not promoted or fitted.

## Interpretation

Removing the categorical field does not materially change predictive performance or the M1-M2 comparison. The categorical lithology coefficient should therefore be described as a weakly identified association with a contact-line class, not as a host-rock effect.

## Outputs

- `paper_1/v15/predictions/oof_M2_cloglog_no_litho.csv`
- `paper_1/v15/metrics/model_metrics.csv`
- `paper_1/v15/metrics/paired_comparisons.csv`
- `paper_1/v15/metrics/paired_bootstrap.csv`
- `paper_1/v15/diagnostics/lithology_assignment_audit.json`
