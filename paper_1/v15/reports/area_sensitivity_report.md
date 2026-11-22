# V15 area sensitivity report

## Models compared

Three specifications separate the effect of clipped-cell area from the effect of changing the link:

| Model | Link | Area treatment | AUC | PR-AUC | Mean log score | Brier |
|---|---|---|---:|---:|---:|---:|
| V14 control | logit | none | 0.7094 | 0.1361 | -0.2526 | 0.06891 |
| M2 domain-distance | cloglog | fixed log-area offset | 0.7340 | 0.1633 | -0.2571 | 0.06922 |
| M2 domain-distance | logit | standardized log-area covariate | 0.7371 | 0.1645 | -0.2487 | 0.06781 |

All three use the same 1,872 cells, folds, predictors and lithology field.

## Paired differences

The area-covariate model improves mean held-out log score relative to the cloglog offset model by 0.00836 per cell (area-covariate minus offset). The paired spatial-block bootstrap interval is [0.00276, 0.01411] and excludes zero. Its AUC is 0.0031 higher.

The no-area V14 control also has a slightly better mean log score than the cloglog offset model (by 0.00452), despite lower AUC. The block-bootstrap interval for that difference includes zero.

## Area-alone discrimination

The projected area-only AUC is 0.583 pooled and 0.410, 0.610, 0.675 and 0.703 by fold. Area is a weak pooled discriminator but a strong one in the north-west (Fold 4), where the V14 control is weakest. (Corrected 22 Nov 2026: earlier text reported 0.5591 because of a clipping bug in `roc_auc`; see `data_and_area_audit.md`.)

## Interpretation

The area offset is geometrically defensible: larger cells have more opportunity to contain a recorded occurrence. It is not an exploration-effort correction.

The choice of area treatment matters. The cloglog offset is not automatically superior to a logit model with log area as a covariate, and the two specifications produce different probability scales and different log scores. The offset model has higher pooled AUC than the no-area control but a worse mean log score. The area-covariate model has the best log score and Brier score of the three.

Accordingly, conclusion D applies: area correction materially affects the quantitative comparison. The primary cloglog ladder remains the pre-specified intensity interpretation, but the area-covariate result must be reported alongside it rather than suppressed.

## Outputs

- `paper_1/v15/predictions/oof_M2_logit_area_covariate.csv`
- `paper_1/v15/predictions/oof_V14_control.csv`
- `paper_1/v15/metrics/model_metrics.csv`
- `paper_1/v15/metrics/paired_comparisons.csv`
- `paper_1/v15/metrics/paired_bootstrap.csv`
