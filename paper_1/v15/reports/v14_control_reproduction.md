# V15 V14 control reproduction

## Purpose

The V14 benchmark remains the frozen comparison. This report checks that the V15 model implementation can reproduce its out-of-fold behaviour before any V15 likelihood or exposure change is interpreted.

## Specification matched

The `V14_control` model uses:

- Four canonical along-belt folds and the same 1,872 complete cases.
- Logistic link.
- No area offset and no area covariate.
- Partially pooled domain intercepts.
- Partially pooled domain-specific fault-distance linear and quadratic coefficients.
- Partially pooled domain-specific contact-distance linear and quadratic coefficients.
- A shared Bouguer-gravity coefficient.
- The V14 categorical lithology field and its contact-distance predictor.

This is a new fit through the V15 implementation, not a byte reproduction of the frozen V14 chains. The V14 trace itself stores a two-dimensional `beta_grav` posterior with shape `(2, 1500)`, confirming that gravity was shared globally. The frozen V14 OOF predictions remain the exact benchmark.

## Sampler

- V15 control: two chains, 1,000 draws, 2,000 tuning steps, target acceptance 0.995.
- V14 benchmark: two chains, 1,500 draws, 2,500 tuning steps, target acceptance 0.99.
- All four control folds have zero divergences, maximum R-hat below 1.0075 and minimum bulk ESS above 523.

## Results

| Model | ROC-AUC | PR-AUC | Mean log score | Brier | Calibration slope |
|---|---:|---:|---:|---:|---:|
| Frozen V14 OOF | 0.7155 | 0.1373 | -0.2510 | 0.06879 | 0.605 |
| V15 V14-control refit | 0.7094 | 0.1361 | -0.2526 | 0.06891 | 0.588 |

| Fold | Frozen AUC | Control AUC | Frozen log score | Control log score |
|---|---:|---:|---:|---:|
| 1 | 0.8753 | 0.8761 | -0.0858 | -0.0866 |
| 2 | 0.7357 | 0.7174 | -0.2518 | -0.2565 |
| 3 | 0.6851 | 0.6855 | -0.2739 | -0.2744 |
| 4 | 0.6031 | 0.6019 | -0.3926 | -0.3929 |
| Pooled | 0.7155 | 0.7094 | -0.2510 | -0.2526 |

## Interpretation

The control reproduces the frozen benchmark closely. Pooled discrimination differs by 0.0062 AUC, held-out log score by 0.0016 per cell, and Brier score by 0.00012. Three of four fold AUCs differ by less than 0.002; Fold 2 differs by 0.018.

The pooled difference is consistent with a re-specified sampler and posterior Monte Carlo variation, not with a changed model definition. The control is therefore suitable as the within-V15 comparator for the cloglog exposure model.

The frozen V14 result must still be cited as the published benchmark. The V15 control should not replace it.

## Outputs

- `paper_1/v15/predictions/oof_V14_control.csv`
- `paper_1/v15/predictions/V14_control_fold_*.csv`
- `paper_1/v15/metrics/model_metrics.csv`
- `paper_1/v15/diagnostics/V14_control_fold_*_summary.csv`
