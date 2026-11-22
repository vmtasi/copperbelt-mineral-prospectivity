# V15 residual spatial autocorrelation report

## Method

Residuals are `y - p` from the held-out out-of-fold predictions. The spatial weights matrix uses eight nearest neighbours, row-standardised, with 999 permutations. The tested alternative is positive spatial autocorrelation.

## Results

| Model | Cells | Moran's I | Permutation p |
|---|---:|---:|---:|
| M0_global | 1,872 | 0.2464 | 0.001 |
| M1_domain_intercept | 1,872 | 0.2237 | 0.001 |
| M2_domain_distance | 1,872 | 0.1804 | 0.001 |
| M3_domain_distance_gravity | 1,872 | 0.1521 | 0.001 |

## Interpretation

All models leave statistically detectable positive residual clustering. Adding domain intercepts, domain-varying distance slopes and domain-varying gravity reduces but does not remove that structure.

The held-out observations are therefore not plausibly independent conditional on the predictors. Fold-level and pooled metrics remain useful descriptive comparisons, but nominal standard errors and unadjusted confidence intervals would understate uncertainty. The paired spatial-block bootstrap is the more appropriate comparison summary, and it too has limited precision with only four folds.

`M3` has the lowest residual autocorrelation, but the difference is a reduction in unexplained clustering, not evidence that the residuals are independent.

## Outputs

- `paper_1/v15/diagnostics/moran_i.csv`
