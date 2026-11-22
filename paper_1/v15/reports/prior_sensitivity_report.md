# V15 prior sensitivity report

## Purpose

The hierarchical model estimates between-domain scales for the intercept and for the fault-distance and contact-distance linear and quadratic coefficients. With only two domains containing positive cells, those scales may be weakly identified and therefore sensitive to prior width. This experiment refits M2 with the hierarchical prior scale multiplied by 0.5 and by 2.0 and compares predictions, held-out scores and posterior scales.

The prior scale multiplies the standard deviations of the population means, the half-normal scales of the between-domain standard deviations, the shared slope priors, the gravity prior, the categorical-lithology prior and the area-covariate prior. It does not change the likelihood, the folds, the data or the predictor transformations.

## Sampler and convergence

Both variants use two chains, 1,000 draws, 2,000 tuning steps, target acceptance 0.995.

| Variant | Divergences | Max R-hat | Min bulk ESS | Min tail ESS | Pass |
|---|---:|---:|---:|---:|---|
| prior 0.5 | 0 | 1.0093 | 714 | 403 | yes |
| prior 1.0 (primary) | 0 | 1.0033 | 2,113 | 1,630 | yes |
| prior 2.0 | 0 | 1.0074 | 416 | 494 | yes |

All prior variants converged. The primary fit has four chains and 1,500 draws; the variants have two chains and 1,000 draws, so their tail estimates are less precise.

## Held-out performance

| Variant | ROC-AUC | PR-AUC | Mean log score | Brier | Calibration slope |
|---|---:|---:|---:|---:|---:|
| prior 0.5 | 0.7378 | 0.1642 | -0.2480 | 0.06807 | 0.572 |
| prior 1.0 (primary) | 0.7340 | 0.1633 | -0.2571 | 0.06922 | 0.489 |
| prior 2.0 | 0.7274 | 0.1590 | -0.2680 | 0.06999 | 0.421 |

| Variant | Fold 1 AUC | Fold 2 AUC | Fold 3 AUC | Fold 4 AUC |
|---|---:|---:|---:|---:|
| prior 0.5 | 0.6040 | 0.7045 | 0.7801 | 0.7152 |
| prior 1.0 | 0.6713 | 0.6958 | 0.7751 | 0.7073 |
| prior 2.0 | 0.6776 | 0.6719 | 0.7739 | 0.7010 |

| Variant | Fold 1 log score | Fold 2 log score | Fold 3 log score | Fold 4 log score |
|---|---:|---:|---:|---:|
| prior 0.5 | -0.1203 | -0.2597 | -0.2447 | -0.3673 |
| prior 1.0 | -0.1120 | -0.2902 | -0.2493 | -0.3769 |
| prior 2.0 | -0.1082 | -0.3294 | -0.2524 | -0.3818 |

Prediction differences from the primary fit:

| Variant | Mean absolute probability difference | Maximum absolute difference | Mean probability |
|---|---:|---:|---:|
| prior 0.5 | 0.0104 | 0.1032 | 0.0742 |
| prior 2.0 | 0.0062 | 0.1383 | 0.0681 |

## Posterior scales

Mean posterior standard deviation across the four folds:

| Parameter | prior 0.5 | prior 1.0 | prior 2.0 |
|---|---:|---:|---:|
| alpha_sigma | 0.645 | 1.033 | 1.726 |
| sigma_beta_f_lin | 0.762 | 1.176 | 1.790 |
| sigma_beta_f_sq | 0.549 | 0.859 | 1.301 |
| sigma_beta_l_lin | 0.623 | 1.045 | 1.734 |
| sigma_beta_l_sq | 0.290 | 0.520 | 0.898 |

The scale posteriors move nearly in proportion to the prior width. A change from 0.5 to 2.0, a factor of four, changes the posterior means by roughly a factor of 2.2 to 3.1. This is the signature of weak identification: the data do not pin down the between-domain scales, so the prior contributes materially to their location.

## Interpretation

Predictions are moderately robust in ranking. Across a four-fold change in prior width, pooled AUC varies from 0.727 to 0.738, a range of 0.011. The direction of the domain-varying model relative to M0 and M1 is not tested here, and the prior variants should not be compared with M0 or M1 as if they were separate pre-registered models.

The probabilistic and interpretative conclusions are less robust.

- Held-out log score improves monotonically as the prior narrows: -0.268 at width 2.0, -0.257 at width 1.0 and -0.248 at width 0.5. Calibration also improves, from slope 0.42 to 0.49 to 0.57. A wider prior therefore makes the model less calibrated and less useful probabilistically.
- The improvement with a narrower prior is consistent with the primary M2 fit over-dispersing domain effects. It does not establish that width 0.5 is scientifically correct. Selecting it after seeing the held-out score would be post-hoc tuning and is not done here.
- The between-domain scales are prior-dominated. Quantities such as the amount of heterogeneity, the implied coefficient ranges and the population stationary point should not be reported as data-determined.
- Domains without positive cells contribute almost no information to the scales. Their coefficients remain prior- and partial-pooling-driven under every variant.

This result strengthens the conclusion that the available data do not support a strong hierarchical claim. The model can rank cells, but it cannot estimate how much the domain-specific relationships really differ. Any manuscript statement about the magnitude of between-domain variation must be removed or explicitly labelled prior-dependent.

## Outputs

- `paper_1/v15/metrics/prior_sensitivity_metrics.csv`
- `paper_1/v15/metrics/prior_sensitivity_predictions.csv`
- `paper_1/v15/diagnostics/prior_sensitivity_scales.csv`
- `paper_1/v15/diagnostics/prior_sensitivity_summary.json`
- `paper_1/v15/predictions/prior0p5__M2_domain_distance_fold_*.csv`
- `paper_1/v15/predictions/prior2p0__M2_domain_distance_fold_*.csv`
