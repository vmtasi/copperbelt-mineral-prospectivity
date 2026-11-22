# V15 buffered spatial validation

## Purpose

The canonical four folds hold out contiguous along-belt sectors, but training cells adjacent to the held-out sector can share short-range geological and reporting structure with it. This experiment removes training cells within 5 km and 10 km of the held-out fold and refits M0 and M2 under the same likelihood, predictors and folds. It is a sensitivity analysis, not a replacement for the canonical validation.

The canonical folds, labels, model specification and V14 benchmark are unchanged.

## Design

- Buffers: 5 km and 10 km around each held-out fold.
- Models: `M0_global` and `M2_domain_distance`.
- Folds: the four canonical folds, 468 test cells each.
- Sampler: 2 chains, 500 draws, 1,000 tuning steps, target acceptance 0.995.
- Training cells inside the buffer are removed before fitting; test cells are never removed.

## Training support removed

| Buffer | Fold | Removed cells | Removed positives | Remaining positives |
|---|---:|---:|---:|---:|
| 5 km | 1 | 22 | 0 | 129 |
| 5 km | 2 | 36 | 2 | 105 |
| 5 km | 3 | 30 | 7 | 94 |
| 5 km | 4 | 21 | 3 | 74 |
| 10 km | 1 | 43 | 1 | 128 |
| 10 km | 2 | 69 | 3 | 104 |
| 10 km | 3 | 65 | 13 | 88 |
| 10 km | 4 | 43 | 4 | 73 |

At 10 km, Fold 3 loses 13 of 101 training positives, including 6 in NRB_3a and 7 in NRB_3b. Fold 1 retains 21 of 22 NRB_3b training positives, so the main loss there is geometric support, not label loss. No fold becomes devoid of positives, but the 10 km Fold 1 fit is supported by only nine held-out positives and a substantially altered training neighbourhood.

## Convergence

| Buffer | Model | Fold | Divergences | Max R-hat | Min bulk ESS | Min tail ESS | Status |
|---|---|---:|---:|---:|---:|---:|---|
| 5 km | M0 | 1 | 0 | 1.0070 | 303 | 341 | pass |
| 5 km | M0 | 2 | 0 | 1.0095 | 304 | 313 | pass |
| 5 km | M0 | 3 | 0 | 1.0117 | 302 | 376 | pass |
| 5 km | M0 | 4 | 0 | 1.0185 | 257 | 391 | pass |
| 5 km | M2 | 1 | 0 | 1.0130 | 222 | 229 | pass |
| 5 km | M2 | 2 | 0 | 1.0173 | 407 | 275 | pass |
| 5 km | M2 | 3 | 0 | 1.0095 | 311 | 288 | pass |
| 5 km | M2 | 4 | 0 | 1.0127 | 361 | **136** | marginal |
| 10 km | M0 | 1 | 0 | 1.0168 | 201 | 281 | pass |
| 10 km | M0 | 2 | 0 | 1.0069 | 256 | 325 | pass |
| 10 km | M0 | 3 | 0 | 1.0058 | 327 | 403 | pass |
| 10 km | M0 | 4 | 0 | 1.0080 | 401 | 507 | pass |
| 10 km | M2 | 1 | 0 | 1.0179 | 196 | **140** | marginal |
| 10 km | M2 | 2 | 0 | 1.0119 | 380 | 292 | pass |
| 10 km | M2 | 3 | 0 | 1.0093 | 507 | 436 | pass |
| 10 km | M2 | 4 | **3** | 1.0120 | 372 | **171** | fail |

The script's status flag uses the fit-level convergence test, which is less strict than the V15 reporting criterion of tail ESS at least 200. Under that stricter criterion, the three entries in bold are not fully converged. Buffered results are therefore descriptive sensitivity evidence. They must not be used as a primary comparison and must not be pooled with the canonical fits.

## Pooled results

| Model | Treatment | ROC-AUC | Mean log score | Brier |
|---|---|---:|---:|---:|
| M0 | canonical | 0.6882 | -0.2725 | 0.07625 |
| M0 | 5 km buffer | 0.6671 | -0.2957 | 0.08450 |
| M0 | 10 km buffer | 0.6566 | -0.2975 | 0.08348 |
| M2 | canonical | 0.7340 | -0.2571 | 0.06922 |
| M2 | 5 km buffer | 0.7267 | -0.2582 | 0.06922 |
| M2 | 10 km buffer | 0.6974 | -0.2696 | 0.07098 |

## Fold results

| Model | Treatment | Fold 1 AUC | Fold 2 AUC | Fold 3 AUC | Fold 4 AUC |
|---|---|---:|---:|---:|---:|
| M0 | canonical | 0.8289 | 0.6886 | 0.7552 | 0.7137 |
| M0 | 5 km | 0.8238 | 0.6858 | 0.7673 | 0.7007 |
| M0 | 10 km | 0.8194 | 0.6787 | 0.7637 | 0.6988 |
| M2 | canonical | 0.6713 | 0.6958 | 0.7751 | 0.7073 |
| M2 | 5 km | 0.9116 | 0.6860 | 0.7877 | 0.7022 |
| M2 | 10 km | **0.4713** | 0.6601 | 0.7782 | 0.7022 |

| Model | Treatment | Fold 1 log score | Fold 2 log score | Fold 3 log score | Fold 4 log score |
|---|---|---:|---:|---:|---:|
| M0 | canonical | -0.1731 | -0.2536 | -0.2711 | -0.3923 |
| M0 | 5 km | -0.2624 | -0.2576 | -0.2601 | -0.4029 |
| M0 | 10 km | -0.2534 | -0.2615 | -0.2702 | -0.4049 |
| M2 | canonical | -0.1120 | -0.2902 | -0.2493 | -0.3769 |
| M2 | 5 km | -0.1069 | -0.2950 | -0.2458 | -0.3851 |
| M2 | 10 km | **-0.1333** | -0.2994 | -0.2573 | -0.3883 |

## Interpretation

The direction of the M0-to-M2 comparison survives buffering. M2 retains a pooled AUC advantage over M0 of +0.060 at 5 km and +0.041 at 10 km, against +0.046 canonically. The corresponding mean log-score advantages are +0.0375 and +0.0279 per cell. This is consistent with the canonical result: the domain-varying distance specification does not depend on training cells immediately adjacent to the held-out sector.

The magnitude is not stable. M2 degrades by 0.007 AUC under a 5 km buffer and by 0.037 under 10 km; M0 degrades by 0.021 and 0.032. The 10 km M2 Fold 1 estimate is uninformative: nine held-out positives, a training partition that has lost almost all of its NRB_3b neighbours, a non-converged fit, and an AUC of 0.471. The 5 km M2 Fold 1 value of 0.912 is equally fragile for the opposite reason, with the same nine positives.

The experiment does not test M1 against M2. It cannot, on its own, show that domain-varying slopes are responsible for the M2 advantage; that comparison retains the canonical paired result and its spatial-block interval.

## Limitations

- Two chains and 500 draws are insufficient for precise tail quantiles and for stable tail ESS in sparse domains.
- Three fits fail the strict V15 tail-ESS criterion; one has three post-tuning divergences.
- The buffer removes different numbers of cells and positives per fold, so fold-level changes mix support loss with model change.
- Buffered folds are not independent replicates. Differences across buffers remain descriptive.

## Outputs

- `paper_1/v15/metrics/buffered_validation_metrics.csv`
- `paper_1/v15/diagnostics/buffered_training_support.csv`
- `paper_1/v15/models/buffered_fit_status.json`
- `paper_1/v15/predictions/buffered_*_fold_*.csv`
- `paper_1/v15/traces/buffered_*/fold_*.nc`
