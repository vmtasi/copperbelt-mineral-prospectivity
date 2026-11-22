# V15 gravity heterogeneity report

## Question

Does the M3 gain justify domain-varying gravity slopes, or is it an artefact of fold composition, correlation with the distance predictors, or weak support in domains without positive observations?

## Fold-level gravity distributions

| Fold | Cells | Positives | Gravity mean | Gravity SD | AUC, high gravity positive | AUC, low gravity positive | Correlation with fault distance | Correlation with contact distance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 468 | 9 | -15.95 | 9.35 | 0.150 | 0.850 | 0.601 | -0.387 |
| 2 | 468 | 31 | -30.03 | 12.50 | 0.217 | 0.783 | 0.385 | 0.565 |
| 3 | 468 | 37 | -6.69 | 10.46 | 0.434 | 0.566 | 0.306 | 0.418 |
| 4 | 468 | 61 | -16.36 | 12.43 | 0.491 | 0.509 | -0.164 | -0.235 |

Within each held-out fold, occurrences favour low Bouguer gravity. The association is strong in the south-east (Folds 1-2: 0.85 and 0.78 with low gravity scored positive) and fades to chance in the north-west (Fold 4: 0.51). This reproduces the preliminary 0.85-to-0.51 pattern. It is a marginal, within-fold description, not a model result, and gravity correlates with fault and contact distance at 0.3-0.6 in Folds 1-3.

**Correction (22 Nov 2026):** an earlier version reported fold AUCs of 0.473, 0.498, 0.423 and 0.487 and stated that gravity was near chance and that the preliminary pattern was not reproduced. Those values came from a bug in `roc_auc`, which clipped the mGal values to [1e-12, 1]: every negative value became 1e-12 and every positive value 1, so nearly all cells tied. The function is fixed and the table is recomputed.

## Posterior coefficients

All domain posterior means for `beta_grav` are negative, but their support differs:

- NRB_3b is the only consistently informative domain. Its posterior mean ranges from -1.09 to -1.54 across folds, with 95% intervals excluding zero in every fold.
- NRB_3a ranges from -0.03 to -0.33. Most intervals include zero.
- CRZ, MMSB, NKB and SRB have wide intervals and no positive cells. Their posterior means are informed mainly by partial pooling, not by local outcomes.

The evidence is consistent with heterogeneous slope magnitude, but the magnitude is prior-dependent and only NRB_3b is consistently informative. There is no demonstrated reversal of sign.

## Predictive change

M3 improves the pooled result relative to M2: ROC-AUC 0.8020 versus 0.7340, and mean held-out log score -0.2267 versus -0.2571. The improvement is not evenly distributed:

| Fold | M2 AUC | M3 AUC | Mean log-score gain (M3 - M2) |
|---|---:|---:|---:|
| 1 | 0.6713 | 0.9695 | +0.0321 |
| 2 | 0.6958 | 0.8138 | +0.0801 |
| 3 | 0.7751 | 0.7608 | -0.0081 |
| 4 | 0.7073 | 0.7290 | +0.0174 |

Fold 1 contains only nine positives, and its AUC moves from weak to almost perfect. Fold 3 gets slightly worse in log score. The pooled spatial-block bootstrap interval for the M3-minus-M2 log-score gain excludes zero, but four spatial folds are too few to treat that interval as a highly precise sampling distribution.

## Interpretation

M3 has incremental predictive value under the canonical folds. The result is conditional:

1. Gravity has strong marginal discrimination (low gravity favoured) in Folds 1-2 and none in Fold 4. M3's gain is concentrated in the same folds, so the marginal pattern and the model result agree. That agreement is consistent with a real along-belt change in the gravity association, but it is not independent confirmation, because both describe the same data.
2. The strongest coefficient arises in NRB_3b, the domain that dominates the negative support.
3. Most of the predictive gain comes from folds 1 and 2.
4. Gravity is strongly correlated with fault distance in fold 1, so the gain cannot be interpreted as an isolated geophysical effect.
5. Four domains have no positive cells, so their varying gravity coefficients are prior-dominated.
6. **Disclosure:** M3 was added to the V15 ladder after fold-level gravity-only results (stored V14-era held-out predictions) had been inspected. It is a data-suggested hypothesis, not a purely pre-specified test, and must be reported as such.

M3 should therefore be reported as a domain-varying gravity sensitivity with conditional predictive benefit, not as evidence for a stable domain-specific gravity law.

## Outputs

- `paper_1/v15/diagnostics/gravity_coefficients.csv`
- `paper_1/v15/diagnostics/gravity_fold_summary.csv`
- `paper_1/v15/traces/M3_domain_distance_gravity/`
