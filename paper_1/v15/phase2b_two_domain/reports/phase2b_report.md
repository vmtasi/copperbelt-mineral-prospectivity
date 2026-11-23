# Phase 2b - two-domain fixed-effect frame

Generated: 2026-11-23T05:13:49.810382Z

## Frame and provenance

The **full 1,872-cell frame is primary**. It includes fixed outside-NRB and fixed NRB_3b terms under the same complementary log-log likelihood and fixed log-area offset as V15. The **1,677-cell NRB-only frame is secondary and descriptive only**.

| Frame | Cells | Positives | Role |
|---|---:|---:|---|
| Full frame | 1,872 | 138 | Primary; source of verdicts |
| NRB_3a + NRB_3b | 1,677 | 138 | Secondary; no verdicts |

Distance to fault was rebuilt from 126 surviving structure lines (`SHA-256 c410fcb33bb0c6a4f9284f54ea5d4a092eb79e5c982910f1dc65530d564d9e89`). It reproduces the stored distance within 1 m for 1,645 cells and differs for 227 cells, including 26 positives. V15 fold IDs were copied unchanged.

## Fold composition

| Fold | Cells | Positives | NRB_3a | NRB_3b | Outside NRB |
|---|---:|---:|---:|---:|---:|
| 1 | 468 | 9 | 0 | 310 | 158 |
| 2 | 468 | 31 | 207 | 240 | 21 |
| 3 | 468 | 37 | 468 | 0 | 0 |
| 4 | 468 | 61 | 452 | 0 | 16 |

| Secondary fold | Cells | Positives |
|---:|---:|---:|
| 1 | 310 | 9 |
| 2 | 447 | 31 |
| 3 | 468 | 37 |
| 4 | 452 | 61 |

The 3a/3b contrast is nearly confounded with fold: **Fold 1 is NRB_3b only (with outside-NRB support), Folds 3-4 are NRB_3a only (Fold 4 includes 16 MMSB cells), and only Fold 2 is mixed.** Fold-specific estimates are therefore descriptive and cannot identify a fold-invariant 3b effect.

## Models

- N0: global terms; fixed outside-NRB term in the primary frame.
- N1: N0 plus the fixed 3b indicator.
- N2: N1 plus 3b x fault, fault squared, contact and contact squared.
- N3: N2 plus 3b x gravity.
- All coefficients have independent Normal(0, 1) priors; no hierarchical scales are estimated.

## Held-out performance

| Frame | Model | Cells | Positives | AUC | PR-AUC | Mean log score | Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| primary | N0_global | 1872 | 138 | 0.7304 | 0.1456 | -0.24738 | 0.06868 |
| primary | N1_fixed_3b | 1872 | 138 | 0.7390 | 0.1560 | -0.24529 | 0.06815 |
| primary | N2_3b_distance_interactions | 1872 | 138 | 0.7448 | 0.1598 | -0.24341 | 0.06776 |
| primary | N3_3b_gravity_interaction | 1872 | 138 | 0.7707 | 0.1993 | -0.24132 | 0.06966 |
| secondary | N0_global | 1677 | 138 | 0.7165 | 0.1564 | -0.27119 | 0.07624 |
| secondary | N1_fixed_3b | 1677 | 138 | 0.7231 | 0.1641 | -0.26903 | 0.07545 |
| secondary | N2_3b_distance_interactions | 1677 | 138 | 0.7478 | 0.1846 | -0.26088 | 0.07359 |
| secondary | N3_3b_gravity_interaction | 1677 | 138 | 0.7732 | 0.2198 | -0.25982 | 0.07648 |

## Primary comparisons

Orientation is upper minus lower. Positive log-score and AUC differences favour the upper model. Intervals use 2,000 resamples of 25-km spatial blocks (117 blocks in the primary frame).

| Comparison | Delta AUC | Mean Delta log score | Spatial-block 95% CI | Positive folds |
|---|---:|---:|---|---:|
| N0_global -> N1_fixed_3b | +0.0087 | +0.00209 | [-0.00520, +0.00947] | 2/4 |
| N1_fixed_3b -> N2_3b_distance_interactions | +0.0057 | +0.00188 | [-0.00923, +0.01205] | 3/4 |
| N2_3b_distance_interactions -> N3_3b_gravity_interaction | +0.0260 | +0.00209 | [-0.01075, +0.01571] | 2/4 |

## Secondary-frame comparisons

These values are descriptive. They are not used for any verdict.

| Comparison | Delta AUC | Mean Delta log score | Spatial-block 95% CI | Positive folds |
|---|---:|---:|---|---:|
| N0_global -> N1_fixed_3b | +0.0065 | +0.00216 | [-0.01005, +0.01291] | 2/4 |
| N1_fixed_3b -> N2_3b_distance_interactions | +0.0247 | +0.00816 | [-0.00163, +0.01694] | 3/4 |
| N2_3b_distance_interactions -> N3_3b_gravity_interaction | +0.0254 | +0.00105 | [-0.02332, +0.02739] | 2/4 |

## Decision

The operational Phase 4 adaptation required: pooled log-score and AUC improvement; a 25-km spatial-block 95% interval for log-score gain excluding zero; improvement in at least 3 of 4 folds; and the stated coefficient-stability condition. For N1 -> N2, at least two of the four added 3b-distance coefficients must exclude zero in at least 3 of 4 folds.

- **N1_fixed_3b->N2_3b_distance_interactions: NOT SUPPORTED.** Positive folds: 3/4; coefficient evidence: {'criterion': 'majority of the four added 3b x distance terms have a 95% interval excluding zero in >= 3 of 4 folds', 'stable_terms': ['beta_3b_f_lin', 'beta_3b_l_lin'], 'stable_term_count': 2}.
- **N2_3b_distance_interactions->N3_3b_gravity_interaction: NOT SUPPORTED.** Positive folds: 2/4; coefficient evidence: {'criterion': '3b x gravity interaction interval excludes zero in >= 3 of 4 folds', 'folds_excluding_zero': 3}.

The focal 3b x gravity coefficient is negative in every fold and its 95% interval excludes zero in 3 of 4 folds. That coefficient criterion is met, but the predictive criteria are not: the spatial-block interval includes zero and the log score improves in only 2 of 4 folds.

| Fold | 3b x gravity posterior mean | 95% HDI | Excludes zero |
|---:|---:|---|---|
| 1 | -0.7712 | [-1.5241, +0.0398] | no |
| 2 | -1.0710 | [-1.8372, -0.3937] | yes |
| 3 | -1.3230 | [-1.8268, -0.7322] | yes |
| 4 | -1.2669 | [-1.8711, -0.7227] | yes |

## Fold 2 descriptive check

| Comparison | Delta AUC | Mean Delta log score |
|---|---:|---:|
| N1_fixed_3b -> N2_3b_distance_interactions | +0.0671 | +0.01368 |
| N2_3b_distance_interactions -> N3_3b_gravity_interaction | +0.0703 | -0.00258 |

## Coefficient stability

| Model | Parameter | Folds with interval excluding zero |
|---|---|---:|
| N0_global | beta_outside_nrb | 4/4 |
| N1_fixed_3b | beta_3b | 4/4 |
| N1_fixed_3b | beta_outside_nrb | 4/4 |
| N2_3b_distance_interactions | beta_3b | 3/4 |
| N2_3b_distance_interactions | beta_3b_f_lin | 4/4 |
| N2_3b_distance_interactions | beta_3b_f_sq | 1/4 |
| N2_3b_distance_interactions | beta_3b_l_lin | 4/4 |
| N2_3b_distance_interactions | beta_3b_l_sq | 0/4 |
| N2_3b_distance_interactions | beta_outside_nrb | 2/4 |
| N3_3b_gravity_interaction | beta_3b | 4/4 |
| N3_3b_gravity_interaction | beta_3b_f_lin | 4/4 |
| N3_3b_gravity_interaction | beta_3b_f_sq | 0/4 |
| N3_3b_gravity_interaction | beta_3b_grav | 3/4 |
| N3_3b_gravity_interaction | beta_3b_l_lin | 3/4 |
| N3_3b_gravity_interaction | beta_3b_l_sq | 0/4 |
| N3_3b_gravity_interaction | beta_outside_nrb | 3/4 |

## Convergence and interpretation

Fits recorded: 32; convergence failures: 0. The convergence screen requires zero divergences, R-hat < 1.05, and bulk and tail ESS >= 200. Each fit used two chains, 500 retained draws, 750 tuning draws and target acceptance 0.95, with chains run two at a time in one model job.

Conclusions are drawn only from the full primary frame. The NRB-only frame is reported separately because four outside-NRB domains have zero positives and therefore cannot inform domain performance. N3 improves pooled ranking and log score relative to N2, but it worsens Brier score and does not meet the predictive stability part of the decision rule. It is therefore not promoted for the primary scientific conclusion.
