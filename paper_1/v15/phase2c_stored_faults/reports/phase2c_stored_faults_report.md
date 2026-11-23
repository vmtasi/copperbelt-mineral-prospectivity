# Phase 2c - stored-fault robustness

Generated: 2026-11-23T06:17:06.809831Z

## Frame and provenance

The **full 1,872-cell frame is primary**. It includes fixed outside-NRB and fixed NRB_3b terms under the same complementary log-log likelihood and fixed log-area offset as V15. The **1,677-cell NRB-only frame is secondary and descriptive only**.

| Frame | Cells | Positives | Role |
|---|---:|---:|---|
| Full frame | 1,872 | 138 | Primary; source of verdicts |
| NRB_3a + NRB_3b | 1,677 | 138 | Secondary; no verdicts |

Distance to fault is the **stored V15 value**. The rebuilt variant differs for 227 cells, including 26 positives. The six added segments in the rebuilt layer are classified AMBIGUOUS and neither fault version is privileged. V15 fold IDs were copied unchanged.

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
| primary | N0_global | 1872 | 138 | 0.7032 | 0.1279 | -0.25958 | 0.07206 |
| primary | N1_fixed_3b | 1872 | 138 | 0.7220 | 0.1432 | -0.25239 | 0.07024 |
| primary | N2_3b_distance_interactions | 1872 | 138 | 0.7298 | 0.1389 | -0.25279 | 0.07077 |
| primary | N3_3b_gravity_interaction | 1872 | 138 | 0.7799 | 0.1876 | -0.23920 | 0.07007 |

## Primary comparisons

Orientation is upper minus lower. Positive log-score and AUC differences favour the upper model. Intervals use 2,000 resamples of 25-km spatial blocks (117 blocks in the primary frame).

| Comparison | Delta AUC | Mean Delta log score | Spatial-block 95% CI | Positive folds |
|---|---:|---:|---|---:|
| N0_global -> N1_fixed_3b | +0.0188 | +0.00719 | [-0.00178, +0.01585] | 2/4 |
| N1_fixed_3b -> N2_3b_distance_interactions | +0.0078 | -0.00040 | [-0.01364, +0.01245] | 3/4 |
| N2_3b_distance_interactions -> N3_3b_gravity_interaction | +0.0501 | +0.01359 | [-0.00270, +0.03100] | 3/4 |

## Fault-layer robustness

The fixed-effect rows use the same primary frame, V15 folds, prior, sampler, seed, spatial-block bootstrap and decision rule. V15 hierarchical rows are reference contrasts using stored fault distances.

| Ladder / fault version | Comparison | Mean log-score gain | 95% spatial-block CI | Positive folds | Verdict |
|---|---|---:|---|---:|---|
| fixed-effect ladder: stored | N1-N2 | -0.00040 | [-0.01364, +0.01245] | 3/4 | NOT SUPPORTED |
| fixed-effect ladder: stored | N2-N3 | +0.01359 | [-0.00270, +0.03100] | 3/4 | NOT SUPPORTED |
| fixed-effect ladder: rebuilt | N1-N2 | +0.00188 | [-0.00923, +0.01205] | 3/4 | NOT SUPPORTED |
| fixed-effect ladder: rebuilt | N2-N3 | +0.00209 | [-0.01075, +0.01571] | 2/4 | NOT SUPPORTED |
| V15 hierarchical reference: stored | M1-M2 | +0.00622 | [-0.01598, +0.02905] | 2/4 | NOT SUPPORTED |
| V15 hierarchical reference: stored | M2-M3 | +0.03038 | [+0.01100, +0.05300] | 3/4 | SUPPORTED |

**Reading:** The fixed-effect verdicts match under both fault versions, so the Phase 2b fixed-effect conclusions are robust to the fault-layer choice. Neither fixed-effect N2-to-N3 comparison meets the rule, whereas the V15 hierarchical M2-to-M3 comparison does; the V15 gravity gain therefore depended on the six-domain hierarchical structure.

## Decision

The operational Phase 4 adaptation required: pooled log-score and AUC improvement; a 25-km spatial-block 95% interval for log-score gain excluding zero; improvement in at least 3 of 4 folds; and the stated coefficient-stability condition. For N1 -> N2, at least two of the four added 3b-distance coefficients must exclude zero in at least 3 of 4 folds.

- **N1_fixed_3b->N2_3b_distance_interactions: NOT SUPPORTED.** Positive folds: 3/4; coefficient evidence: {'criterion': 'majority of the four added 3b x distance terms have a 95% interval excluding zero in >= 3 of 4 folds', 'stable_terms': ['beta_3b_f_lin', 'beta_3b_l_lin'], 'stable_term_count': 2}.
- **N2_3b_distance_interactions->N3_3b_gravity_interaction: NOT SUPPORTED.** Positive folds: 3/4; coefficient evidence: {'criterion': '3b x gravity interaction interval excludes zero in >= 3 of 4 folds', 'folds_excluding_zero': 3}.

The focal 3b x gravity coefficient is negative in 4/4 folds and its 95% interval excludes zero in 3/4 folds. The N2-to-N3 verdict is not supported; the log score improves in 3/4 folds.

| Fold | 3b x gravity posterior mean | 95% HDI | Excludes zero |
|---:|---:|---|---|
| 1 | -0.7568 | [-1.6256, +0.0151] | no |
| 2 | -1.1286 | [-1.8359, -0.3548] | yes |
| 3 | -1.2644 | [-1.8289, -0.7420] | yes |
| 4 | -1.1783 | [-1.7616, -0.5578] | yes |

## Fold 2 descriptive check

| Comparison | Delta AUC | Mean Delta log score |
|---|---:|---:|
| N1_fixed_3b -> N2_3b_distance_interactions | +0.0968 | +0.01312 |
| N2_3b_distance_interactions -> N3_3b_gravity_interaction | +0.0862 | +0.02444 |

## Coefficient stability

| Model | Parameter | Folds with interval excluding zero |
|---|---|---:|
| N0_global | beta_outside_nrb | 4/4 |
| N1_fixed_3b | beta_3b | 4/4 |
| N1_fixed_3b | beta_outside_nrb | 3/4 |
| N2_3b_distance_interactions | beta_3b | 4/4 |
| N2_3b_distance_interactions | beta_3b_f_lin | 4/4 |
| N2_3b_distance_interactions | beta_3b_f_sq | 1/4 |
| N2_3b_distance_interactions | beta_3b_l_lin | 4/4 |
| N2_3b_distance_interactions | beta_3b_l_sq | 0/4 |
| N2_3b_distance_interactions | beta_outside_nrb | 3/4 |
| N3_3b_gravity_interaction | beta_3b | 4/4 |
| N3_3b_gravity_interaction | beta_3b_f_lin | 3/4 |
| N3_3b_gravity_interaction | beta_3b_f_sq | 0/4 |
| N3_3b_gravity_interaction | beta_3b_grav | 3/4 |
| N3_3b_gravity_interaction | beta_3b_l_lin | 3/4 |
| N3_3b_gravity_interaction | beta_3b_l_sq | 0/4 |
| N3_3b_gravity_interaction | beta_outside_nrb | 3/4 |

## Convergence and interpretation

Fits recorded: 16; convergence failures: 0. The convergence screen requires zero divergences, R-hat < 1.05, and bulk and tail ESS >= 200. Each fit used two chains, 500 retained draws, 750 tuning draws and target acceptance 0.95, with chains run two at a time in one model job.

Conclusions are drawn only from the full primary frame. The NRB-only frame was not refitted in Phase 2c; its composition is shown for transparency.

**Fault-layer reading:** The fixed-effect verdicts match under both fault versions, so the Phase 2b fixed-effect conclusions are robust to the fault-layer choice. Neither fixed-effect N2-to-N3 comparison meets the rule, whereas the V15 hierarchical M2-to-M3 comparison does; the V15 gravity gain therefore depended on the six-domain hierarchical structure.
