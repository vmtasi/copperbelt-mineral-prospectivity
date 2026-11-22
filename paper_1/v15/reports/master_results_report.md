# V15 master results report — Paper 1

**Status:** analysis complete; no V14, Phase 6/7 or manuscript file modified. No commit or push.
**Branch:** `paper1-completion`
**HEAD:** `74b436c`
**Date:** 2026-11-22

> **Erratum (22 Nov 2026, post-review).** Three statements in the first version of this report were wrong and are corrected below.
> 1. *Area-alone AUC.* `src/v15/metrics.py::roc_auc` clipped its input to [1e-12, 1], which ties every cell above 1 km2. The correct projected values are 0.410 / 0.610 / 0.675 / 0.703, pooled 0.583. The earlier explanation (a degree-area artefact) was wrong: degree areas are all below 1 and escaped the clipping. The function is fixed, and `test_roc_auc_ranks_raw_covariates_without_clipping` guards it.
> 2. *Gravity-alone AUC.* The same bug turned mGal values into two tied groups. Correctly computed, occurrences favour low gravity, with AUC 0.850 / 0.783 / 0.566 / 0.509 by fold when low gravity is scored positive. The preliminary 0.85-to-0.51 pattern **is** reproduced.
> 3. *The 0.674 global reference.* Its out-of-fold predictions **are** stored and checksummed (`figures/audit/phase4_oof_predictions_v14_centroid.csv`, column `M7_All_Global_Prediction`). They are produced by `src/phase4_tectonic_heterogeneity.py`, an L2-penalised (C = 1) scikit-learn logistic regression with the same quadratic distance terms, folds and lithology handling, and `tests/test_manuscript_numerical_consistency.py` recomputes them. The valid caveat is narrower: it is a penalised maximum-likelihood fit, not a Bayesian one, and has no area term.
>
> Model metrics, paired comparisons, bootstraps and perturbation results use model probabilities, which never fall outside the clipping range, so they are unaffected. Only `area_auc_by_fold.csv` and `gravity_fold_summary.csv` changed when the diagnostics were rerun.

## 1. What was done

V15 asks whether domain-varying geological relationships add defensible predictive value beyond domain-specific baseline rates, after accounting for unequal spatial support. The programme contains:

- a data and cell-area audit;
- a V14 control reproduction;
- a four-model ablation ladder (M0–M3) under identical four-fold spatial validation;
- a complementary log-log exposure model with log clipped-cell area as a fixed offset;
- area, lithology, gravity, prior, buffer, label-perturbation and fault-provenance sensitivity analyses;
- a random-forest comparator;
- residual spatial-autocorrelation diagnostics;
- checksums, tests and this handover.

No model was selected by maximising AUC. The conclusion is negative where the evidence is negative.

## 2. Repository, benchmark and provenance

### 2.1 Frozen V14 benchmark

- `pytest -q tests/test_v14_benchmark_integrity.py`: **82 passed**.
- 61 tracked artefacts in `paper_1/benchmark/SHA256SUMS` verified byte-for-byte.
- 20 external-archive artefacts in `SHA256SUMS_external` verified; none skipped.
- `git diff --stat` is empty. No tracked file was modified.
- The working V14 modelling table hash is `161ad50684950f7f106c079079eacbae9de09c1851dc15867b014c6b88bbd878`.

### 2.2 Frozen-file drift

The previously reported drift is real as a source-provenance gap, but it is **not an uncommitted working-tree change**:

- `src/validation_strategies.py` in the working tree equals `HEAD` (`44a43dad1f22…`). The V14 OOF manifest records `dee9c6add0d1…`. Neither the V14 hash nor the pre-V14 baseline hash is present as a blob in current Git history. The V14 run therefore used an uncommitted intermediate version that was not preserved by Git.
- `AI_PROJECT_CONTEXT.md` is untracked and ignored at `.gitignore:68`; it has no HEAD version. Its warning is relative to a former MCP baseline, not to Git.
- The prior audit found the differences to be a BOM, a PCA import path, lazy imports and docstrings, with zero fold mismatches across all 1,872 cells. The frozen numerical artefacts are intact.
- **Implication:** V14 is usable as a numerical control because its artefacts are verified, but its exact source-level pipeline cannot be reconstructed from the repository. This was not fixed or recaptured. It remains an unresolved reproducibility limitation.

### 2.3 V15 experiment specification

- File: `paper_1/v15/specification/v15_experiment_specification.json`
- Current SHA-256: `7caae51d39c6bc5f842d3637b7cd5df3a5f8fa82b809e9289d1d9aa616b05156`
- The specification was regenerated at `2026-11-22T18:06:50Z` to record the executed M0/M1 and M2/M3 sampler settings and the corrected no-lithology equation. The earlier recorded hash `2427e5e1…` refers to the superseded version. The `.sha256` sidecar has been corrected. No model was refitted as a result.
- The specification still lists `M2_logit_no_area` as a sensitivity model. It is specification-identical to `V14_control` and was deliberately not fitted twice. Its partial run is archived under `paper_1/v15/models/aborted_duplicate_M2_logit_no_area/` and must not be evaluated.

## 3. Data, outcome and folds

| Item | Value |
|---|---:|
| V14 grid rows | 1,899 |
| Unique grid IDs | 1,899 |
| Complete-case cells | 1,872 |
| Excluded rows (missing domain) | 27 |
| Positive cells before exclusion | 139 |
| Positive cells after exclusion | 138 |
| Negative cells | 1,734 |
| Prevalence | 0.07372 |

| Fold | Cells | Positives |
|---|---:|---:|
| 1 | 468 | 9 |
| 2 | 468 | 31 |
| 3 | 468 | 37 |
| 4 | 468 | 61 |

The excluded positive cell is grid 7074 (Seembere). The outcome is binary: at least one compiled USGS occurrence in the cell. It is not a deposit count and not a point-process realisation.

USGS occurrence accounting:

- 187 source records.
- 186 in complete-case cells.
- 184 in the 138 model-positive cells.
- 2 in model-negative cells: Kanaronga (grid 986), Luishia (grid 4288).
- 1 in excluded positive cell 7074: Seembere.
- 140 complete-case cells contain at least one record; 138 positive cells contain records; maximum 5 records in one cell; 33 positive cells contain multiple records (79 records).

The 147 RDC.ROCKS observations are **not** in the modelling inventory and are **not** model positives. They remain a separate observation corpus with zero confirmed cross-source identities.

## 4. Cell area

- Definition: full 5 km cell footprint intersected with the union of the five USGS permissive tracts.
- CRS: EPSG:32735; unit km².
- Full cell area: 25 km².
- Total clipped area: 21,790.3558 km².
- Maximum discrepancy against an independent cell-box/tract intersection: 4.3 × 10⁻¹⁴ km².
- 205 complete-case cells below 1 km²; **zero** positives among them.
- Positive mean area 13.34 km² (median 12.02); negative mean 11.38 km² (median 9.43).

Projected area-alone AUC:

| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Pooled |
|---:|---:|---:|---:|---:|
| 0.4103 | 0.6095 | 0.6745 | 0.7031 | 0.5834 |

Area discriminates weakly in the pooled sample but strongly in the north-west (Fold 4), where the V14 control is weakest. (Corrected; see the erratum.)

Area is geometrically justified as spatial support. It is **not** an exploration-effort or reporting-intensity correction.

## 5. Model equations

Let `z_f`, `z_l` and `z_g` be training-fold standardized fault distance, contact distance and Bouguer gravity; `z_f²` and `z_l²` their squares; `r_i` the categorical-lithology indicator vector; and `A_i` the clipped area in km².

All primary models use a complementary log-log likelihood:

```
N_i ~ Poisson(lambda_i * A_i)
Y_i = 1(N_i >= 1)
P(Y_i = 1) = 1 - exp(-lambda_i * A_i)
cloglog(P(Y_i = 1)) = log(lambda_i) + log(A_i)
```

Equivalently `P(Y_i = 1) = 1 - exp(-exp(eta_i))`, with `eta_i` containing `log(A_i)` as a fixed offset. The offset coefficient is one, not estimated. `lambda_i` is an intensity per km², and the returned probability is the probability of at least one recorded occurrence in the cell.

- **M0 (global):**
  `eta_i = alpha + beta_f z_f + beta_f2 z_f² + beta_l z_l + beta_l2 z_l² + beta_g z_g + beta_rock' r_i + log(A_i)`
- **M1 (domain intercept):**
  `eta_i = alpha_{d(i)} + [shared distance, gravity and lithology terms] + log(A_i)`
- **M2 (domain-varying distance):**
  `eta_i = alpha_{d(i)} + beta_{f,d(i)} z_f + beta_{f2,d(i)} z_f² + beta_{l,d(i)} z_l + beta_{l2,d(i)} z_l² + beta_g z_g + beta_rock' r_i + log(A_i)`
- **M3 (domain-varying distance and gravity):**
  as M2 with `beta_{g,d(i)} z_g`.
- **V14 control:** logistic link, no area term, M2-style domain-varying distance slopes and shared gravity. This is the like-for-like V14 likelihood.

Priors for domain-varying quantities:

```
alpha_mu ~ Normal(base, 1)
alpha_sigma ~ HalfNormal(1)
alpha_offset_d ~ Normal(0, 1)
alpha_d = alpha_mu + alpha_offset_d * alpha_sigma
mu_beta ~ Normal(0, 1)
sigma_beta ~ HalfNormal(1)
beta_d = mu_beta + offset_d * sigma_beta,  offset_d ~ Normal(0, 1)
shared slopes ~ Normal(0, 1)
```

`base` is an empirical centring value: `log((sum(y)+0.5)/(sum(A)+0.5))` for cloglog and the corresponding log-odds for logit. The Bernoulli probability is clipped to `[1e-12, 1-1e-12]`.

Sensitivity models: `M2_logit_area_covariate` replaces the offset with an estimated `beta_area * z(log A)`; `M2_cloglog_no_litho` removes `r_i`.

## 6. Primary results

| Model | AUC | PR-AUC | Mean log score | Total log score | Brier | Brier skill | Calibration slope |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0_global | 0.6882 | 0.1204 | -0.27250 | -510.12 | 0.07625 | -0.097 | 0.398 |
| M1_domain_intercept | 0.7087 | 0.1350 | -0.26332 | -492.94 | 0.07382 | -0.062 | 0.462 |
| M2_domain_distance | 0.7340 | 0.1633 | -0.25711 | -481.30 | 0.06922 | +0.004 | 0.489 |
| M3_domain_distance_gravity | 0.8020 | 0.2105 | -0.22672 | -424.43 | 0.06545 | +0.058 | 0.752 |
| V14_control | 0.7094 | 0.1361 | -0.25258 | -472.84 | 0.06891 | +0.008 | 0.588 |
| V14_frozen | 0.7155 | 0.1373 | -0.25103 | -469.93 | 0.06879 | — | 0.605 |
| RF_comparator | 0.7528 | 0.1574 | -0.23521 | -440.32 | 0.06488 | +0.066 | 0.952 |

Brier skill is relative to the training-partition prevalence baseline.

### Central paired comparisons

Orientation is `upper - lower`; a positive mean log-score difference means the upper model is better.

| Comparison | ΔAUC | Mean Δ log score | Total Δ log score | Spatial-block 95% CI | P(upper better) |
|---|---:|---:|---:|---|---:|
| M0 → M1 | +0.0205 | +0.00918 | +17.18 | [-0.00083, +0.01944] | 0.962 |
| M1 → M2 | +0.0253 | +0.00622 | +11.64 | [-0.01598, +0.02905] | 0.714 |
| M2 → M3 | +0.0680 | +0.03038 | +56.88 | [+0.010996, +0.05300] | 0.999 |

Spatial-block bootstrap: 25 km blocks, 117 blocks, 2,000 replicates. (Corrected 22 Nov 2026: the first two intervals were printed in left-minus-right orientation, opposite to the column definition; `paired_bootstrap.csv` stores left minus right. Conclusions are unchanged.)

The central test of the Paper 1 headline claim is **M1 → M2**. Its interval includes zero. The descriptive point estimate favours M2, but the data do not establish that domain-varying slopes beat domain intercepts alone.

### Fold-level results

| Model | Fold 1 AUC | Fold 2 AUC | Fold 3 AUC | Fold 4 AUC |
|---|---:|---:|---:|---:|
| M0 | 0.8289 | 0.6886 | 0.7552 | 0.7137 |
| M1 | 0.8378 | 0.6326 | 0.7505 | 0.7146 |
| M2 | 0.6713 | 0.6958 | 0.7751 | 0.7073 |
| M3 | 0.9695 | 0.8138 | 0.7608 | 0.7290 |
| RF | 0.7657 | 0.7253 | 0.7640 | 0.6765 |
| V14 frozen | 0.8753 | 0.7357 | 0.6851 | 0.6031 |

M2 is worse than M1 in Fold 1 and Fold 4 and better in Folds 2 and 3. The pooled gain is not a uniform along-belt gain.

M3's largest gain is Fold 1, which contains only nine positives. Its Fold 3 log score is slightly worse than M2 (-0.2574 vs -0.2493).

## 7. Area sensitivity

| Model | Link / area treatment | AUC | Mean log score | Calibration slope |
|---|---|---:|---:|---:|
| M2_domain_distance | cloglog / offset | 0.7340 | -0.25711 | 0.489 |
| M2_logit_area_covariate | logit / covariate | 0.7371 | -0.24874 | 0.571 |
| V14_control | logit / none | 0.7094 | -0.25258 | 0.588 |
| M2_cloglog_no_litho | cloglog / offset | 0.7347 | -0.25709 | 0.489 |

The area-covariate model beats the cloglog-offset M2 on held-out log score by 0.00836 per cell (M2 minus area-covariate = -0.00836; spatial-block interval `[-0.01411, -0.00276]`, excluding zero). The offset formulation is therefore not the only defensible exposure treatment, and the area treatment materially changes the probabilistic result. This supports conclusion D.

The area covariate is not a bias correction. It is an alternative description of how support varies with cell geometry.

## 8. Lithology

The stored V14 categorical field is a contact-line class, not a centroid host-lithology class:

| Assignment rule | Matches | Share |
|---|---:|---:|
| Raster class at centroid | 798 / 1,899 | 42.02% |
| Majority raster class in cell | 858 / 1,889 | 45.42% |
| Nearest contact to centroid | 942 / 1,899 | 49.61% |
| Any tied nearest contact to cell polygon | 1,812 / 1,899 | 95.42% |

- 1,890 cells have tied nearest contacts of different classes.
- 87 cells are unmatched by the tied-contact rule, including 5 positives.
- No source supports a defensible unique centroid host-lithology assignment.

Removing categorical lithology changes pooled AUC by about 0.0007 and mean log score by about 0.00002 per cell. The contact-distance predictor is retained. The categorical field contributes essentially nothing and should not be described as a host-lithology control.

## 9. Gravity

Within each fold, occurrences favour low Bouguer gravity. With low gravity scored positive, the gravity-alone AUC is 0.850, 0.783, 0.566 and 0.509 for Folds 1-4: strong in the south-east and absent in the north-west. This reproduces the preliminary 0.85-to-0.51 pattern (corrected; see the erratum). Gravity correlates with fault and contact distance at 0.3-0.6 in Folds 1-3, so the marginal pattern is not an isolated gravity effect.

M3's pooled gain is real in this sample but not a stable gravity law:

- all domain posterior means for `beta_grav` are negative;
- only NRB_3b is consistently informative (mean -1.09 to -1.54, intervals excluding zero);
- NRB_3a is weak (-0.03 to -0.33, most intervals include zero);
- CRZ, MMSB, NKB and SRB have no positive cells and are prior-dominated;
- M3's advantage is concentrated in Folds 1 and 2;
- M3 was added to the ladder after fold-level gravity results had been inspected; it is a data-suggested hypothesis and must be disclosed as such;
- M3 versus RF has mean log-score gain +0.00849 with interval `[-0.00801, +0.02624]`, which includes zero.

M3 should be reported as a conditional predictive result, not as evidence that gravity has a domain-specific geological effect.

## 10. Prior sensitivity

| Prior width | AUC | Mean log score | Calibration slope |
|---|---:|---:|---:|
| 0.5 | 0.7378 | -0.24798 | 0.572 |
| 1.0 (primary) | 0.7340 | -0.25711 | 0.489 |
| 2.0 | 0.7274 | -0.26795 | 0.421 |

The between-domain scale posteriors track prior width almost proportionally. From width 0.5 to 2.0, `alpha_sigma` moves from 0.645 to 1.726, `sigma_beta_f_lin` from 0.762 to 1.790, and `sigma_beta_l_sq` from 0.290 to 0.898.

Ranking is moderately stable (AUC range 0.011), but probabilistic calibration and the magnitude of heterogeneity are prior-dependent. The narrower prior gives the best held-out log score; selecting it post hoc would be tuning and is not done. Statements about how much domain relationships vary must be removed or labelled prior-dependent.

## 11. Buffered validation

| Model | Canonical AUC | 5 km AUC | 10 km AUC | Canonical log score | 5 km log score | 10 km log score |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 0.6882 | 0.6671 | 0.6566 | -0.2725 | -0.2957 | -0.2975 |
| M2 | 0.7340 | 0.7267 | 0.6974 | -0.2571 | -0.2582 | -0.2696 |

The M0-to-M2 direction survives: M2 retains a +0.060 AUC advantage at 5 km and +0.041 at 10 km. Magnitudes are unstable. Three buffered M2 fits fail the strict tail-ESS criterion; the 10 km Fold 1 AUC of 0.471 is uninformative with nine held-out positives. Buffered results are descriptive sensitivity evidence, not primary inference. See `buffered_validation_report.md`.

## 12. Residual spatial dependence

Moran's I on OOF residuals `y - p`, 8 nearest neighbours, row-standardised, 999 permutations:

| Model | Moran's I | p |
|---|---:|---:|
| M0 | 0.2464 | 0.001 |
| M1 | 0.2237 | 0.001 |
| M2 | 0.1804 | 0.001 |
| M3 | 0.1521 | 0.001 |

All models retain positive residual autocorrelation. Held-out cells are not independent, and the OOF metrics are optimistic relative to prediction in an unsampled area. The reduction from M0 to M3 is descriptive; it does not establish independent residuals.

## 13. Positional perturbation and fault provenance

- 500 simulations, 1 km radius, 187 source points; 184 used.
- Mean changed cells 40.07 (median 40, p95 51.05); mean positives 138.65.
- Mean point edge-crossing probability 0.174; maximum 0.64.
- Mean AUC under perturbation: M0 0.679, M1 0.696, M2 0.723, M3 0.794, RF 0.742, V14 frozen 0.714.
- The ordering is stable; the absolute values move by about 0.01.

Fault-distance provenance:

- 1,672 cells reproduce within 1 m from the surviving 126-line structure layer.
- 227 cells do not; 26 are positive. Maximum absolute difference 33.08 km.
- Non-reproduced by fold: Fold 1 = 69 cells/9 positives; Fold 2 = 158/17.
- By domain: NKB 15/0, NRB_3a 94/6, NRB_3b 118/20.

The stored predictor is not reproducible for a non-random subset concentrated in Folds 1–2 and NRB_3b. The complete-case frame was not changed. This is a cartographic provenance limitation.

## 14. Answers to the required conclusions

**A. Varying slopes provide convincing predictive improvement — No.** M2 over M1: +0.025 AUC and +0.0062 log score per cell, with a spatial-block interval that includes zero. Fold effects are mixed.

**B. Varying slopes provide limited or conditional improvement — Yes.** The pooled point estimate is positive, and the broader M0-to-M2 contrast survives 5 km and 10 km buffering, but M1-to-M2 was not buffered. The gain is concentrated in Fold 2, reverses in Folds 1 and 4, and is not separated from zero under spatial-block uncertainty.

**C. Domain-specific intercepts explain most of the improvement — Partly.** From M0 to M2 the AUC gain is +0.0458; M0→M1 supplies +0.0205 (45%) and M1→M2 +0.0253 (55%). On the log-score scale, M0→M1 supplies 60% and M1→M2 40%. The answer depends on the metric; intercepts alone do not explain all of it.

**D. Area correction materially changes the original conclusions — Yes.** Projected area alone reaches 0.583 AUC pooled and 0.703 in Fold 4; adding area raises M0's Fold 4 AUC to 0.714 against 0.603 for V14. The logit area-covariate model beats the cloglog-offset M2 on log score with an interval excluding zero. Exposure treatment changes the probabilistic ranking.

**E. Varying gravity effects offer additional value — Conditional only.** M3 improves on M2 in this sample with an interval excluding zero, but the gain is concentrated in Folds 1-2, especially the nine-positive Fold 1, where the marginal low-gravity association is also strongest; M3 was data-suggested, the coefficients are not stable across domains, and M3 versus RF includes zero.

**F. The available data do not support a strong hierarchical claim — Yes.** Sparse-domain scales are prior-dominated; M2 versus M1 is not separated from zero; residual autocorrelation remains; and the categorical lithology field is not defensible.

## 15. What survives of the original Paper 1 claim

Survives, with qualification:

- A domain-varying hierarchical model can be implemented and reproduces the frozen V14 benchmark closely (V15 control 0.7094 AUC versus frozen 0.7155).
- Domain-varying specifications produce a higher descriptive pooled AUC than a matched global model under the V15 likelihood (M2 0.734 versus M0 0.688).
- Predictive skill is spatially uneven, and the along-belt folds are the correct primary axis.

Does not survive as stated:

- That domain-varying slopes are convincingly superior. The central M1→M2 interval includes zero.
- That the DVHM's advantage over the 0.674 global reference reflects domain-varying slopes. The 0.674 predictions are stored and reproducible (erratum item 3), but that reference is a penalised maximum-likelihood fit without area, and the V15 ladder shows that the matched slope contrast (M1→M2) is not separated from zero. Keep 0.674 only as the historical V14 reference; use the V15 ladder for the matched comparison.
- That gravity has an established domain-specific geological effect. The along-belt fading of the low-gravity association is reproduced, and M3 captures it, but M3 was data-suggested and its gain is concentrated in two folds.
- Any statement about the magnitude of between-domain variation. Those scales are prior-dependent.
- Any implication that probabilities are well calibrated. Calibration slopes range from 0.398 to 0.752 for the Bayesian models.

## 16. Files and outputs

Specification and data:

- `paper_1/v15/specification/v15_experiment_specification.json`
- `paper_1/v15/specification/v15_experiment_specification.sha256`
- `paper_1/v15/data/v15_model_frame.csv`
- `paper_1/v15/diagnostics/data_build_audit.json`

Models and traces:

- `paper_1/v15/models/fit_status_recomputed.json`
- `paper_1/v15/models/nonconverged_initial/` (archived, not used)
- `paper_1/v15/models/aborted_duplicate_M2_logit_no_area/` (archived, not used)
- `paper_1/v15/traces/{M0_global,M1_domain_intercept,M2_domain_distance,M3_domain_distance_gravity,V14_control,M2_logit_area_covariate,M2_cloglog_no_litho}/fold_*.nc`
- `paper_1/v15/traces/{buffered_5km,buffered_10km}/...`
- `paper_1/v15/traces/{prior0p5_,prior2p0_}M2_domain_distance/...`

Predictions and metrics:

- `paper_1/v15/predictions/oof_*.csv`
- `paper_1/v15/predictions/*_fold_*.csv`
- `paper_1/v15/metrics/model_metrics.csv`
- `paper_1/v15/metrics/model_summary.csv`
- `paper_1/v15/metrics/central_comparisons.csv`
- `paper_1/v15/metrics/paired_comparisons.csv`
- `paper_1/v15/metrics/paired_by_fold.csv`
- `paper_1/v15/metrics/paired_bootstrap.csv`
- `paper_1/v15/metrics/comparator_metrics.csv`
- `paper_1/v15/metrics/buffered_validation_metrics.csv`
- `paper_1/v15/metrics/prior_sensitivity_metrics.csv`
- `paper_1/v15/metrics/prior_sensitivity_predictions.csv`

Diagnostics:

- `paper_1/v15/diagnostics/sampling_diagnostics.csv`
- `paper_1/v15/diagnostics/sampling_parameter_summary.csv`
- `paper_1/v15/diagnostics/area_*.csv`
- `paper_1/v15/diagnostics/small_cells.csv`
- `paper_1/v15/diagnostics/occurrence_point_cell_support.csv`
- `paper_1/v15/diagnostics/occurrence_support_summary.json`
- `paper_1/v15/diagnostics/gravity_coefficients.csv`
- `paper_1/v15/diagnostics/gravity_fold_summary.csv`
- `paper_1/v15/diagnostics/fault_distance_*.csv`
- `paper_1/v15/diagnostics/label_perturbation_*.csv`
- `paper_1/v15/diagnostics/perturbation_point_assignment.csv`
- `paper_1/v15/diagnostics/moran_i.csv`
- `paper_1/v15/diagnostics/lithology_assignment_audit.json`
- `paper_1/v15/diagnostics/prior_sensitivity_scales.csv`
- `paper_1/v15/diagnostics/prior_sensitivity_summary.json`
- `paper_1/v15/diagnostics/buffered_training_support.csv`

Reports:

- `paper_1/v15/reports/data_and_area_audit.md`
- `paper_1/v15/reports/v14_control_reproduction.md`
- `paper_1/v15/reports/area_sensitivity_report.md`
- `paper_1/v15/reports/gravity_heterogeneity_report.md`
- `paper_1/v15/reports/moran_i_report.md`
- `paper_1/v15/reports/label_perturbation_report.md`
- `paper_1/v15/reports/fault_layer_sensitivity_report.md`
- `paper_1/v15/reports/lithology_provenance_audit.md`
- `paper_1/v15/reports/lithology_sensitivity_report.md`
- `paper_1/v15/reports/comparator_report.md`
- `paper_1/v15/reports/sampling_diagnostics.md`
- `paper_1/v15/reports/buffered_validation_report.md`
- `paper_1/v15/reports/prior_sensitivity_report.md`
- `paper_1/v15/reports/master_results_report.md`
- `paper_1/v15/reports/proposed_manuscript_revision.md`
- `paper_1/v15/reports/independent_review_checklist.md`

Code and tests:

- `src/v15/*.py`
- `tests/test_v15_models.py`

## 17. Tests run

- `pytest -q tests/test_v15_models.py`: **7 passed in 6.41 s**. Covers exposure likelihood equivalence, model nesting and design matrices, fold integrity, posterior-prediction range, log-score computation and specification identity.
- `pytest -q tests/test_v14_benchmark_integrity.py`: **82 passed in 0.89 s**. Covers all 61 tracked and 20 external benchmark artefacts.

## 18. Reproduction commands

```powershell
.\.venv\Scripts\python.exe -m src.v15.build_data
.\.venv\Scripts\python.exe -m src.v15.fit --models M0_global M1_domain_intercept --folds 0 1 2 3 --draws 1000 --tune 1500 --chains 2 --cores 2 --target-accept 0.95
.\.venv\Scripts\python.exe -m src.v15.fit --models M2_domain_distance M3_domain_distance_gravity --folds 0 1 2 3 --draws 1500 --tune 2500 --chains 4 --cores 4 --target-accept 0.995
.\.venv\Scripts\python.exe -m src.v15.fit --models V14_control M2_logit_area_covariate M2_cloglog_no_litho --folds 0 1 2 3 --draws 1000 --tune 2000 --chains 2 --cores 2 --target-accept 0.995
.\.venv\Scripts\python.exe -m src.v15.fit --models M2_domain_distance --folds 0 1 2 3 --draws 1000 --tune 2000 --chains 2 --cores 2 --target-accept 0.995 --prior-scale 0.5 --tag prior0p5_
.\.venv\Scripts\python.exe -m src.v15.fit --models M2_domain_distance --folds 0 1 2 3 --draws 1000 --tune 2000 --chains 2 --cores 2 --target-accept 0.995 --prior-scale 2.0 --tag prior2p0_
.\.venv\Scripts\python.exe -m src.v15.buffered --buffers-km 5 10 --models M0_global M2_domain_distance --folds 0 1 2 3 --draws 500 --tune 1000 --chains 2 --cores 2 --target-accept 0.995
.\.venv\Scripts\python.exe -m src.v15.evaluate
.\.venv\Scripts\python.exe -m src.v15.prior_sensitivity
.\.venv\Scripts\python.exe -m src.v15.sampling_diagnostics
.\.venv\Scripts\python.exe -m src.v15.gravity
.\.venv\Scripts\python.exe -m src.v15.moran
.\.venv\Scripts\python.exe -m src.v15.fault_sensitivity
.\.venv\Scripts\python.exe -m src.v15.perturbation
.\.venv\Scripts\python.exe -m src.v15.prior_predictive
.\.venv\Scripts\python.exe -m src.v15.comparator
.\.venv\Scripts\python.exe -m pytest -q tests/test_v15_models.py
.\.venv\Scripts\python.exe -m pytest -q tests/test_v14_benchmark_integrity.py
.\.venv\Scripts\python.exe -m src.v15.checksums
.\.venv\Scripts\python.exe -m src.v15.checksums --verify
```

## 19. Unresolved issues

1. **V14 reference 0.674 is a penalised maximum-likelihood fit.** Its predictions are stored and reproducible (erratum item 3), but it is not a Bayesian, area-aware or link-matched counterpart of the V15 models. Report it as the historical reference only.
2. **V14 source snapshot is not in Git.** The manifest hash for `src/validation_strategies.py` does not match HEAD or prior commits. The numerical artefacts are verified; the source-level pipeline is not exactly reconstructible.
3. **Fault-distance provenance.** 227 cells, including 26 positives, are not reproducible from the surviving structure layer, with errors up to 33 km.
4. **Categorical lithology.** No defensible centroid host-lithology assignment exists in the acquired data. The current field is a tied-contact class.
5. **Gravity.** The along-belt fading of the low-gravity association is reproduced (0.85 to 0.51). M3's gain matches it but is data-suggested and concentrated in Folds 1-2; it needs an independent test, such as a held-out district.
6. **Prior dependence.** Between-domain scales are prior-dominated; the magnitude of heterogeneity is not identified.
7. **Buffered convergence.** Three M2 buffered fits fail the strict tail-ESS criterion; the 10 km Fold 1 result is uninformative.
8. **Complete-case exclusion.** 27 cells, including one positive, are excluded for missing domain. No missing-data model was fitted.
9. **Positional uncertainty.** The 1 km radius is a working assumption, not a per-record survey uncertainty.
10. **Occurrence process.** The compiled inventory is not a complete point pattern; area is support, not effort.
11. **Four folds.** Four spatial units cannot give a precise sampling distribution. All intervals are block-bootstrap summaries, not repeated independent validation.
12. **Residual autocorrelation.** All primary models leave positive Moran's I; OOF scores are optimistic for unsampled areas.

## 20. Independent review checklist

See `paper_1/v15/reports/independent_review_checklist.md`. It contains the file-level, numerical and interpretation checks, including the exact values above and the restrictions on what may be claimed.
