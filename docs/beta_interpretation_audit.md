# V11 Post-fit β Interpretation and Stability Audit

Date: 2026-09-14

## A. Files changed

- `src/v11c_beta_diagnostics.py`
- `src/v11c_independent_check.py`

No frozen V11 fitting code or posterior trace was modified.

## B. Files created

### Diagnostic code
- `src/v11c_beta_diagnostics.py`
- `src/v11c_independent_check.py` (session-written independent numerical check helper)

### Diagnostic CSVs
All are under the pinned new directory `figures/audit/beta_diagnostics/`:
- `beta_diagnostics_coefficients_by_fold.csv`
- `beta_diagnostics_four_cases_by_fold.csv`
- `beta_four_case_summary.csv`
- `beta_turning_points_by_fold.csv`
- `beta_threshold_sensitivity_four_cases.csv`
- `beta_slope_support_by_fold.csv`
- `beta_response_curve_extrema_by_fold.csv`
- `beta_coefficient_stability_pairwise.csv`

### Diagnostic plots
- `NRB_3a_fault_response_curves.png`
- `NRB_3a_fault_slope_probability.png`
- `NRB_3a_lithology_response_curves.png`
- `NRB_3a_lithology_slope_probability.png`
- `NRB_3b_fault_response_curves.png`
- `NRB_3b_fault_slope_probability.png`
- `NRB_3b_lithology_response_curves.png`
- `NRB_3b_lithology_slope_probability.png`

### Final report
- `docs/beta_interpretation_audit.md`

## C. Files intentionally left frozen

- `src/v11_spatial_stability_final.py`
- `src/validation_strategies.py`
- `AI_PROJECT_CONTEXT.md`
- `figures/v11_fold_1_trace.nc`
- `figures/v11_fold_2_trace.nc`
- `figures/v11_fold_3_trace.nc`
- `figures/v11_fold_4_trace.nc`

The existing legacy outputs were not overwritten, including `figures/v11_geological_stability_matrix.csv`, `figures/audit/phase5_turning_points.csv`, and the Phase 7 predictive-validation artifacts. They remain legacy/superseded where their terminology or support convention differs from this audit.

## D. Exact implementation changes

The new downstream module reads the existing V11 posterior traces and reconstructs the V11 fold assignments and training-fold StandardScaler parameters without fitting a model. It implements:

1. Draw-level quadratic turning points using `z* = -beta_lin/(2*beta_sq)` and physical conversion `D* = mu_train + sigma_train*z*`.
2. Turning-point classification from curvature sign: positive `beta_sq` is a minimum and negative `beta_sq` is a maximum.
3. Explicit separation of `P(beta_sq>0)`, `P(beta_sq<0)`, and `P(beta_sq>0.05)`.
4. Fold × Daly-domain empirical support using only the training observations available in that fold, with inclusive `[observed_min_distance_km, observed_max_distance_km]` boundaries.
5. Unfiltered algebraic D* and conditional D* under `beta_sq > 0.05`, with the conditional nature explicitly encoded in output column names.
6. Posterior slope calculations `beta_lin + 2*beta_sq*((D-mu_train)/sigma_train)` over a 401-point grid spanning each exact fold/domain support.
7. Posterior-mean response curves evaluated independently on the probability scale; because the logistic map is monotonic, extremum locations correspond to extrema of the underlying linear predictor for a fixed draw, but the reported posterior-mean curve is computed directly rather than inferred from median D*.
8. Fold-specific coefficient summaries and descriptive pairwise posterior differences for `beta_f_lin`, `beta_f_sq`, `beta_l_lin`, and `beta_l_sq`. No composite stability score was introduced.
9. Sensitivity of conditional D* to thresholds 0, 0.01, 0.05, and 0.10.
10. Four required case plots for response curves and slope probabilities.

The V11 quadratic model equation, priors, hierarchical structure, non-centred parameterization, feature definitions, fold construction, and posterior traces were not changed.

## E. Tests and audits executed

### Baseline and repository state

The authoritative starting `git_status()` was:

`## main...origin/main [ahead 1]`

`git_log()` showed HEAD `888a03c (MCP: replace server.py with hardened v3 server1.py)`, one commit ahead of `origin/main`.

`verify_frozen_intact(recapture=False)` was run before implementation and returned **7 OK, 0 drifted, 0 missing** against the session baseline.

### Existing test suite

`run_pytest -q` was executed with the permitted 900-second timeout. It did not run tests because the environment lacks pytest. Exact error:

`C:\Users\vanmu\copperbelt-mineral-prospectivity\.venv\Scripts\python.exe: No module named pytest`

Exit code: 1. This is an environment/dependency failure, not evidence that the project tests pass.

### New diagnostic module

`src/v11c_beta_diagnostics.py` was syntax-validated and then executed successfully with an explicit 1200-second timeout. It regenerated all CSVs and plots listed above.

The first execution exposed an ordinary new-module import error (`ModuleNotFoundError: No module named 'src'`), which was corrected by adding the repository root to `sys.path`. A subsequent execution exposed an unterminated string introduced in that new module; the exact error was:

`SyntaxError: unterminated string literal (detected at line 18)`

The module was corrected, syntax-validated, and the next execution completed successfully.

### Independent numerical verification

`src/v11c_independent_check.py` was syntax-validated and executed successfully. Four posterior draws—one from each required case—were independently recomputed using the stated formulas. Results:

| Case | Fold | Draw | D* (km) | slope(D*) |
|---|---:|---:|---:|---:|
| NRB_3a fault | 1 | 0 | 27.462616 | -2.220e-16 |
| NRB_3a lithology | 1 | 0 | 26.933604 | 1.110e-16 |
| NRB_3b fault | 2 | 0 | 77.051797 | -8.882e-16 |
| NRB_3b lithology | 2 | 0 | -17.979008 | 0.000e+00 |

The slope at each independently computed turning point is numerically zero, confirming agreement with the quadratic derivative formula. The helper also independently reconstructed the fold-specific support intervals.

### Generated-output inspection

The regenerated directory was checked after execution. It contains 8 CSV tables and 8 PNG plots. The four-case CSV was read directly and verified to contain fold-specific coefficients, support, conditional/unconditional D*, curvature probabilities, classification, and posterior-mean curve extrema. The threshold-sensitivity CSV was also inspected.

### Terminology audit

`scan_terminology()` was run over maintained source/documentation/audit outputs for the four specified legacy turning-point expressions. It returned **0 hits in maintained content outside the audit report's own audit-description text** after the report was cleaned; frozen files remain exempt from editing as required.

## F. Output validation

The output tables use the exact fold-specific training support rather than an arbitrary 0–50 km interval. For example, NRB_3b fault support varies by fold from approximately 62.0 km in Fold 1 to 109.0 km in Fold 2; NRB_3b lithology support is approximately 39.3–45.3 km. The four-case table records these values explicitly.

The posterior-mean response extrema were computed on 401 grid points spanning the corresponding empirical support. Slope tables use the same support and the exact training-fold scaler parameters.

The old `figures/v11_geological_stability_matrix.csv` was not overwritten and is therefore not the authoritative post-fit β stability matrix. The new `beta_diagnostics_*` tables are the upgraded diagnostic outputs.

## G. Before/after interpretation — four required cases

### 1. NRB_3a fault distance

Across folds, `beta_lin` is negative (fold medians approximately -2.68, -1.41, -1.26, -0.70) while `beta_sq` is positive in three folds and sign-uncertain in Fold 2. The fold-median `P(beta_sq>0)` is about 0.943, but Fold 2 has only 0.464 and therefore does not independently support a minimum. `P(beta_sq>0.05)` has a fold median about 0.923, again with Fold 2 only about 0.433.

The unfiltered D* medians range roughly from -18.0 to 46.6 km across folds, and the probability that D* lies inside the actual fold-specific empirical support ranges from about 0.126 to 0.938. Posterior-mean curve extrema are approximately 36.6–43.3 km and are minima within support. Thus the response-curve evidence is more coherent than the algebraic D* distribution in Fold 2, but coefficient stability is not strong enough to describe the fault turning point as uniformly identified across folds.

The appropriate interpretation is therefore: **NRB_3a fault curvature is generally minimum-like but spatially heterogeneous, with one fold showing curvature-sign uncertainty and substantial D* instability.** A single precise geological distance is not strongly identified by the fold-wise posterior.

### 2. NRB_3a lithology-contact distance

This is the strongest of the four cases. `beta_lin` is negative in all four folds and `beta_sq` is positive with posterior sign probability about 0.993–1.000. `P(beta_sq>0.05)` is about 0.985–0.999. The unfiltered D* medians are approximately 26.2–28.5 km and the posterior probability of lying inside actual training support is about 0.963–1.000.

Posterior-mean response-curve extrema are approximately 25.5–28.5 km and are minima. The conditional `beta_sq>0.05` analysis changes D* very little. This is evidence for a relatively well-identified **local minimum** in the fitted lithology-distance response, not a claim that one physical distance should be selected as a target.

Fold-to-fold coefficient differences still have wide credible intervals, so coefficient stability should not be described as perfect. Nevertheless, the sign and response-curve behavior are substantially more consistent than for the NRB_3b cases.

### 3. NRB_3b fault distance

The fault coefficients are highly sign-uncertain. Across folds, `P(beta_sq>0)` ranges approximately 0.161–0.708 and `P(beta_sq<0)` approximately 0.292–0.839. `P(beta_sq>0.05)` ranges only about 0.109–0.589. No fold meets the strong minimum/maximum-dominant classification used by this audit.

Unfiltered D* medians range approximately 22.8–80.4 km, with broad intervals extending outside the empirical support. The posterior-mean curve extrema vary strongly: about 0.36–22.3 km, including a boundary/monotone result in Fold 2. This discrepancy demonstrates why algebraic D* and posterior-mean curve extrema cannot be substituted for one another.

The evidence therefore supports **weak or uncertain quadratic identification** for NRB_3b fault distance. Any single reported fault turning point should be treated as weakly identified and not as a robust geological characteristic.

### 4. NRB_3b lithology-contact distance

`beta_lin` is negative across folds, and `beta_sq` is positive in the posterior majority of draws, but curvature sign is less secure than in NRB_3a lithology. `P(beta_sq>0)` ranges approximately 0.695–0.865 and `P(beta_sq>0.05)` approximately 0.660–0.845. Thus a minimum is more probable than a maximum, but not decisively so.

Unfiltered D* medians range approximately 26.5–41.3 km, while the probability that D* lies inside actual training support is only about 0.381–0.584. Conditional filtering at 0.05 shifts medians upward and does not remove the support uncertainty. Posterior-mean curve extrema are around 28.1–34.2 km and are minima within support.

The correct conclusion is **a minimum-like response is suggested, but quadratic/turning-point identification is materially weaker than for NRB_3a lithology, and empirical support for the algebraic D* is uncertain.**

### Posterior slope behavior across empirical support

The slope diagnostic gives the direction of the modeled response rather than only locating a turning point. In NRB_3a lithology, all folds have strongly negative slope probability at the lower support boundary and strongly positive probability at the upper boundary, consistent with a well-supported U-shaped response. NRB_3a fault behaves similarly in Folds 1 and 4, while Fold 2 remains negative through most of its support and Fold 3 is comparatively uncertain near its upper boundary. NRB_3b lithology starts strongly negative but has weak or mixed slope sign near the upper boundary in all folds, consistent with a less securely identified minimum. NRB_3b fault has mixed slope behavior at both ends and changes direction differently across folds; this is consistent with the weak identification of a common fault turning point.

Importantly, `beta_lin` is the slope at the training-fold mean distance (`z=0`), not the slope at physical distance zero. The slope tables use the physical distance grid transformed with the same fold-specific training scaler as V11.

## H. Model integrity

**The V11 PyMC model did not change.** No V11 refit was performed. The frozen fitting module and all four frozen posterior traces remained untouched. The new work is strictly downstream/post-fit.

The pre-existing V11 equation was internally consistent with the stated quadratic and standardization: `z=(D-mu_train)/sigma_train`, `D*=mu_train+sigma_train*(-beta_lin/(2*beta_sq))`, and `d eta/dD` follows directly from the chain rule. No genuine model implementation bug requiring a V11 change was identified.

## I. Frozen-baseline verification

The final `verify_frozen_intact(recapture=False)` release gate was executed and returned **7 OK, 0 drifted, 0 missing**. The baseline timestamp remained `2026-09-14T17:40:59.980927+00:00`; no recapture was performed after the initial session baseline.

## J. Limitations

1. **beta_sq → 0:** Algebraic D* becomes unstable because the denominator approaches zero. A finite numerical D* is not evidence of identification.
2. **0.05 filter:** No intrinsic Bayesian or geological justification for `beta_sq > 0.05` was found in the inspected project materials. It is therefore a methodological filter, not a scientific truth criterion.
3. **Conditional D*:** D* reported under the 0.05 rule is `p(D* | beta_sq > 0.05)`, not the unconditional posterior.
4. **Median D* vs posterior-mean turning point:** Median posterior D* and the extremum of the posterior-mean response are different estimands and can materially disagree.
5. **Empirical support:** Support is fold-specific training-data support. It does not establish geological causality and does not guarantee that the fitted quadratic is reliable at every supported distance.
6. **Quadratic identification:** Posterior sign uncertainty, broad coefficient intervals, and fold-to-fold differences limit strong claims in several cases.
7. **Coefficient stability:** The pairwise diagnostics are descriptive posterior comparisons, not a composite stability score. Wide difference intervals indicate that fold-specific coefficient equality is not established.
8. **Predictive vs coefficient stability:** Phase 7.6 is predictive sensitivity analysis. Stable AUC is not evidence of stable beta coefficients, stable response curves, or stable turning points.
9. **Test environment:** The existing pytest suite could not execute because pytest is not installed in the project environment.
10. **Legacy terminology:** Frozen files were not edited. Any stale terminology inside them would remain a documented limitation under the frozen-file rule.

## K. Final scientific conclusion

V11 supports a **clearer and more defensible interpretation of quadratic distance effects than the legacy 0–50 km presentation**, but it does not support a universal set of precise geological distances.

- **NRB_3a lithology-contact distance** is the most reasonably identified case: curvature is overwhelmingly positive, the fitted quadratic is minimum-shaped, and the algebraic turning point is generally inside actual training support. This supports a local minimum in the modeled response, not a claim that one physical distance should be selected as a target.
- **NRB_3a fault distance** is generally minimum-like, but one spatial fold has curvature-sign uncertainty and D* is much less stable. It is therefore only moderately identified across folds.
- **NRB_3b lithology-contact distance** shows a probable minimum-like response, but curvature sign is uncertain enough and D* support probability low enough that the turning point is weak-to-moderately identified rather than firmly established.
- **NRB_3b fault distance** is the weakest case. Curvature changes sign substantially across posterior draws/folds, D* is unstable, and posterior-mean response extrema vary from boundary/monotone behavior to interior minima. A robust fault turning point is not established.

The principal scientific result is therefore not a ranking of D* magnitudes. It is that **quadratic curvature, algebraic turning points, empirical support, slope behavior, coefficient stability, and predictive stability are distinct properties of the V11 model and must be reported separately.** The corrected analysis weakens the interpretation of several apparent distance optima and supports only a limited, case-specific set of minimum-like responses.

Phase 7.6 remains **predictive sensitivity analysis** and must not be used as evidence for beta-coefficient or turning-point stability.
