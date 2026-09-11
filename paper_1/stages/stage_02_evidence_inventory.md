# Stage 02: Evidence Inventory
# Stage 2: Evidence Inventory (Revised)

## Claim 1: Predictor relationships are spatially heterogeneous
*   **Question:** Do geological predictors have the same predictive relationship with mineralization across tectonic domains?
*   **Evidence:** Domain-specific Bayesian posterior coefficients from V11.
*   **Analysis:** Comparison of posterior distributions for coefficients (e.g., $\beta_{fault,d}$, $\beta_{lith,d}$) across Daly domains.
*   **Evidence Product:** Table 1 — Domain-specific posterior coefficients (Domain, Predictor, Posterior median, 95% HDI, Direction).
*   **Code Reference:** `v11_spatial_cv.py`
*   **Status:** Empirical result.

## Claim 2: The spatial scale of predictor influence varies between regions
*   **Question:** Does a geological predictor operate at the same spatial scale throughout the Copperbelt?
*   **Evidence:** Posterior turning points ($D^*$). For example, NRB_3a shows an identifiable lithological response around ~28 km, while NRB_3b exhibits an extremely broad, unconstrained posterior distribution.
*   **Evidence Product:** Table 2 — Domain-specific response scales (Domain, Predictor, Median $D^*$, 95% HDI, $P(D^* \leq D_{\max})$).
*   **Code Reference:** `phase5_robust_response.py`
*   **Status:** Empirical result.

## Claim 3: Relative model performance is geographically dependent
*   **Question:** Does the identity of the better-performing model remain constant across spatial regions?
*   **Evidence:** Strict along-belt OOF predictions demonstrating a sign reversal in $\Delta AUC$ (e.g., $\Delta AUC_{F2} < 0$ favoring M5, but $\Delta AUC_{F3} > 0$ favoring V11).
*   **Evidence Product:** Table/Figure 3 — Spatial OOF model comparison (ROC-AUC, PR-AUC, Brier score, $\Delta AUC$ by fold).
*   **Code Reference:** `phase7_final_validation.py`
*   **Status:** Empirical result.

## Claim 4: The performance reversal is robust to spatial dependence
*   **Question:** Could the apparent reversal simply be an artifact of treating spatially correlated cells as independent observations?
*   **Evidence:** Spatial block bootstrap tested across multiple spatial scales (10x10, 15x15, 20x20, 25x25 grids). The direction and magnitude of the regional performance differences persist regardless of block resolution.
*   **Evidence Product:** Table/Figure 4 — Multi-scale spatial robustness (Block scale, $\Delta AUC$ bootstrap median, 95% CI, Bootstrap Proportion $\Delta AUC > 0$).
*   **Code Reference:** `phase7_multiscale_robustness.py`
*   **Status:** Empirical robustness result.

## Claim 5: Pooled global metrics can conceal regional predictive failure
*   **Question:** Does a strong global evaluation metric guarantee reliable prediction in all key subregions?
*   **Evidence:** The pooled OOF evaluation for V11 yields a highly competitive AUC (~0.689). However, within Fold 4—which contains ~44% of the deposits—the model drops to ~0.561 AUC. This demonstrate a single performance metric can conceal substantial regional varation in predictive performance.
*   **Evidence Product:** Figure 5 — Pooled versus regional performance (Visual contrast of Pooled Copperbelt AUC vs. Fold 4 AUC).
*   **Code Reference:** `phase7_final_validation.py`
*   **Status:** Empirical result.

## Claim 6: Regional failure is not explained by missing predictor observations
*   **Question:** Could the poor regional performance simply result from missing survey data?
*   **Evidence:** The missingness audit indicates 100% complete cases for the modeling variables within the evaluated dataset.
*   **Interpretive Explanation (For Discussion Section):** The absence of missing predictor observations suggests that the regional reduction in discrimination is not attributable to incomplete measurements. One possible explanation is that the selected regional-scale proxies contain insufficient transferable information in that region.
*   **Evidence Product:** Table 6 — Spatial observation audit (Domain, cell counts, deposit counts, complete-case percentages).
*   **Code Reference:** `phase6_missingness_audit.py`
*   **Status:** Empirical diagnostic.