# Stage 05: Interpretation

## 5.1 Multivariate predictor structure and conditional inference

The V11 model must be interpreted fundamentally as a multivariate mineral prospectivity system rather than a univariate distance-vectoring exercise. By integrating distance to fault, distance to lithology contact, Bouguer gravity anomaly, and categorical host stratigraphy into a single linear predictor, the model evaluates each predictor conditionally:

$$
\eta_i = \alpha_{d(i)} + \eta_{f, i} + \eta_{l, i} + \eta_{g, i} + \eta_{rock, i}.
$$

This joint formulation carries critical implications for geological interpretation:
1. **Conditional Parameter Interpretation:** A fitted coefficient describes an association conditional on the other included predictors and the model structure. It does not isolate a causal physical mechanism or account for unmeasured geology, sampling, or errors in the mapped evidence layers.
2. **Role of Bouguer Gravity:** Bouguer gravity enters as a global standardized linear predictor ($\beta_g$). While the absence of a quadratic gravity term precludes the calculation of an algebraic stationary point ($D^*$), this reflects a deliberate modeling choice for regional background stabilization, not an empirical finding that gravity is secondary to proximity predictors.
3. **Host Lithology Term:** Retained host-lithology indicators are included as global predictors, with categories supported by both classes in the training fold. Their inclusion does not ensure that distance coefficients isolate transport or other geological mechanisms.

## 5.2 Spatial heterogeneity in SD-normalized component shares

The decomposition shows that SD-normalized shares of additive components in the posterior-mean OOF linear predictor vary across the four folds. These shares describe model-predictor dispersion; they are not outcome-variance explained, marginal predictor importance, or causal effects:

- **Fold 1 ($S_f = 86.6\%$):** The fault-distance component has the largest SD-normalized share; its component standard deviation is 7.46. Contact distance and gravity have shares of 11.0% and 2.0%, respectively.
- **Fold 2 ($S_f = 46.8\%$, $S_l = 28.0\%$, $S_{rock} = 19.6\%$):** Fault distance has the largest share. Host lithology and contact distance together account for 47.6% of the normalized shares.
- **Fold 3 ($S_f = 30.5\%$, $S_l = 32.8\%$, $S_{rock} = 22.4\%$, $S_g = 14.4\%$):** Contact distance has the largest share, and the four component shares are more distributed than in Fold 1.
- **Fold 4 ($S_f = 33.2\%$, $S_g = 28.7\%$, $S_l = 23.2\%$, $S_{rock} = 14.9\%$):** Gravity and fault distance have the two largest shares; their component standard deviations are 0.57 and 0.66, respectively.

These are descriptive component-dispersion patterns for the specified folds. They do not establish that particular geological processes control mineralization or explain fold-level predictive performance. Geological interpretations are hypotheses for follow-up, not findings identified by this decomposition.

## 5.3 Distance-response representation sensitivity: recurring association versus representation-dependent form

The $2 \times 2$ factorial sensitivity analysis resolves a central scientific question: which aspects of the distance response recur across representations, and which are representation-dependent?

### 1. Recurring predictive association across representations
Across the four evaluated models (Model A raw-quadratic, Model B log-quadratic, Model C raw-linear, Model D log-linear), distance associations recur, but their magnitude and form vary by predictor, domain, and fold. The negative NRB_3a trends are not a uniform belt-wide result; reported NRB_3b fault-distance linear coefficient medians are positive. The quadratic specifications also permit non-monotone responses, so a universal monotonic decline is not established.

This association is not uniform in every domain-specific posterior: NRB_3b fault-distance linear coefficients are positive in the reported fold summaries. The recurring association therefore describes the evaluated representations in aggregate, not a uniformly negative relationship across the entire belt.

### 2. Sensitivity of quadratic curvature
In contrast to the recurring predictive distance associations, quadratic curvature is sensitive to predictor transformation:
- In raw physical distance space (Model A), the distribution of cell distances is severely right-skewed: most deposits lie within $0\!-
!15\ \mathrm{km}$ of contacts, while the background basin grid extends to $60\!-
!80\ \mathrm{km}$. A quadratic polynomial fitted to this skewed distribution bends upward in the sparse distal tail to avoid overly penalizing distal non-deposit cells.
- Under Model B, the log transform compresses upper-tail distances and posterior curvature support changes. For NRB_3a contact distance, positive-curvature support drops from $\ge 0.998$ under Model A to $0.674$--$0.930$ under Model B; fault-curvature support also changes. These are representation differences and do not show that the transform identifies the uniquely correct response shape.

### 3. Model parsimony and predictive transferability
Across the evaluated specifications, mean spatial OOF ROC-AUC ranges from $0.688$ to $0.719$. Model D has the highest mean ROC-AUC ($0.7188$) and mean PR-AUC ($0.1967$), compared with $0.1365$ mean PR-AUC for Model A; Model C's Fold 1 ROC-AUC is $0.9298$, a single-fold result. The quadratic term improved mean ROC-AUC under raw distance (Model A $0.7116$ vs Model C $0.6876$) but not under log distance (Model B $0.7042$ vs Model D $0.7188$). These comparisons do not establish universal superiority of any functional form.

## 5.4 Diagnostic status and unidentifiability of $D^*$

Historically, the algebraic stationary point $D^* = \mu_{train} + \sigma_{train} z^*$ was hypothesized to reflect a characteristic distance or optimal structural trap offset. The completed empirical evidence does not support this interpretation:

1. **Failure of Operational Robustness Criteria:** The pre-registered robustness protocol satisfied all three criteria in only **1 of 48 evaluable cases (2.1%; Fold 1, NRB_3a lithological contact)**. Stationary-point agreement within 25% occurred in 2 of 48 cases, with an average relative discrepancy of $60.6\%$ (median $64.1\%$). Curvature probabilities were discordant in 64.6% of cases. Overall robustness was therefore not general across representations.
2. **Population Singularity:** Reconstructing population-level distributions from the frozen V11 posterior traces demonstrates that population quadratic parameters heavily overlap zero (fold-level $P(\mu_{f, sq} > 0) = 0.27$--$0.53$; $P(\mu_{l, sq} > 0) = 0.73$--$0.82$). When posterior curvature mass crosses zero, the algebraic ratio $z^* = -\mu_{lin} / (2\mu_{sq})$ suffers from denominator singularity, producing explosive Cauchy-like credible intervals spanning $[-376, +379]\ \mathrm{km}$. A belt-wide population $D^*$ is mathematically unidentifiable. This population-level unidentifiability is distinct from the domain/fold stationary points, which are representation-dependent and unstable rather than automatically meaningless in every local setting.
3. **Fold-Level Instability:** At the domain level, apparent stationary points vary across spatial folds (e.g., NRB_3a fault $D^*$ moving from $-10.4\ \mathrm{km}$ in Fold 2 to $+46.9\ \mathrm{km}$ in Fold 3). In CRZ, MMSB, NKB, and SRB, no positive observations inform within-domain positive-class behavior; any algebraic stationary point there is not evidence of an observed threshold.
4. **Tail Truncation Sensitivity:** Removing distal background cells ($p_{95}$ truncation) reduced Fold 1 AUC from $0.846 \to 0.616$ for Model A and from $0.829 \to 0.656$ for Model B, and mean AUC from $0.712 \to 0.667$ and $0.704 \to 0.665$, respectively; the fitted predictions and discrimination are sensitive to the distal tail under both models.

Consequently, $D^*$ must be interpreted strictly as an unstable mathematical diagnostic of a specific polynomial representation, not as a physical trap distance, an optimal exploration vector, or a universal metallogenic property.

## 5.5 Pooled and fold-level OOF discrimination

Four-fold along-belt spatial holdout evaluates transfer among the specified sectors; it does not guarantee independence across fold boundaries or performance in all future regions:

- **Pooled OOF discrimination ($\Delta\mathrm{AUC}=+0.147$):** Pooled V11 ROC-AUC is $0.707$ compared with $0.559$ for M5; the paired row-bootstrap CI is $[+0.073,+0.217]$ and $P(\Delta>0)=1.000$. Because the specifications differ in predictor set and model structure, this comparison does not isolate the contribution of any single component or guarantee transfer beyond these folds.
- In **Fold 3**, the paired row-bootstrap V11-minus-M5 ROC-AUC difference is $+0.056$ [$-0.002$,$+0.112$], an interval that includes zero; this score difference does not identify a geological cause.
- In **Fold 2**, the paired row-bootstrap V11-minus-M5 ROC-AUC difference is $-0.118$ [$-0.224$,$-0.011$]; this score difference does not explain why M5 scores higher.
- In **Folds 1 and 4**, the 95% row-bootstrap intervals for the differences span zero.
- The separate multiscale spatial block bootstrap evaluates metric-difference sensitivity across occupied-coordinate block scales from $10\times10$ to $25\times25$. The pooled interval included zero at $10\times10$ but excluded zero at $15\times15$, $20\times20$, and $25\times25$; Fold 2 remained negative at $15\times15$ and $25\times25$. It is not a cell-level posterior uncertainty interval.

These results show that pooled and fold-level discrimination differ in the specified evaluation. The analysis does not establish which geological factors cause the observed performance differences.

## 5.6 Daly-domain stratification of frozen predictions

Stratifying the frozen out-of-fold predictions by Daly domain provides descriptive performance summaries for the strata containing both classes:

- **NRB_3a (107 positive cells):** V11 domain-stratified ROC-AUC is $0.6824$ compared with $0.5772$ for M5 ($\Delta\mathrm{AUC}=+0.1052$); this is a descriptive difference in frozen OOF scores.
- **NRB_3b (31 positive cells):** M5 domain-stratified ROC-AUC is $0.8780$ compared with $0.6089$ for V11 ($\Delta\mathrm{AUC}=-0.2691$); the comparison does not isolate why the scores differ.
- **Zero-positive strata (CRZ, NKB, SRB, MMSB):** ROC-AUC is undefined because these domains have only zero labels in this modeling frame. This single-class limitation does not establish true geological absence, model failure, or calibration behavior.


This secondary stratification summarizes frozen spatial OOF scores without refitting by domain; it is not LODO or fold-by-domain validation. ROC-AUC is defined only for strata containing both positive and negative cells, and the procedure does not establish domain-level transfer to held-out domains.

## 5.7 Qualified implications for Daly's hypothesis

The empirical findings describe variation across model groups indexed by the supplied Daly-domain labels; this analysis does not independently test or validate Daly's geological framework:

1. **Observed model variation:** Posterior associations and domain-stratified scores vary across the supplied labels. Four labels have no positive observations, and their effects depend more strongly on the shared hierarchical structure. These results do not demonstrate distinct causal regimes or test the geological validity of the labels.
2. **Not supported: A universal invariant distance interpretation:** The evidence does not support interpreting Daly's domains as characterized by fixed, representation-invariant distance thresholds or stationary points. Curvature is representation-dependent and population $D^*$ is unidentifiable; the evaluated ROC-AUC values are similar across the distance representations, while PR-AUC differs.
3. **Synthesis:** In this analysis, the labels provide a grouping structure for partial pooling and descriptive summaries. The fitted component shares do not estimate the relative importance of ore-forming processes or calibrate physical distance thresholds.
