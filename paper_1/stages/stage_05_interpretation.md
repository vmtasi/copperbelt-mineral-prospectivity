# Stage 05: Interpretation

## 5.1 Multivariate predictor structure and conditional inference

The V11 model must be interpreted fundamentally as a multivariate mineral prospectivity system rather than a univariate distance-vectoring exercise. By integrating distance to fault, distance to lithology contact, Bouguer gravity anomaly, and categorical host stratigraphy into a single linear predictor, the model evaluates each predictor conditionally:

$$
\eta_i = \alpha_{d(i)} + \eta_{f, i} + \eta_{l, i} + \eta_{g, i} + \eta_{rock, i}.
$$

This joint formulation carries critical implications for geological interpretation:
1. **Conditional Parameter Interpretation:** Any fitted distance coefficient reflects the marginal association of structural or lithological proximity *given* the concurrent effects of regional gravity and host-rock lithology. Distance responses must never be interpreted as isolated physical mechanisms operating in the absence of regional basin configuration or stratigraphic reactivity.
2. **Role of Bouguer Gravity:** Bouguer gravity enters as a global standardized linear predictor ($\beta_g$). While the absence of a quadratic gravity term precludes the calculation of an algebraic stationary point ($D^*$), this reflects a deliberate modeling choice for regional background stabilization, not an empirical finding that gravity is secondary to proximity predictors.
3. **Host Stratigraphy Adjustment:** Retained lithological units account for local host-rock suitability, ensuring that proximity parameters capture spatial transport and gradient effects rather than merely rediscovering favorable ore-bearing formations.

## 5.2 Spatial heterogeneity in predictor importance

The linear predictor variance decomposition provides evidence that the contributions of model components to the decomposed OOF linear predictor vary along the strike of the Central African Copperbelt. These shares are not direct causal or geological importance measures. Rather than establishing a single invariant mechanism, the model captures a geographic shift in predictor contributions across spatial folds:

- **Fold 1 ($S_f = 86.5\%$):** In the northwestern fold, the fault-distance component accounts for over six-sevenths of the variance in the decomposed OOF linear predictor ($\sigma = 7.36$). This contribution pattern is compatible with structural architecture being particularly informative in the evaluated fold; it does not identify a causal mechanism. Lithology distance ($11.1\%$) and Bouguer gravity ($2.0\%$) contribute modest marginal variance.
- **Transitional Multi-Factor Architecture (Fold 2, $S_f = 50.3\%$, $S_l = 23.9\%$, $S_{rock} = 20.3\%$):** In the north-central fold, the fault-distance component contributes the largest share, while host-rock lithology and contact proximity contribute 44% of the decomposed linear-predictor variance combined.
- **Balanced Co-Dominance (Fold 3, $S_f = 33.1\%$, $S_l = 28.2\%$, $S_{rock} = 23.4\%$, $S_g = 15.3\%$):** In the central-southern fold, variance in the decomposed linear predictor is distributed across faults, contacts, host stratigraphy and Bouguer gravity.
- **Geophysical Co-Dominance (Fold 4, $S_f = 34.9\%$, $S_g = 31.1\%$, $S_l = 17.7\%$, $S_{rock} = 16.3\%$):** In the southern fold, regional Bouguer gravity contributes nearly a third of the decomposed linear-predictor variance alongside fault distance. In this sector, the variation of the fault-distance component is lower ($\sigma = 0.65$ compared to $7.36$ in Fold 1).

These shifts must be understood as contribution patterns in the decomposed OOF linear predictor rather than as simple causal effects. As an interpretation, they are compatible with structural architecture being particularly informative in the north and with regional sub-basin geometry and stratigraphic contacts contributing more in the southeastern evaluated folds. The present analysis does not identify the underlying causal mechanism.

## 5.3 Distance-response representation sensitivity: recurring association versus representation-dependent form

The $2 \times 2$ factorial sensitivity analysis resolves a central scientific question: which aspects of the distance response recur across representations, and which are representation-dependent?

### 1. Recurring predictive association across representations
Across all four evaluated models (Model A raw-quadratic, Model B log-quadratic, Model C raw-linear, Model D log-linear), distance to structural faults and distance to lithological contacts consistently exhibit negative linear relationships with mineralization log-odds. Whether evaluated on raw or logarithmic scales, prospectivity monotonically decreases away from structural and lithological contacts across the evaluated spatial folds. This is a recurring predictive association across the evaluated representations, with magnitude and transferability varying by spatial fold.

This association is not uniform in every domain-specific posterior: NRB_3b fault-distance linear coefficients are positive in the reported fold summaries. The recurring association therefore describes the evaluated representations in aggregate, not a uniformly negative relationship across the entire belt.

### 2. Sensitivity of quadratic curvature
In contrast to the monotonic distance association, quadratic curvature is highly sensitive to predictor transformation:
- In raw physical distance space (Model A), the distribution of cell distances is severely right-skewed: most deposits lie within $0\text{--}15\text{ km}$ of contacts, while the background basin grid extends to $60\text{--}80\text{ km}$. A quadratic polynomial fitted to this skewed distribution bends upward in the sparse distal tail to avoid overly penalizing distal non-deposit cells.
- When the distance predictor is log-transformed ($\log(1 + x_{\text{km}})$, Model B), the extreme right tail is compressed, naturally linearizing the relationship on the logit scale. Consequently, quadratic curvature parameters collapse toward zero ($P(\beta_{l^2} > 0)$ drops from $\ge 0.993$ to near $0.50$), and fault curvature in NRB_3a inverts from convex to concave.

### 3. Model parsimony and predictive transferability
The out-of-fold predictive results show no demonstrable predictive advantage for quadratic terms over simpler linear specifications. The mean spatial OOF ROC-AUC across models is virtually identical ($0.671\text{--}0.686$). Furthermore, hierarchical log-linear Model D achieves the highest mean PR-AUC ($0.1446$) across spatial folds—substantially outperforming raw-quadratic Model A ($0.1069$)—while raw-linear Model C achieves an OOF ROC-AUC of $0.9041$ in Fold 1. Under the evaluated spatial comparisons, the linear and log-linear representations are more parsimonious and at least comparably effective for mineral prospectivity mapping.

## 5.4 Diagnostic status and unidentifiability of $D^*$

Historically, the algebraic stationary point $D^* = \mu_{train} + \sigma_{train} z^*$ was hypothesized to reflect a characteristic distance or optimal structural trap offset. The completed empirical evidence does not support this interpretation:

1. **Failure of Operational Robustness Criteria:** The pre-registered robustness protocol revealed that Model A and Model B failed to agree on $D^*$ in **100% of evaluable cases (0/48)**, with an average relative discrepancy of $67.2\%$. Curvature probabilities were discordant in 73% of cases. Overall robustness was satisfied in 0 of 48 cases ($0\%$).
2. **Population Singularity:** Reconstructing population-level distributions from the frozen V11 posterior traces demonstrates that population quadratic parameters heavily overlap zero ($P(\mu_{f, sq} > 0) = 38.4\%$; $P(\mu_{l, sq} > 0) = 74.3\%$). When posterior curvature mass crosses zero, the algebraic ratio $z^* = -\mu_{lin} / (2\mu_{sq})$ suffers from denominator singularity, producing explosive Cauchy-like credible intervals spanning $[-279, +366]\text{ km}$. A belt-wide population $D^*$ is mathematically unidentifiable.
2. **Population Singularity:** Reconstructing population-level distributions from the frozen V11 posterior traces demonstrates that population quadratic parameters heavily overlap zero ($P(\mu_{f, sq} > 0) = 38.4\%$; $P(\mu_{l, sq} > 0) = 74.3\%$). When posterior curvature mass crosses zero, the algebraic ratio $z^* = -\mu_{lin} / (2\mu_{sq})$ suffers from denominator singularity, producing explosive Cauchy-like credible intervals spanning $[-279, +366]\text{ km}$. A belt-wide population $D^*$ is mathematically unidentifiable. This population-level unidentifiability is distinct from the domain/fold stationary points, which are representation-dependent and unstable rather than automatically meaningless in every local setting.
3. **Fold-Level Instability:** At the domain level, apparent stationary points swing dramatically across spatial folds (e.g., NRB_3a fault $D^*$ moving from $-18.0\text{ km}$ in Fold 2 to $+46.6\text{ km}$ in Fold 3). In unmineralized domains, $D^*$ is completely undefined.
4. **Tail Truncation Sensitivity:** Removing distal background cells ($p_{95}$ truncation) causes Fold 1 predictive discrimination to collapse from $0.817 \to 0.595$; the fitted predictions and discrimination are sensitive to the distal tail under both Model A and Model B.

Consequently, $D^*$ must be interpreted strictly as an unstable mathematical diagnostic of a specific polynomial representation, not as a physical trap distance, an optimal exploration vector, or a universal metallogenic property.

## 5.5 Predictive transferability and geographic non-stationarity

Evaluating models under four-fold along-belt spatial holdout validation provides an honest, rigorous measure of prospective transferability:

- **Belt-Wide Pooled Transferability ($\Delta\text{AUC} = +0.164$):** When out-of-fold predictions are pooled across all four spatial folds, V11 achieves an OOF ROC-AUC of $0.689$ compared to $0.525$ for the compact M5 baseline. The pooled OOF results show higher discrimination for V11 than M5 under the specified comparison ($P(\Delta > 0) = 1.000$). Because V11 and M5 differ in multiple architectural and predictor components, this comparison does not isolate the contribution of any single component.
- **Regional Divergence and Geographic Non-Stationarity:** The pooled advantage masks pronounced regional contrasts:
  - In **Fold 3**, V11 substantially outperforms M5 ($\Delta\text{AUC} = +0.091$ [$+0.024$, $+0.159$]), demonstrating superior predictive transfer in the central-southern transitional regime.
  - In **Fold 2**, M5 outperforms V11 ($\Delta\text{AUC} = -0.136$ [$-0.250$, $-0.020$]), showing that a compact global specification can perform well in specific local structural contexts.
  - In **Folds 1 and 4**, performance between the two models is comparable within 95% bootstrap intervals.
- **Stability Across Spatial Block Scales:** The spatial block bootstrap supports this pattern across the evaluated bootstrap block dimensions ($10\times 10$ to $25\times 25$ units), indicating that the observed regional divergence persists within the evaluated spatial framework.

These results emphasize that predictive performance must be evaluated regionally. A model with high pooled skill can exhibit substantial variability in local transferability depending on which geological factors govern mineralization in the target sector.

## 5.6 Daly-domain stratification of frozen predictions

Stratifying the frozen out-of-fold predictions by Daly domain highlights the geological divergence between the two mineralized tectonic sectors:

- **NRB_3a (Western/Central Sector, 107 deposits):** V11 achieves an OOF ROC-AUC of $0.6617$ compared to $0.5358$ for M5 ($\Delta\text{AUC} = +0.1259$), indicating higher discrimination for V11 under this frozen OOF comparison within the evaluated domain.
- **NRB_3b (Eastern/Southeastern Sector, 31 deposits):** M5 achieves an OOF ROC-AUC of $0.8669$ compared to $0.5838$ for V11 ($\Delta\text{AUC} = -0.2831$). In this sector, the simple global combination of contact proximity and Bouguer gravity captures the local spatial gradient more effectively than the partially pooled domain parameters.
- **Unmineralized Domains (CRZ, NKB, SRB, MMSB):** ROC-AUC is undefined because these domains contain zero documented deposits in the modeling frame. This is a mathematical consequence of evaluating binary ranking metrics in homogeneous negative partitions. It does not establish model failure or calibration behavior in those domains.

This secondary stratification provides valuable geological context without violating spatial validation protocols: the model was never refit on domains, and no domain-specific leakage was introduced.

## 5.7 Qualified implications for Daly's hypothesis

The empirical findings offer qualified support for aspects of Daly's tectonic framework while redefining how that framework should be statistically understood:

1. **Support for Tectonic Heterogeneity:** The data strongly support Daly's core premise that the Central African Copperbelt is not a monolithic metallogenic province. Tectonic domains exhibit distinct base rates, varying structural versus stratigraphic controls, and differing predictive transferability. The hierarchical Bayesian model successfully operationalizes this heterogeneity through partial pooling.
2. **Rejection of Universal Invariant Distances:** The data decisively refute the notion that Daly's domains are characterized by invariant, domain-specific distance thresholds or "optima" ($D^*$). Curvature is representation-dependent, population $D^*$ is unidentifiable, and predictive transferability is equally well captured by linear and log-linear formulations.
3. **Synthesis:** The primary value of Daly's tectonic classification lies in recognizing **spatial non-stationarity in the relative importance of ore-forming processes**—structural fluid flow, stratigraphic trapping, and regional basin architecture—rather than in calibrating fixed geometric buffer zones.
