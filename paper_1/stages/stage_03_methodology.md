# Stage 03: Methodology, Study Design and Analytical Framework

## 3.1 Study area and modeling data

The study area encompasses the Central African Copperbelt tract represented by the project spatial modeling grid. Each observation is one $5\,\mathrm{km}\times5\,\mathrm{km}$ cell with centroid coordinates $(x_i,y_i)$ in WGS 84 / UTM Zone 35S (EPSG:32735) and a binary label $Y_i\in\{0,1\}$. The V11 complete-case frame contains 1,872 cells after the filtering described below.

The prepared modeling table supplies three continuous predictors alongside host lithology:
1. **Distance to mapped faults ($D_{fault}$):** Precomputed distance to supplied structural linework, stored on the projected spatial basis; V11 consumes the field and does not reconstruct the GIS distance.
2. **Distance to lithological contacts ($D_{lith}$):** Precomputed distance to supplied contact linework; the original geological map and its scale are not identified in the inspected project files.
3. **Bouguer gravity anomaly ($X_{grav}$):** A prepared regional gravity value. An executed notebook reads a local text grid, transforms its coordinates from EPSG:4326 to EPSG:32735, and assigns the nearest grid-point value; the original grid and publisher metadata are unavailable.
4. **Host lithological class ($\mathbf{x}_{rock}$):** The V11 predictor is the prepared field `litho_contact_litho_class`, not the separate `Africa_Surface_Lithology.tif` raster.

The V11 fit reads prepared fields from `data/copperbelt_training_v5_with_tectonic_domain.csv`; it does not reproject source layers or calculate raw GIS distances. The supplied fault layer's `source` attribute identifies `Selley_et_al_2005_Fig1` for the mapped structural features. The source map for the contact linework could not be identified. The input `domain` strings are mapped to six model labels by substring rules; V11 does not perform point-in-polygon assignment.

Within the 1,872-cell modeling frame, 138 cells have a positive label and 1,734 are background cells. The input table contains 107 positive cells in NRB_3a and 31 in NRB_3b; CRZ, NKB, SRB, and MMSB have no positive labels in this frame. These counts describe the compiled occurrence inventory, not the true absence of mineralization in the four single-class domains.

The available base-grid table contains 210 positive rows across 139 unique cell IDs, including 71 repeated positive rows. The executed feature notebook deduplicates positive base-grid records by cell ID, retaining the first row, before merging the binary presence flag onto unique grid cells. Thus repeated rows do not become repeated V11 observations. The original occurrence inventory is unavailable, so the repeated rows cannot be classified further as separate deposits, overlapping footprints, or duplicate records. The input modeling table has 139 positive labels before V11 filtering; the complete-case/domain filter leaves 138.

## 3.2 Daly-domain classification and hierarchical stratification

Daly's tectonic domains provide a six-class geological stratification of the Copperbelt:
- **CRZ:** Congo River Zone
- **NKB:** North Kundelungu Basin
- **SRB:** South Roan Basin
- **MMSB:** Mwembeshi Shear Belt
- **NRB_3a:** Northern Roan Basin (Western/Central sector)
- **NRB_3b:** Northern Roan Basin (Eastern/Southeastern sector)

In the hierarchical model, domain labels index domain-varying intercepts and distance coefficients. These domains represent regional tectonic regimes with differing structural evolution, basement involvement, and stratigraphic preservation. 

The domains are not treated as independent cross-validation blocks. A standard classification metric such as ROC-AUC requires both positive and negative labels in the evaluation sample. Consequently, domain-stratified ROC-AUC can be evaluated only in NRB_3a and NRB_3b. In CRZ, NKB, SRB, and MMSB, all labels are zero, so ROC-AUC is undefined. These are single-class strata, not failed validation tests or evidence of true geological absence.

## 3.3 Spatial blocking and preprocessing protocol

The 1,872 cell centroids are projected onto the first principal component, sorted by projection, and split into four approximately equal contiguous along-belt folds of 468 cells each ($B_1, B_2, B_3, B_4$). Validation holds out one fold at a time:

$$
\mathcal{D}^{(k)}_{train} = \mathcal{D} \setminus B_k, \qquad \mathcal{D}^{(k)}_{test} = B_k \quad (k \in \{1, 2, 3, 4\}).
$$

Fold-specific preprocessing is fitted on the training partition and applied to the held-out fold:
1. **Standardization:** Continuous predictors ($D_{fault}, D_{lith}, X_{grav}$) are standardized to zero mean and unit variance using parameters $(\mu_{train}, \sigma_{train})$ calculated solely from the training partition $\mathcal{D}^{(k)}_{train}$:

   $$
   z_{i} = \frac{X_i - \mu_{train}}{\sigma_{train}}.
   $$

   Test-block observations are transformed using the frozen training parameters.
2. **Quadratic Construction:** Quadratic terms are constructed from the standardized variables: $z_{f, i}^2$ and $z_{l, i}^2$. Polynomial terms are never squared in raw physical units prior to standardization.
3. **Categorical Filtering:** A host-lithology one-hot category is retained only when the training partition contains at least one positive and at least one negative cell in that category. This support check uses training labels only.

This design assesses geographic transfer among the four specified sectors and reduces local train-test overlap relative to random-cell splitting. It does not guarantee independence across fold boundaries or establish performance in all unmapped or frontier sectors.

Before fitting, V11 excludes rows missing `centroid_x`, `centroid_y`, `domain`, `litho_contact_litho_class`, `distance_to_fault`, `distance_to_lithology_contact`, or `bouguer`, then removes domain strings that do not map to one of the six recognized labels. `deposit_present` is not included in the `dropna` list; the supplied modeling table contains labels for retained rows. These operations produce the 1,872-cell frame described above. Continuous predictors are standardized using training-fold statistics, and quadratic terms are formed after standardization.

## 3.4 V11 hierarchical Bayesian model specification

V11 is formulated as a hierarchical Bayesian logistic regression with partial pooling across Daly domains. For observation $i$ located in Daly domain $d(i) \in \{1, \dots, 6\}$:

$$
Y_i \sim \mathrm{Bernoulli}(p_i), \qquad \mathrm{logit}(p_i) = \eta_i,
$$

with the linear predictor defined as:

$$
\eta_i = \alpha_{d(i)} + \beta_{f, d(i)} z_{f, i} + \beta_{f^2, d(i)} z_{f, i}^2 + \beta_{l, d(i)} z_{l, i} + \beta_{l^2, d(i)} z_{l, i}^2 + \beta_g z_{g, i} + \mathbf{x}_{rock, i}^{\mathsf T}\boldsymbol{\beta}_{rock}.
$$

The model architecture specifies:
- **Hierarchical Distance Effects:** Linear and quadratic coefficients for fault distance ($\beta_{f, d}, \beta_{f^2, d}$) and lithology contact distance ($\beta_{l, d}, \beta_{l^2, d}$) vary by Daly domain, partially pooled toward belt-wide population distributions.
- **Global Geophysical & Lithological Effects:** Bouguer gravity anomaly enters through a single global linear coefficient $\beta_g$, with no quadratic term. The retained host-lithology indicators enter via a global parameter vector $\boldsymbol{\beta}_{rock}$.
- **Hierarchical Intercept:** Base-rate intercepts $\alpha_d$ vary across domains around a population mean anchored to the empirical training log-odds.

### Prior distributions and sampling parameterization

Priors are specified weakly informative to regularize inference without dominating the likelihood:

$$
\begin{aligned}
\mu_{f, lin}, \mu_{f, sq}, \mu_{l, lin}, \mu_{l, sq} &\sim \mathcal{N}(0, 1), \\
\sigma_{f, lin}, \sigma_{f, sq}, \sigma_{l, lin}, \sigma_{l, sq} &\sim \mathrm{HalfNormal}(1), \\
\mathrm{offset}_{p, d} &\sim \mathcal{N}(0, 1), \\
\mu_\alpha &\sim \mathcal{N}\left(\mathrm{logit}(\hat{p}_{train}), 1\right), \\
\sigma_\alpha &\sim \mathrm{HalfNormal}(1), \\
\beta_g &\sim \mathcal{N}(0, 1), \\
\boldsymbol{\beta}_{rock} &\sim \mathcal{N}(\mathbf{0}, \mathbf{I}).
\end{aligned}
$$

Domain-specific coefficients are constructed using the non-centered parameterization:

$$
\beta_{p, d} = \mu_p + \sigma_p \cdot \mathrm{offset}_{p, d},
$$

for each distance parameter $p \in \{f_{lin}, f_{sq}, l_{lin}, l_{sq}\}$ and domain $d$. The non-centered parameterization avoids pathological funnel geometries in the posterior geometry when sample sizes and event counts within domains are modest.

### Posterior prediction and uncertainty scope

Sampling uses NUTS with two chains, 2,500 tuning steps, and 1,500 retained draws per chain (3,000 parameter draws per fold), `target_accept=0.99`, one core, and random seed 42. For each held-out cell and posterior draw, the prediction code calculates $p_i^{(s)}=\mathrm{logit}^{-1}(\eta_i^{(s)})$ and stores the mean $\bar p_i=3000^{-1}\sum_s p_i^{(s)}$ in the frozen OOF table. The draw-wise probabilities represent posterior uncertainty in the conditional probability given model parameters and predictors; the workflow does not sample future Bernoulli outcomes. A reconstruction from the four frozen traces reproduced all 1,872 OOF means with maximum absolute difference approximately $9.97\times10^{-17}$ and did not refit the model.

### Compact reference baseline (M5)

To benchmark predictive skill, V11 is evaluated alongside a compact, non-hierarchical baseline model (M5). M5 is a global logistic regression containing standardized Bouguer gravity, standardized lithology contact distance, and its quadratic term ($z_g, z_l, z_l^2$), omitting fault distance and domain hierarchy. M5 is trained and tested on the exact same four along-belt spatial partitions.

## 3.5 Distance representation sensitivity framework ($2 \times 2$ factorial grid)

Distance predictors in mineral exploration are strongly right-skewed: most cells are relatively close to faults or contacts, but a long spatial tail extends tens of kilometres into regional basins. To determine whether inferred distance responses, quadratic curvatures, and predictive performance are robust to mathematical representation choices, a pre-registered $2 \times 2$ factorial sensitivity analysis was implemented across all four spatial folds.

Four alternative models were evaluated:
- **Model A (Raw-Quadratic, V11 specification):** Standardized raw distance with linear and quadratic terms: $\beta_1 z + \beta_2 z^2$.
- **Model B (Log-Quadratic):** Standardized log-transformed distance, $x_{\log} = \log(1 + x_{\mathrm{km}})$, standardized within training folds, with linear and quadratic terms: $\beta_1 z_{\log} + \beta_2 z_{\log}^2$.
- **Model C (Raw-Linear):** Standardized raw distance with a linear term only: $\beta_1 z$.
- **Model D (Log-Linear):** Standardized log-transformed distance with a linear term only: $\beta_1 z_{\log}$.

### Distal tail perturbation ($p_{95}$ truncation)

To assess sensitivity to observations in the distant right tail, an 8-fit perturbation truncated training observations exceeding the 95th percentile ($p_{95}$) of fault or contact distance. It removes approximately 141 distal non-deposit cells per fold while retaining more than 90% of positive cells. This changes the fitted data and scores but does not isolate the mechanism behind that change.

### Operational robustness criteria for stationary points ($D^*$)

To evaluate whether a derived stationary point ($D^*$) represents a stable feature across distance representations rather than a representation-dependent property of polynomial fitting, three quantitative robustness criteria were pre-registered across all 48 evaluable $(\mathrm{fold} \times \mathrm{domain} \times \mathrm{predictor})$ combinations:
1. **Criterion 1 (Curvature Probability Agreement):** The posterior probability of positive curvature must be consistent between raw and log representations:

   $$
   |\Delta \Pr(\beta_2 > 0)| = |\Pr(\beta_{2, \mathrm{raw}} > 0) - \Pr(\beta_{2, \mathrm{log}} > 0)| < 0.15.
   $$

2. **Criterion 2 (Quantitative Stationary-Point Agreement):** The posterior medians of back-transformed $D^*$ under Model A ($D^*_A$) and Model B ($D^*_B$) must agree within 25% of their mid-point:

   $$
   \frac{|D^*_A - D^*_B|}{0.5(D^*_A + D^*_B)} \le 0.25.
   $$

3. **Criterion 3 (Posterior Support Concentration):** Both models must place the majority of their posterior stationary-point mass within the observed empirical training support:

   $$
   \Pr(D^*_A \in \mathrm{Support}_A) \ge 0.50 \quad \mathrm{and} \quad \Pr(D^*_B \in \mathrm{Support}_B) \ge 0.50.
   $$


A turning-point diagnostic is classified as representation-robust if and only if all three criteria are satisfied simultaneously.

## 3.6 Mathematical formulation of the stationary-point diagnostic ($D^*$)

For a quadratic distance component on the standardized scale,

$$
\eta(z) = \alpha + \beta_{lin} z + \beta_{sq} z^2,
$$

the first and second derivatives with respect to $z$ are:

$$
\frac{d\eta}{dz} = \beta_{lin} + 2\beta_{sq} z, \qquad \frac{d^2\eta}{dz^2} = 2\beta_{sq}.
$$

For any posterior MCMC draw where $\beta_{sq} \neq 0$, an algebraic stationary point exists at:

$$
z^* = -\frac{\beta_{lin}}{2\beta_{sq}}.
$$

This stationary point is back-transformed to physical kilometres using the training fold's standardization parameters:

$$
D^* = \mu_{train} + \sigma_{train} z^*.
$$

### Evidential separation of diagnostic quantities

To prevent misinterpreting algebraic diagnostics as physical exploration targets, the analytical framework enforces strict separation among six distinct quantities:
1. **Curvature sign and strength:** Evaluated by $P(\beta_{sq} > 0)$ and the magnitude of $\beta_{sq}$. A positive coefficient indicates convex (minimum-shaped) curvature on the logit scale; a negative coefficient indicates concave (maximum-shaped) curvature.
2. **Algebraic stationary-point existence:** A finite $z^*$ exists for every draw where $\beta_{sq} \neq 0$. However, when posterior mass spans zero ($\beta_{sq} \approx 0$), division by values near zero produces an explosive Cauchy-like distribution with extreme, unidentifiable tails.
3. **Empirical support check:** A stationary point is within empirical support if $D_{min, train} \le D^* \le D_{max, train}$. Posterior draws falling outside this interval represent mathematical extrapolation beyond observed data.
4. **Posterior concentration:** Evaluated via the 95% highest posterior density interval or credible interval of $D^*$. Broad intervals spanning negative distances or hundreds of kilometres indicate non-identifiability.
5. **Posterior-mean response shape:** Evaluated by plotting the posterior expectation $\mathbb{E}[P(Y=1 \mid D)]$ over the observed domain. The extremum of this expected curve need not coincide with the median of draw-level $D^*$ ratios due to Jensen's inequality and ratio skewness.
6. **Predictive discrimination:** Evaluated by out-of-fold ROC-AUC, PR-AUC, and Brier score. A model may achieve high discrimination regardless of whether its quadratic stationary point is stable.

Bouguer gravity enters linearly and has no quadratic term  ($\beta_{g^2} \equiv 0$); therefore, it possesses no algebraic $D^*$. This is a property of the model parameterization, not an indication that gravity is less influential in prospectivity discrimination.

## 3.7 SD-normalized shares of OOF linear-predictor components

For each held-out fold, the implementation computes posterior-mean additive components for fault distance, contact distance, Bouguer gravity, and host lithology. The intercept is omitted from the four-component share denominator. Let $c_{m,i}$ denote the posterior-mean contribution of component $m$ for held-out cell $i$ in fold $k$. Its dispersion across the $N_k$ held-out cells is:

$$
\mathrm{SD}_{m,k}=\sqrt{\frac{1}{N_k}\sum_{i\in B_k}(c_{m,i}-\bar{c}_{m,k})^2},\qquad m\in\{fault,lith,grav,rocks\}.
$$

The reported share is:

$$
S_{m,k}=\frac{\mathrm{SD}_{m,k}}{\sum_{j\in\{fault,lith,grav,rocks\}}\mathrm{SD}_{j,k}}\times100\%.
$$

These are SD-normalized shares of dispersion in the decomposed posterior-mean OOF linear predictor. They are not proportions of outcome variance explained, Shapley values, or causal geological importance measures.

## 3.8 Predictive evaluation metrics and secondary domain stratification

Model performance is evaluated across the four along-belt spatial holdout partitions using three complementary metrics:
1. **Receiver Operating Characteristic Area Under the Curve (ROC-AUC):** Measures the ranking discrimination across all classification thresholds, independent of class prevalence.
2. **Precision-Recall Area Under the Curve (PR-AUC):** Evaluates precision across recall levels, providing a stringent assessment under severe class imbalance (138 deposits out of 1,872 cells, ~7.4% base rate).
3. **Brier Score:** Evaluates mean squared probability calibration error: $\frac{1}{N} \sum_{i=1}^N (\hat{p}_i - Y_i)^2$.

The primary pooled and fold-level metric intervals use 1,000 row-bootstrap resamples of fixed OOF scores. Separately, the multiscale spatial block bootstrap resamples occupied coordinate blocks at $10\times10$, $15\times15$, $20\times20$, and $25\times25$ grid-unit scales to assess robustness of performance differences. These procedures quantify metric uncertainty and are distinct from posterior uncertainty in cell-conditional probabilities.

### Secondary domain stratification protocol

Following the generation of primary out-of-fold predictions from the four spatial holdout models, predictions are matched to their corresponding Daly domain labels. This secondary stratification groups the frozen spatial OOF predictions by domain to assess regional performance differences. 

Critically:
- The model is **never refit** by Daly domain.
- This is **not** Leave-One-Daly-Domain-Out (LODO) validation, nor is it a Fold $\times$ Domain cross-validation scheme.
- ROC-AUC is reported exclusively for domains containing both deposits and non-deposits (NRB_3a and NRB_3b), with 95% bootstrap percentile intervals.
