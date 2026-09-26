# Stage 03: Methodology, Study Design and Analytical Framework

## 3.1 Study area and modeling data

The study area encompasses the Central African Copperbelt tract represented by the project spatial modeling grid. Each grid observation corresponds to a discrete spatial cell with centroid coordinates $(x_i, y_i)$ and a binary indicator $Y_i \in \{0, 1\}$ denoting the presence or absence of a sediment-hosted stratiform copper-cobalt deposit. The analysis uses the complete-case modeling frame of 1,872 grid observations, matching the exact filtering and feature availability criteria established in the frozen V11 baseline.

The modeling framework integrates three continuous geological and geophysical predictors alongside host-rock stratigraphy:
1. **Distance to major faults ($D_{fault}$):** Euclidean distance in kilometres from cell centroids to regional-scale faults and structural lineaments.
2. **Distance to lithological contacts ($D_{lith}$):** Euclidean distance in kilometres from cell centroids to mapped stratigraphic contacts.
3. **Bouguer gravity anomaly ($X_{grav}$):** Regional Bouguer gravity values (in mGal) reflecting basement topography, sub-basin architecture, and crustal density variations.
4. **Host lithological classes ($\mathbf{x}_{rock}$):** One-hot encoded categorical indicators representing mapped stratigraphic units.

Within the 1,872-cell modeling frame, exactly 138 deposit-positive observations are documented. The deposit distribution is geographically concentrated across Daly's tectonic domains: 107 deposits occur in NRB_3a and 31 in NRB_3b. The remaining four domains (CRZ, NKB, SRB, and MMSB) contain zero deposit-positive occurrences in this modeling frame. These deposit counts reflect the spatial distribution of documented occurrences within the compiled tract; they do not imply that unmineralized domains lack tectonic significance.

## 3.2 Daly-domain classification and hierarchical stratification

Daly's tectonic domains provide a six-class geological stratification of the Copperbelt:
- **CRZ:** Congo River Zone
- **NKB:** North Kundelungu Basin
- **SRB:** South Roan Basin
- **MMSB:** Mwembeshi Shear Belt
- **NRB_3a:** Northern Roan Basin (Western/Central sector)
- **NRB_3b:** Northern Roan Basin (Eastern/Southeastern sector)

In the hierarchical model, domain labels index domain-varying intercepts and distance coefficients. These domains represent regional tectonic regimes with differing structural evolution, basement involvement, and stratigraphic preservation. 

The domains are not treated as independent cross-validation blocks. A standard classification metric such as ROC-AUC requires the presence of both positive (deposit) and negative (non-deposit) instances in the evaluation sample. Consequently, domain-stratified ROC-AUC can be meaningfully evaluated only in NRB_3a and NRB_3b. In CRZ, NKB, SRB, and MMSB, where deposits are absent ($Y_i = 0$ for all cells), the receiver operating characteristic is undefined. These domains are not "failed" validation tests; rather, the discrimination estimand is mathematically undefined in single-class samples.

## 3.3 Spatial blocking and preprocessing protocol

To eliminate spatial data leakage caused by spatial autocorrelation, the modeling domain is partitioned into four contiguous, along-belt spatial blocks ($B_1, B_2, B_3, B_4$). Validation proceeds by holding out one spatial block at a time:

$$
\mathcal{D}^{(k)}_{train} = \mathcal{D} \setminus B_k, \qquad \mathcal{D}^{(k)}_{test} = B_k \quad (k \in \{1, 2, 3, 4\}).
$$

Strict preprocessing isolation is enforced across folds:
1. **Standardization:** Continuous predictors ($D_{fault}, D_{lith}, X_{grav}$) are standardized to zero mean and unit variance using parameters $(\mu_{train}, \sigma_{train})$ calculated solely from the training partition $\mathcal{D}^{(k)}_{train}$:
   $$
   z_{i} = \frac{X_i - \mu_{train}}{\sigma_{train}}.
   $$
   Test-block observations are transformed using the frozen training parameters.
2. **Quadratic Construction:** Quadratic terms are constructed from the standardized variables: $z_{f, i}^2$ and $z_{l, i}^2$. Polynomial terms are never squared in raw physical units prior to standardization.
3. **Categorical Filtering:** Host lithology indicators are subjected to a train-only support check; categories without representation in $\mathcal{D}^{(k)}_{train}$ are omitted.

This along-belt spatial holdout strategy provides an honest assessment of geographic predictive transferability to unmapped or frontier sectors along the orogen, avoiding the artificial optimism of random-cell cross-validation.

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
\text{offset}_{p, d} &\sim \mathcal{N}(0, 1), \\
\mu_\alpha &\sim \mathcal{N}\left(\mathrm{logit}(\hat{p}_{train}), 1\right), \\
\sigma_\alpha &\sim \mathrm{HalfNormal}(1), \\
\beta_g &\sim \mathcal{N}(0, 1), \\
\boldsymbol{\beta}_{rock} &\sim \mathcal{N}(\mathbf{0}, \mathbf{I}).
\end{aligned}
$$

Domain-specific coefficients are constructed using the non-centered parameterization:
$$
\beta_{p, d} = \mu_p + \sigma_p \cdot \text{offset}_{p, d},
$$
for each distance parameter $p \in \{f_{lin}, f_{sq}, l_{lin}, l_{sq}\}$ and domain $d$. The non-centered parameterization avoids pathological funnel geometries in the posterior geometry when sample sizes and event counts within domains are modest.

### Compact reference baseline (M5)

To benchmark predictive skill, V11 is evaluated alongside a compact, non-hierarchical baseline model (M5). M5 is a global logistic regression containing standardized Bouguer gravity, standardized lithology contact distance, and its quadratic term ($z_g, z_l, z_l^2$), omitting fault distance and domain hierarchy. M5 is trained and tested on the exact same four along-belt spatial partitions.

## 3.5 Distance representation sensitivity framework ($2 \times 2$ factorial grid)

Distance predictors in mineral exploration are strongly right-skewed: most cells are relatively close to faults or contacts, but a long spatial tail extends tens of kilometres into regional basins. To determine whether inferred distance responses, quadratic curvatures, and predictive performance are robust to mathematical representation choices, a pre-registered $2 \times 2$ factorial sensitivity analysis was implemented across all four spatial folds.

Four alternative models were evaluated:
- **Model A (Raw-Quadratic, V11 specification):** Standardized raw distance with linear and quadratic terms: $\beta_1 z + \beta_2 z^2$.
- **Model B (Log-Quadratic):** Standardized log-transformed distance, $x_{\log} = \log(1 + x_{\text{km}})$, standardized within training folds, with linear and quadratic terms: $\beta_1 z_{\log} + \beta_2 z_{\log}^2$.
- **Model C (Raw-Linear):** Standardized raw distance with a linear term only: $\beta_1 z$.
- **Model D (Log-Linear):** Standardized log-transformed distance with a linear term only: $\beta_1 z_{\log}$.

### Distal tail perturbation ($p_{95}$ truncation)

To test whether quadratic curvature is driven by sparse observations in the distant right tail, an 8-fit sensitivity test was conducted by truncating training observations exceeding the 95th percentile ($p_{95}$) of distance to fault or contact. This perturbation removes approximately 141 distal non-deposit grid cells per fold while retaining more than 90% of deposit occurrences, isolating the influence of distal background cells on model curvature.

### Operational robustness criteria for stationary points ($D^*$)

To evaluate whether a derived stationary point ($D^*$) represents a stable feature across distance representations rather than a representation-dependent property of polynomial fitting, three quantitative robustness criteria were pre-registered across all 48 evaluable $(\text{fold} \times \text{domain} \times \text{predictor})$ combinations:
1. **Criterion 1 (Curvature Probability Agreement):** The posterior probability of positive curvature must be consistent between raw and log representations:
   $$
   |\Delta P(\beta_2 > 0)| = |P(\beta_{2, \text{raw}} > 0) - P(\beta_{2, \log} > 0)| < 0.15.
   $$
2. **Criterion 2 (Quantitative Stationary-Point Agreement):** The posterior medians of back-transformed $D^*$ under Model A ($D^*_A$) and Model B ($D^*_B$) must agree within 25% of their mid-point:
   $$
   \frac{|D^*_A - D^*_B|}{0.5(D^*_A + D^*_B)} \le 0.25.
   $$
3. **Criterion 3 (Posterior Support Concentration):** Both models must place the majority of their posterior stationary-point mass within the observed empirical training support:
   $$
   P(D^*_A \in \text{Support}_A) \ge 0.50 \quad \text{and} \quad P(D^*_B \in \text{Support}_B) \ge 0.50.
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

Bouguer gravity enters linearly and has no quadratic term ($\beta_{g^2} \equiv 0$); therefore, it possesses no algebraic $D^*$. This is a property of the model parameterization, not an indication that gravity is less influential in prospectivity discrimination.

## 3.7 Predictor variance contribution decomposition

To quantify how the relative importance of geological predictors varies along the strike of the Copperbelt without relying on causal assumptions, we perform a linear predictor variance decomposition. For each held-out spatial test fold $B_k$, the out-of-fold linear predictor is decomposed into its constituent additive terms:

$$
\eta_i = \alpha_{d(i)} + \eta_{f, i} + \eta_{l, i} + \eta_{g, i} + \eta_{rock, i},
$$
where:
- $\eta_{f, i} = \beta_{f, d(i)} z_{f, i} + \beta_{f^2, d(i)} z_{f, i}^2$ (fault distance component),
- $\eta_{l, i} = \beta_{l, d(i)} z_{l, i} + \beta_{l^2, d(i)} z_{l, i}^2$ (lithology contact distance component),
- $\eta_{g, i} = \beta_g z_{g, i}$ (Bouguer gravity component),
- $\eta_{rock, i} = \mathbf{x}_{rock, i}^{\mathsf T}\boldsymbol{\beta}_{rock}$ (host lithology component).

The sample variance of each additive component across test cells in $B_k$ is computed:
$$
s_j^2 = \mathrm{Var}\left(\{\eta_{j, i}\}_{i \in B_k}\right) \quad \text{for } j \in \{fault, lith, grav, rocks\}.
$$
The relative variance contribution share for predictor $j$ in fold $k$ is defined as:
$$
S_{j, k} = \frac{s_{j, k}^2}{\sum_{m} s_{m, k}^2} \times 100\%.
$$
This metric describes each predictor component's contribution to variation in the decomposed OOF linear predictor within each geographical fold. It provides a model-based description of spatial non-stationarity in predictor contributions; it is not a direct causal or geological importance measure.

## 3.8 Predictive evaluation metrics and secondary domain stratification

Model performance is evaluated across the four along-belt spatial holdout partitions using three complementary metrics:
1. **Receiver Operating Characteristic Area Under the Curve (ROC-AUC):** Measures the ranking discrimination across all classification thresholds, independent of class prevalence.
2. **Precision-Recall Area Under the Curve (PR-AUC):** Evaluates precision across recall levels, providing a stringent assessment under severe class imbalance (138 deposits out of 1,872 cells, ~7.4% base rate).
3. **Brier Score:** Evaluates mean squared probability calibration error: $\frac{1}{N} \sum_{i=1}^N (\hat{p}_i - Y_i)^2$.

To quantify uncertainty in predictive metrics, 1,000-iteration spatial block bootstrap distributions are computed across varying spatial block scales ($10\times 10$, $15\times 15$, $20\times 20$, and $25\times 25$ grid units).

### Secondary domain stratification protocol

Following the generation of primary out-of-fold predictions from the four spatial holdout models, predictions are matched to their corresponding Daly domain labels. This secondary stratification groups the frozen spatial OOF predictions by domain to assess regional performance differences. 

Critically:
- The model is **never refit** by Daly domain.
- This is **not** Leave-One-Daly-Domain-Out (LODO) validation, nor is it a Fold $\times$ Domain cross-validation scheme.
- ROC-AUC is reported exclusively for domains containing both deposits and non-deposits (NRB_3a and NRB_3b), with 95% bootstrap percentile intervals.
