# Stage 3: Methodology

## 3.1 Deposit Data and Spatial Representation

- **Response Variable:** Binary indicator of Cu-Co mineralization,
  \(Y \in \{0,1\}\).

- **Dataset:** The analysis was conducted on the complete-case spatial
  dataset after exclusion of observations with missing values in the
  response, spatial coordinates, domain information, lithological class,
  or predictor variables.

- **Positive Observations:** The evaluated dataset contained 138
  positive mineralization observations together with the corresponding
  non-deposit grid cells across the evaluated Copperbelt region.

- **Spatial Representation:** Each observation was represented by its
  spatial centroid coordinates \((x,y)\), allowing spatial partitioning
  and geographically structured validation.

- **Missingness:** The Phase 6 missingness audit was used to assess
  whether regional predictive differences could plausibly be attributed
  to differential missingness in the predictor data.

## 3.2 Geological and Geophysical Predictors

A deliberately constrained predictor space was used to isolate spatial
heterogeneity in predictive relationships from increases in algorithmic
complexity or feature quantity.

The core predictors were:

- **Bouguer gravity anomaly** (\(X_{grav}\))
- **Distance to lithological contact** (\(X_{lith}\))
- **Distance to major fault structures** (\(X_{fault}\))

Predictors were standardized independently within each training fold.
The corresponding test observations were transformed using parameters
estimated exclusively from the training data, preventing information
from the held-out region from entering model fitting.

Quadratic terms were introduced for selected distance variables to allow
non-linear relationships between mineralization probability and
geological proximity.

## 3.3 Spatial Partitioning and Validation Framework

The study region was partitioned into four along-belt spatial folds
(Folds 1–4) using the predefined spatial-fold procedure.

For each iteration, one spatial block was withheld as the test region
while the remaining three blocks constituted the training dataset.

This design was adopted instead of random cell-level cross-validation
because neighboring spatial observations may share geological and
environmental structure. Random partitioning could therefore allow
spatially proximate observations from the same geological environment
to appear in both training and testing sets.

The resulting evaluation therefore focuses on geographic transferability
rather than interpolation among randomly separated cells.

## 3.4 Competing Model Formulations

### M5: Global Baseline

M5 represents the selected compact global logistic-regression
formulation. A single set of coefficients is estimated for the entire
training region:

\[
P(Y=1|X)=
\operatorname{logit}^{-1}
\left(
\beta_0+
\beta_{grav}X_{grav}+
\beta_{lith}X_{lith}+
\beta_{lith^2}X_{lith}^{2}
\right).
\]

The coefficients are therefore assumed to be spatially invariant across
the evaluation region.

### V11: Domain-Aware Hierarchical Model

V11 is a Bayesian hierarchical logistic-regression formulation in which
the intercept and selected predictor effects are allowed to vary across
Daly tectonic domains.

For domain \(d\), the model takes the general form:
# Stage 03: Methodology, Study Design and Analytical Framework

## 3.1 Study area and modeling data

The study concerns the Central African Copperbelt tract represented by the project modeling grid. Each grid observation has centroid coordinates $(x_i,y_i)$ and a binary response $Y_i\in\{0,1\}$ indicating deposit presence. The analysis uses the complete-case modeling population after applying the same response, coordinate, domain, lithological-class and predictor availability requirements used by the V11 implementation.

The predictor set is deliberately constrained to three geological or geophysical variables: distance to major faults ($D_{fault}$), distance to lithological contacts ($D_{lith}$), and Bouguer gravity anomaly ($X_{grav}$). The complete modeling frame contains 138 deposit-positive observations: 107 in NRB_3a and 31 in NRB_3b. CRZ, NKB, SRB and MMSB contain zero deposit-positive observations in this frame. These counts describe the available modeling data; they do not imply that the zero-positive domains lack geological importance.

## 3.2 Daly-domain classification

Daly's geological domains are used as a geological stratification of the Copperbelt tract. The six fixed labels are CRZ, NKB, SRB, MMSB, NRB_3a and NRB_3b. The classification is used to examine whether fitted behavior and already-generated predictive discrimination are consistent across geological settings.

The domains are not treated as six independent predictive-validation datasets. In particular, a conventional ROC-AUC requires both deposit-positive and non-deposit observations. Consequently, domain-level ROC-AUC is estimable in the current modeling frame for NRB_3a and NRB_3b, while the four zero-positive domains cannot yield a conventional positive-versus-negative ROC-AUC. They are not considered failed validation domains; the estimand is undefined there under this data composition.

## 3.3 Predictor preprocessing and spatial folds

For each of four predefined along-belt spatial folds, one block $B_k$ is held out and the remaining observations form the training set:

\[
\mathcal{D}^{(k)}_{train}=\mathcal{D}\setminus B_k,\qquad
\mathcal{D}^{(k)}_{test}=B_k.
\]

Each continuous predictor is standardized using parameters estimated only from $\mathcal{D}^{(k)}_{train}$. The held-out observations are transformed with those training parameters. Quadratic distance terms are constructed after standardization. This preserves the spatial separation between training and test observations and prevents test-fold information from entering preprocessing.

The four-fold along-belt design is the primary validation framework. It evaluates geographic transferability rather than random-cell interpolation, where neighboring cells could place closely related geological environments in both training and test data.

## 3.4 V11 hierarchical Bayesian model

V11 is a Bayesian hierarchical logistic-regression model. For observation $i$ in spatial/geological unit $d(i)$, its linear predictor has the form

\[
\begin{aligned}
\eta_i ={}& \alpha_{d(i)}
 + \beta_{f, d(i)}z_{f,i}
 + \beta_{f^2,d(i)}z_{f,i}^{2}\\
&+ \beta_{l,d(i)}z_{l,i}
 + \beta_{l^2,d(i)}z_{l,i}^{2}
 + \beta_g z_{g,i}
 + \mathbf{x}_{rock,i}^{\mathsf T}\boldsymbol{\beta}_{rock},
\end{aligned}
\]

with

\[
P(Y_i=1\mid\eta_i)=\operatorname{logit}^{-1}(\eta_i).
\]

The intercept and distance coefficients are unit-specific but arise from shared population-level distributions through a non-centred hierarchical parameterization and partial pooling. This allows spatial/geological variation while retaining information shared across units. The hierarchy is a modeling representation of possible non-stationarity, not proof that every estimated difference is geological in origin. Posterior sampling provides uncertainty distributions for coefficients, curvature and predicted probabilities; V11 is not refit as part of the domain-stratified analysis.

The compact M5 baseline is a global logistic regression using Bouguer gravity, standardized lithology-contact distance and its quadratic term. It supplies the comparative predictive reference for the same four spatial test partitions.

## 3.5 Interpretation of curvature and $D^*$

For a distance-specific quadratic component,

\[
\eta(z)=\alpha+\beta_{lin}z+\beta_{sq}z^2,
\]

the derivative and curvature are

\[
\frac{d\eta}{dz}=\beta_{lin}+2\beta_{sq}z,
\qquad
\frac{d^2\eta}{dz^2}=2\beta_{sq}.
\]

When $\beta_{sq}\neq0$, the algebraic stationary point is

\[
z^*=-\frac{\beta_{lin}}{2\beta_{sq}},
\qquad
D^*=\mu_{train}+\sigma_{train}z^*.
\]

Here $D^*$ is the mathematical location implied by one posterior draw or a summary of those draws. Positive curvature is minimum-shaped and negative curvature is maximum-shaped. A finite $D^*$ does not by itself establish a meaningful turning point: draws near $\beta_{sq}=0$ can produce unstable values, and a stationary point outside the observed support is extrapolative for that modeling population. Accordingly, the analysis separates curvature sign, posterior uncertainty in $D^*$, response-curve shape, and the probability that $D^*$ lies within the inclusive observed distance support.

## 3.6 Primary spatial OOF validation

For each fold, V11 is estimated using only the three training blocks and generates predictions for the held-out block. The four held-out prediction sets are concatenated into frozen V11 spatial OOF predictions. Each observation therefore receives a prediction from a model that did not use its own spatial block for fitting. M5 is evaluated on the same four-fold partition using fold-specific training preprocessing.

Predictive discrimination is summarized with ROC-AUC, with PR-AUC and Brier score providing complementary information under class imbalance. The direct comparison is

\[
\Delta AUC=AUC_{V11}-AUC_{M5}.
\]

Fold-level results are retained alongside pooled OOF results because pooled performance can conceal geographically varying transferability.

## 3.7 Domain-stratified spatial OOF analysis

After V11 fitting and OOF prediction are complete, the existing frozen V11 OOF artifact is aligned to the original Phase 7 modeling observations using the verified observation identity of the artifact. The retained predictions are then stratified by the six Daly domains. This analysis does not refit V11, does not fit a separate model within each domain, does not use Leave-One-Daly-Domain-Out validation, and does not create Fold-by-Domain validation cells.

For each domain, the analysis reports cell counts, deposits, non-deposits, M5 OOF AUC, V11 OOF AUC, their difference and bootstrap 95% confidence intervals where both response classes are present. Thus domain-stratified performance describes how the existing spatially held-out predictions behave across geological settings; it is a secondary stratification of the primary four-fold spatial validation, not a replacement for it.

## 3.8 Scope of the analytical framework

The framework is designed to test a specific geological and predictive proposition: whether a nonlinear fault-distance response and its spatial variation are supported by the V11 posterior and by geographically separated prediction. It does not equate a large domain representation with universal Copperbelt generalization, and it does not convert an algebraic $D^*$ into a geological target without considering uncertainty, curvature classification and empirical support.