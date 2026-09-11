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

\[
\operatorname{logit}(P(Y_i=1))
=
\alpha_d
+
\beta_{fault,d}X_{fault,i}
+
\beta_{fault^2,d}X_{fault,i}^{2}
+
\beta_{lith,d}X_{lith,i}
+
\beta_{lith^2,d}X_{lith,i}^{2}
+
\beta_{grav}X_{grav,i}
+
\text{lithological effects}.
\]

Domain-specific coefficients are hierarchically regularized around
shared population-level distributions, allowing regional effects to
differ while retaining partial pooling between domains.

The central methodological contrast is therefore between a formulation
with globally shared predictive relationships and one that explicitly
permits those relationships to vary across geological domains.

## 3.5 Out-of-Fold Prediction Strategy

For each spatial fold \(k\), the models were fitted exclusively using

\[
D_{train}^{(k)} = \mathcal{D}\setminus B_k
\]

and predictions were generated for the withheld spatial block

\[
D_{test}^{(k)} = B_k.
\]

This procedure was repeated for all four folds. The resulting held-out
predictions were then concatenated to form a complete OOF prediction
set in which each observation was predicted by a model that had not
been trained on that observation's spatial block.

## 3.6 Evaluation Metrics

Three complementary metrics were used:

- **ROC-AUC:** measures ranking discrimination between mineralized and
  non-mineralized observations.
- **PR-AUC:** provides an additional discrimination measure under the
  substantial class imbalance present in the dataset.
- **Brier Score:** evaluates the accuracy of predicted probabilities.

For direct model comparison, the primary contrast was

\[
\Delta AUC =
AUC_{V11}-AUC_{M5}.
\]

Because both models were evaluated on the same held-out observations
within each spatial fold, this constitutes a paired spatial comparison
of predictive performance.

Both fold-level and pooled OOF metrics were examined. Fold-level
evaluation was particularly important because pooling geographically
distinct observations can conceal regional differences in predictive
performance.

## 3.7 Spatial Robustness Analysis

Uncertainty in model-performance differences was additionally examined
using spatially structured block bootstrap resampling.

Rather than treating individual grid cells as independent bootstrap
units, observations were grouped into spatial blocks and these blocks
were resampled with replacement.

The robustness analysis was repeated using four block-grid resolutions:

- 10 × 10
- 15 × 15
- 20 × 20
- 25 × 25

For each scale, the distribution of

\[
\Delta AUC=AUC_{V11}-AUC_{M5}
\]

was examined.

The purpose of this analysis was to determine whether the observed
direction and magnitude of model-performance differences were sensitive
to the spatial scale used to define bootstrap blocks.

Particular attention was given to the observed reversal in relative
model performance between spatial folds, rather than assuming that the
globally superior model was uniformly superior throughout the study
region.