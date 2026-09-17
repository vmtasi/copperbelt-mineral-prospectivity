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
