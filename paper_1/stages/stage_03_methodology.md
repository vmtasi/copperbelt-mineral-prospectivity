# Stage 03: Methodology, Study Design and Analytical Framework

## 3.1 Study area and modeling data

The study concerns the Central African Copperbelt tract represented by the project modeling grid. Each grid observation has centroid coordinates $(x_i,y_i)$ and a binary response $Y_i\in\{0,1\}$ indicating deposit presence. The analysis uses the complete-case modeling population after applying the same response, coordinate, Daly-domain, lithological-class and continuous-predictor availability requirements used by the V11 implementation.

The three continuous predictors are distance to major faults ($D_{fault}$), distance to lithological contacts ($D_{lith}$), and Bouguer gravity anomaly ($X_{grav}$). The complete modeling frame contains 138 deposit-positive observations: 107 in NRB_3a and 31 in NRB_3b. CRZ, NKB, SRB and MMSB contain zero deposit-positive observations in this frame. These counts describe the available modeling data; they do not imply that the zero-positive domains lack geological importance.

## 3.2 Daly-domain classification

Daly's geological domains are used as a geological stratification of the Copperbelt tract. The six fixed labels are CRZ, NKB, SRB, MMSB, NRB_3a and NRB_3b. In V11, these domain labels determine the index used by the domain-varying intercept and distance coefficients.

The domains are not treated as six independent predictive-validation datasets. In particular, a conventional ROC-AUC requires both deposit-positive and non-deposit observations. Consequently, domain-level ROC-AUC is estimable in the current modeling frame for NRB_3a and NRB_3b, while the four zero-positive domains cannot yield a conventional positive-versus-negative ROC-AUC. They are not considered failed validation domains; the estimand is undefined there under this data composition.

## 3.3 Predictor preprocessing and spatial folds

For each of four predefined along-belt spatial folds, one block $B_k$ is held out and the remaining observations form the training set:

\[
\mathcal{D}^{(k)}_{train}=\mathcal{D}\setminus B_k,\qquad
\mathcal{D}^{(k)}_{test}=B_k.
\]

Each continuous predictor is standardized using parameters estimated only from $\mathcal{D}^{(k)}_{train}$. The held-out observations are transformed with those training parameters. The quadratic distance terms are then constructed from the standardized distance variables, so the model uses $z_{fault}^2$ and $z_{lith}^2$, not the square of an unstandardized physical distance. This preserves spatial separation between training and test observations and prevents test-fold information from entering preprocessing.

The four-fold along-belt design is the primary validation framework. It evaluates geographic transferability rather than random-cell interpolation, where neighboring cells could place closely related geological environments in both training and test data.

## 3.4 V11 hierarchical Bayesian model

V11 is a Bayesian hierarchical logistic-regression model. For observation $i$ in Daly domain $d(i)$, the linear predictor implemented in the model is

\[
\begin{aligned}
\eta_i ={}& \alpha_{d(i)}
 + \beta_{f,d(i)}z_{f,i}
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

Thus, the model contains three continuous predictors: fault distance, lithology-contact distance and Bouguer gravity. The first two have linear and quadratic terms; Bouguer gravity has one global linear coefficient and no quadratic term. The lithological-class indicators retained after the train-only validity check enter through the global coefficient vector $\boldsymbol{\beta}_{rock}$.

The intercept and the four distance coefficients are domain-specific and hierarchically partially pooled. The implemented population-level priors are

\[
\begin{aligned}
\mu_{f,lin},\mu_{f,sq},\mu_{l,lin},\mu_{l,sq} &\sim \operatorname{Normal}(0,1),\\
\sigma_{f,lin},\sigma_{f,sq},\sigma_{l,lin},\sigma_{l,sq} &\sim \operatorname{HalfNormal}(1),\\
\text{offset}_{k} &\sim \operatorname{Normal}(0,1).
\end{aligned}
\]

The intercept population mean uses the training-fold log-odds base rate,

\[
\mu_\alpha\sim\operatorname{Normal}(\operatorname{logit}(\hat p_{train}),1),
\]

with $\alpha$ variation governed by $\sigma_\alpha\sim\operatorname{HalfNormal}(1)$ and standard-normal domain offsets. The global Bouguer-gravity coefficient and global retained lithological-class coefficients have $\operatorname{Normal}(0,1)$ priors. The domain-specific coefficients are constructed from their population means, scale parameters and standard-normal offsets; V11 samples these relationships using the corresponding non-centred parameterization. This is the implemented sampling representation of partial pooling rather than a separate model specification.

The compact M5 baseline is a global logistic regression using Bouguer gravity, standardized lithology-contact distance and its quadratic term. It supplies the comparative predictive reference for the same four spatial test partitions.

## 3.5 Interpretation of curvature and $D^*$

For either distance predictor, the quadratic component can be written on the standardized scale as

\[
\eta(z)=\alpha+\beta_{lin}z+\beta_{sq}z^2.
\]

Its slope and curvature are

\[
\frac{d\eta}{dz}=\beta_{lin}+2\beta_{sq}z,
\qquad
\frac{d^2\eta}{dz^2}=2\beta_{sq}.
\]

A finite algebraic stationary point for a draw with $\beta_{sq}\neq0$ is

\[
z^*=-\frac{\beta_{lin}}{2\beta_{sq}},
\qquad
D^*=\mu_{train}+\sigma_{train}z^*.
\]

The transformation uses the same training-fold mean and scale that standardized the corresponding physical distance. Therefore, reported $D^*$ values in kilometres are back-transformed physical distances; the underlying coefficient calculation is performed in standardized $z$-space.

The curvature sign determines the shape of the quadratic component: $\beta_{sq}>0$ gives a minimum-shaped quadratic and $\beta_{sq}<0$ gives a maximum-shaped quadratic. When $\beta_{sq}$ is near zero, $D^*$ becomes numerically unstable and a finite algebraic value should not be interpreted as strong turning-point identification. The analysis therefore distinguishes posterior curvature sign, existence of a finite stationary point, stationary-point classification, empirical support, posterior concentration and the extremum of the posterior-mean response curve.

These $D^*$ calculations apply only to fault distance and lithology-contact distance because only those predictors have quadratic terms in V11. Bouguer gravity has no quadratic term and consequently has no $D^*$ under this model.

For a distance $D$, posterior slope behavior is evaluated through

\[
slope(D)=\beta_{lin}+2\beta_{sq}\left(\frac{D-\mu_{train}}{\sigma_{train}}\right).
\]

In particular, $\beta_{lin}$ is the local slope at $z=0$, corresponding to the training-fold mean physical distance, not the slope at physical distance zero.

## 3.6 Empirical support and response-curve quantities

Empirical distance support is defined from the actual training observations for each fold and relevant domain/variable. If $D_{min,train}$ and $D_{max,train}$ denote the observed minimum and maximum training distances, a draw is counted as within support when

\[
D_{min,train}\le D^*\le D_{max,train},
\]

using inclusive boundaries. The held-out fold does not define this support. A within-support probability indicates that a stationary point lies within the observed predictor range represented by the relevant training data; it does not establish geological causality or validation of a geological threshold.

The manuscript distinguishes two further quantities. First, the posterior median of draw-level algebraic $D^*$ values summarizes the posterior distribution of stationary points. Second, the turning point of the posterior-mean response curve is obtained from the independently constructed mean fitted response over the empirical domain. These need not coincide because taking a nonlinear ratio draw by draw and then taking its median is not equivalent to finding the extremum of the posterior-mean curve.

The logistic transformation is monotonic, so an interior extremum occurs at the same distance on the linear-predictor and probability scales. Response-curve interpretation therefore reports the scale being plotted while preserving the location of the corresponding extremum.

## 3.7 Predictive validation and domain stratification

Predictive validation uses the four predefined along-belt spatial folds as the primary evaluation. V11 predictions are generated for held-out observations and compared with the compact M5 baseline using the existing ROC-AUC and Brier outputs, with V11 PR-AUC reported as an additional V11-only summary. The pooled OOF result aggregates predictions across the four held-out folds.

Daly-domain results are a secondary descriptive stratification of those already-generated frozen four-fold OOF predictions. V11 is not refit by Daly domain, and no Leave-One-Daly-Domain-Out or Fold-by-Domain validation design is introduced. Domain-level ROC-AUC is reported only where both classes occur in the modeling frame.

## 3.8 Analytical interpretation framework

The analytical framework keeps six evidential quantities separate: (1) coefficient magnitude and uncertainty; (2) quadratic curvature; (3) algebraic stationary-point existence and classification; (4) empirical support and posterior concentration of $D^*$; (5) posterior slope and response-curve behavior; and (6) held-out predictive discrimination. Coefficient stability, response-curve stability, turning-point stability and predictive stability are therefore distinct concepts. The hierarchy and spatial OOF design are interpreted together rather than allowing any one quantity to stand in for the others.
