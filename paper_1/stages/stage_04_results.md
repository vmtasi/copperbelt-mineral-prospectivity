# Stage 04: Results

## 4.1 V11 posterior coefficients and nonlinear distance responses

V11 estimates three continuous predictors jointly: distance to fault, distance to lithology contact, and Bouguer gravity. The two distance predictors have domain-specific linear and quadratic coefficients, whereas the Bouguer-gravity coefficient is global in the implemented V11 specification. The retained lithological-class indicators also enter as global coefficients. The results below therefore describe a multivariate model rather than a fault-distance-only analysis.

For NRB_3a fault distance, fold-specific median $\beta_{fault}$ values ranged from $-2.681$ to $-0.703$, while median $\beta_{fault^2}$ ranged from $-0.057$ to $1.601$. The posterior probability of positive fault curvature, $P(\beta_{fault^2}>0)$, ranged from $0.464$ to $0.996$ across the four folds; $P(\beta_{fault^2}>0.05)$ ranged from $0.433$ to $0.993$. Folds 1, 3 and 4 were predominantly positive-curvature, whereas Fold 2 was sign-uncertain under the posterior probability criterion. This fold-level description follows the stored diagnostic probabilities rather than treating every positive posterior median as established curvature.

For NRB_3b fault distance, median $\beta_{fault}$ ranged from $0.181$ to $1.548$ and median $\beta_{fault^2}$ ranged from $-0.273$ to $0.086$. The corresponding $P(\beta_{fault^2}>0)$ ranged from $0.161$ to $0.708$, and $P(\beta_{fault^2}>0.05)$ ranged from $0.109$ to $0.589$. These values indicate substantially weaker and more variable curvature evidence than the corresponding NRB_3a lithology-contact response.

Lithology-contact distance showed the more consistent nonlinear pattern in NRB_3a. Median $\beta_{lith}$ ranged from $-0.849$ to $-0.451$, median $\beta_{lith^2}$ ranged from $0.293$ to $0.334$, and $P(\beta_{lith^2}>0)$ ranged from $0.993$ to $1.000$. In NRB_3b, median $\beta_{lith}$ ranged from $-1.902$ to $-0.819$, median $\beta_{lith^2}$ ranged from $0.209$ to $0.425$, and $P(\beta_{lith^2}>0)$ ranged from $0.695$ to $0.865$.

Bouguer gravity is a third, integral predictor of the V11 linear predictor but has no quadratic term. The existing V11 result artifacts do not provide a dedicated fold-by-fold posterior summary for the global $\beta_g$ comparable to the distance-coefficient tables. Accordingly, this stage records its implemented role and global status without inferring a gravity-specific effect size or spatial pattern from artifacts belonging to other model specifications. Its absence from the $D^*$ analysis reflects model structure, not an assertion that gravity is unimportant.

The coefficient probabilities above describe posterior curvature sign. They are distinct from the existence of a finite algebraic stationary point, its classification, and whether that point lies within observed distance support.

## 4.2 Posterior turning-point $D^*$ results

For NRB_3a fault distance, the unfiltered fold-median $D^*$ values ranged from $-18.024$ to $46.619$ km. The probability that posterior draws of $D^*$ lay within the fold-specific empirical support ranged from $0.126$ to $0.938$. The posterior-mean response-curve extrema ranged from $36.645$ to $43.349$ km and were classified as minima on the corresponding support grids, although the algebraic posterior was substantially less stable in Fold 2.

For NRB_3b fault distance, unfiltered fold-median $D^*$ values ranged from $22.780$ to $80.438$ km, with within-support probabilities from $0.625$ to $0.881$. The posterior-mean response extrema ranged from $0.358$ to $22.326$ km; one fold was classified as boundary/monotone while the other fold-level curves included minimum or maximum behavior. This distinction records different estimands: the median algebraic stationary point is not the same quantity as the extremum of the posterior-mean response curve.

For NRB_3a lithology-contact distance, unfiltered fold-median $D^*$ values ranged from $26.189$ to $28.534$ km, and within-support probabilities ranged from $0.963$ to $1.000$. For NRB_3b lithology-contact distance, fold medians ranged from $26.505$ to $41.340$ km, while within-support probabilities ranged from $0.381$ to $0.584$. The corresponding posterior-mean curve extrema were approximately $25.498$–$28.467$ km for NRB_3a and $28.098$–$34.175$ km for NRB_3b.

The diagnostic outputs also report conditional $D^*$ distributions after the methodological filter $\beta_{sq}>0.05$. These are conditional posterior summaries, not unconditional $D^*$ distributions. The filter was not treated as a geological truth criterion.

## 4.3 Empirical support, response shape and slope behavior

The fold-specific empirical support intervals were calculated from training observations only and used inclusive lower and upper boundaries. A positive quadratic coefficient corresponds to a minimum-shaped fitted component, while a negative coefficient corresponds to a maximum-shaped component. The classification is distinct from the existence of a finite algebraic stationary point and from whether that point lies within observed support.

The support and response-shape results did not always coincide with the algebraic summaries. NRB_3a fault posterior-mean curves were classified as minima in all folds, although Fold 2 had sign-uncertain curvature and only 0.126 of unfiltered $D^*$ draws within support. NRB_3b fault included a boundary/monotone response curve and a maximum-shaped curve among its fold-level outputs, alongside broad and sign-uncertain algebraic summaries. NRB_3a lithology was classified as a minimum across folds with high support probabilities, whereas NRB_3b lithology was classified as a minimum over the support grid despite lower curvature and support certainty.

The posterior slope diagnostics evaluate the fitted derivative across the relevant empirical support rather than reducing a quadratic relationship to a single coefficient. For the standardized distance variable, the slope is $\beta_{lin}+2\beta_{sq}z$, with $z=(D-\mu_{train})/\sigma_{train}$. Thus $\beta_{lin}$ is the local slope at the training-fold mean distance ($z=0$), not the effect of moving outward from physical distance zero.

## 4.4 Spatial variation/non-stationarity and coefficient stability

The fold-level coefficient and response summaries show spatial variation in both coefficient values and response shape. For NRB_3a fault distance, Folds 1, 3 and 4 were predominantly positive-curvature while Fold 2 remained sign-uncertain; the associated $D^*$ support probabilities nevertheless differed markedly across folds. NRB_3b fault curvature remained less clearly identified, with posterior sign probabilities spanning both directions and a boundary/monotone posterior-mean curve among the fold-level outputs. Lithology-contact responses were minimum-shaped in the posterior-mean curves across the reported NRB_3a and NRB_3b folds, but posterior curvature and empirical-support evidence were stronger in NRB_3a.

The pairwise coefficient table is descriptive rather than a composite stability score. It reports posterior differences between folds and whether their intervals include zero. Coefficient stability is therefore treated separately from predictive stability and from the stability of derived $D^*$ values.

## 4.5 Four-fold along-belt spatial OOF predictive performance

The primary four-fold spatial OOF comparison produced the following results:

| Scope | M5 ROC-AUC | V11 ROC-AUC | V11-M5 $\Delta$AUC | V11 PR-AUC | V11 Brier |
|---|---:|---:|---:|---:|---:|
| Fold 1 | 0.855 [0.695, 0.961] | 0.817 [0.748, 0.868] | -0.038 [-0.132, 0.069] | 0.048 | 0.019 |
| Fold 2 | 0.824 [0.753, 0.887] | 0.688 [0.599, 0.769] | -0.136 [-0.250, -0.020] | 0.103 | 0.063 |
| Fold 3 | 0.585 [0.493, 0.668] | 0.676 [0.605, 0.745] | +0.091 [0.024, 0.159] | 0.133 | 0.074 |
| Fold 4 | 0.581 [0.506, 0.655] | 0.561 [0.495, 0.626] | -0.020 [-0.083, 0.044] | 0.145 | 0.118 |
| Pooled OOF | 0.525 [0.470, 0.581] | 0.689 [0.647, 0.727] | +0.164 [0.084, 0.236] | 0.125 | 0.069 |

The fold-level V11 Brier scores were 0.019, 0.063, 0.074 and 0.118, respectively. The corresponding M5 Brier scores were 0.023, 0.060, 0.076 and 0.119. The V11–M5 AUC difference was negative in Folds 1, 2 and 4 and positive in Fold 3, while the pooled difference was positive.

The spatial block bootstrap output retained this fold pattern across 10×10, 15×15, 20×20 and 25×25 block grids. For example, the observed Fold 2 difference was $-0.136$ and the observed Fold 3 difference was $+0.091$ at every reported block scale; pooled bootstrap median differences ranged from $0.163$ to $0.167$.

## 4.6 Daly-domain stratification of frozen OOF predictions

The existing frozen four-fold V11 OOF predictions were aligned to 1,872 modeling observations: 1,872 artifact rows, 1,872 aligned rows, zero unmatched model rows, zero unmatched artifact rows, zero duplicated model rows, zero duplicated artifact rows and zero missing V11 predictions. The aligned predictions were then stratified by Daly domain; V11 was not refit by domain.

| Daly domain | Cells | Deposits | Non-deposits | M5 OOF AUC | V11 OOF AUC | V11-M5 $\Delta$AUC |
|---|---:|---:|---:|---:|---:|---:|
| CRZ | 154 | 0 | 154 | Undefined | Undefined | Undefined |
| NKB | 24 | 0 | 24 | Undefined | Undefined | Undefined |
| SRB | 1 | 0 | 1 | Undefined | Undefined | Undefined |
| MMSB | 16 | 0 | 16 | Undefined | Undefined | Undefined |
| NRB_3a | 1,127 | 107 | 1,020 | 0.535752 [0.484879, 0.586426] | 0.661701 [0.615874, 0.705728] | +0.125948 |
| NRB_3b | 550 | 31 | 519 | 0.866897 [0.811545, 0.913898] | 0.583815 [0.491894, 0.678182] | -0.283082 |

ROC-AUC is undefined in CRZ, NKB, SRB and MMSB because each contains only non-deposit observations in the modeling frame. These rows are data-composition limitations, not failed validation results. The intervals in the domain table are 95% bootstrap percentile intervals. The two estimable domains show different discrimination patterns in the same primary four-fold spatially held-out predictions.
