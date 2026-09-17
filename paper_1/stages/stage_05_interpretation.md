# Stage 05: Interpretation

## 5.1 Multivariate predictor structure

V11 should be interpreted first as a three-predictor mineral-prospectivity model. Fault distance, lithology-contact distance and Bouguer gravity enter the same hierarchical logistic predictor, so the fitted distance relationships are conditional components of a multivariate model rather than isolated univariate curves. Fault distance and lithology-contact distance have domain-specific linear and quadratic coefficients, while Bouguer gravity has a global linear coefficient. The retained lithological-class terms provide additional global categorical adjustment.

This structure matters for interpretation. A distance coefficient describes the fitted contribution of that predictor conditional on the other model terms, not an independent geological mechanism. Likewise, the absence of a $D^*$ for gravity follows from the absence of a gravity quadratic term in V11; it should not be read as evidence that gravity is less important than either distance predictor.

## 5.2 Nonlinear distance responses

The two distance predictors show different degrees of nonlinear evidence. Positive quadratic coefficients imply minimum-shaped fitted components, while negative coefficients imply maximum-shaped components. NRB_3a lithology-contact distance provides the clearest combined example: posterior $P(\beta_{lith^2}>0)$ was $0.993$–$1.000$ across folds, fold-median $D^*$ values were about $26$–$29$ km, and within-support probabilities were $0.963$–$1.000$. Its posterior-mean response was minimum-shaped across the reported folds.

NRB_3a fault distance is also generally minimum-like in the posterior-mean response, but its curvature evidence is more heterogeneous. Three folds were predominantly positive-curvature, while Fold 2 was sign-uncertain; its within-support probability ranged from $0.126$ to $0.938$. NRB_3b fault distance provides weaker evidence for a common response shape, with curvature probabilities spanning both signs and posterior-mean curves including boundary/monotone and interior-extremum behavior. NRB_3b lithology-contact distance was more often positive-curvature, but its within-support probability was only $0.381$–$0.584$.

These results indicate that nonlinear distance response is setting-dependent. They do not imply that either distance predictor has one invariant response throughout the Copperbelt.

## 5.3 $D^*$, classification and empirical support

For each distance predictor, $D^*$ is the algebraic stationary point of the fitted standardized quadratic after back-transformation to physical distance. It is useful as one diagnostic of nonlinear response, but it must be separated from curvature, stationary-point existence, classification, empirical support, posterior concentration and posterior-mean response shape.

A finite draw-level $D^*$ shows that a particular coefficient draw has a stationary point; it does not establish that the stationary point is well identified. This is especially important when the quadratic coefficient is near zero, because the ratio $-\beta_{lin}/(2\beta_{sq})$ can become unstable. Conversely, a high probability that $D^*$ lies within the observed training support makes the stationary point less extrapolative, but empirical support is a range criterion rather than geological validation.

The evidence is strongest for NRB_3a lithology-contact distance because its curvature is consistently positive, its $D^*$ distribution is comparatively concentrated, its support probability is high, and its posterior-mean response has the corresponding minimum shape. Fault-distance $D^*$ evidence is less uniform, particularly across NRB_3a folds and throughout NRB_3b. These distinctions are why a stationary point is not converted into a recommended or universal geological distance.

## 5.4 Spatial non-stationarity

The differing fold-level coefficients, curvature probabilities, support probabilities and response-curve extrema are consistent with spatially varying fitted relationships. The hierarchical formulation provides partial pooling so that domain-specific distance effects can vary while sharing information through population-level distributions.

This modeled variation should not be interpreted automatically as proof of distinct geological mechanisms. The evidence for non-stationarity is statistical and predictive: coefficient distributions and response shapes differ across spatial settings, and the spatial OOF results also vary by held-out region. Those observations establish context dependence of the fitted relationships without identifying causation.

## 5.5 Predictive performance and geographic transferability

The primary predictive evidence comes from the four-fold along-belt spatial OOF evaluation. V11's AUC difference relative to M5 was negative in Folds 1, 2 and 4 and positive in Fold 3, while the pooled OOF difference was positive. Thus the pooled result summarizes heterogeneous geographic behavior rather than a uniform advantage in every held-out region.

The multiscale spatial bootstrap retained the observed sign pattern of the Fold 2 and Fold 3 differences across the reported block scales. This is evidence about the robustness of the completed predictive comparison; it is not evidence that the underlying coefficients, response curves or geological relationships are equally stable.

Predictive performance should therefore be kept separate from coefficient stability, turning-point stability and geological interpretation. A model can transfer differently across regions even when its underlying predictors are defined consistently.

## 5.6 Daly-domain stratification

The secondary stratification of the frozen spatial OOF predictions shows different predictive behavior in the two Daly domains for which conventional ROC-AUC is estimable. In NRB_3a, V11 achieved AUC $0.661701$ versus $0.535752$ for M5, a difference of $+0.125948$. In NRB_3b, V11 achieved $0.583815$ versus $0.866897$ for M5, a difference of $-0.283082$.

These are stratified summaries of the same already-generated four-fold spatial OOF predictions. They are not independent domain refits, Leave-One-Daly-Domain-Out validation or Fold-by-Domain validation. CRZ, NKB, SRB and MMSB contain no positive observations in the modeling frame, so conventional ROC-AUC is undefined there rather than evidence of poor performance.

## 5.7 Relationship to Daly's hypothesis

Taken together, the evidence is consistent with aspects of Daly's proposition but does not constitute proof of it. The nonlinear fitted relationships of the two distance predictors show that proximity effects need not be represented as uniformly monotonic functions. The particularly well-supported NRB_3a lithology-contact response provides a clear example of an interior minimum-shaped modeled response that is represented within the observed training support. Fault-distance behavior is more heterogeneous, and the contrast in predictive transferability across spatial folds and Daly domains further indicates that a single invariant relationship is not supported by the completed analysis.

Bouguer gravity remains an integral third predictor in this interpretation even though its V11 specification is linear and therefore has no quadratic stationary point. The existing artifacts do not provide a dedicated V11 fold-by-fold posterior summary for its global coefficient, so no stronger gravity-specific effect claim is warranted here. Its inclusion in the multivariate model and in the predictive OOF results is nevertheless part of the evidence base.

The appropriate interpretation is therefore that the V11 results provide qualified empirical evidence of nonlinear and spatially varying prospectivity relationships that are compatible with aspects of Daly's proposition. They do not establish a universal distance, a causal structural mechanism, or Daly's hypothesis in its entirety.
