# Stage 02: Research Gap, Problem, Hypothesis, Aim and Objectives

## 2.1 Research gap

Existing mineral-prospectivity studies commonly evaluate whether geological predictors are associated with mineralization, but association alone does not establish whether proximity relationships are nonlinear, how their curvature varies spatially, whether an algebraic stationary point is supported by observations, or whether a multivariate relationship transfers to held-out parts of a long heterogeneous belt. A second gap is interpretive: the nonlinear distance terms can generate a derived stationary-point quantity, $D^*$, but curvature, stationary-point existence, empirical support, response shape and predictive discrimination are distinct quantities and should not be treated as interchangeable evidence.

The specific gap addressed here is therefore a unified treatment of three-predictor mineral prospectivity—fault distance, lithology-contact distance and Bouguer gravity—within a partially pooled hierarchical model, while separately characterizing nonlinear distance responses and evaluating predictive transferability under along-belt spatial separation. Daly's geological domains provide a geological framework for describing this evidence without assuming that all domains contain sufficient positive observations for independent discrimination estimates.

## 2.2 Research problem

It is not sufficient to determine whether an individual geological variable enters a predictive model. The scientific problem is to determine how the combined fault-distance, lithology-contact-distance and Bouguer-gravity predictors relate to mineralization under spatially varying geological conditions, whether the two distance relationships contain identifiable nonlinear structure, and whether the resulting multivariate predictions transfer to held-out parts of the Copperbelt. The problem also requires separating mathematical response features from empirical support and from geological interpretation.

## 2.3 Falsifiable hypothesis

The primary hypothesis is that the relationship between mineralization probability and the three-predictor geological/geophysical feature set is spatially non-stationary, with the two distance predictors potentially exhibiting domain-dependent nonlinear components. If supported, V11 should yield posterior evidence of differing distance coefficients or response shapes across spatial/geological units while incorporating Bouguer gravity as a common component of the multivariate model, and its out-of-fold discrimination should vary across the four along-belt test regions rather than behaving as a uniformly transferable global relationship. If not supported, the posterior and spatial OOF results should be compatible with substantially common distance responses and comparatively uniform predictive behavior.

This hypothesis does not predict that V11 must outperform the compact M5 baseline in every fold or Daly domain. Variation in predictive performance remains informative about geographic transferability even when the direction of the V11–M5 difference changes across regions.

## 2.4 Aim

To evaluate whether a partially pooled hierarchical Bayesian model combining fault distance, lithology-contact distance and Bouguer gravity captures nonlinear distance relationships and spatially varying mineral-prospectivity behavior, and whether its predictions transfer across the Copperbelt under along-belt spatial out-of-fold evaluation.

## 2.5 Objectives

1. Estimate the linear and quadratic effects of fault distance and lithology-contact distance together with the Bouguer-gravity and retained lithological-class effects using the existing V11 hierarchical Bayesian formulation.
2. Quantify posterior uncertainty in the nonlinear distance responses, including the algebraic stationary point $D^*$ where mathematically defined, and distinguish curvature, turning-point existence, turning-point classification, empirical support and posterior-mean response shape.
3. Determine whether the fitted coefficients and response behavior vary across spatial/geological units in a manner consistent with modeled spatial non-stationarity, without treating hierarchical variation as proof of geological causation.
4. Compare V11 with the compact M5 baseline using the existing four-fold along-belt spatial OOF predictions and fold-level and pooled predictive metrics.
5. Describe how the frozen V11 OOF predictive discrimination is distributed across the six Daly domains, calculating domain-specific ROC-AUC only where both deposit-positive and non-deposit observations are present.

## 2.6 Research questions

1. What do the combined posterior coefficients for fault distance, lithology-contact distance and Bouguer gravity indicate about mineral-prospectivity relationships across the Copperbelt?
2. Do the fault-distance and lithology-contact-distance components exhibit identifiable nonlinear curvature, and what response shapes do those terms imply?
3. Where finite algebraic stationary points are supported, what are their posterior distributions and how often do they fall within the observed distance support of the relevant modeling population?
4. Do the estimated relationships and response shapes vary across spatial/geological units along the Copperbelt?
5. How does V11's out-of-fold predictive discrimination compare with M5 across the four held-out along-belt folds and in pooled OOF evaluation?
6. Among Daly domains containing both classes, how does V11's frozen OOF discrimination compare with the M5 OOF baseline?

## 2.7 Deliberate boundaries

The study does not refit V11 separately within each Daly domain, use Leave-One-Daly-Domain-Out validation as its primary design, or create Fold-by-Domain validation cells. The domain-stratified analysis is a secondary stratification of the existing frozen four-fold V11 OOF predictions. Stage 4 reports the numerical results; this stage defines the questions and estimands without presuming their outcomes.
