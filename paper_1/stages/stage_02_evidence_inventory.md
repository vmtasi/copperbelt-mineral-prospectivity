# Stage 02: Research Gap, Problem, Hypothesis, Aim and Objectives

## 2.1 Research gap

Existing prospectivity studies commonly evaluate whether geological predictors are associated with mineralization, but a general association does not resolve whether fault distance has a nonlinear response, where the stationary point of that response lies, how uncertain that point is, or whether it is supported by observations in the relevant geological setting. These questions are especially important in a long, heterogeneous belt where spatial separation can expose failures of geographic transferability that pooled or randomly partitioned metrics conceal.

The specific gap addressed here is therefore the absence of a unified quantitative treatment of nonlinear fault-distance response, uncertainty in its algebraic turning point, empirical support for that turning point, spatially varying relationships and predictive validation under along-belt spatial separation. Daly's geological domains provide a geological framework for examining this evidence without assuming that all domains contain sufficient positive observations for independent discrimination estimates.

## 2.2 Research problem

It is not sufficient to determine whether distance to a fault enters a predictive model. The scientific problem is to determine whether a nonlinear fault-distance component is identifiable and interpretable under spatially varying geological conditions, and whether the corresponding predictive behavior transfers to held-out parts of the Copperbelt. The problem includes separating a mathematical stationary point from an empirically supported response feature and separating model discrimination from a geological claim about a preferred distance.

## 2.3 Falsifiable hypothesis

The primary hypothesis is that the relationship between mineralization probability and fault distance is spatially non-stationary and may contain a domain-dependent nonlinear component. If this hypothesis is supported, V11 should yield posterior evidence of differing fault-distance coefficients or response shapes across spatial/geological units, and its out-of-fold discrimination should vary across the four along-belt test regions rather than behaving as a uniformly transferable global relationship. If it is not supported, the posterior and spatial OOF results should be compatible with a substantially common response and comparatively uniform predictive behavior.

This hypothesis does not predict that V11 must outperform M5 in every fold or Daly domain. A result in which V11 is competitive overall but its advantage varies spatially remains informative about transferability rather than constituting automatic confirmation of the geological hypothesis.

## 2.4 Aim

To evaluate whether nonlinear fault-distance relationships associated with Cu-Co mineralization are identifiable, empirically supported and geographically transferable across the Copperbelt when estimated with a partially pooled hierarchical Bayesian model and evaluated using along-belt spatial out-of-fold prediction.

## 2.5 Objectives

1. Estimate the linear and quadratic fault-distance response, together with the corresponding lithology-contact, gravity and lithological effects, using the existing V11 hierarchical Bayesian formulation.
2. Quantify posterior uncertainty in the algebraic stationary point $D^*$ and distinguish curvature, turning-point existence, response classification and empirical support.
3. Determine whether fitted coefficients and response behavior vary across spatial/geological units in a manner consistent with spatial non-stationarity, without treating hierarchical variation as proof of geological causation.
4. Compare V11 with the compact M5 baseline using the existing four-fold along-belt spatial OOF predictions and fold-level and pooled predictive metrics.
5. Describe how the frozen V11 OOF predictive discrimination is distributed across the six Daly domains, calculating domain-specific ROC-AUC only where both deposit-positive and non-deposit observations are present.

## 2.6 Research questions

1. Does the V11 posterior support a nonlinear fault-distance response, and what curvature does that response imply?
2. What is the posterior distribution of the algebraic turning point $D^*$, and how often does it fall within the observed fault-distance support of the relevant modeling population or domain?
3. Do the estimated relationships and response shapes vary across spatial/geological units along the Copperbelt?
4. How does V11's out-of-fold predictive discrimination compare with M5 across the four held-out along-belt folds and in pooled OOF evaluation?
5. Among Daly domains containing both classes, how does V11's frozen OOF discrimination compare with the M5 OOF baseline?

## 2.7 Deliberate boundaries

The study does not refit V11 separately within each Daly domain, use Leave-One-Daly-Domain-Out validation as its primary design, or create Fold-by-Domain validation cells. The domain-stratified analysis is a secondary stratification of the existing frozen four-fold V11 OOF predictions. Stage 4 will report the numerical results; this stage defines the questions and estimands without presuming their outcomes.