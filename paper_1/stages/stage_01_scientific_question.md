# Stage 01: Introduction, Background and Scientific Context

## 1.1 Geological context

The Central African Copperbelt contains substantial Cu-Co mineralization distributed through a structurally and lithologically complex belt. Mineralization is not expected to be controlled by a single measurable property. Faults, lithological contacts and regional geophysical structure can influence fluid pathways, permeability, host-rock architecture and the preservation of mineralized systems. These controls may operate differently along the belt because geological histories, structural configurations and deposit-bearing environments vary spatially.

This setting makes the Copperbelt a useful natural laboratory for asking whether relationships learned in one part of the belt transfer to another. It also makes a spatially separated evaluation scientifically necessary: a model can perform well when neighboring cells are mixed between training and testing while failing to transfer to a distinct along-belt region.

## 1.2 Fault proximity and nonlinear geological response

Fault distance is a geologically interpretable proxy for structural access, but its relationship with mineralization need not be monotonic. Very large distances may indicate limited structural influence, whereas locations immediately adjacent to a mapped fault may not always be the most favorable because mineralization can depend on the interaction of faults with host lithology, alteration, fluid chemistry and local structural geometry. A nonlinear response is therefore a scientifically meaningful possibility rather than merely an algorithmic embellishment.

The V11 analysis represents fault distance with linear and quadratic terms. This permits the fitted response to bend and, when supported by the posterior, to contain a stationary point. The stationary point is summarized by

\[
D^* = -\frac{\beta_{fault}}{2\beta_{fault^2}},
\]

after transforming from the standardized distance scale to physical distance. In this paper, $D^*$ is treated as the algebraic turning point of the fitted quadratic component. It is not automatically an optimal, preferred or universal geological distance. Its scientific relevance depends on curvature uncertainty, spatial consistency and whether it lies within the observed fault-distance support of the relevant modeling population.

## 1.3 Spatial non-stationarity and hierarchical inference

The geological processes represented by the predictors may not be spatially stationary along the Copperbelt. A single global relationship can therefore obscure regional variation, while fitting entirely separate models can discard information shared across the belt. V11 addresses this tension with a Bayesian hierarchical logistic-regression framework: selected intercepts and predictor effects vary by geological/spatial unit while being estimated from shared population-level distributions through partial pooling.

The hierarchy provides a statistical framework for estimating spatially varying effects; it does not, by itself, prove geological heterogeneity. Evidence for non-stationarity must be evaluated from posterior coefficient behavior, response shape, spatially separated predictive performance and the consistency of those patterns across the relevant geological settings.

## 1.4 Scientific question

This study asks whether the relationship between Cu-Co mineralization probability and fault distance contains a nonlinear structure consistent with the geological hypothesis under examination, and whether that relationship and its predictive discrimination remain transferable across spatially distinct parts of the Copperbelt when spatial non-stationarity is represented through the V11 hierarchical framework.

The study also asks whether the resulting out-of-fold predictive behavior is distributed consistently across Daly geological domains. Daly domains are used as a geological stratification for interpretation of model behavior, not as a claim that every domain provides an independently estimable validation problem.

## 1.5 Scope limitations

This study does not assume that Daly's hypothesis is proven. It tests whether the fitted nonlinear response, its posterior turning-point distribution, empirical support and spatially separated predictive behavior provide evidence consistent with or contrary to that hypothesis. It does not claim that V11 universally outperforms the compact global baseline, that one $D^*$ applies throughout the Copperbelt, or that the selected predictors exhaust the geological controls on mineralization. The results concern the defined modeling grid, predictors, frozen V11 posterior and four-fold along-belt validation design.