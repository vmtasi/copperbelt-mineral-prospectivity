# Stage 01: Introduction, Background and Scientific Context

## 1.1 Geological context

The Central African Copperbelt contains substantial Cu-Co mineralization distributed through a structurally and lithologically complex belt. Mineralization is not expected to be controlled by a single measurable property. The V11 framework therefore considers three principal continuous geological or geophysical predictors together: distance to major faults, distance to lithological contacts, and Bouguer gravity anomaly. Faults and lithological contacts can influence fluid pathways, permeability, host-rock architecture and mineralization sites, while regional gravity structure provides complementary geophysical information. These controls may operate differently along the belt because geological histories, structural configurations and deposit-bearing environments vary spatially.

This setting makes the Copperbelt a useful natural laboratory for asking whether relationships learned in one part of the belt transfer to another. It also makes a spatially separated evaluation scientifically necessary: a model can perform well when neighboring cells are mixed between training and testing while failing to transfer to a distinct along-belt region.

## 1.2 Geological predictors and nonlinear distance response

Distance to a fault and distance to a lithological contact are geologically interpretable proximity variables, but neither relationship need be monotonic. Very large distances may indicate limited structural or lithological influence, whereas locations immediately adjacent to a feature need not always be the most favorable because mineralization can depend on interactions among faults, host lithology, alteration, fluid chemistry and local geometry. A nonlinear response is therefore a scientifically meaningful possibility rather than merely an algorithmic embellishment.

The V11 analysis represents both distance predictors with linear and quadratic terms, allowing their fitted responses to bend and, when supported by the posterior, to contain a stationary point. Bouguer gravity enters as a linear standardized predictor and has no quadratic term in V11; consequently, it has no corresponding quadratic turning-point quantity $D^*$. The absence of $D^*$ for gravity is a property of the model specification, not evidence that gravity is unimportant.

For either distance predictor, the quadratic component can be written on the standardized scale as

\[
\eta(z)=\alpha+\beta_{1}z+\beta_{2}z^2,
\]

with stationary point

\[
z^*=-\frac{\beta_{1}}{2\beta_{2}},
\qquad
D^*=\mu_{train}+\sigma_{train}z^*.
\]

Here $D^*$ is the algebraic posterior turning point after back-transformation to physical distance. It is not automatically an optimal, preferred or universal geological distance. Its relevance depends separately on curvature uncertainty, whether a finite stationary point is supported, the classification implied by the curvature sign, the empirical distance range represented by the relevant training observations, posterior concentration, and the behavior of the posterior-mean response curve.

## 1.3 Spatial non-stationarity and hierarchical inference

The three predictors may not have spatially stationary relationships with mineralization along the Copperbelt. A single global relationship can obscure regional variation, while fitting entirely separate models can discard information shared across the belt. V11 addresses this tension with a Bayesian hierarchical logistic-regression framework in which the intercept and the linear and quadratic coefficients for the two distance predictors vary by Daly geological unit while being estimated through shared population-level distributions and partial pooling. Bouguer gravity and the retained lithological-class coefficients are global in the implemented V11 specification.

The hierarchy provides a statistical framework for estimating spatially varying effects; it does not, by itself, prove geological heterogeneity. Evidence for non-stationarity must be evaluated from posterior coefficient behavior, response shape, spatially separated predictive performance and the consistency of those patterns across relevant geological settings.

## 1.4 Scientific question

This study asks whether a multivariate hierarchical model combining fault distance, lithology-contact distance and Bouguer gravity can represent spatially varying mineral-prospectivity relationships, including nonlinear distance responses, and whether those relationships retain predictive discrimination when evaluated on spatially separated parts of the Copperbelt.

A secondary question concerns the nonlinear distance components specifically: where supported, what do their posterior turning points and response shapes indicate, and how closely are those features represented within the empirical distance ranges of the relevant training populations? Daly domains provide a geological stratification for interpreting the already-generated spatial OOF predictions rather than six independent validation problems.

## 1.5 Scope limitations

This study does not assume that Daly's hypothesis is proven. It evaluates whether the combined fitted relationships among the three predictors, their spatial variation, nonlinear distance responses and spatially separated predictive behavior provide evidence consistent with or contrary to aspects of that proposition. It does not claim that one predictor alone controls mineralization, that one $D^*$ applies throughout the Copperbelt, that Bouguer gravity should have a turning point despite its linear V11 specification, or that the selected predictors exhaust the geological controls on mineralization. The results concern the defined modeling grid, predictor set, frozen V11 posterior and four-fold along-belt validation design.
