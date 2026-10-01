# Stage 02: Research Gap, Problem, Hypothesis, Aim and Objectives

## 2.1 Research gap

Existing mineral prospectivity studies commonly focus on maximizing predictive accuracy using machine-learning algorithms evaluated under random cross-validation. However, four critical scientific gaps remain unaddressed in regional prospectivity mapping:

1. **Spatial Transferability in Heterogeneous Belts:** Random cell splitting can place spatially related observations in training and test sets, potentially overstating performance for a new-region prediction target. The geographic transferability of multivariate prospectivity relationships along elongated belts is therefore an important empirical question, although spatial blocking does not guarantee independence across block boundaries.
2. **Spatial Non-Stationarity in Predictor Relationships:** Most regional workflows assume a globally stationary relationship between mineralization and geological predictors. Few studies explicitly test whether the relative contributions of structural, stratigraphic, and geophysical controls vary across distinct crustal or tectonic domains within a single metallogenic province.
3. **Uncertainty Quantification in Prospectivity Mapping:** Prospectivity outputs can conflate parameter posterior uncertainty, uncertainty in a cell's conditional probability, future outcome variability, and uncertainty in performance metrics. These quantities require separate definitions; a posterior mean probability map alone does not report all of them or quantify uncertainty in the input evidence layers.
4. **Representation Sensitivity of Distance Responses:** Proximity to geological structures is frequently parameterized using alternative functional forms (linear, logarithmic, or polynomial). When nonlinear terms imply a mathematical stationary point ($D^*$), studies can reify this value into an assumed "geological optimum" without testing whether the curvature and derived distance are robust to alternative mathematical representations or sensitive to upper-tail observations.

Addressing these combined gaps requires an integrated framework that unites multivariate geological predictors, hierarchical spatial partial pooling, along-belt spatial validation, and formal representation sensitivity analysis.

## 2.2 Research problem

It is not sufficient to demonstrate that proximity to faults, contacts, or gravity anomalies correlates with mineralization in a localized district. The core scientific problem is:
- To determine how the combined trio of fault distance, lithology-contact distance, and Bouguer gravity anomaly relates to copper-cobalt prospectivity across the diverse tectonic domains of the Central African Copperbelt;
- To establish whether the predictive influence of these earth-system variables remains stable or shifts systematically between along-belt sectors;
- To evaluate whether proximity relationships and inferred nonlinear response shapes are robust features of the underlying geological data or artifacts of specific mathematical parameterizations; and
- To separate algebraic turning-point diagnostics ($D^*$) from empirical support, response shape, and causal geological mechanisms.

## 2.3 Falsifiable hypotheses

### Primary Hypothesis (Spatial Heterogeneity):
> **Fitted associations and predictive discrimination for the multivariate geological/geophysical predictor set vary across the evaluated along-belt sectors and Daly-domain strata.**
> 
> *Testable Predictions:* Evidence consistent with the hypothesis includes fold- or domain-varying posterior coefficient summaries, varying SD-normalized shares of additive components in the posterior-mean OOF linear predictor, and differences in held-out discrimination. These summaries are descriptive model results and do not identify causal geological mechanisms. The four-fold design does not guarantee independence across fold boundaries.

### Secondary Hypothesis (Distance-Response Representation Robustness):
> **The structural and lithological proximity relationships exhibit stable, representation-robust functional forms across alternative mathematical specifications.**
> 
> *Testable Predictions:* If the inferred nonlinear distance responses and stationary points ($D^*$) reflect genuine geological thresholds, they will remain consistent under alternative scaling (logarithmic versus raw physical distances) and functional forms (linear versus quadratic), meeting pre-registered operational criteria for curvature agreement and stationary-point stability. If representation-dependent, curvature sign certainty will fluctuate, algebraic $D^*$ distributions will diverge significantly between raw and logarithmic forms, and linear models will achieve equivalent or superior spatial predictive accuracy without quadratic turning points.

## 2.4 Aim

To evaluate a partially pooled hierarchical Bayesian model combining fault distance, lithology-contact distance, Bouguer gravity, and retained host-lithology classes; assess geographic transfer among four along-belt holdout folds; and determine which distance-response conclusions persist across the evaluated mathematical representations.

## 2.5 Objectives

1. **Multivariate Parameter Estimation:** Estimate the joint linear and quadratic effects of fault distance and lithology-contact distance alongside global Bouguer gravity and host-rock lithological classes using a non-centered hierarchical Bayesian logistic framework.
2. **Spatial Transferability Evaluation:** Describe V11 and compact M5 OOF ROC-AUC, PR-AUC, and Brier scores across the four along-belt folds and pooled predictions. Interpret their difference as a comparison of these model specifications, not an isolated effect of hierarchical pooling.
3. **Spatial Heterogeneity Characterization:** Describe the SD-normalized shares of the standard deviations of additive posterior-mean OOF linear-predictor components across the spatial folds; do not interpret these shares as outcome-variance explained or causal importance.
4. **Distance Representation Sensitivity Analysis:** Systematically evaluate distance-response robustness across a $2 \times 2$ factorial grid (raw versus logarithmic distance $\times$ quadratic versus linear functional form) using pre-specified criteria for curvature probability agreement, $D^*$ numerical stability, and empirical distance support.
5. **Stationary-Point Diagnostic Audit:** Formally evaluate whether derived algebraic stationary points ($D^*$) are identifiable at the population level and representation-robust at the domain level, distinguishing mathematical curvature from causal geological thresholds.
6. **Secondary Domain Stratification:** Describe the distribution of frozen out-of-fold predictions across Daly's tectonic domains where positive and negative observations permit conventional discrimination analysis.

## 2.6 Research questions

1. What do the joint posterior distributions for fault distance, lithology-contact distance, Bouguer gravity, and host-rock classes indicate about regional prospectivity relationships across the Copperbelt?
2. How do the SD-normalized shares of additive posterior-mean OOF linear-predictor components vary across along-belt spatial folds?
3. How does the predictive discrimination of the hierarchical model compare with the compact M5 baseline when evaluated on spatially separated along-belt test partitions?
4. Are proximity associations and apparent quadratic curvatures stable when distance variables are represented logarithmically or evaluated with purely linear functional forms?
5. Does the population-level or domain-level posterior support an identifiable, representation-robust stationary-point diagnostic ($D^*$) that can be interpreted as a characteristic geological distance?
6. Among Daly domains containing both classes, how does frozen spatial OOF discrimination reflect regional geological context?

## 2.7 Deliberate boundaries

The study does not fit six independent domain models, implement Leave-One-Daly-Domain-Out (LODO) cross-validation, or evaluate Fold $\times$ Domain factorial validation cells. The Daly-domain analysis is strictly a secondary descriptive stratification of the existing, frozen four-fold along-belt spatial OOF predictions. The study avoids conflating model-derived stationary points with physical exploration targets and does not claim that mathematical curvature demonstrates Daly's hypothesis in its entirety.
