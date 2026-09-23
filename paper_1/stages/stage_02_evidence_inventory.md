# Stage 02: Research Gap, Problem, Hypothesis, Aim and Objectives

## 2.1 Research gap

Existing mineral prospectivity studies commonly focus on maximizing predictive accuracy using machine-learning algorithms evaluated under random cross-validation. However, four critical scientific gaps remain unaddressed in regional prospectivity mapping:

1. **Spatial Transferability in Heterogeneous Belts:** Evaluating models via random cell splitting notoriously masks spatial autocorrelation leakage. There is a lack of rigorous empirical studies evaluating whether multivariate prospectivity relationships transfer along the regional strike of elongated orogenic belts under strict spatial out-of-fold (OOF) separation.
2. **Spatial Non-Stationarity in Predictor Relationships:** Most regional workflows assume a globally stationary relationship between mineralization and geological predictors. Few studies explicitly test whether the relative contributions of structural, stratigraphic, and geophysical controls vary across distinct crustal or tectonic domains within a single metallogenic province.
3. **Uncertainty Quantification in Prospectivity Mapping:** Traditional prospectivity algorithms produce deterministic probability surfaces that obscure parametric uncertainty and spatial extrapolation risk. Uncertainty-aware Bayesian formulations that partially pool regional regimes while propagating posterior uncertainty remain rare in regional exploration.
4. **Representation Sensitivity of Distance Responses:** Proximity to geological structures is frequently parameterized using arbitrary functional forms (linear, logarithmic, or polynomial). When nonlinear terms imply a mathematical stationary point ($D^*$), studies routinely reify this value into an assumed "geological optimum" without testing whether the curvature and derived distance are robust to alternative mathematical representations or sensitive to upper-tail observations.

Addressing these combined gaps requires an integrated framework that unites multivariate geological predictors, hierarchical spatial partial pooling, along-belt spatial validation, and formal representation sensitivity analysis.

## 2.2 Research problem

It is not sufficient to demonstrate that proximity to faults, contacts, or gravity anomalies correlates with mineralization in a localized district. The core scientific problem is:
- To determine how the combined trio of fault distance, lithology-contact distance, and Bouguer gravity anomaly relates to copper-cobalt prospectivity across the diverse tectonic domains of the Central African Copperbelt;
- To establish whether the predictive influence of these earth-system variables remains stable or shifts systematically between along-belt sectors;
- To evaluate whether proximity relationships and inferred nonlinear response shapes are robust features of the underlying geological data or artifacts of specific mathematical parameterizations; and
- To separate algebraic turning-point diagnostics ($D^*$) from empirical support, response shape, and causal geological mechanisms.

## 2.3 Falsifiable hypotheses

### Primary Hypothesis (Spatial Heterogeneity):
> **The relationships between copper-cobalt mineralization and the multivariate geological/geophysical predictor set are spatially heterogeneous along the Central African Copperbelt.**
> 
> *Testable Predictions:* If supported, the hierarchical Bayesian model will reveal systematic shifts in domain-level distance slopes and regional predictor variance shares across the four along-belt folds, accompanied by geographically variable predictive transferability (varying spatial OOF ROC-AUC). If not supported, the posterior evidence will be compatible with uniform, belt-wide distance coefficients, stable predictor contribution shares, and homogeneous spatial predictive performance.

### Secondary Hypothesis (Distance-Response Representation Robustness):
> **The structural and lithological proximity relationships exhibit stable, representation-robust functional forms across alternative mathematical specifications.**
> 
> *Testable Predictions:* If the inferred nonlinear distance responses and stationary points ($D^*$) reflect genuine geological thresholds, they will remain consistent under alternative scaling (logarithmic versus raw physical distances) and functional forms (linear versus quadratic), meeting pre-registered operational criteria for curvature agreement and stationary-point stability. If representation-dependent, curvature sign certainty will fluctuate, algebraic $D^*$ distributions will diverge significantly between raw and logarithmic forms, and linear models will achieve equivalent or superior spatial predictive accuracy without quadratic turning points.

## 2.4 Aim

To evaluate whether a partially pooled hierarchical Bayesian model combining fault distance, lithology-contact distance, Bouguer gravity, and host stratigraphy captures spatially heterogeneous mineral-prospectivity relationships across the Central African Copperbelt, transfers reliably under along-belt spatial cross-validation, and yields distance-response inferences that are robust to mathematical representation choices.

## 2.5 Objectives

1. **Multivariate Parameter Estimation:** Estimate the joint linear and quadratic effects of fault distance and lithology-contact distance alongside global Bouguer gravity and host-rock lithological classes using a non-centered hierarchical Bayesian logistic framework.
2. **Spatial Transferability Evaluation:** Evaluate the predictive discrimination (ROC-AUC, PR-AUC, Brier score) of the hierarchical model against the compact M5 baseline across four geographically held-out along-belt spatial folds and in pooled global out-of-fold evaluation.
3. **Spatial Heterogeneity Characterization:** Quantify the relative contribution of each structural, lithological, and geophysical predictor to variation in the linear predictor across the along-belt spatial folds to characterize regional shifts in predictive control.
4. **Distance Representation Sensitivity Analysis:** Systematically evaluate distance-response robustness across a $2 \times 2$ factorial grid (raw versus logarithmic distance $\times$ quadratic versus linear functional form) using pre-specified criteria for curvature probability agreement, $D^*$ numerical stability, and empirical distance support.
5. **Stationary-Point Diagnostic Audit:** Formally evaluate whether derived algebraic stationary points ($D^*$) are identifiable at the population level and representation-robust at the domain level, distinguishing mathematical curvature from causal geological thresholds.
6. **Secondary Domain Stratification:** Describe the distribution of frozen out-of-fold predictions across Daly's tectonic domains where positive and negative observations permit conventional discrimination analysis.

## 2.6 Research questions

1. What do the joint posterior distributions for fault distance, lithology-contact distance, Bouguer gravity, and host-rock classes indicate about regional prospectivity relationships across the Copperbelt?
2. How do the relative variance contributions of structural, stratigraphic, and geophysical predictors vary across along-belt spatial sectors?
3. How does the predictive discrimination of the hierarchical model compare with the compact M5 baseline when evaluated on spatially separated along-belt test partitions?
4. Are proximity associations and apparent quadratic curvatures stable when distance variables are represented logarithmically or evaluated with purely linear functional forms?
5. Does the population-level or domain-level posterior support an identifiable, representation-robust stationary-point diagnostic ($D^*$) that can be interpreted as a characteristic geological distance?
6. Among Daly domains containing both classes, how does frozen spatial OOF discrimination reflect regional geological context?

## 2.7 Deliberate boundaries

The study does not fit six independent domain models, implement Leave-One-Daly-Domain-Out (LODO) cross-validation, or evaluate Fold $\times$ Domain factorial validation cells. The Daly-domain analysis is strictly a secondary descriptive stratification of the existing, frozen four-fold along-belt spatial OOF predictions. The study avoids conflating model-derived stationary points with physical exploration targets and does not claim that mathematical curvature demonstrates Daly's hypothesis in its entirety.
