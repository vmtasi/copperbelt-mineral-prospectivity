# Stage 06: Discussion

## 6.1 Principal findings

The completed analysis supports a three-predictor interpretation of Copperbelt mineral prospectivity. V11 jointly represents distance to fault, distance to lithology contact and Bouguer gravity within a partially pooled hierarchical logistic model, with additional retained lithological-class terms. The two distance predictors have domain-specific linear and quadratic effects, whereas Bouguer gravity has a global linear effect. This multivariate structure provides the basis for interpreting nonlinear distance responses without reducing the study to a single predictor or to the derived quantity $D^*$.

Several findings follow from that framework. First, the distance relationships contain nonlinear fitted components whose curvature strength and sign certainty vary by setting. Second, algebraic stationary points can be calculated for the quadratic distance terms, but their uncertainty and empirical support differ substantially. Third, NRB_3a lithology-contact distance provides the clearest combined evidence for a supported interior nonlinear response. Fourth, fault-distance evidence is more heterogeneous, particularly across folds. Fifth, Bouguer gravity remains an integral predictor of the multivariate model even though the V11 specification gives it no quadratic term and therefore no $D^*$. Sixth, spatial OOF performance varies across the along-belt partitions and across the two Daly domains with estimable ROC-AUC.

The scientific contribution is therefore the joint interpretation of predictor effects, nonlinear distance response, spatial variation and held-out predictive behavior rather than the identification of one distance value.

## 6.2 Implications for the three-predictor model

The fitted coefficients should be read as conditional components of the multivariate prospectivity model. Fault distance and lithology-contact distance describe potentially nonlinear proximity relationships, while Bouguer gravity supplies complementary geophysical information through its global standardized linear coefficient. Because the predictors enter the same linear predictor, the distance responses should not be interpreted as if gravity or the other geological terms were absent.

The implemented hierarchy allows the distance coefficients and intercept to vary by Daly domain while borrowing strength through population-level distributions. This is useful for representing spatially varying relationships, but hierarchical variation alone does not establish distinct geological mechanisms. The available evidence supports context dependence of the fitted model rather than a causal decomposition of mineralization controls.

The absence of a gravity $D^*$ is a specification consequence: V11 contains a linear global Bouguer-gravity term but no gravity quadratic term. It should not be interpreted as evidence that gravity contributes less information than a distance predictor. The existing manuscript artifacts do not contain a dedicated fold-by-fold posterior summary for the global V11 gravity coefficient, so the discussion does not assign it an unsupported numerical effect size.

## 6.3 Nonlinear distance responses and $D^*$

The two distance predictors provide different levels of evidence for nonlinear response. NRB_3a lithology-contact distance is the clearest case: positive curvature has posterior probability $0.993$–$1.000$ across folds, fold-median $D^*$ values are approximately $26$–$29$ km, and the probability that $D^*$ lies within the observed training support is $0.963$–$1.000$. Its posterior-mean response is minimum-shaped across the reported folds. These quantities together make the modeled interior response comparatively well represented by the available predictor range.

The fault-distance results are less uniform. NRB_3a fault response is generally minimum-shaped in the posterior-mean curves, but curvature sign is uncertain in Fold 2 and the support probability of the algebraic $D^*$ varies from $0.126$ to $0.938. NRB_3b fault curvature is weaker and more variable, with fold-level response shapes including boundary/monotone and interior-extremum behavior. NRB_3b lithology-contact distance is more often positive-curvature but has only moderate within-support probabilities of $0.381$–$0.584$.

These results reinforce the need to distinguish curvature, finite stationary-point existence, minimum/maximum classification, empirical support, posterior concentration and posterior-mean response shape. An algebraic $D^*$ is not automatically a geological target, and a point within empirical support is not automatically a validated geological threshold.

## 6.4 Spatial non-stationarity

Variation in posterior coefficients, curvature probabilities, support probabilities and response shapes indicates that the modeled relationships are not identical across spatially separated settings. The partial-pooling hierarchy provides a principled representation of this variation without requiring separate models for every unit.

The evidence should nevertheless be described as statistical non-stationarity rather than assumed geological causation. Spatially varying coefficients can reflect genuine geological heterogeneity as well as limitations of sampling, predictor representation or the available modeling frame. The completed analysis establishes that the fitted relationships vary; it does not by itself identify why every difference occurs.

## 6.5 Geographic transferability

The primary predictive validation is the four-fold along-belt spatial OOF analysis. V11 exceeded M5 in Fold 3 but underperformed M5 in Folds 1, 2 and 4, while the pooled OOF AUC difference was positive. The resulting pattern shows that predictive behavior depends on the held-out geographic region rather than being uniformly transferable.

The completed multiscale spatial bootstrap retained the observed sign pattern for the Fold 2 and Fold 3 contrasts. This concerns predictive robustness of the completed OOF comparison. It does not establish coefficient stability, response-curve stability or turning-point stability. Predictive stability and parameter stability are distinct properties.

## 6.6 Daly-domain stratification

The secondary stratification of the frozen four-fold OOF predictions reveals heterogeneous predictive behavior between NRB_3a and NRB_3b. In NRB_3a, V11 AUC was $0.661701$ versus $0.535752$ for M5, giving $\Delta AUC=+0.125948$. In NRB_3b, V11 AUC was $0.583815$ versus $0.866897$ for M5, giving $\Delta AUC=-0.283082$.

These values describe predictions generated under the primary spatial OOF design and subsequently grouped by Daly domain. They are not domain refits, Leave-One-Daly-Domain-Out validation or Fold-by-Domain validation. CRZ, NKB, SRB and MMSB contain zero positive observations in the modeling frame, so conventional ROC-AUC is undefined in those domains rather than evidence of poor predictive performance.

The contrast between NRB_3a and NRB_3b therefore adds geological context to the transferability result while retaining the primary four-fold validation design.

## 6.7 Implications for Daly's hypothesis

Several aspects of the completed evidence are compatible with Daly's geological proposition. The two proximity predictors are not uniformly monotonic in the fitted model, and the particularly clear NRB_3a lithology-contact response shows a supported interior minimum-shaped response. Fault-distance behavior is more heterogeneous, with weaker curvature identification and less stable stationary-point evidence across some folds. The spatial OOF results likewise show that predictive behavior changes across geographic partitions, and the domain-stratified results differ between NRB_3a and NRB_3b.

These findings should not be converted into a claim that Daly's hypothesis has been proven. Mathematical curvature is not geological causation. An algebraic stationary point is not a universal geological distance. Empirical support only indicates representation within an observed predictor range; it does not validate a mechanism. Predictive improvement measures discrimination on held-out observations and does not independently establish the geological process responsible for that discrimination.

The most defensible interpretation is therefore qualified: the completed V11 analysis provides empirical evidence consistent with aspects of Daly's proposition, while also demonstrating spatial heterogeneity that prevents a single Copperbelt-wide distance relationship from being treated as established.

## 6.8 Statistical evidence and geological inference

The analysis separates several evidential levels. Coefficient magnitude and credible intervals describe posterior parameter uncertainty. Quadratic curvature describes the sign and strength of the nonlinear component. $D^*$ describes the algebraic stationary point for individual distance-coefficient draws. Response-curve classification describes the shape of the posterior-mean response over the empirical domain. Empirical support describes whether a stationary point falls within the observed training range. Posterior slope behavior describes where the fitted response is increasing or decreasing. Predictive discrimination describes ranking performance on spatially held-out observations. None of these quantities alone establishes a geological mechanism.

Coefficient stability must likewise remain distinct from predictive stability. Fold-to-fold similarity of predictive metrics does not establish similarity of the underlying coefficients, and a stable coefficient does not automatically imply a stable $D^*$. The completed evidence is strongest when several quantities point in the same direction, as in NRB_3a lithology-contact distance, and weaker when curvature or support is uncertain, as in several fault-distance settings.

## 6.9 Data composition and limitations

The complete modeling frame contains 138 positive observations: 107 in NRB_3a and 31 in NRB_3b. CRZ, NKB, SRB and MMSB contain zero positive observations. This composition permits conventional domain-level ROC-AUC only in NRB_3a and NRB_3b. The undefined AUCs in the other four domains do not indicate geological irrelevance or predictive failure; they indicate that the current modeling frame contains no positive-versus-negative contrast with which ROC-AUC could be calculated.

The four-fold along-belt design evaluates transfer across the selected spatial partitions and does not establish transferability to every possible Copperbelt setting. The observations are observational rather than experimental, so statistical associations cannot by themselves establish geological causation. $D^*$ can be unstable when the quadratic coefficient approaches zero, and fold-specific support and curvature uncertainty limit claims of a single precise distance. The $D^*$ summaries produced under the methodological $\beta_{sq}>0.05$ filter are conditional distributions and should not be confused with the unconditional posterior. The threshold itself is a methodological filter rather than an intrinsic Bayesian truth criterion.

## 6.10 Implications for mineral prospectivity research

The results support a reporting practice in which the three-predictor model is presented first, followed by nonlinear distance behavior, stationary-point diagnostics, empirical support, posterior slope behavior, spatial variation and held-out prediction. This prevents a derived $D^*$ from becoming a substitute for the broader multivariate model and prevents pooled predictive performance from concealing geographic heterogeneity.

The study also illustrates why model interpretation should retain the distinction between mathematical response features and geological claims. A supported interior stationary point can be a useful description of the fitted response while remaining non-causal. Similarly, a predictive difference between spatially held-out regions can reveal transferability patterns without identifying a geological mechanism.

## 6.11 Final synthesis

Paper 1 establishes that a partially pooled three-predictor V11 model can represent mineral-prospectivity relationships that vary across the Copperbelt and that the two distance predictors can exhibit nonlinear fitted responses in relevant settings. The strongest combined evidence for an interior response occurs in NRB_3a lithology-contact distance, where positive curvature, a comparatively concentrated stationary-point distribution, high empirical-support probability and the posterior-mean response shape agree. Fault-distance nonlinear evidence is more heterogeneous and less uniformly identified. Bouguer gravity remains an integral global predictor of the multivariate model despite having no quadratic stationary point under the V11 specification.

The primary four-fold along-belt OOF results demonstrate heterogeneous geographic transferability, while secondary stratification of the same frozen OOF predictions shows different predictive behavior between NRB_3a and NRB_3b. Together, these findings provide qualified empirical evidence consistent with aspects of Daly's proposition, but they do not establish a universal geological distance, a causal structural mechanism, or Daly's hypothesis in full. The principal result is therefore a context-dependent, spatially varying three-predictor prospectivity relationship in which nonlinear distance diagnostics are informative components of interpretation rather than the objective of the study itself.
