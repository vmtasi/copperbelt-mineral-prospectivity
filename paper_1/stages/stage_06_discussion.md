# Stage 06: Discussion

## 6.1 Contribution of the study

This study contributes an integrated treatment of nonlinear distance effects and geographic transferability in Copperbelt mineral prospectivity. The V11 framework estimates linear and quadratic distance effects within a partially pooled hierarchical Bayesian model, allowing posterior uncertainty to be carried through coefficient, curvature and predicted-probability summaries. The derived $D^*$ quantity makes the stationary point of a fitted quadratic explicit, while the empirical-support analysis prevents that mathematical quantity from being interpreted without reference to the observed distance range.

The study also combines hierarchical representation of spatial variation with four-fold along-belt spatial OOF validation. The validation asks whether relationships transfer to held-out regions rather than only whether predictions rank nearby cells in a randomly mixed sample. Finally, the existing frozen V11 OOF predictions are stratified by Daly geological domain, adding geological context to predictive performance without changing the primary validation design or refitting the model.

## 6.2 Implications for Daly's hypothesis

Several components of the evidence are compatible with Daly's geological proposition. The fitted responses are not uniformly linear, and the strongest case, NRB_3a lithology-contact distance, shows positive curvature, a concentrated stationary-point distribution and high empirical-support probability. The fault-distance results and the fold-level predictive differences also show that modeled behavior can vary across spatial settings.

The evidence remains limited in important ways. A mathematical stationary point is not automatically a geological target. Empirical support makes a stationary point less extrapolative but does not establish causality. Predictive discrimination measures ranking performance, not the geological mechanism that produced it. Likewise, spatially varying hierarchical coefficients are evidence of modeled heterogeneity, not automatically causal geological heterogeneity. The results therefore support a qualified, testable interpretation of Daly's hypothesis rather than a claim that it has been proven.

## 6.3 NRB_3a and NRB_3b

The contrast between NRB_3a and NRB_3b is central to the transferability result. In NRB_3a, V11's domain-stratified OOF AUC was approximately $0.662$, compared with approximately $0.536$ for M5. In NRB_3b, V11's AUC was approximately $0.584$, compared with approximately $0.867$ for M5. These are not rankings of universally better models. They show that the hierarchical formulation's predictive behavior depends on the geological setting represented by the held-out observations.

The contrast also cautions against interpreting partial pooling as a guarantee of improved generalization. V11 can improve discrimination in one domain and reduce it in another while remaining competitive in pooled OOF evaluation. The scientific implication is not that one model should be selected as the winner everywhere, but that model transferability must be evaluated in relation to geological and spatial context.

## 6.4 Data composition and domain limitations

The complete modeling frame contains 138 positive observations: 107 in NRB_3a and 31 in NRB_3b. CRZ, NKB, SRB and MMSB contain zero positive observations. This composition permits conventional domain-level ROC-AUC only in NRB_3a and NRB_3b. The undefined AUCs in the other four domains do not indicate geological irrelevance, predictive failure or absence of mineralization in a broader geological sense. They indicate that the current modeling frame contains no positive-versus-negative contrast with which ROC-AUC could be calculated.

The smaller positive count in NRB_3b also contributes to wider uncertainty around its domain-level AUC and makes its result more sensitive to the limited positive sample than the larger NRB_3a result. The domain table should therefore be read as a description of the current spatial OOF evidence, not as an equal evidential comparison across all six domains.

## 6.5 Statistical evidence and geological inference

The analysis separates several claims that are often conflated. Curvature concerns the sign and uncertainty of a quadratic coefficient. $D^*$ concerns the algebraic stationary point implied by that coefficient. Empirical support concerns whether that point falls inside the observed distance range. Response-curve classification concerns the shape of the posterior-mean response over the support. Predictive discrimination concerns ranking performance on held-out observations. None of these quantities alone establishes a geological mechanism.

The results therefore support statements such as “minimum-shaped modeled response” or “spatially variable OOF discrimination” where the stored evidence warrants them. They do not support converting a stationary point into a recommended fault distance or treating a predictive advantage as proof of a structural process.

## 6.6 Limitations

The principal limitations are those imposed by the data and design. Domain imbalance means that four Daly domains cannot contribute a conventional ROC-AUC in the current modeling frame. NRB_3b contains fewer positive observations than NRB_3a, producing greater uncertainty for its domain-level estimate. The four-fold along-belt design evaluates transfer across the selected spatial partitions; it does not establish transferability to every possible Copperbelt setting. The observations are observational rather than experimental, so statistical associations cannot by themselves establish geological causation. Finally, $D^*$ can be unstable when the quadratic coefficient is near zero, and fold-specific support and curvature uncertainty limit claims of a single precise distance.

## 6.7 Final synthesis

The central question was whether the mineralization–fault-distance relationship contains nonlinear structure relevant to the geological hypothesis and whether its predictive behavior transfers uniformly across the Copperbelt. The completed evidence indicates that nonlinear, minimum-shaped responses can be identified in some settings, but that their curvature, stationary points, empirical support and predictive usefulness vary across folds and domains. V11 provides a useful partially pooled framework for representing this variation, yet its pooled advantage does not imply uniform regional superiority. The most defensible conclusion is that Copperbelt prospectivity relationships are context-dependent: the model provides case-specific evidence consistent with nonlinear and spatially variable geological control, while the available data do not justify a universal geological distance or a claim that Daly's hypothesis has been proven in full.
