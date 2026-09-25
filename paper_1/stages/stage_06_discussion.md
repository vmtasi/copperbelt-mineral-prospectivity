# Stage 06: Discussion

## 6.1 Principal findings and surviving scientific contributions

This study establishes an uncertainty-aware Bayesian framework for mineral prospectivity mapping along the Central African Copperbelt, evaluating how multivariate geological and geophysical controls transfer across spatially separated sectors. By reframing the investigation around multivariate prediction, spatial heterogeneity, and representation robustness, the analysis yields six core scientific findings:

1. **Multivariate Geological Associations:** Copper-cobalt mineralization across the evaluated modeling population is associated with the joint predictor pattern involving fault distance, lithological contact distance, Bouguer gravity anomaly, and host-rock lithology. Proximity effects cannot be interpreted in isolation from this broader multivariate model.
2. **Spatial Heterogeneity in Predictor Importance:** The relative importance of geological controls shifts systematically along the strike of the orogen. Linear predictor variance decomposition reveals that prospectivity discrimination is overwhelmingly fault-dominated in the northwest (Fold 1: 86.5% variance share), transitions to a balanced multi-factor system in the central sectors (Folds 2 and 3), and becomes co-dominated by regional Bouguer gravity in the southeast (Fold 4: 31.1% variance share).
3. **Recurring Distance Associations Across Representations:** Proximity to faults and lithological contacts exhibits a recurring predictive association with mineralization log-odds across the evaluated representations, with magnitude and transferability varying by spatial fold.
4. **Representation Sensitivity of Quadratic Curvature:** While distance decay is robust, quadratic curvature is sensitive to mathematical scaling. Curvature in raw distance models reflects polynomial accommodation of the long, right-skewed spatial tail. When distances are log-transformed, the relationship linearizes on the logit scale, causing quadratic terms to collapse toward zero or invert in sign.
5. **Diagnostic Limits of $D^*$ as a Physical Quantity:** The derived algebraic stationary point $D^*$ fails all pre-registered operational robustness criteria across representations (0 of 48 cases passed simultaneously). At the population level, $D^*$ suffers from denominator singularity with credible intervals spanning $[-279, +366]\text{ km}$. Quadratic curvature is representation-dependent, population-level $D^*$ is not identifiable, and domain/fold stationary points should not be interpreted as invariant geological distances.
6. **Model Parsimony in Prospective Transfer:** Simpler linear and log-linear hierarchical models (Models C and D) achieve out-of-fold spatial discrimination ($AUC = 0.671\text{--}0.683$) comparable to quadratic models ($AUC = 0.673\text{--}0.686$), while the hierarchical log-linear model achieves the highest precision-recall performance ($PR\text{-}AUC = 0.145$).

## 6.2 Geological implications: shifting metallogenic regimes along strike

The empirical shift in predictor variance shares aligns closely with modern tectonic interpretations of the Central African Copperbelt as a heterogeneous, multi-stage orogenic system. 

In the northwestern Lufilian Arc (represented in Fold 1), intense shortening, nappe emplacement, and steep structural imbrication localize hydrothermal fluids into narrow, fault-bounded corridors. Here, proximity to major faults accounts for 86.5% of linear predictor variance ($\sigma = 7.36$), providing sharp spatial discrimination. In this structural regime, fault mapping is the preeminent exploration vector.

Progressing southeastward into the Zambian Copperbelt (Folds 2, 3, and 4), structural styles transition toward basement-cored domes, broader fold geometries, and extensive sub-basin depocentres. Accordingly, the decomposed OOF linear predictor assigns contributions to stratigraphic contact proximity (up to 28.2%), host-rock units (up to 23.4%), and regional Bouguer gravity anomalies (up to 31.1%). In Fold 4, where the variation of the fault-distance component is lower ($\sigma = 0.65$), Bouguer gravity is a co-dominant model component. These contribution patterns are compatible with a setting in which regional architecture is informative; the present analysis does not identify the underlying causal mechanism.

This geographic non-stationarity indicates why exploration models calibrated in one sector of a metallogenic belt may transfer variably to another. The present results show changing model contributions along the evaluated tectonic strike, but do not identify the underlying physical processes.

## 6.3 Representation sensitivity and model parsimony in spatial data

The sensitivity analysis highlights a critical methodological hazard in data-driven mineral prospectivity modeling: the conflation of polynomial curvature with geological process.

Distance measurements in regional exploration grids are inevitably right-skewed: a few cells lie within prospective mineralized corridors ($< 5\text{ km}$), while the vast majority extend across tens of kilometres of unmineralized regional basin. Fitting a quadratic polynomial ($\beta_1 z + \beta_2 z^2$) to such raw physical distances forces the model to balance the steep proximal slope against the broad distal background. If the model is penalized for over-predicting in the distant tail, the quadratic term can bend upward at large distances as a representation-dependent consequence of polynomial rigidity.

This interpretation is supported by two empirical findings in this study:
1. **Logarithmic Linearization:** Applying a log transformation ($\log(1 + x_{\text{km}})$) naturally compresses the distal tail. Under this transformation (Model B), quadratic curvature largely vanishes ($P(\beta_{sq} > 0)$ drops to $\approx 50\%$), and the fitted response is adequately represented by a monotonic log-linear decay (Model D).
2. **Tail Truncation Collapse:** Truncating training observations at the 95th percentile ($p_{95}$) removes distal background cells, causing Fold 1 discrimination to drop from $0.817 \to 0.595$. The fitted predictions and discrimination are sensitive to the distal tail under both Model A and Model B.

Because hierarchical linear models (Models C and D) match or exceed the predictive discrimination of quadratic models under spatial cross-validation—with log-linear Model D achieving the highest PR-AUC ($0.145$)—the principle of model parsimony indicates that linear distance representations are superior for operational exploration vectoring.

## 6.4 The diagnostic limits of $D^*$ and avoidance of false reification

Historically, the identification of a stationary point $D^*$ at $25\text{--}35\text{ km}$ from faults or contacts was tempting to interpret as a "sweet spot" or optimal structural offset for fluid flow and mineral deposition. The rigorous evaluation conducted here indicates that such reification is not supported:

1. **Scale Sensitivity:** Changing the distance scale from raw physical kilometres to logarithmic units shifts the median stationary point by an average of $67.2\%$, causing NRB_3a lithology medians to collapse from $26\text{--}28\text{ km}$ down to $7\text{--}19\text{ km}$, and fault medians to collapse to boundary values ($0\text{ km}$) or explode beyond $80\text{ km}$.
2. **Denominator Singularity:** In unconstrained Bayesian inference, whenever the posterior distribution of the quadratic parameter $\beta_{sq}$ includes density near zero, the ratio $z^* = -\beta_{lin} / (2\beta_{sq})$ produces explosive, fat-tailed Cauchy-like distributions. At the population level, this produces 95% credible intervals spanning $[-279, +366]\text{ km}$, indicating population-level non-identifiability.
3. **Absence in Geophysical Predictors:** Bouguer gravity enters linearly and has no $D^*$ diagnostic, despite contributing up to 31.1% of predictive variance in Fold 4.

$D^*$ must therefore be recognized as a model-derived diagnostic whose interpretation is dependent on the distance representation rather than as an intrinsic property of the Copperbelt hydrothermal system. Domain/fold stationary points should not be interpreted as invariant geological distances, and the analysis does not support establishing fixed spatial buffer corridors from these polynomial diagnostics.

## 6.5 Honest spatial evaluation versus historical linear baselines

A notable question arising in Copperbelt prospectivity modeling is why early linear logistic regression models reportedly performed near chance levels ($AUC \le 0.50$), whereas the modern hierarchical Raw-Linear Model C evaluated here achieves a spatial OOF ROC-AUC of $0.671$ (and $0.904$ in Fold 1) across the evaluated folds.

Forensic analysis reveals that this is an apples-to-oranges comparison stemming from fundamentally different modeling architectures:
- **Early Linear Baselines:** Early models typically used unpooled, non-hierarchical logistic regressions fit to crude tract-boundary buffers, lacking geological domain stratification, omitting regional gravity, and failing to account for regional baseline odds. Furthermore, many were evaluated on poorly isolated or spatially confounded validation sets.
- **Hierarchical Linear Model C:** In contrast, Model C incorporates:
  1. High-fidelity geological proxies (distances to mapped structural lineaments and lithological contacts);
  2. Hierarchical partial pooling across Daly tectonic domains, allowing regional intercepts and slopes to adapt while sharing belt-wide strength;
  3. Regional Bouguer gravity and host stratigraphy as stabilizing multivariate controls;
  4. Rigorous Bayesian regularization preventing over-fitting.

When linear distance predictors are embedded in an appropriate hierarchical spatial architecture, they provide recurring predictive associations across the evaluated spatial folds without requiring polynomial complexity.

## 6.6 Re-evaluating Daly's tectonic hypothesis

The completed evidence reframes how Daly's tectonic domain hypothesis should be utilized in quantitative resource assessment:

- **Supported: Regional heterogeneity:** The data support the interpretation that the Central African Copperbelt comprises distinct tectonic regimes that should not automatically be treated as a single homogeneous metallogenic province. Base rates, decomposed predictor contribution shares, and out-of-fold transferability differ between domains such as NRB_3a and NRB_3b.
- **Not supported: A universal invariant distance interpretation:** The empirical evidence does not support the interpretation that Daly's domains are characterized by fixed, invariant turning points or characteristic distances. Proximity associations recur across evaluated representations, while their mathematical form and predictive contribution vary geographically.

The primary value of Daly's framework is as a principled geological prior for **spatial hierarchical stratification**, allowing Bayesian models to pool information across sub-basins without imposing false uniformity.

## 6.7 Methodological limitations

To ensure rigorous interpretation, several study limitations must be explicitly noted:

1. **Observational Modeling Frame:** The analysis is based on a 2D spatial grid of 1,872 cells with 138 documented deposits. The absence of deposits in CRZ, NKB, SRB, and MMSB reflects compiled surface occurrences, rendering binary ranking metrics (ROC-AUC) mathematically undefined in those domains.
2. **Spatial Holdout Resolution:** While four-fold along-belt spatial holdout provides rigorous protection against spatial leakage, four folds provide a discrete discretization of a continuous tectonic gradient.
3. **Dimensionality Constraints:** Geological predictors represent surface and near-surface mapped features and regional gravity. The model does not include 3D structural geometries, subsurface seismic reflections, or depth-to-basement models, which may account for unexplained spatial variance.
4. **Conditional Posterior Filtering:** Diagnostic summaries evaluating $\beta_{sq} > 0.05$ represent conditional methodological filters rather than unconditional Bayesian posteriors, and must not be interpreted as physical validation criteria.

## 6.8 Best practice recommendations for quantitative prospectivity modeling

The findings of this study suggest four concrete methodological recommendations for quantitative mineral prospectivity research:

1. **Mandate Spatially Separated Holdout Validation:** Random-cell cross-validation must be abandoned in regional prospectivity studies. Models must be validated by holding out contiguous geographical blocks along orogenic strike to assess genuine exploration transferability.
2. **Test Representation Sensitivity:** Proximity relationships should routinely be evaluated across raw, logarithmic, and linear representations. Researchers must verify whether apparent non-linearities or turning points persist under monotonic transformations before inferring physical trapping mechanisms.
3. **Adopt Hierarchical Bayesian Architectures:** Rather than choosing between over-generalized global models and data-starved local models, prospectivity workflows should employ hierarchical models with partial pooling across tectono-stratigraphic domains.
4. **Report Predictor Contribution Shares:** Researchers should report linear predictor variance decompositions across spatial test folds to expose geographic non-stationarity in geological controls, rather than relying solely on global performance metrics.

## 6.9 Final synthesis

Paper 1 shows that Bayesian mineral prospectivity mapping across the Central African Copperbelt can be formulated as an uncertainty-aware, multivariate hierarchical system that explicitly accounts for spatial heterogeneity.

Distance to major faults and distance to lithological contacts provide recurring predictive associations across the evaluated representations, with variable magnitude and transferability. However, the apparent quadratic curvature and derived stationary points (\(D^*\)) observed in raw-distance models are representation-dependent under the evaluated specifications and do not satisfy the pre-registered robustness criteria for representation-robust curvature and stationary-point inference.

By shifting attention from universal physical turning points to spatial non-stationarity, the study shows that decomposed predictor contributions shift from fault-associated patterns in the northwest to more distributed multi-factor and gravity-associated patterns in the southeast. Hierarchical linear and log-linear models capture these recurring associations parsimoniously within the evaluated spatial folds
