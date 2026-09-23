# Stage 06: Discussion

## 6.1 Principal findings and surviving scientific contributions

This study establishes an uncertainty-aware Bayesian framework for mineral prospectivity mapping along the Central African Copperbelt, evaluating how multivariate geological and geophysical controls transfer across spatially separated sectors. By reframing the investigation around multivariate prediction, spatial heterogeneity, and representation robustness, the analysis yields six core scientific findings:

1. **Multivariate Geological Controls:** Copper-cobalt mineralization across the Copperbelt is governed by the joint action of structural conduits (fault distance), stratigraphic traps (lithological contact distance), regional crustal architecture (Bouguer gravity anomaly), and host-rock lithology. Proximity effects cannot be understood in isolation from this broader multivariate system.
2. **Spatial Heterogeneity in Predictor Importance:** The relative importance of geological controls shifts systematically along the strike of the orogen. Linear predictor variance decomposition reveals that prospectivity discrimination is overwhelmingly fault-dominated in the northwest (Fold 1: 86.5% variance share), transitions to a balanced multi-factor system in the central sectors (Folds 2 and 3), and becomes co-dominated by regional Bouguer gravity in the southeast (Fold 4: 31.1% variance share).
3. **Robust Distance Associations Across Representations:** Proximity to faults and lithological contacts exhibits a consistent, robust negative association with mineralization log-odds across all spatial folds and mathematical representations (raw distance, logarithmic distance, linear forms, and quadratic forms).
4. **Representation Sensitivity of Quadratic Curvature:** While distance decay is robust, quadratic curvature is sensitive to mathematical scaling. Curvature in raw distance models reflects polynomial accommodation of the long, right-skewed spatial tail. When distances are log-transformed, the relationship linearizes on the logit scale, causing quadratic terms to collapse toward zero or invert in sign.
5. **Diagnostic Failure of $D^*$ as a Physical Quantity:** The derived algebraic stationary point $D^*$ fails all pre-registered operational robustness criteria across representations (0 of 48 cases passed simultaneously). At the population level, $D^*$ suffers from denominator singularity with credible intervals spanning $[-279, +366]\text{ km}$. $D^*$ is a mathematical diagnostic of polynomial parameterization, not an identifiable physical trap distance or optimal exploration vector.
6. **Model Parsimony in Prospective Transfer:** Simpler linear and log-linear hierarchical models (Models C and D) achieve out-of-fold spatial discrimination ($AUC = 0.671\text{--}0.683$) comparable to quadratic models ($AUC = 0.673\text{--}0.686$), while the hierarchical log-linear model achieves the highest precision-recall performance ($PR\text{-}AUC = 0.145$).

## 6.2 Geological implications: shifting metallogenic regimes along strike

The empirical shift in predictor variance shares aligns closely with modern tectonic interpretations of the Central African Copperbelt as a heterogeneous, multi-stage orogenic system. 

In the northwestern Lufilian Arc (represented in Fold 1), intense shortening, nappe emplacement, and steep structural imbrication localize hydrothermal fluids into narrow, fault-bounded corridors. Here, proximity to major faults accounts for 86.5% of linear predictor variance ($\sigma = 7.36$), providing sharp spatial discrimination. In this structural regime, fault mapping is the preeminent exploration vector.

Progressing southeastward into the Zambian Copperbelt (Folds 2, 3, and 4), structural styles transition toward basement-cored domes, broader fold geometries, and extensive sub-basin depocentres. Accordingly, the model demonstrates that predictive variance redistributes toward stratigraphic contact proximity (up to 28.2%), reactive host-rock units (up to 23.4%), and regional Bouguer gravity anomalies (up to 31.1%). In Fold 4, where the discriminatory contrast of fault distance attenuates ($\sigma = 0.65$), Bouguer gravity emerges as a primary co-dominant predictor, reflecting the fundamental control of regional basement architecture and rift-basin sub-sills on hydrothermal circulation.

This geographic non-stationarity demonstrates why exploration models calibrated in one sector of a metallogenic belt frequently perform erratically when applied to another: the dominant physical processes localizing mineralization evolve along the tectonic strike.

## 6.3 Representation sensitivity and model parsimony in spatial data

The sensitivity analysis highlights a critical methodological hazard in data-driven mineral prospectivity modeling: the conflation of polynomial curvature with geological process.

Distance measurements in regional exploration grids are inevitably right-skewed: a few cells lie within prospective mineralized corridors ($< 5\text{ km}$), while the vast majority extend across tens of kilometres of unmineralized regional basin. Fitting a quadratic polynomial ($\beta_1 z + \beta_2 z^2$) to such raw physical distances forces the model to balance the steep proximal slope against the broad distal background. If the model is penalized for over-predicting in the distant tail, the quadratic term can bend upward at large distances simply as a mathematical artifact of polynomial rigidity.

This mechanism is confirmed by two empirical findings in this study:
1. **Logarithmic Linearization:** Applying a log transformation ($\log(1 + x_{\text{km}})$) naturally compresses the distal tail. Under this transformation (Model B), quadratic curvature largely vanishes ($P(\beta_{sq} > 0)$ drops to $\approx 50\%$), and the relationship is fully captured by a monotonic log-linear decay (Model D).
2. **Tail Truncation Collapse:** Truncating training observations at the 95th percentile ($p_{95}$) removes distal background cells, causing Fold 1 discrimination to drop from $0.817 \to 0.595$. The model relies on the distant background cells to define its spatial contrast.

Because hierarchical linear models (Models C and D) match or exceed the predictive discrimination of quadratic models under spatial cross-validation—with log-linear Model D achieving the highest PR-AUC ($0.145$)—the principle of model parsimony indicates that linear distance representations are superior for operational exploration vectoring.

## 6.4 The diagnostic limits of $D^*$ and avoidance of false reification

Historically, the identification of a stationary point $D^*$ at $25\text{--}35\text{ km}$ from faults or contacts was tempting to interpret as a "sweet spot" or optimal structural offset for fluid flow and mineral deposition. The rigorous evaluation conducted here demonstrates that such reification is scientifically untenable:

1. **Scale Sensitivity:** Changing the distance scale from raw physical kilometres to logarithmic units shifts the median stationary point by an average of $67.2\%$, causing NRB_3a lithology medians to collapse from $26\text{--}28\text{ km}$ down to $7\text{--}19\text{ km}$, and fault medians to collapse to boundary values ($0\text{ km}$) or explode beyond $80\text{ km}$.
2. **Denominator Singularity:** In unconstrained Bayesian inference, whenever the posterior distribution of the quadratic parameter $\beta_{sq}$ includes density near zero, the ratio $z^* = -\beta_{lin} / (2\beta_{sq})$ produces explosive, fat-tailed Cauchy-like distributions. At the population level, this produces 95% credible intervals spanning $[-279, +366]\text{ km}$, demonstrating complete parameter unidentifiability.
3. **Absence in Geophysical Predictors:** Bouguer gravity enters linearly and has no $D^*$ diagnostic, despite contributing up to 31.1% of predictive variance in Fold 4.

$D^*$ must therefore be recognized as a mathematical diagnostic of a specific polynomial parameterization rather than an intrinsic property of the Copperbelt hydrothermal system. Exploration programs should avoid establishing fixed spatial buffer corridors based on unverified polynomial stationary points.

## 6.5 Honest spatial evaluation versus historical linear baselines

A notable question arising in Copperbelt prospectivity modeling is why early linear logistic regression models reportedly performed near chance levels ($AUC \le 0.50$), whereas the modern hierarchical Raw-Linear Model C evaluated here achieves a robust spatial OOF ROC-AUC of $0.671$ (and $0.904$ in Fold 1).

Forensic analysis reveals that this is an apples-to-oranges comparison stemming from fundamentally different modeling architectures:
- **Early Linear Baselines:** Early models typically used unpooled, non-hierarchical logistic regressions fit to crude tract-boundary buffers, lacking geological domain stratification, omitting regional gravity, and failing to account for regional baseline odds. Furthermore, many were evaluated on poorly isolated or spatially confounded validation sets.
- **Hierarchical Linear Model C:** In contrast, Model C incorporates:
  1. High-fidelity geological proxies (distances to mapped structural lineaments and lithological contacts);
  2. Hierarchical partial pooling across Daly tectonic domains, allowing regional intercepts and slopes to adapt while sharing belt-wide strength;
  3. Regional Bouguer gravity and host stratigraphy as stabilizing multivariate controls;
  4. Rigorous Bayesian regularization preventing over-fitting.

When linear distance predictors are embedded in an appropriate hierarchical spatial architecture, they provide powerful, highly transferable prospectivity predictions without requiring polynomial complexity.

## 6.6 Re-evaluating Daly's tectonic hypothesis

The completed evidence reframes how Daly's tectonic domain hypothesis should be utilized in quantitative resource assessment:

- **Confirmed: Regional Tectonic Heterogeneity:** The data strongly confirm Daly's fundamental insight that the Central African Copperbelt comprises distinct tectonic regimes that cannot be treated as a single homogeneous metallogenic province. Base rates, structural versus stratigraphic variance shares, and out-of-fold transferability differ markedly between domains such as NRB_3a and NRB_3b.
- **Refuted: Universal Geometric Optima:** The empirical evidence decisively refutes the hypothesis that Daly's domains are characterized by fixed, invariant turning points or characteristic distances. Proximity matters everywhere, but its mathematical representation is linear or log-linear, and its predictive contribution varies geographically.

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

Paper 1 demonstrates that Bayesian mineral prospectivity mapping across the Central African Copperbelt is most effectively formulated as an uncertainty-aware, multivariate hierarchical system that explicitly accounts for spatial heterogeneity. 

Distance to major faults and distance to lithological contacts provide robust, transferable vectoring signals that consistently exhibit negative associations with deposit probability. However, the apparent quadratic curvature and derived stationary points ($D^*$) observed in raw distance models are artifacts of polynomial parameterization and distal tail compression, failing pre-registered robustness criteria. 

By replacing the quest for universal physical turning points with a rigorous investigation of spatial non-stationarity, the study shows that mineralization controls shift from fault-dominated structural conduits in the northwest to multi-factor and gravity-influenced systems in the southeast. Hierarchical linear and log-linear models capture these relationships parsimoniously, providing a rigorous, transferable foundation for mineral prospectivity assessment across complex metallogenic terranes.
