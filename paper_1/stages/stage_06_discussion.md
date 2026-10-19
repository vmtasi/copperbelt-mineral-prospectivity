# Stage 06: Discussion

## 6.1 Principal findings and surviving scientific contributions

This study evaluates a hierarchical Bayesian prospectivity model using four along-belt holdout folds and a distance-representation sensitivity analysis. The frozen OOF scores are posterior mean conditional probabilities, while the reconstructed posterior-uncertainty artifact reports cell-level 95% credible intervals for $p_i$; no future Bernoulli outcomes are sampled. The results support the following scoped findings:

1. **Multivariate Geological Associations:** Copper-cobalt mineralization across the evaluated modeling population is associated with the joint predictor pattern involving fault distance, lithological contact distance, Bouguer gravity anomaly, and host-rock lithology. Proximity effects cannot be interpreted in isolation from this broader multivariate model.
2. **Fold variation in component dispersion:** SD-normalized shares of additive components in the posterior-mean OOF linear predictor vary across folds (fault-distance share: 86.6% in Fold 1; gravity share: 28.7% in Fold 4). These are not shares of outcome variance, causal effects, or measures of isolated discrimination.
3. **Recurring Distance Associations Across Representations:** Proximity to faults and lithological contacts exhibits a recurring predictive association with mineralization log-odds across the evaluated representations, with magnitude and transferability varying by spatial fold.
4. **Representation sensitivity of curvature:** Posterior support for quadratic curvature differs between raw and log distance representations. The sensitivity analysis establishes this difference for the evaluated specifications but does not identify a unique response form or the geological mechanism behind the difference.
5. **Diagnostic Limits of $D^*$ as a Physical Quantity:** The derived algebraic stationary point $D^*$ fails the pre-registered operational robustness criteria in 47 of 48 cases; only Fold 1 NRB_3a lithological contact satisfied all three criteria simultaneously (1 of 48, 2.1%). At the population level, $D^*$ suffers from denominator singularity with credible intervals spanning $[-376, +379]\ \mathrm{km}$. Quadratic curvature is representation-dependent, population-level $D^*$ is not identifiable, and domain/fold stationary points should not be interpreted as invariant geological distances.
6. **Discrimination across evaluated specifications:** Mean OOF ROC-AUC ranges from $0.688$ to $0.719$ across the four distance representations; log-linear Model D has the highest mean ROC-AUC ($0.7188$) and mean PR-AUC ($0.197$). These fold-specific comparisons do not establish a universally preferable model form.

## 6.2 Interpreting spatial variation cautiously

The fold-wise decomposition describes how the standard deviations of posterior-mean additive predictor components compare within the held-out cells. Fault distance has the largest SD-normalized share in Fold 1 (86.6%); the gravity share is 28.7% in Fold 4. Contact distance and host lithology have larger shares in Folds 2 and 3 than in Fold 1. These quantities characterize the model's decomposed OOF linear predictor, not mineralization mechanisms or outcome variance.

Regional geological literature provides context for considering spatially varying associations, but these results do not map folds directly to unique tectonic mechanisms, identify causal fluid pathways, or explain why discrimination differs. They motivate geological follow-up rather than establish a geological process from the component decomposition.

## 6.3 Representation sensitivity and model parsimony in spatial data

The sensitivity analysis illustrates why polynomial curvature should not be equated with a geological process.

The evaluated raw and log specifications differ in how distance is represented, and the posterior curvature summaries change across those representations. Tail truncation also changes the observed Fold 1 AUC under the specified sensitivity perturbation. These results establish sensitivity in the evaluated design; they do not determine whether the cause is tail leverage, model form, or another feature of the data.

Two empirical results motivate this caution:
1. **Representation change:** Applying $\log(1+x_{\mathrm{km}})$ compresses upper-tail values, and the posterior curvature support differs between Models A and B. This difference does not establish that the log-linear model is the true response function.
2. **Tail perturbation:** Under the specified $p_{95}$ training-data truncation, Fold 1 AUC changes from $0.846$ to $0.616$ (Model A) and from $0.829$ to $0.656$ (Model B). This demonstrates sensitivity to that perturbation, not a general causal role for distal background cells.

Model D has the highest mean ROC-AUC ($0.7188$) and mean PR-AUC ($0.197$). The two linear specifications bracket the quadratic specifications in mean ROC-AUC, so the comparison supports evaluating simpler forms but does not establish that they are universally preferable.

## 6.4 The diagnostic limits of $D^*$ and avoidance of false reification

Historically, the identification of a stationary point $D^*$ at $25\!-
!35\ \mathrm{km}$ from faults or contacts was tempting to interpret as a "sweet spot" or optimal structural offset for fluid flow and mineral deposition. The rigorous evaluation conducted here indicates that such reification is not supported:

1. **Scale Sensitivity:** Changing the distance scale from raw physical kilometres to logarithmic units shifts the median stationary point by an average of $60.6\%$ (median $64.1\%$). In NRB_3a, lithology-contact medians of $26.3$--$29.7\ \mathrm{km}$ under raw distance shift to $13.4$--$36.4\ \mathrm{km}$ under log distance, while fault medians shift to approximately $0\ \mathrm{km}$ in three folds or to $95.5\ \mathrm{km}$ in the fourth.
2. **Denominator Singularity:** In unconstrained Bayesian inference, whenever the posterior distribution of the quadratic parameter $\beta_{sq}$ includes density near zero, the ratio $z^* = -\beta_{lin} / (2\beta_{sq})$ produces explosive, fat-tailed Cauchy-like distributions. At the population level, this produces 95% credible intervals spanning $[-376, +379]\ \mathrm{km}$, indicating population-level non-identifiability.
3. **Linear gravity term:** Bouguer gravity enters linearly and has no $D^*$ diagnostic. Its SD-normalized component share is 28.7% in Fold 4; this is not a share of predictive variance or evidence of causal importance.

$D^*$ must therefore be recognized as a model-derived diagnostic whose interpretation is dependent on the distance representation rather than as an intrinsic property of the Copperbelt hydrothermal system. Domain/fold stationary points should not be interpreted as invariant geological distances, and the analysis does not support establishing fixed spatial buffer corridors from these polynomial diagnostics.

## 6.5 Comparing model specifications without attributing gains

V11 and M5 differ in predictor set and model structure, so their OOF score differences do not isolate the effect of a single feature or of hierarchical pooling. V11's pooled ROC-AUC is $0.707$ versus $0.559$ for M5, with a pooled difference of $+0.147$; the fold-level differences vary, including $+0.056$ in Fold 3 and $-0.118$ in Fold 2. These are results for the specified frozen predictions, not evidence about why one specification scores higher.

The distance-representation comparison is separate from the V11--M5 comparison. Model C has mean OOF ROC-AUC $0.688$ and Fold 1 ROC-AUC $0.930$, illustrating why fold-level results should accompany the mean. The available results do not establish a historical performance trend or support attributing differences from unrelated studies to any one modeling choice.

## 6.6 Re-evaluating Daly's tectonic hypothesis

The completed evidence reframes how Daly's tectonic domain hypothesis should be utilized in quantitative resource assessment:

- **Observed variation:** Posterior summaries and descriptive OOF scores vary across the supplied domain labels. The four zero-positive domains provide no within-domain positive-class information, and their model effects rely more strongly on hierarchical pooling.
- **Scope of the domain analysis:** The labels index model terms and post-hoc strata; this study does not formally test or validate Daly's tectonic framework. The sensitivity results do not support universal, representation-invariant distance thresholds.

In this model, the supplied labels serve as a grouping variable for partial pooling. The analysis does not independently establish that they are a validated geological prior or that every labeled group represents a distinct causal regime.

## 6.7 Methodological limitations

To ensure rigorous interpretation, several study limitations must be explicitly noted:

1. **Observational Modeling Frame:** The analysis is based on a 2D spatial grid of 1,872 cells with 138 documented deposits. The absence of deposits in CRZ, NKB, SRB, and MMSB reflects compiled surface occurrences, rendering binary ranking metrics (ROC-AUC) mathematically undefined in those domains.
2. **Spatial Holdout Resolution:** The PCA-ordered four-fold design reduces local train-test overlap but cannot guarantee independence across fold boundaries. Four folds discretize a continuous spatial gradient and assess only the specified holdout design.
3. **Dimensionality Constraints:** Geological predictors represent surface and near-surface mapped features and regional gravity. The model does not include 3D structural geometries, subsurface seismic reflections, or depth-to-basement models, which may account for unexplained spatial variance.
4. **Conditional Posterior Filtering:** Diagnostic summaries evaluating $\beta_{sq} > 0.05$ represent conditional methodological filters rather than unconditional Bayesian posteriors, and must not be interpreted as physical validation criteria.

## 6.8 Best practice recommendations for quantitative prospectivity modeling

The findings of this study suggest four concrete methodological recommendations for quantitative mineral prospectivity research:

1. **Match validation to the prediction target:** For a new-region prediction task, use geographically separated evaluation appropriate to the intended transfer and report the blocking design and its limitations; no single block design guarantees independence or universal transferability.
2. **Test Representation Sensitivity:** Proximity relationships should routinely be evaluated across raw, logarithmic, and linear representations. Researchers must verify whether apparent non-linearities or turning points persist under monotonic transformations before inferring physical trapping mechanisms.
3. **Consider partial pooling when justified:** Hierarchical models can share information across groups, but their use and grouping structure should be justified by the data and scientific question.
4. **Label component summaries precisely:** If reporting additive predictor-component shares, state their normalization, compute them on held-out predictions, and avoid interpreting them as causal importance or outcome variance explained.

## 6.9 Final synthesis

Paper 1 evaluates a hierarchical Bayesian model for the Central African Copperbelt under a specified four-fold along-belt holdout design. It quantifies parameter posterior uncertainty and reconstructs cell-level posterior uncertainty for held-out conditional probabilities; the frozen OOF table contains the posterior mean probabilities, while the accompanying uncertainty artifact provides 95% credible intervals for $p_i$ and does not include sampled future outcomes.

Distance to major faults and distance to lithological contacts provide recurring predictive associations across the evaluated representations, with variable magnitude and transferability. However, the apparent quadratic curvature and derived stationary points (\(D^*\)) observed in raw-distance models are representation-dependent under the evaluated specifications and do not satisfy the pre-registered robustness criteria for representation-robust curvature and stationary-point inference.

The SD-normalized component shares vary across folds, and discrimination also varies between the specified held-out sectors. These are descriptive predictive results, not evidence that causal controls shift geographically. The evaluated distance associations and response forms should be interpreted within this modeling frame; the contact-map and original Bouguer-grid provenance remain incomplete in the available project files.
