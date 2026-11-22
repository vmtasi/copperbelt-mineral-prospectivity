> **Superseded (22 Nov 2026).** This proposal relied on three errors: the area AUC, the gravity AUC and the provenance of the 0.674 reference (see the erratum in `master_results_report.md`). The manuscript was revised directly from the corrected results; this file is kept as a record only.

# Proposed manuscript revision — for approval only

**Status:** proposal. The frozen manuscript in `paper_1/manuscript/` has not been modified. No figure or table has been replaced. This document identifies the claims that V15 supports, qualifies or contradicts, and provides replacement text for later approval.

## 1. Required factual corrections

| Current statement | Problem | Required action |
|---|---|---|
| "a predictor-matched global logistic model with the same four predictors provides a reference at 0.674" | No global-logit OOF predictions or trace are stored in the benchmark. The number cannot be reproduced from the repository. It also mixes a logit model with the DVHM specification. | Remove the 0.674 comparison, or fit and archive a matched global-logit model before retaining it. Use the V15 M0–M2 ladder for the matched-likelihood comparison. |
| "Hierarchical partial pooling also gives a fold-mean ROC-AUC of 0.725 against 0.691 for the reference model" | Same unverified reference. | Remove or replace with the V15 fold table. |
| "The comparison provides additional context for the predictive performance of the heterogeneous formulation" | Implies a verified gain over the reference. The central M1→M2 interval includes zero. | Rewrite as a descriptive comparison with the interval stated. |
| "fault distance dominates in the south-east, whereas the share of Bouguer gravity increases toward the north-west" | Component shares are descriptive and depend on within-sector predictor spread. M3's gravity gain is not stable. | Retain only as a descriptive decomposition, explicitly not a geological effect. |
| "The model therefore demonstrates useful regional discrimination" | Discrimination is real but spatially uneven, miscalibrated and autocorrelated. | Qualify: "descriptive ranking skill, uneven across sectors". |
| Calibration slope 0.61 as the only calibration statement | V15 shows slopes from 0.398 (global) to 0.752 (M3); the V14 control is 0.588. | Report the ladder, not a single value. |
| No statement on cell area | Area alone reaches 0.559 AUC; 205 small cells have no positives; the area-covariate model has the best M2-family log score. | Add the area audit and the area sensitivity. |
| No statement on prior dependence | Between-domain scales move almost proportionally with prior width. | Add prior sensitivity and remove claims about the magnitude of heterogeneity. |
| No statement on residual autocorrelation | Moran's I is 0.15–0.25 at p = 0.001 for all primary models. | Add the residual-dependence limitation. |

## 2. Proposed abstract replacement

Replace the current abstract paragraph with the following. It keeps the frozen V14 result as the benchmark but does not pair it with the unverified 0.674 reference.

> Regional mineral prospectivity mapping often assumes that predictor–occurrence relationships are constant across a mineralized belt. We evaluate this assumption in the Central African Copperbelt with a partially pooled Bayesian model in which intercepts and the linear and quadratic coefficients of fault distance and lithological-contact distance vary by Daly tectonic domain. The model is applied to a 1,872-cell nominal 5 km × 5 km grid (138 cells with a compiled occurrence record) and evaluated by holding out each of four contiguous along-belt sectors in turn. We compare four specifications under an identical complementary log–log likelihood, with log clipped-cell area as a fixed exposure offset: a global model, a domain-intercept model, a domain-varying distance model, and a domain-varying distance-and-gravity model. Pooled out-of-fold ROC-AUC rises from 0.688 for the global model to 0.709 with domain intercepts, 0.734 with domain-varying distances and 0.802 with domain-varying distances and gravity. The gain from domain-varying distances over domain intercepts is +0.025 ROC-AUC and +0.0062 mean held-out log score per cell, but its spatial-block bootstrap interval includes zero. The larger gain from domain-varying gravity (+0.068 ROC-AUC, +0.030 log score) is concentrated in the two southern folds, especially Fold 1 (nine positive cells), is not accompanied by a stable gravity-alone signal, and is not separated from a random-forest comparator. A logit model with area as a covariate rather than an offset improves held-out log score by 0.0084 per cell relative to the offset formulation. Between-domain scale parameters are prior-dependent, categorical lithology contributes negligibly, and residuals remain spatially autocorrelated. The frozen V14 benchmark (ROC-AUC 0.716) is reproduced to within 0.006 by a like-for-like control. We conclude that domain-varying relationships can be represented and can improve descriptive ranking, but the present inventory does not establish a stable predictive advantage over domain-specific baseline rates.

## 3. Proposed Results replacement

Replace the paragraph beginning "Pooled over all 1,872 cells, DVHM ROC-AUC is 0.716" and the following comparator table with the ladder below. Retain the frozen V14 sentence as a separate benchmark statement.

> Pooled over all 1,872 cells, the frozen V14 model achieves ROC-AUC 0.716. A like-for-like V15 control reproduces this result at 0.709, a difference of 0.006 attributable to respecification of the sampler. Under a common complementary log–log likelihood with log clipped-cell area as a fixed offset, the global model achieves 0.688, the domain-intercept model 0.709, the domain-varying distance model 0.734 and the domain-varying distance-and-gravity model 0.802. Mean held-out log scores are -0.2725, -0.2633, -0.2571 and -0.2267 per cell respectively.
>
> The pre-specified central comparison is the domain-varying distance model against the domain-intercept model. The descriptive gain is +0.025 ROC-AUC and +0.0062 mean log score per cell, with a 25 km spatial-block bootstrap interval for the log-score difference of -0.0291 to +0.0160. The interval includes zero. The gain is not uniform: the domain-varying model is worse in Folds 1 and 4 and better in Folds 2 and 3. The data therefore do not establish that varying the distance slopes improves prediction beyond domain-specific baseline rates.
>
> The domain-varying gravity model improves on the distance model by +0.068 ROC-AUC and +0.0304 mean log score per cell, with an interval of +0.0110 to +0.0530. This gain is concentrated in Folds 1 and 2, especially Fold 1, which contains nine positive cells; gravity alone has fold AUCs of 0.473, 0.498, 0.423 and 0.487; and the difference from the random forest has an interval of -0.0080 to +0.0262. The gravity result is therefore conditional and should not be interpreted as a stable geological effect.
>
> Cell area, defined as the 5 km cell footprint intersected with the permissive-tract union in EPSG:32735, has a pooled AUC of 0.559. The 205 complete-case cells below 1 km2 contain no positives. Replacing the fixed log-area offset with an estimated area covariate under a logit link raises pooled AUC to 0.737 and improves held-out log score by 0.0084 per cell relative to the offset model (offset minus covariate = -0.0084; interval -0.0141 to -0.0028). Exposure treatment therefore affects the probabilistic ranking and must be reported.
>
> Categorical lithology contributes negligibly: removing it changes AUC by about 0.0007 and mean log score by about 0.00002 per cell. The contact-distance predictor is retained. Between-domain scale parameters are prior-dependent: changing the prior width from 0.5 to 2.0 moves the posterior mean of `sigma_beta_f_lin` from 0.76 to 1.79 and changes held-out log score from -0.248 to -0.268. Residual Moran's I remains 0.15–0.25 at p = 0.001 for all primary models.

Proposed table:

| Model | ROC-AUC | PR-AUC | Mean log score | Brier | Calibration slope |
|---|---:|---:|---:|---:|---:|
| Global (M0) | 0.688 | 0.120 | -0.2725 | 0.0763 | 0.398 |
| Domain intercept (M1) | 0.709 | 0.135 | -0.2633 | 0.0738 | 0.462 |
| Domain-varying distance (M2) | 0.734 | 0.163 | -0.2571 | 0.0692 | 0.489 |
| Domain-varying distance and gravity (M3) | 0.802 | 0.211 | -0.2267 | 0.0654 | 0.752 |
| V14 control | 0.709 | 0.136 | -0.2526 | 0.0689 | 0.588 |
| Frozen V14 | 0.716 | 0.137 | -0.2510 | 0.0688 | 0.605 |
| Random forest | 0.753 | 0.157 | -0.2352 | 0.0649 | 0.952 |

## 4. Proposed Discussion replacement

Replace the paragraph beginning "Spatially separated validation shows that predictive skill varies by sector" with:

> Spatially separated validation shows that predictive skill varies by sector, but it does not identify a stable advantage for domain-varying slopes. The central contrast between the domain-intercept and domain-varying distance models has a point estimate favouring the latter and an interval that includes zero. The contrast is also fold-dependent, reversing in two of four sectors. The appropriate conclusion is descriptive: the model can represent along-belt heterogeneity, and the resulting ranking is competitive, but the present inventory does not establish that the added slope heterogeneity improves transfer.
>
> Two further results constrain interpretation. First, the between-domain scales are not identified by these data. Their posteriors track prior width, and domains without positive cells contribute almost no information. Statements about how much relationships vary, or about population-level stationary points, are therefore not data-determined. Second, residuals remain spatially autocorrelated under every specification. Out-of-fold scores are optimistic relative to prediction in unsampled ground, and the four folds are four spatial units, not a precise sampling distribution.
>
> The gravity result requires particular caution. The preliminary impression of a large fold-to-fold reversal in gravity discrimination is not reproduced with the stored Bouguer field. The domain-varying gravity model achieves a higher pooled score, but its gain is concentrated in the smallest fold, gravity alone is near chance, and most domain coefficients are prior-dominated. It should be presented as a conditional predictive result, not as evidence for a domain-specific gravity law.
>
> The area analysis changes the probabilistic conclusion. Area is a genuine support measure, not an exploration-effort correction, and the offset formulation is geometrically defensible. Nevertheless, an estimated area covariate under a logit link fits better on held-out log score. The choice of exposure treatment is therefore a modelling decision with material consequences, not a neutral technical detail.
>
> Finally, the 184 USGS records in positive cells are not 184 independent observations. The model has 138 positive cells; multiple records in one cell are not independent, and two records fall in model-negative cells. The 147 RDC.ROCKS observations are a separate corpus with zero confirmed cross-source identities and are not model positives.

## 5. Proposed Conclusions replacement

Replace the conclusion paragraphs with:

> This study asked whether domain-varying geological relationships add defensible predictive value beyond domain-specific baseline rates in the Central African Copperbelt. Under four-fold spatial validation, a domain-varying distance model achieved higher descriptive pooled discrimination than a matched global model (0.734 versus 0.688) and a small further gain over a domain-intercept model (0.734 versus 0.709). The central slope-heterogeneity contrast was not separated from zero under spatial-block uncertainty, reversed in two folds, and was accompanied by prior-dependent between-domain scales. The data therefore support the representation of heterogeneity, but not a strong claim that varying slopes improve transfer.
>
> A domain-varying gravity model achieved the highest pooled score (0.802), but the gain was concentrated in Folds 1 and 2, especially the nine-positive Fold 1, gravity alone was near chance, and the difference from a random forest included zero. It is a conditional result, not a stable gravity law. Categorical lithology contributed negligibly. Exposure treatment mattered: an area covariate under a logit link improved held-out log score relative to the complementary log–log offset. Residual spatial autocorrelation remained positive in every model, so the out-of-fold scores should be read as optimistic relative to prediction in unsampled ground.
>
> The frozen V14 benchmark remains reproducible at ROC-AUC 0.716 and is reproduced to within 0.006 by a like-for-like control. The earlier comparison with a 0.674 global reference is not retained because no stored predictions or trace support that number. Future work should prioritise a matched global-logit control, a defensible host-lithology field, reconstruction of the fault geometry, and a study-area decision based on geological architecture rather than the extent of the compiled inventory.

## 6. Sentences to delete or rewrite

Delete or rewrite these exact strings in `paper_1/manuscript/main.tex`:

1. Abstract: "a predictor-matched global logistic model with the same four predictors provides a reference at 0.674."
2. Results: "A predictor-matched global logistic model with the same four predictors provides the reference at 0.674 (difference +0.041; 95% CI +0.006 to +0.077; Table ...)."
3. Results: "Hierarchical partial pooling also gives a fold-mean ROC-AUC of 0.725 against 0.691 for the reference model; this is a mean of four fold values."
4. Results: "Relative to the reference model, fold-level differences vary along the belt" and the following fold-difference sentence, unless the reference model's predictions are restored and archived.
5. Discussion: "The reference model differs from the DVHM in both domain intercepts and domain slopes, so the pooled difference cannot be attributed to slope heterogeneity alone." Replace with the M1→M2 contrast, which isolates slope heterogeneity.
6. Conclusions: "As a reference, the matched global logistic model achieves 0.674 using the same four predictors. The comparison provides additional context for the predictive performance of the heterogeneous formulation."
7. Any statement that the between-domain scales, the amount of heterogeneity or the stationary point are estimated by the data.

## 7. What must not change

- Do not alter the frozen V14 numbers, traces, OOF predictions, figures or benchmark.
- Do not replace the 0.716 benchmark with a V15 number. Report 0.716 as frozen and 0.709 as the V15 control.
- Do not describe the 184 USGS records as independent positives.
- Do not describe the 147 RDC.ROCKS observations as model positives.
- Do not describe area as an exploration-effort correction.
- Do not claim a stable domain-specific gravity effect.
- Do not claim residual independence.

## 8. Approval gates

Before any manuscript edit:

1. Approve removal or replacement of the 0.674 reference.
2. Decide whether to fit and archive a matched global-logit model, or to use only the V15 M0–M2 ladder.
3. Decide whether the area-covariate result is reported as a sensitivity or promoted to a primary specification.
4. Approve the prior-sensitivity wording, which removes claims about the magnitude of heterogeneity.
5. Approve the new limitations paragraph on residual autocorrelation and fault provenance.

No manuscript file will be changed until these are approved.
