# Stage 01: Scientific Question

## 1.1 Central Research Question
Does the relationship between geological proxies and Cu-Co mineralization transfer uniformly across spatially distinct tectonic regions of the Central African Copperbelt, and, if not, does this spatial non-stationarity produce geographically varying predictive performance?

## 1.2 Core Hypothesis
The relationship between fundamental geological predictors and mineralization is spatially non-stationary across tectonically distinct regions of the Copperbelt. Consequently, a single global predictive formulation will not exhibit uniform predictive performance across geographic evaluation regions. A domain-aware hierarchical formulation is expected to recover regional structure where global relationships fail, though neither approach is hypothesized to be universally superior across all regions.

## 1.3 Scientific Objectives
*   **Objective 1:** Establish spatial heterogeneity in predictor relationships (Determine whether the estimated relationships between fundamental geological predictors and mineralization vary across tectonic domains).
*   **Objective 2:** Establish spatial heterogeneity in physical response scales (Determine whether the characteristic spatial scales associated with geological predictors are consistent across domains).
*   **Objective 3:** Test geographic transferability (Compare a compact global model (M5) and a domain-aware hierarchical model (V11) using strict along-belt spatial OOF validation).
*   **Objective 4:** Test whether relative model performance changes geographically (Evaluate whether model superiority remains consistent across spatial folds or reverses between regions).
*   **Objective 5:** Test the robustness of observed spatial differences (Determine whether observed differences in model performance persist under spatially structured bootstrap resampling at multiple spatial scales).
*   **Objective 6:** Assess whether global metrics adequately characterize regional predictive behavior (Compare pooled OOF performance with geographically isolated performance to determine whether global metrics conceal regional failures).

## 1.4 Scope Limitations (What We Are NOT Claiming)
*   We do not claim that V11 universally outperforms M5, nor that hierarchical models are inherently superior everywhere.
*   We do not claim that the predictive failure in Fold 4 is definitively proven to have one specific geological cause.
*   We do not claim that spatial non-stationarity has been proven for *every* possible geological predictor, nor that our three selected proxies exhaust all geological information.
*   We do not claim that the Copperbelt can *never* be modeled globally.
*   **The precise proposition:** We test whether predictive relationships and predictive performance can reasonably be treated as spatially stationary across tectonically distinct regions.