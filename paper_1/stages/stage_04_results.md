# Stage 04: Results

## 4.1 Multivariate Bayesian posterior structure and coefficient distributions

The V11 hierarchical model jointly estimates the parameters of three continuous geological and geophysical predictors alongside categorical host-rock units. Proximity to faults and proximity to lithological contacts enter with domain-specific linear and quadratic terms, Bouguer gravity anomaly enters through a global linear coefficient, and retained host-rock units enter via a global parameter vector.

### Lithology contact distance posterior parameters
In the mineralized domain NRB_3a, lithology contact distance displays consistent negative linear coefficients and positive quadratic coefficients across all four along-belt spatial folds:
- Linear coefficient median ($\beta_{l}$): ranged from $-0.849$ to $-0.451$.
- Quadratic coefficient median ($\beta_{l^2}$): ranged from $+0.293$ to $+0.334$.
- Posterior probability of positive curvature, $P(\beta_{l^2} > 0)$: $1.000$ in Fold 1, $1.000$ in Fold 2, $0.993$ in Fold 3, and $0.995$ in Fold 4.

In NRB_3b, lithology contact distance also shows negative linear slopes and positive quadratic medians, though with wider posterior dispersion:
- Linear coefficient median ($\beta_{l}$): ranged from $-1.902$ to $-0.819$.
- Quadratic coefficient median ($\beta_{l^2}$): ranged from $+0.209$ to $+0.425$.
- Posterior probability of positive curvature, $P(\beta_{l^2} > 0)$: ranged from $0.695$ to $0.865$.

### Fault distance posterior parameters
Fault distance exhibits substantial fold-to-fold variation in both slope and curvature. In NRB_3a:
- Linear coefficient median ($\beta_{f}$): ranged from $-2.681$ to $-0.703$.
- Quadratic coefficient median ($\beta_{f^2}$): $-0.057$ in Fold 2, and positive ($+0.410$ to $+1.601$) in Folds 1, 3, and 4.
- Posterior probability of positive curvature, $P(\beta_{f^2} > 0)$: $0.984$ in Fold 1, $0.464$ in Fold 2 (sign-uncertain), $0.902$ in Fold 3, and $0.996$ in Fold 4.

In NRB_3b, fault distance shows weak and highly variable curvature:
- Linear coefficient median ($\beta_{f}$): ranged from $+0.181$ to $+1.548$.
- Quadratic coefficient median ($\beta_{f^2}$): ranged from $-0.273$ to $+0.086$.
- Posterior probability of positive curvature, $P(\beta_{f^2} > 0)$: ranged from $0.161$ to $0.708$, indicating absence of consistent curvature evidence.

### Bouguer gravity and host lithology
Regional Bouguer gravity anomaly acts as a stabilizing global linear predictor ($\beta_g$). Because the V11 model specification parameterizes gravity as strictly linear without a quadratic term, it does not possess an algebraic stationary point ($D^*$). Its lack of a $D^*$ diagnostic is an architectural feature of the model, not an empirical finding that gravity is secondary to distance predictors.

## 4.2 Spatial variation in predictor importance (linear predictor variance decomposition)

Evaluating the out-of-fold linear predictor components ($\eta_i = \alpha_{d(i)} + \eta_{f, i} + \eta_{l, i} + \eta_{g, i} + \eta_{rock, i}$) across test observations reveals marked spatial non-stationarity in the relative importance of geological controls along the Copperbelt strike:

| Spatial Test Fold | Test Cells | Known Deposits | Fault Dist Share ($S_f$) | Lith Dist Share ($S_l$) | Bouguer Gravity Share ($S_g$) | Host Rock Share ($S_{rock}$) | Dominant Geological Driver |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Fold 1** (Northern) | 468 | 9 | **86.5%** ($\sigma=7.36$) | 11.1% ($\sigma=0.95$) | 2.0% ($\sigma=0.17$) | 0.4% ($\sigma=0.03$) | **Fault Proximity (86.5%)** |
| **Fold 2** (North-Central) | 468 | 31 | **50.3%** ($\sigma=1.17$) | 23.9% ($\sigma=0.56$) | 5.6% ($\sigma=0.13$) | 20.3% ($\sigma=0.47$) | **Fault Proximity (50.3%)** |
| **Fold 3** (Central-South) | 468 | 37 | **33.1%** ($\sigma=0.79$) | 28.2% ($\sigma=0.68$) | 15.3% ($\sigma=0.37$) | 23.4% ($\sigma=0.56$) | **Multi-factor Co-dominant** |
| **Fold 4** (Southern) | 468 | 61 | **34.9%** ($\sigma=0.65$) | 17.7% ($\sigma=0.33$) | **31.1%** ($\sigma=0.58$) | 16.3% ($\sigma=0.30$) | **Fault & Gravity Co-dominant** |

This decomposition highlights a pronounced geographic gradient:
- In **Fold 1**, prospectivity discrimination is overwhelmingly governed by proximity to major fault structures (86.5% variance share), with mineralization sharply localized along structural corridors.
- In **Folds 2 and 3**, control transitions toward a balanced, multi-factor system where lithological contact proximity (23.9%–28.2%) and host stratigraphy (20.3%–23.4%) play major roles alongside faults.
- In **Fold 4**, regional Bouguer gravity variations surge to account for 31.1% of linear predictor variance, co-dominating with fault distance (34.9%), while the discriminatory power of fault distance alone attenuates markedly ($\sigma = 0.65$ vs $\sigma = 7.36$ in Fold 1).

## 4.3 Distance representation sensitivity analysis ($2 \times 2$ factorial grid)

The $2 \times 2$ sensitivity analysis evaluated whether distance associations, quadratic curvatures, and predictive performance depend on mathematical representation (raw physical distance versus logarithmic distance, and quadratic versus linear functional forms).

### Predictive discrimination across representations
Out-of-fold predictive performance across all four spatial folds is summarized below:

| Model Specification | Distance Transform | Functional Form | Fold 1 OOF AUC | Fold 2 OOF AUC | Fold 3 OOF AUC | Fold 4 OOF AUC | Mean OOF ROC-AUC | Mean OOF PR-AUC | Pooled OOF ROC-AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A** (V11) | Raw ($z$) | Quadratic | 0.8168 | 0.6880 | 0.6761 | 0.5614 | **0.6856 (~0.686)** | **0.1069 (~0.107)** | 0.6887 |
| **Model B** | Log ($z_{\log}$) | Quadratic | 0.8054 | 0.7085 | 0.6492 | 0.5289 | **0.6730 (~0.673)** | **0.1227 (~0.123)** | 0.7009 |
| **Model C** | Raw ($z$) | Linear | 0.9041 | 0.6494 | 0.6338 | 0.4978 | **0.6713 (~0.671)** | **0.1299 (~0.130)** | 0.6665 |
| **Model D** | Log ($z_{\log}$) | Linear | 0.8751 | 0.6768 | 0.6513 | 0.5297 | **0.6832 (~0.683)** | **0.1446 (~0.145)** | 0.6975 |

Key findings from the representation grid:
1. **Predictive Invariance to Functional Form:** Mean spatial OOF ROC-AUC is virtually indistinguishable across all four models ($0.671$ to $0.686$). Adding quadratic terms provides no meaningful improvement in spatial transferability over hierarchical linear models.
2. **Superior Precision-Recall of Log-Linear Formulation:** Hierarchical log-linear Model D achieves the highest mean PR-AUC ($0.1446$) across spatial folds—substantially outperforming quadratic Model A ($0.1069$)—while maintaining a highly competitive mean ROC-AUC ($0.6832$).
3. **Linear Model Discriminatory Power:** In Fold 1, raw-linear Model C achieves an exceptional OOF ROC-AUC of $0.9041$, demonstrating that linear proximity captures the essential structural vectoring signal.

### Curvature sensitivity to logarithmic transformation
While linear distance decay is preserved across all models, quadratic curvature is highly sensitive to the transformation:
- Under raw distance (Model A), NRB_3a lithology contact distance exhibits near-certain positive curvature ($P(\beta_{l^2} > 0) \ge 0.993$ across all folds).
- Under logarithmic transformation (Model B), positive curvature evidence collapses toward ambiguity: $P(\beta_{l^2} > 0)$ drops to $0.700$ (Fold 1), $0.641$ (Fold 2), $0.519$ (Fold 3), and $0.542$ (Fold 4).
- For fault distance in NRB_3a, Model B flips the curvature sign from positive to negative: $P(\beta_{f^2} > 0)$ drops from $0.984 \to 0.093$ (Fold 1), $0.464 \to 0.126$ (Fold 2), and $0.902 \to 0.072$ (Fold 3), indicating concave response in log space.
Because the logarithmic transformation $\log(1 + x_{\text{km}})$ compresses the extended upper tail, it naturally linearizes the distance relationship, rendering quadratic curvature parameters redundant or sign-inverted.

### Pre-registered operational robustness criteria evaluation
Across all 48 evaluable $(\text{fold} \times \text{domain} \times \text{predictor})$ combinations comparing Model A and Model B:
- **Criterion 1 (Curvature Probability Agreement, $|\Delta P| < 0.15$):** Passed in **13 of 48 cases (27.1%)**.
- **Criterion 2 ($D^*$ Median Agreement within 25%):** Passed in **0 of 48 cases (0.0%)**. The mean relative shift in median $D^*$ between raw and log models was $67.2\%$.
- **Criterion 3 (Posterior Support Concentration, $P(D^* \in \text{Support}) \ge 0.50$ in both):** Passed in **15 of 48 cases (31.3%)**.
- **Overall Robustness (Simultaneous satisfaction of all 3 criteria):** **0 of 48 cases (0.0%)**.

Under the pre-registered protocol, the derived stationary point $D^*$ fails the test of representation robustness. It is an artifact of polynomial fitting to skewed raw distance predictors rather than an invariant physical optimum.

### Distal tail perturbation ($p_{95}$ truncation)
Truncating training observations above the 95th percentile (removing ~141 distal non-deposit cells per fold while preserving >90% of deposits) demonstrated marked sensitivity:
- In Fold 1, Model A OOF ROC-AUC collapsed from $0.8168 \to 0.595$, and Model B collapsed from $0.8054 \to 0.576$.
- Mean OOF ROC-AUC dropped from $0.686 \to 0.640$ (Model A) and $0.673 \to 0.622$ (Model B).
This confirms that distal background cells provide essential contrast required to calibrate base rates and define spatial gradients away from prospective corridors.

## 4.4 Population-level reconstruction and $D^*$ diagnostic failure

Reconstructing belt-wide population distributions from the frozen V11 posterior traces confirms that non-linear distance curvature cannot be identified at the population level:

| Population Parameter | Fold 1 Mean | Fold 2 Mean | Fold 3 Mean | Fold 4 Mean | Pooled Mean | Posterior Probability of Curvature |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fault linear ($\mu_{f, lin}$)** | $-0.454$ | $-0.405$ | $-0.572$ | $-0.361$ | **$-0.448$** | $P(\mu < 0) > 85\%$ (Consistent decline) |
| **Fault quadratic ($\mu_{f, sq}$)** | $-0.260$ | $-0.339$ | $-0.094$ | $-0.001$ | **$-0.174$** | $P(\mu > 0) = \mathbf{38.4\%}$ (Predominantly flat/negative) |
| **Lithology linear ($\mu_{l, lin}$)** | $-0.640$ | $-0.471$ | $-0.863$ | $-0.695$ | **$-0.667$** | $P(\mu < 0) > 90\%$ (Consistent decline) |
| **Lithology quadratic ($\mu_{l, sq}$)** | $+0.221$ | $+0.141$ | $+0.105$ | $+0.212$ | **$+0.170$** | $P(\mu > 0) = \mathbf{74.3\%}$ (Ambiguous curvature) |

Because the population quadratic parameters heavily overlap zero, calculating the population algebraic stationary point $z^* = -\mu_{lin} / (2\mu_{sq})$ produces severe division-by-zero singularities:
- For fault distance, the 95% posterior credible interval for population $D^*$ spans from **$-279.36\text{ km}$ to $+366.38\text{ km}$** (Fold 3).
- For lithology contact distance, the 95% credible interval spans from **$-161.69\text{ km}$ to $+152.34\text{ km}$** (Fold 4).
These posterior distributions place heavy density on negative kilometres (physically impossible) and hundreds of kilometres beyond the geographic boundary of the basin. At the population level, $D^*$ is mathematically unidentifiable.

### Domain-level $D^*$ diagnostics within V11 (Model A)
At the domain level within Model A:
- For NRB_3a lithology contact distance, fold-median $D^*$ values clustered tightly between $26.189$ and $28.534\text{ km}$, with within-support probabilities of $0.963$ to $1.000$. However, in Model B, this median shifted to $6.8\text{--}19.1\text{ km}$ ($19.1\text{ km}$ in Fold 1, $12.5\text{ km}$ in Fold 2, $11.4\text{ km}$ in Fold 3, and $6.8\text{ km}$ in Fold 4), demonstrating scale dependence.
- For NRB_3a fault distance, fold medians in Model A swung wildly from $-18.024\text{ km}$ (Fold 2) to $+46.619\text{ km}$ (Fold 3), with within-support probabilities ranging from $0.126$ to $0.938$. In Model B, fault medians collapsed to boundary values ($\approx 0\text{ km}$) or exceeded $80\text{ km}$.
- In unmineralized domains (CRZ, MMSB, NKB, SRB), $P(\beta_{sq} > 0)$ remained below $0.50$, rendering $D^*$ completely undefined.

## 4.5 Four-fold along-belt spatial OOF predictive performance (V11 vs M5)

The primary evaluation of prospective transferability compares the full hierarchical V11 model against the compact M5 baseline across the four predefined along-belt spatial holdout partitions:

| Spatial Evaluation Partition | M5 ROC-AUC [95% CI] | V11 ROC-AUC [95% CI] | Difference ($\Delta$AUC) [95% CI] | V11 PR-AUC | V11 Brier Score | M5 Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** (Northern) | 0.855 [0.695, 0.961] | 0.817 [0.748, 0.868] | $-0.038$ [$-0.132$, $+0.069$] | 0.048 | 0.019 | 0.023 |
| **Fold 2** (North-Central) | 0.824 [0.753, 0.887] | 0.688 [0.599, 0.769] | $-0.136$ [$-0.250$, $-0.020$] | 0.103 | 0.063 | 0.060 |
| **Fold 3** (Central-South) | 0.585 [0.493, 0.668] | 0.676 [0.605, 0.745] | $\mathbf{+0.091}$ [$+0.024$, $+0.159$] | 0.133 | 0.074 | 0.076 |
| **Fold 4** (Southern) | 0.581 [0.506, 0.655] | 0.561 [0.495, 0.626] | $-0.020$ [$-0.083$, $+0.044$] | 0.145 | 0.118 | 0.119 |
| **Pooled Spatial OOF** | 0.525 [0.470, 0.581] | 0.689 [0.647, 0.727] | $\mathbf{+0.164}$ [$+0.084$, $+0.236$] | 0.125 | 0.069 | 0.070 |

Key predictive transferability findings:
1. **Pooled Predictive Superiority:** Aggregated across the entire strike of the Copperbelt, V11 achieves an out-of-fold ROC-AUC of $0.689$ [0.647, 0.727], substantially outperforming M5 ($0.525$ [0.470, 0.581]), yielding a significant pooled improvement of $\Delta\text{AUC} = +0.164$ with $P(\Delta > 0) = 1.000$.
2. **Regional Transferability Divergence:** Across individual folds, relative performance is heterogeneous:
   - In Fold 3, V11 significantly outperforms M5 by $+0.091$ [$+0.024$, $+0.159$].
   - In Fold 2, M5 outperforms V11 by $+0.136$ [$+0.020$, $+0.250$].
   - In Folds 1 and 4, the 95% bootstrap intervals for $\Delta\text{AUC}$ span zero ($-0.038$ and $-0.020$, respectively).
3. **Multiscale Spatial Bootstrap Invariance:** Spatial block bootstrap tests across $10\times 10$, $15\times 15$, $20\times 20$, and $25\times 25$ grid blocks consistently confirmed the observed fold-level sign pattern and pooled median differences ($+0.163$ to $+0.167$).

## 4.6 Secondary Daly-domain stratification of frozen OOF predictions

Aligning the frozen four-fold out-of-fold predictions to the 1,872 grid observations and stratifying by Daly domain provides regional context for model behavior without refitting:

| Daly Domain | Total Cells | Deposit Occurrences | Non-Deposit Cells | M5 OOF ROC-AUC [95% CI] | V11 OOF ROC-AUC [95% CI] | Domain Difference ($\Delta$AUC) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CRZ** | 154 | 0 | 154 | Undefined | Undefined | Undefined |
| **NKB** | 24 | 0 | 24 | Undefined | Undefined | Undefined |
| **SRB** | 1 | 0 | 1 | Undefined | Undefined | Undefined |
| **MMSB** | 16 | 0 | 16 | Undefined | Undefined | Undefined |
| **NRB_3a** | 1,127 | 107 | 1,020 | 0.5358 [0.4849, 0.5864] | 0.6617 [0.6159, 0.7057] | **+0.1259** |
| **NRB_3b** | 550 | 31 | 519 | 0.8669 [0.8115, 0.9139] | 0.5838 [0.4919, 0.6782] | **-0.2831** |

Key insights from domain stratification:
- In **NRB_3a**, which hosts 77.5% of known deposits, V11 demonstrates strong predictive transferability, outperforming the baseline by $+0.1259$ ($0.6617$ vs $0.5358$).
- In **NRB_3b**, M5 achieves higher discrimination ($0.8669$ vs $0.5838$), reflecting the fact that M5's global specification aligns closely with the local gradient in this sector.
- In CRZ, NKB, SRB, and MMSB, ROC-AUC is undefined due to the lack of deposit occurrences. These domains do not represent model failures, but rather the intrinsic limitation of binary classification metrics in unmineralized evaluation partitions.
