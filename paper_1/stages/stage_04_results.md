# Stage 04: Results

## 4.1 Multivariate Bayesian posterior structure and coefficient distributions

The V11 hierarchical model jointly estimates the parameters of three continuous geological and geophysical predictors alongside categorical host-rock units. Proximity to faults and proximity to lithological contacts enter with domain-specific linear and quadratic terms, Bouguer gravity anomaly enters through a global linear coefficient, and retained host-rock units enter via a global parameter vector. For each held-out cell, the frozen V11 posterior draws support a cell-specific distribution for the conditional occurrence probability $p_i$, and the reconstructed uncertainty artifact reports posterior mean, median, standard deviation, and 95% credible intervals for these cell-level probabilities.

### Lithology contact distance posterior parameters
In the mineralized domain NRB_3a, lithology contact distance displays consistent negative linear coefficients and positive quadratic coefficients across all four along-belt spatial folds:
- Linear coefficient median ($\beta_{l}$): ranged from $-0.977$ to $-0.618$.
- Quadratic coefficient median ($\beta_{l^2}$): ranged from $+0.352$ to $+0.394$.
- Posterior probability of positive curvature, $P(\beta_{l^2} > 0)$: $1.000$ in Fold 1, $1.000$ in Fold 2, $0.998$ in Fold 3, and $0.999$ in Fold 4.

In NRB_3b, lithology contact distance also shows negative linear slopes and positive quadratic medians, though with wider posterior dispersion:
- Linear coefficient median ($\beta_{l}$): ranged from $-2.005$ to $-0.970$.
- Quadratic coefficient median ($\beta_{l^2}$): ranged from $+0.301$ to $+0.481$.
- Posterior probability of positive curvature, $P(\beta_{l^2} > 0)$: ranged from $0.766$ to $0.896$.

### Fault distance posterior parameters
Fault distance exhibits substantial fold-to-fold variation in both slope and curvature. In NRB_3a:
- Linear coefficient median ($\beta_{f}$): ranged from $-2.622$ to $-0.705$.
- Quadratic coefficient median ($\beta_{f^2}$): $-0.018$ in Fold 2, and positive ($+0.380$ to $+1.627$) in Folds 1, 3, and 4.
- Posterior probability of positive curvature, $P(\beta_{f^2} > 0)$: $0.985$ in Fold 1, $0.488$ in Fold 2 (sign-uncertain), $0.917$ in Fold 3, and $0.996$ in Fold 4.

In NRB_3b, fault distance shows weak and highly variable curvature:
- Linear coefficient median ($\beta_{f}$): ranged from $+0.091$ to $+1.435$.
- Quadratic coefficient median ($\beta_{f^2}$): ranged from $-0.242$ to $+0.114$.
- Posterior probability of positive curvature, $P(\beta_{f^2} > 0)$: ranged from $0.191$ to $0.737$, indicating absence of consistent curvature evidence.

### Bouguer gravity and host lithology
Regional Bouguer gravity anomaly acts as a stabilizing global linear predictor ($\beta_g$). Because the V11 model specification parameterizes gravity as strictly linear without a quadratic term, it does not possess an algebraic stationary point ($D^*$). Its lack of a $D^*$ diagnostic is an architectural feature of the model, not an empirical finding that gravity is secondary to distance predictors.

## 4.2 Spatial variation in SD-normalized component shares

Across test observations, the SD-normalized shares of posterior-mean additive components in the OOF linear predictor vary across the Copperbelt folds. These shares describe component dispersion, not outcome variance explained or causal importance:

| Spatial Test Fold | Test Cells | Positive Cells | Fault SD-Share ($S_f$) | Contact SD-Share ($S_l$) | Gravity SD-Share ($S_g$) | Host-Rock SD-Share ($S_{rock}$) | Largest SD-normalized Share |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Fold 1** (Northern) | 468 | 9 | **86.6%** ($\sigma=7.46$) | 11.0% ($\sigma=0.95$) | 2.0% ($\sigma=0.17$) | 0.4% ($\sigma=0.04$) | **Fault distance (86.6%)** |
| **Fold 2** (North-Central) | 468 | 31 | **46.8%** ($\sigma=1.14$) | 28.0% ($\sigma=0.68$) | 5.6% ($\sigma=0.14$) | 19.6% ($\sigma=0.48$) | **Fault distance (46.8%)** |
| **Fold 3** (Central-South) | 468 | 37 | 30.5% ($\sigma=0.80$) | **32.8%** ($\sigma=0.85$) | 14.4% ($\sigma=0.37$) | 22.4% ($\sigma=0.58$) | **Contact distance (32.8%)** |
| **Fold 4** (Southern) | 468 | 61 | **33.2%** ($\sigma=0.66$) | 23.2% ($\sigma=0.46$) | 28.7% ($\sigma=0.57$) | 14.9% ($\sigma=0.29$) | **Fault distance (33.2%)** |

This decomposition highlights a pronounced geographic gradient:
- In **Fold 1**, the fault-distance component has the largest SD-normalized share (86.6%); its component standard deviation is 7.46.
- In **Folds 2 and 3**, the shares are distributed across fault distance, contact distance, host lithology, and gravity, with contact distance at 28.0% and 32.8%, and host lithology at 19.6% and 22.4%, respectively.
- In **Fold 4**, the gravity share is 28.7% and the fault-distance share is 33.2%; their component standard deviations are 0.57 and 0.66, respectively. These figures do not measure either predictor's isolated discrimination or explain why fold performance differs.

## 4.3 Distance representation sensitivity analysis ($2 \times 2$ factorial grid)

The $2 \times 2$ sensitivity analysis evaluated whether distance associations, quadratic curvatures, and predictive performance depend on mathematical representation (raw physical distance versus logarithmic distance, and quadratic versus linear functional forms).

### Predictive discrimination across representations
Out-of-fold predictive performance across all four spatial folds is summarized below:

| Model Specification | Distance Transform | Functional Form | Fold 1 OOF AUC | Fold 2 OOF AUC | Fold 3 OOF AUC | Fold 4 OOF AUC | Mean OOF ROC-AUC | Mean OOF PR-AUC | Pooled OOF ROC-AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A** (V11) | Raw ($z$) | Quadratic | 0.8456 | 0.7092 | 0.6870 | 0.6048 | **0.7116 (~0.712)** | **0.1365 (~0.136)** | 0.7066 |
| **Model B** | Log ($z_{\log}$) | Quadratic | 0.8293 | 0.7447 | 0.6608 | 0.5818 | **0.7042 (~0.704)** | **0.1619 (~0.162)** | 0.7256 |
| **Model C** | Raw ($z$) | Linear | 0.9298 | 0.6726 | 0.6393 | 0.5087 | **0.6876 (~0.688)** | **0.1791 (~0.179)** | 0.6783 |
| **Model D** | Log ($z_{\log}$) | Linear | 0.9133 | 0.7248 | 0.6636 | 0.5735 | **0.7188 (~0.719)** | **0.1967 (~0.197)** | 0.7251 |

Key findings from the representation grid:
1. **Mean ROC-AUC Across Specifications:** Mean spatial OOF ROC-AUC ranges from $0.688$ to $0.719$. Model D (log-linear) has the highest mean ($0.7188$), followed by Model A ($0.7116$); the difference is modest and does not establish universal superiority of a functional form.
2. **Mean PR-AUC:** Model D has the highest mean PR-AUC ($0.1967$), compared with $0.1365$ for Model A and $0.1619$ for Model B. This is a descriptive comparison across the evaluated folds.
3. **Fold 1 Model C Result:** Raw-linear Model C achieves OOF ROC-AUC $0.9298$ in Fold 1; its mean OOF ROC-AUC is $0.6876$, so the single-fold result should not be generalized to all sectors.

### Curvature sensitivity to logarithmic transformation
Posterior curvature support is sensitive to the distance transformation:
- Under raw distance (Model A), NRB_3a lithology contact distance exhibits near-certain positive curvature ($P(\beta_{l^2} > 0) \ge 0.998$ across all folds).
- Under logarithmic transformation (Model B), positive curvature evidence collapses toward ambiguity: $P(\beta_{l^2} > 0)$ drops to $0.870$ (Fold 1), $0.793$ (Fold 2), $0.674$ (Fold 3), and $0.930$ (Fold 4).
- For fault distance in NRB_3a, Model B gives $P(\beta_{f^2} > 0) = 0.101$ (Fold 1), $0.143$ (Fold 2), $0.070$ (Fold 3), and $0.720$ (Fold 4), with quadratic medians of $-0.163$, $-0.266$, $-0.292$, and $+0.097$, indicating a predominantly concave response in log space.
The logarithmic transformation $\log(1 + x_{\mathrm{km}})$ compresses upper-tail distances, and the fitted curvature support changes between the raw and log representations. This sensitivity analysis does not identify the mechanism producing that change.

### Pre-registered operational robustness criteria evaluation
Across all 48 evaluable $(\mathrm{fold} \times \mathrm{domain} \times \mathrm{predictor})$ combinations comparing Model A and Model B:
- **Criterion 1 (Curvature Probability Agreement, $|\Delta P| < 0.15$):** Passed in **17 of 48 cases (35.4%)**.
- **Criterion 2 ($D^*$ Median Agreement within 25%):** Passed in **2 of 48 cases (4.2%)**. The mean relative shift in median $D^*$ between raw and log models was $60.6\%$ (median $64.1\%$).
- **Criterion 3 (Posterior Support Concentration, $P(D^* \in \mathrm{Support}) \ge 0.50$ in both):** Passed in **17 of 48 cases (35.4%)**.
- **Overall Robustness (Simultaneous satisfaction of all 3 criteria):** **1 of 48 cases (2.1%)**: Fold 1, NRB_3a lithological contact.

Under the pre-registered protocol, the derived stationary point $D^*$ fails representation robustness in 47 of 48 cases. The single exception (Fold 1, NRB_3a lithological contact) does not establish a general invariant threshold; $D^*$ remains representation-dependent under the evaluated raw-distance polynomial specification and is not an invariant physical quantity.

### Distal tail perturbation ($p_{95}$ truncation)
Truncating training observations above the 95th percentile of either distance (removing 141--142 cells per fold while retaining 87--95% of deposits) demonstrated marked sensitivity:
- In Fold 1, Model A OOF ROC-AUC fell from $0.8456 \to 0.6163$, and Model B fell from $0.8293 \to 0.6558$.
- Mean OOF ROC-AUC dropped from $0.7116 \to 0.6670$ (Model A) and $0.7042 \to 0.6647$ (Model B).
These results show that the evaluated predictions and discrimination change when distal background cells are removed under this perturbation. They do not isolate why performance changes or establish that the removed cells are universally essential.

## 4.4 Population-level reconstruction and $D^*$ diagnostic failure

Reconstructing belt-wide population distributions from the frozen V11 posterior traces confirms that non-linear distance curvature cannot be identified at the population level:

| Population Parameter | Fold 1 Mean | Fold 2 Mean | Fold 3 Mean | Fold 4 Mean | Pooled Mean | Posterior Probability of Curvature |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fault linear ($\mu_{f, lin}$)** | $-0.451$ | $-0.458$ | $-0.619$ | $-0.388$ | **$-0.479$** | $P(\mu < 0)$ = 0.73--0.88 (Consistent decline) |
| **Fault quadratic ($\mu_{f, sq}$)** | $-0.272$ | $-0.307$ | $-0.057$ | $-0.008$ | **$-0.161$** | $P(\mu > 0)$ = 0.27--0.53 (Predominantly flat/negative) |
| **Lithology linear ($\mu_{l, lin}$)** | $-0.730$ | $-0.573$ | $-0.929$ | $-0.774$ | **$-0.751$** | $P(\mu < 0)$ = 0.89--0.92 (Consistent decline) |
| **Lithology quadratic ($\mu_{l, sq}$)** | $+0.270$ | $+0.194$ | $+0.154$ | $+0.300$ | **$+0.230$** | $P(\mu > 0)$ = 0.73--0.82 (Ambiguous curvature) |

Because the population quadratic parameters heavily overlap zero, calculating the population algebraic stationary point $z^* = -\mu_{lin} / (2\mu_{sq})$ produces severe division-by-zero singularities:
- For fault distance, the widest fold-level 95% posterior credible interval for population $D^*$ spans from **$-376.30\ \mathrm{km}$ to $+379.47\ \mathrm{km}$** (Fold 3).
- For lithology contact distance, the widest fold-level 95% credible interval spans from **$-135.38\ \mathrm{km}$ to $+165.27\ \mathrm{km}$** (Fold 3).
These posterior distributions place heavy density on negative kilometres (physically impossible) and hundreds of kilometres beyond the geographic boundary of the basin. At the population level, $D^*$ is mathematically unidentifiable.

### Domain-level $D^*$ diagnostics within V11 (Model A)
At the domain level within Model A:
- For NRB_3a lithology contact distance, fold-median $D^*$ values clustered between $26.33$ and $29.65\ \mathrm{km}$, with within-support probabilities of $0.983$ to $1.000$.
- For NRB_3a fault distance, fold medians in Model A ranged from $-10.41\ \mathrm{km}$ (Fold 2) to $+46.94\ \mathrm{km}$ (Fold 3), with within-support probabilities ranging from $0.312$ to $0.982$. The negative Fold 2 median reflects a sign-uncertain and effectively flat quadratic coefficient ($P(\beta_{f^2} > 0) = 0.488$).
- The corresponding Model B log-distance diagnostics and the paired raw-versus-log robustness assessment are reported in Section 4.3.
- In CRZ, MMSB, NKB, and SRB, no positive observations are available to inform within-domain positive-class behavior. Domain effects remain estimated through the hierarchical model and depend more strongly on shared population structure; any algebraic $D^*$ in these domains is not evidence of an observed positive-class threshold. Domain-stratified ROC-AUC is undefined in these single-class data.

## 4.5 Four-fold along-belt spatial OOF predictive performance (V11 vs M5)

The primary evaluation compares frozen OOF scores from V11 and M5 across the four predefined along-belt spatial holdout partitions. The table's fold and pooled 95% intervals are row-bootstrap intervals over fixed prediction scores; they are distinct from the separate multiscale spatial block-bootstrap assessment:

| Spatial Evaluation Partition | M5 ROC-AUC [95% CI] | V11 ROC-AUC [95% CI] | Difference ($\Delta$AUC) [95% CI] | V11 PR-AUC | V11 Brier Score | M5 Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** (Northern) | 0.890 [0.765, 0.984] | 0.846 [0.801, 0.887] | $-0.044$ [$-0.124$, $+0.056$] | 0.110 | 0.019 | 0.024 |
| **Fold 2** (North-Central) | 0.827 [0.759, 0.891] | 0.709 [0.631, 0.782] | $-0.118$ [$-0.224$, $-0.011$] | 0.126 | 0.063 | 0.059 |
| **Fold 3** (Central-South) | 0.631 [0.537, 0.716] | 0.687 [0.616, 0.758] | $+0.056$ [$-0.002$, $+0.112$] | 0.147 | 0.077 | 0.077 |
| **Fold 4** (Southern) | 0.625 [0.557, 0.693] | 0.605 [0.544, 0.666] | $-0.020$ [$-0.080$, $+0.040$] | 0.163 | 0.118 | 0.119 |
| **Pooled Spatial OOF** | 0.559 [0.509, 0.615] | 0.707 [0.667, 0.744] | $\mathbf{+0.147}$ [$+0.073$, $+0.217$] | 0.135 | 0.069 | 0.070 |

Key predictive transferability findings:
1. **Pooled OOF Discrimination:** Aggregated across the entire strike of the Copperbelt, V11 achieves an out-of-fold ROC-AUC of $0.707$ [0.667, 0.744], outperforming M5 ($0.559$ [0.509, 0.615]), yielding a pooled improvement of $\Delta\mathrm{AUC} = +0.147$ with $P(\Delta > 0) = 1.000$.
2. **Regional Transferability Divergence:** Across individual folds, relative performance is heterogeneous:
   - In Fold 3, the paired row-bootstrap V11-minus-M5 ROC-AUC difference is $+0.056$ [$-0.002$, $+0.112$]; the interval includes zero.
   - In Fold 2, M5 outperforms V11 by $+0.118$ [$+0.011$, $+0.224$].
   - In Folds 1 and 4, the 95% bootstrap intervals for $\Delta\mathrm{AUC}$ span zero ($-0.044$ and $-0.020$, respectively).
3. **Multiscale Spatial Block-Bootstrap Assessment:** Resampling occupied coordinate blocks at $10\times10$, $15\times15$, $20\times20$, and $25\times25$ grid scales is a separate robustness assessment. The pooled $\Delta\mathrm{AUC}$ interval included zero at $10\times10$ ($[-0.015,+0.277]$) but excluded zero at $15\times15$ ($[+0.030,+0.251]$), $20\times20$ ($[+0.035,+0.248]$), and $25\times25$ ($[+0.048,+0.238]$). Fold 2 $\Delta\mathrm{AUC}$ remained negative at the $15\times15$ and $25\times25$ scales. These results apply to the evaluated block definitions and do not replace the fixed-score row-bootstrap intervals in the table or represent cell-level posterior uncertainty.

## 4.6 Secondary Daly-domain stratification of frozen OOF predictions

Aligning the frozen four-fold out-of-fold predictions to the 1,872 grid observations and stratifying by Daly domain provides regional context for model behavior without refitting:

| Daly Domain | Total Cells | Positive Cells | Non-Positive Cells | M5 OOF ROC-AUC [95% CI] | V11 OOF ROC-AUC [95% CI] | Domain Difference ($\Delta$AUC) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CRZ** | 154 | 0 | 154 | Undefined | Undefined | Undefined |
| **NKB** | 24 | 0 | 24 | Undefined | Undefined | Undefined |
| **SRB** | 1 | 0 | 1 | Undefined | Undefined | Undefined |
| **MMSB** | 16 | 0 | 16 | Undefined | Undefined | Undefined |
| **NRB_3a** | 1,127 | 107 | 1,020 | 0.5772 [0.5326, 0.6286] | 0.6824 [0.6363, 0.7241] | **+0.1052** |
| **NRB_3b** | 550 | 31 | 519 | 0.8780 [0.8267, 0.9197] | 0.6089 [0.5188, 0.6987] | **-0.2691** |

Key insights from domain stratification:
- In **NRB_3a**, which contains 107 of the 138 positive cells, V11 has higher domain-stratified ROC-AUC than M5 ($0.6824$ vs $0.5772$; difference $+0.1052$).
- In **NRB_3b**, M5 has higher domain-stratified ROC-AUC than V11 ($0.8780$ vs $0.6089$; difference $-0.2691$); the comparison does not establish why the scores differ.
- In CRZ, NKB, SRB, and MMSB, ROC-AUC is undefined because these strata contain no positive labels in this modeling frame. This is a single-class metric limitation, not evidence of model failure or of true geological absence.
