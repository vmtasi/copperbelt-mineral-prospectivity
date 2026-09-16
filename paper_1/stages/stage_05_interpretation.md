# Stage 05: Interpretation

## 5.1 Meaning of curvature and $D^*$

The V11 results support treating distance response as a posterior distribution of possible curves rather than as a single deterministic distance effect. A positive quadratic coefficient corresponds to minimum-shaped curvature in the fitted linear predictor, while a negative coefficient corresponds to maximum-shaped curvature. The algebraic quantity $D^*$ identifies the stationary point implied by a particular coefficient draw. It does not, by itself, establish a geological target.

The distinction is important in these results. NRB_3a lithology-contact distance has positive curvature with posterior probabilities of $0.993$–$1.000$ across folds, fold-median $D^*$ values near $26$–$29$ km, and within-support probabilities of $0.963$–$1.000$. The stationary point is therefore more interpretable as a feature of the fitted response than it would be if it were entirely outside the observed support. Even here, empirical support strengthens statistical interpretability but does not establish causality or a universal preferred distance.

NRB_3a fault distance is generally minimum-like but less uniform: one fold has $P(\beta_{fault^2}>0)=0.464$, and the fold-median within-support probability ranges from $0.126$ to $0.938$. NRB_3b fault distance is weaker still, with posterior curvature sign probabilities that do not establish a common minimum or maximum and response curves that vary from boundary/monotone to interior extrema. NRB_3b lithology-contact distance is more often positive-curvature, but its within-support probability is only $0.381$–$0.584$ across folds. Thus curvature, finite stationary-point calculation, response classification and empirical support are separate evidential steps.

## 5.2 Spatial non-stationarity

The differing fold-level coefficients, curvature probabilities, support probabilities and response-curve extrema are consistent with spatially varying fitted relationships. The OOF results reinforce this pattern: V11 exceeded M5 in Fold 3 but underperformed M5 in Folds 1, 2 and 4. The pooled V11 advantage therefore does not describe a uniform local advantage.

This is evidence that the modeled relationship and its transferability vary across the along-belt partitions. It is not proof that every coefficient difference is caused by a distinct geological mechanism. The hierarchical model supplies partial pooling and a way to estimate variation; geological causation requires evidence beyond coefficient variation alone.

## 5.3 Predictive transferability

The fold-level results indicate that geographic transferability is conditional on the held-out region. V11 was strongest relative to M5 in Fold 3, where $\Delta AUC=+0.091$, and weakest in Fold 2, where $\Delta AUC=-0.136$. Fold 1 showed a smaller negative difference and Fold 4 was close to parity but slightly negative. The pooled difference of $+0.164$ therefore summarizes an average over heterogeneous regional behavior rather than a guarantee of superiority in each region.

The multiscale spatial bootstrap retained the same sign pattern for the observed Fold 2 and Fold 3 differences. This supports the empirical stability of the reported regional contrast under the completed block-bootstrap analysis, while not converting predictive stability into coefficient or geological stability.

## 5.4 Daly-domain interpretation

The domain-stratified results show a strong contrast between the two domains in which conventional ROC-AUC is estimable. In NRB_3a, V11 achieved AUC $0.661701$ compared with $0.535752$ for M5, a difference of $+0.125948$. In NRB_3b, V11 achieved $0.583815$ compared with $0.866897$ for M5, a difference of $-0.283082$.

These results demonstrate that the frozen spatially held-out predictive behavior is not uniform across NRB_3a and NRB_3b. They do not represent independent domain refits, Leave-One-Daly-Domain-Out validation or Fold-by-Domain validation. CRZ, NKB, SRB and MMSB contain no positive observations in this modeling frame, so their ROC-AUC is undefined rather than poor.

## 5.5 Relationship to Daly's hypothesis

The results provide partial and qualified evidence relevant to Daly's geological proposition. The presence of nonlinear fitted responses, especially the well-supported minimum-shaped NRB_3a lithology-contact response, is consistent with the proposition that proximity relationships need not be monotonic. The variation in fault-distance curvature, response shape and spatial OOF discrimination is consistent with the proposition that relationships may not transfer uniformly along the Copperbelt.

However, the evidence does not establish a single Daly-wide fault-distance stationary point, a universal geological optimum or a causal structural mechanism. The weaker and more variable NRB_3b results, the fold-specific uncertainty in NRB_3a fault distance and the zero-positive domains constrain the strength of any general claim. The appropriate interpretation is therefore evidence for case-specific nonlinear and spatially variable modeled responses, not proof of Daly's hypothesis in its entirety.
