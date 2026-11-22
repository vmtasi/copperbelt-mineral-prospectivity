# V15 random-forest comparator

## Design

The comparator uses the same 1,872 complete cases and canonical four spatial folds. Training-only standardization, training-only lithology support and the same fault, contact and gravity terms are used. `log_area_km2` is supplied as an ordinary covariate because a tree ensemble has no fixed exposure offset. Hyperparameters are selected by three-fold grouped cross-validation within each training partition using 25 km spatial blocks.

The comparator has access to the same information as the Bayesian models but not the same likelihood or exposure interpretation. It is a performance benchmark, not a replacement model.

## Results

| Sample | n | Positives | ROC-AUC | PR-AUC | Mean log score | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Fold 1 | 468 | 9 | 0.7657 | 0.2783 | -0.0919 | 0.0189 |
| Fold 2 | 468 | 31 | 0.7253 | 0.1157 | -0.2262 | 0.0614 |
| Fold 3 | 468 | 37 | 0.7640 | 0.2004 | -0.2485 | 0.0688 |
| Fold 4 | 468 | 61 | 0.6765 | 0.1997 | -0.3743 | 0.1105 |
| Pooled | 1,872 | 138 | 0.7528 | 0.1574 | -0.2352 | 0.0649 |

## Paired comparison

Against M2, the random forest has a better pooled log score by 0.0219 per cell, but the spatial-block bootstrap interval is [-0.0479, 0.0016] and includes zero.

Against M3, the random forest is worse by 0.0085 per cell, with interval [-0.0080, 0.0262], also including zero. Fold 3 favours the forest slightly; folds 1, 2 and 4 favour M3.

## Interpretation

No model dominates across the comparison. The forest beats M2 on pooled discrimination and log score, so a claim that the M1/M2 hierarchical formulation is the superior predictive model is not supported. M3 beats the forest in pooled metrics, but not decisively under spatial-block uncertainty.

The comparator therefore supports a conditional conclusion: domain-varying distance terms add descriptive and modest predictive value over M1, while the largest gain comes from the less stable domain-varying gravity extension. A flexible non-hierarchical learner remains a credible alternative for pooled prediction.

## Outputs

- `paper_1/v15/metrics/comparator_metrics.csv`
- `paper_1/v15/predictions/RF_comparator_fold_*.csv`
- `paper_1/v15/predictions/oof_RF_comparator.csv`
- `paper_1/v15/metrics/paired_comparisons.csv`
- `paper_1/v15/metrics/paired_bootstrap.csv`
- `paper_1/v15/diagnostics/reliability_RF_comparator.csv`
