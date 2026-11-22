# V15 positional-uncertainty perturbation report

## Design

The experiment perturbs the 184 USGS records that fall inside the 138 positive cells within a 1 km radius, reassigns them to full 5 km cell polygons, and repeats the label construction 500 times. Records in model-negative cells and outside the modelled grid are not used to create positives.

The 1 km radius is the project's working positional-uncertainty assumption. The USGS table has no per-record uncertainty field, so this is a label-churn sensitivity experiment, not a survey-grade relocation model.

## Label churn

| Quantity | Result |
|---|---:|
| Simulations | 500 |
| Radius | 1 km |
| Mean changed cells | 40.07 |
| Median changed cells | 40 |
| 95th percentile changed cells | 51.05 |
| Mean positive cells | 138.65 |
| Mean point edge-crossing probability | 0.174 |
| Maximum point edge-crossing probability | 0.64 |

The total positive count is almost unchanged, but roughly 40 cell identities change per realisation. The binary representation is therefore sensitive to positional uncertainty even when prevalence is stable.

## Model stability

| Model | Mean AUC | AUC SD | Mean log score | Log-score SD |
|---|---:|---:|---:|---:|
| M0_global | 0.6790 | 0.0071 | -0.2761 | 0.0046 |
| M1_domain_intercept | 0.6961 | 0.0077 | -0.2673 | 0.0045 |
| M2_domain_distance | 0.7226 | 0.0070 | -0.2617 | 0.0046 |
| M3_domain_distance_gravity | 0.7944 | 0.0061 | -0.2299 | 0.0036 |
| RF comparator | 0.7415 | 0.0077 | -0.2388 | 0.0037 |
| V14 frozen | 0.7143 | 0.0046 | -0.2523 | 0.0034 |

The ordering of models is stable under this perturbation. Absolute AUC falls by roughly 0.01 relative to the unperturbed predictions, and M3 retains the advantage over M2.

## Limits

The experiment assumes a circular 1 km uniform relocation and does not use documented per-record uncertainty. It preserves record identity and the modelled study frame, but it does not resolve geological mislocation, coordinate-datum errors or historical locality ambiguity.

## Outputs

- `paper_1/v15/diagnostics/label_perturbation_summary_v1.json`
- `paper_1/v15/diagnostics/label_perturbation_simulations_v1.csv`
- `paper_1/v15/diagnostics/label_perturbation_model_metrics_v1.csv`
- `paper_1/v15/diagnostics/perturbation_point_assignment.csv`
