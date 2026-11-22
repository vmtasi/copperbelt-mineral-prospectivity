# V15 sampling diagnostics

## Method

Diagnostics were recomputed directly from every saved V15 trace with ArviZ, not taken from the shared `fit_status.json`. That file was written concurrently by several runs and contains stale labels for the archived non-converged M2/M3 fits. The recomputed table is `paper_1/v15/diagnostics/sampling_diagnostics.csv` and the machine-readable status is `paper_1/v15/models/fit_status_recomputed.json`.

A trace is called converged only when all of the following hold:

- zero post-tuning divergences;
- maximum R-hat below 1.05;
- minimum bulk ESS at least 200;
- minimum tail ESS at least 200.

Variables excluded from the R-hat and ESS scan are the raw non-centred offsets (`alpha_offset`, `offset_*`), which are auxiliary parameterizations rather than interpretable quantities. All reported scale and effect parameters are included.

Pilot traces and the archived initial non-converged M2/M3 run are excluded. The pilot exclusion matches any path component containing `pilot`.

## Primary models

| Model | Folds | Chains x draws | Tuning | Target accept | Divergences | Max R-hat | Min bulk ESS | Min tail ESS | Max MCSE mean | Min BFMI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0_global | 4 | 2 x 1,000 | 1,500 | 0.95 | 0 | 1.0050 | 658 | 870 | 0.0173 | 0.814 |
| M1_domain_intercept | 4 | 2 x 1,000 | 1,500 | 0.95 | 0 | 1.0126 | 705 | 530 | 0.0388 | 0.910 |
| M2_domain_distance | 4 | 4 x 1,500 | 2,500 | 0.995 | 0 | 1.0033 | 2,113 | 1,630 | 0.0257 | 0.830 |
| M3_domain_distance_gravity | 4 | 4 x 1,500 | 2,500 | 0.995 | 0 | 1.0029 | 1,886 | 1,463 | 0.0225 | 0.828 |

All sixteen primary fold fits pass. M2 and M3 were refitted with four chains after the initial two-chain runs produced divergences and unacceptable R-hat. The initial runs are archived under `paper_1/v15/models/nonconverged_initial/` and are not used for inference. Their traces are not read by the diagnostics.

M0 and M1 retain the original two-chain specification. They have no divergences, R-hat below 1.013 and tail ESS above 530. Their lower chain count is a limitation for tail quantiles but not a convergence failure.

## Sensitivity models

| Model | Chains x draws | Divergences | Max R-hat | Min bulk ESS | Min tail ESS | Pass |
|---|---|---:|---:|---:|---:|---|
| V14_control | 2 x 1,000 | 0 | 1.0075 | 523 | 271 | yes |
| M2_logit_area_covariate | 2 x 1,000 | 0 | 1.0072 | 625 | 356 | yes |
| M2_cloglog_no_litho | 2 x 1,000 | 0 | 1.0082 | 521 | 385 | yes |
| prior 0.5 x M2 | 2 x 1,000 | 0 | 1.0093 | 714 | 403 | yes |
| prior 2.0 x M2 | 2 x 1,000 | 0 | 1.0074 | 416 | 494 | yes |

All sensitivity fits pass. The V14 control and the two prior variants use 2,000 tuning steps at target acceptance 0.995, not the primary M2/M3 setting of 2,500.

## Buffered fits

| Model | Folds | Divergences | Max R-hat | Min bulk ESS | Min tail ESS | Pass |
|---|---:|---:|---:|---:|---:|---|
| 5 km M0 | 4 | 0 | 1.0185 | 257 | 391 | yes |
| 5 km M2 | 4 | 0 | 1.0173 | 222 | 136 | **no** |
| 10 km M0 | 4 | 0 | 1.0168 | 201 | 281 | yes |
| 10 km M2 | 4 | 3 | 1.0179 | 196 | 140 | **no** |

The failing rows are the 5 km M2 Fold 4 tail ESS of 136, the 10 km M2 Fold 1 tail ESS of 140, and the 10 km M2 Fold 4 fit with three divergences and tail ESS 171. These are reported as limited-convergence sensitivity results. They are not used for primary inference.

## Interpretation

No primary or sensitivity model used for inference has a convergence failure. The three buffered failures point to identifiable sparse-support problems: the buffered M2 models must estimate domain-varying quadratic slopes after removing nearby training cells, and the affected folds have few positives. Increasing draws or target acceptance would be required before treating those specific buffered fits as stable.

The contrast between the strict recomputed status and the script-level PASS flag is itself informative. `buffered.py` records PASS when the fitter's own convergence check passes, which does not include the V15 tail-ESS criterion. The stricter recomputed table is authoritative for reporting.

## Outputs

- `paper_1/v15/diagnostics/sampling_diagnostics.csv`
- `paper_1/v15/diagnostics/sampling_parameter_summary.csv`
- `paper_1/v15/models/fit_status_recomputed.json`
- `paper_1/v15/traces/**/fold_*.nc`
