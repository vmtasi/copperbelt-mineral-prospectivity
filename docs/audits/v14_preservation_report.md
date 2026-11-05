# V14 Preservation and Benchmark Report

**Audit date:** 2026-11-05
**Purpose:** preserve the V14 historical benchmark, separate historical-reported values from independently reproduced values, audit the spatial folds, and resolve the north-west Fold 4 claim.
**Mode:** read-only. No model was refitted and no MCMC was run.

---

## 1. Canonical artefacts

| Artefact | Path | Bytes | SHA-256 | Tracked by Git? |
|---|---|---:|---|---|
| V14 training table | `data/copperbelt_training_v14_centroid_contact.csv` | 2,232,259 | `161ad50684950f7f106c079079eacbae9de09c1851dc15867b014c6b88bbd878` (manifest) | **No** (ignored) |
| V14 canonical manifest | `data/derived/canonical_training_manifest_v14_centroid_contact.json` | 5,270 | `656be6077ff1499d6fabbe74a7a0a2951e68ae4049b9316bffd405e290623974` | Yes |
| Fold 1 trace | `figures/v14_centroid_contact/v14_centroid_fold_1_trace.nc` | 2,088,740 | `05f72bf404faf17e44da6ef0d4ce00315e2df0323912e7d955e81d3ebcf045fc` | **No** (ignored) |
| Fold 2 trace | `.../v14_centroid_fold_2_trace.nc` | 2,090,488 | `13737735b7a233d9c30043eb9eebb521b51b96910e51ba1ce34a1210f3ab38b5` | **No** |
| Fold 3 trace | `.../v14_centroid_fold_3_trace.nc` | 2,088,544 | `47be84198a80feddfb11ced96f0d60a0be1c6da4e0a5d9a293cfe3eb19a6a183` | **No** |
| Fold 4 trace | `.../v14_centroid_fold_4_trace.nc` | 2,089,114 | `20833b82d2f86268ceec5b5b7949bc4d976890839f64bf4f1a8b9c75ef9b1dbd` | **No** |
| Fold metrics | `.../v14_centroid_fold_metrics.csv` | 148 | `a79f11753e616a21e5c58df62c359c2c67e2b1e872c7884d73511a0666bada98` | **No** |
| Stability matrix | `.../v14_centroid_geological_stability_matrix.csv` | 1,103 | `710d352027d3448d7bcd2148085dee8ad29802f57d2d1d4e64d531c968276da3` | **No** |
| V14 OOF predictions | `.../v14_centroid_oof_predictions.csv` | 113,824 | `f1713d20fc8d6675cafae87cbdf8e7687945b16fcf7b21f013317bb88f03cf8a` | **No** |
| M7 matched-global OOF | `figures/audit/phase4_oof_predictions_v14_centroid.csv` | 329,795 | `1A6283D7108BC9274E7AB9CEB6FB61BB1B0AC0A6598C413ABA603DA4DF987BA3` | **No** |
| Phase 7 validation summary | `figures/audit/phase7_spatial_validation_v14_centroid.csv` | 836 | `D188CFF3E85E918F2E65477AF0501818ED474C284CBC24F280D7DAABC4EBE436` | **No** |
| Post-refit validation | `figures/audit/v14_model_output_validation_post_refit_20261021.json` | 8,174 | internal hashes | **No** |

The V14 training CSV and V13 table were locked by another process during this session. Their hashes above are the values recorded in the canonical manifest and the pre-V14 snapshot; all other hashes were recomputed and matched.

### 1.1 Preservation status

The V14 benchmark is **scientifically preserved but not version-controlled**. `data/*.csv`, `figures/` and `/archive/` are ignored. A fresh clone would contain the scripts and manifests but not the traces, OOF predictions, training table or archived V5 base. This must be corrected by an approved archive/backup decision before the repository is treated as a durable benchmark.

---

## 2. Run configuration and diagnostics

### 2.1 Configuration

```text
model: frozen V11 specification executed through an I/O-only V14 wrapper
folds: 4
draws: 1,500 per chain
tuning: 2,500
chains: 2
cores: 2
target acceptance: 0.99
random seed: 42
```

The wrapper changed only the data path, trace names and output names. The model specification was not changed. A numerical compatibility shim replaced `StandardScaler` and `roc_auc_score` because the QGIS Python 3.12 runtime could not load the CPython 3.10 scikit-learn extension; the manifest records `model_specification_changed: false`.

### 2.2 Diagnostics

| Fold | Training cells | Training positives | Held-out cells | Held-out positives | Divergences | Max R-hat | Min ESS bulk | Min ESS tail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,404 | 129 | 468 | 9 | 0 | 1.0 | 1,675 | 1,444 |
| 2 | 1,404 | 107 | 468 | 31 | 0 | 1.0 | 1,331 | 1,528 |
| 3 | 1,404 | 101 | 468 | 37 | 0 | 1.0 | 1,875 | 1,284 |
| 4 | 1,404 | 77 | 468 | 61 | 0 | 1.0 | 1,939 | 1,310 |

All labels match the modelling table. Validation status is `PASS`.

The per-trace diagnostics above are the diagnostics of the V14 DVHM. The file
`phase4_heterogeneity_hyperparameters_v14_centroid.csv` is **not** a summary of
that model: it is an output of `src/phase_4b.py`, a separate full-data
descriptive hierarchical fit with domain-varying gravity, `target_accept=0.95`
and 2,000 tuning draws. Its minimum `ess_bulk` of 944 for `mu_l` and minimum
`ess_tail` of 1,094 for `sigma_g` therefore describe the Phase-4B descriptive
model, not the V14 DVHM.

---

## 3. Historical benchmark versus independent reproduction

Historical-reported values are the values the prior study reported. Independently reproduced values were recomputed in this audit from the existing OOF files. They are kept separate; historical values are not overwritten.

| Metric | Historical reported | Independently reproduced | Status |
|---|---:|---:|---|
| Pooled DVHM ROC-AUC | 0.716 | 0.7155483677 | agrees to 3 decimals |
| Matched-global M7 ROC-AUC | 0.674 | 0.6742766160 | agrees to 3 decimals |
| Pooled difference | +0.041 | +0.0409639249 | agrees to 3 decimals |
| Pooled DVHM average precision | 0.137 | 0.1373326329 | agrees |
| Pooled M7 average precision | 0.108 | 0.1079619289 | agrees |
| Pooled prevalence | 0.074 | 0.0737179487 | agrees |
| DVHM Brier | 0.069 | 0.0687864169 | agrees |
| M7 Brier | 0.079 | 0.0790541904 | agrees |
| Constant-prevalence Brier baseline | approximately 0.068 | not recomputed | historical only |
| Fold ROC-AUC | 0.875 / 0.736 / 0.685 / 0.603 | 0.8753328492 / 0.7356610320 / 0.6850818336 / 0.6030531276 | agrees |
| Fold positives | 9 / 31 / 37 / 61 | 9 / 31 / 37 / 61 | agrees |
| Residual Moran's I | 0.145, p = 0.001 | artifact not found | **unresolved historical** |
| Calibration slope | 0.61 | artifact not found | **unresolved historical** |
| Calibration intercept | -0.84 | artifact not found | **unresolved historical** |
| Prior block-bootstrap range | -0.022 to +0.103 | artifact not found | **unresolved historical** |

The historical Moran's I, calibration and prior block-bootstrap values are retained as reported evidence. They cannot be independently verified from the current repository because no artifact containing them was found. They must not be silently replaced.

### 3.1 Pooled bootstrap interval

The manuscript reports `+0.006` to `+0.077`. An independent 1,000-resample paired row bootstrap of the fixed OOF predictions, seeded at 42, gave approximately `+0.0061` to `+0.0766`. The small difference from the handoff value (`+0.006964` to `+0.076392`) is due to RNG consumption/implementation details of the bootstrap, not to different data. Both agree with the manuscript to three decimals.

The bootstrap resamples rows, not spatial blocks. It therefore does **not** account for spatial autocorrelation. A block bootstrap exists in `phase7_multiscale_robustness_v14_centroid.csv`, but it uses M5 as the comparator, not M7, so it is not equivalent to the historical matched-model block bootstrap.

### 3.2 Distance-representation sensitivity record

The V14 sensitivity analysis evaluates four hierarchical specifications under
the same four folds and the same six-domain hierarchy:

| Model | Representation | Distance terms | Mean ROC-AUC | Mean AP |
|---|---|---|---:|---:|
| A (DVHM) | raw distance, quadratic | linear + quadratic | 0.725 | 0.126 |
| B | log(1 + distance in km), quadratic | linear + quadratic | 0.707 | 0.154 |
| C | raw distance, linear | linear | 0.687 | 0.178 |
| D | log(1 + distance in km), linear | linear | 0.722 | 0.200 |

The values are unweighted means of four fold-specific metrics, not pooled
out-of-fold metrics. Across 48 fold x domain x distance-predictor cases, the
curvature criterion passed 15 times, the D* median-agreement criterion 4 times,
the training-support criterion 18 times, and all three simultaneously once
(Fold 1, NRB_3a, contact distance). The population-level 95% intervals for D*
span -376 to +379 km for fault distance and -135 to +165 km for contact
distance. The data therefore do not constrain an operationally interpretable
population D*. This is a preserved analytical result, not a reason to discard
the distance predictors.

### 3.3 Spatial block-bootstrap record

The existing V14 block-bootstrap file is
`figures/audit/phase7_multiscale_robustness_v14_centroid.csv`. It compares the
DVHM against **M5**, the compact gravity-plus-contact baseline, not against the
matched-global M7 model. Its pooled observed delta is +0.159 at all four block
scales, with 95% intervals of [0.010, 0.287] at 10x10, [0.048, 0.263] at
15x15, [0.052, 0.258] at 20x20, and [0.064, 0.248] at 25x25. Fold 4 intervals
include zero at all four scales, and Fold 2 remains negative at the 15x15 and
25x25 scales. These results support the historical conclusion that pooled
ranking is positive but sector-level transfer is unstable. They cannot be used
as a matched DVHM-versus-M7 block-bootstrap test.

### 3.4 Complete benchmark envelope

| Required benchmark item | Preserved value or artifact | Reproduced in audit? | Disposition |
|---|---|---|---|
| Pooled out-of-fold ROC-AUC and global reference | DVHM 0.7155483677; M7 0.6742766160 | yes | verified |
| Average precision and prevalence | DVHM 0.1373326329; M7 0.1079619289; prevalence 0.0737179487 | yes | verified |
| Brier score and skill relative to prevalence | DVHM 0.0687864169; M7 0.0790541904; prevalence baseline approximately 0.068 | Brier values yes; baseline not independently reconstructed | preserved with qualification |
| Calibration slope and intercept | historical 0.61 and -0.84 | no artifact found | unresolved historical |
| Residual Moran's I | historical 0.145, p = 0.001 | no artifact found | unresolved historical |
| Spatial block-bootstrap | `phase7_multiscale_robustness_v14_centroid.csv`; M5 comparator | yes, summarized above | preserved |
| Fold ROC-AUC and positive counts | 0.8753/0.7357/0.6851/0.6031; 9/31/37/61 positives | yes | verified |
| Fold composition and coverage | 468 cells each; domains and coordinate ranges recorded in Section 4 | yes | verified |
| Sensitivity findings | four-model metrics; 15/4/18/1 criteria passes | yes from existing CSV | verified |
| Hierarchical parameter uncertainty | raw V14 traces; no dedicated summary found | traces not resampled in this audit | unresolved display; trace preserved |

---

## 4. Fold audit

### 4.1 Fold table

| Fold | Cells | Positives | Prevalence | DVHM AUC | M7 AUC | Delta (DVHM - M7) | DVHM AP | M7 AP | Domains |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 (south-east) | 468 | 9 | 1.923% | 0.8753328492 | 0.8041636408 | +0.0711692084 | 0.078100 | 0.052786 | NRB_3b 310, CRZ 146, NKB 12 |
| 2 (central) | 468 | 31 | 6.624% | 0.7356610320 | 0.6872370266 | +0.0484240053 | 0.122522 | 0.110480 | NRB_3b 240, NRB_3a 207, NKB 12, CRZ 8, SRB 1 |
| 3 (central-north) | 468 | 37 | 7.906% | 0.6850818336 | 0.6702828118 | +0.0147990218 | 0.143368 | 0.130751 | NRB_3a 468 |
| 4 (north-west) | 468 | 61 | 13.034% | 0.6030531276 | 0.6038184235 | -0.0007652958 | 0.160479 | 0.163910 | NRB_3a 452, MMSB 16 |

Per-fold Brier scores for the DVHM are 0.018508, 0.062687, 0.076486 and 0.117465. They rise with prevalence, so they are not directly comparable across folds.

### 4.2 Fold geography

| Fold | Easting range (m) | Northing range (m) | Sector |
|---|---|---|---|
| 1 | 538,121--715,465 | 8,384,711--8,589,373 | south-east |
| 2 | 529,383--685,637 | 8,491,951--8,720,125 | central |
| 3 | 409,933--605,600 | 8,685,082--8,830,461 | central-north |
| 4 | 282,675--504,594 | 8,756,582--8,859,512 | north-west |

The fold assignment is a first-principal-component ordering of cell centroids, divided into four contiguous groups of 468. It is not a domain-stratified split.

### 4.3 Fold composition and confounding

The folds are balanced by cell count but not by positives, prevalence, domain composition, area or predictor spread:

- Fold 1 has 9 positives and a prevalence of 1.9%; Fold 4 has 61 positives and 13.0%.
- Folds 3 and 4 are almost entirely NRB_3a; Fold 1 contains CRZ, NKB and NRB_3b.
- MMSB occurs only in Fold 4; SRB occurs only in Fold 2.
- Average precision rises from 0.078 to 0.160, but so does prevalence; the AP-to-prevalence ratio falls from about 4.1 to about 1.2.

Sector-level performance differences therefore combine geological transferability, occurrence density, domain composition, predictor spread and area support. They cannot be attributed to geology alone.

### 4.4 Does the four-fold design provide sufficient evidence?

**For a descriptive test of transfer among four named sectors: yes.** It shows that ranking skill is not uniform and that the north-west is the weakest sector.

**For a stable ranking of model specifications, or for geological transferability to frontier areas: no.** The design has:

- no buffer zones between training and validation;
- only four partitions, each with one held-out realization;
- fold-level intervals that all include zero;
- prevalence and domain composition confounded with sector;
- two folds dominated by a single domain;
- no explicit spatial random effect to absorb residual autocorrelation.

The original four-fold design should be retained for historical comparison. Any redesign should add buffered validation, alternative partitions, and preferably a larger positive sample or a point-process formulation.

---

## 5. North-west Fold 4 audit

### 5.1 Source files and columns

DVHM OOF:

```text
figures/v14_centroid_contact/v14_centroid_oof_predictions.csv
columns: id, fold, deposit_present, daly_domain,
         centroid_x, centroid_y, prob_v14
```

Matched-global OOF:

```text
figures/audit/phase4_oof_predictions_v14_centroid.csv
columns used: deposit_present, daly_domain, spatial_block,
              centroid_x, centroid_y, M7_All_Global_Prediction
```

Canonical validation table:

```text
figures/audit/phase7_spatial_validation_v14_centroid.csv
columns: Fold, M5_AUC_Obs, M7_AUC_Obs, V14_AUC_Obs,
         V14-M5_Delta_AUC, V14-M7_Delta_AUC,
         M5_Brier_Obs, V14_Brier_Obs, M5_PRAUC, V14_PRAUC
```

**Qualification:** the matched-global file has no stable `id` column. Its 1,872-row label vector, fold vector and row ordering agree with the DVHM OOF file, and its centroids coincide. Exact linkage is by row position and coincident centroids, not by a validated one-to-one cell ID.

### 5.2 Question 1 -- does the canonical CSV report 0.603 for both?

Yes. The Fold 4 row contains:

```text
M7_AUC_Obs:   0.603 [0.537, 0.668]
V14_AUC_Obs:  0.603 [0.543, 0.665]
V14-M7_Delta_AUC: -0.000 [-0.046, 0.050]
```

### 5.3 Question 2 -- same held-out observations?

Yes, with the ID qualification above:

```text
both files: 1,872 rows
labels equal: True
fold vectors equal: True
centroids equal: True
Fold 4: 468 cells, 61 positives
prediction coverage: 468 non-null predictions in each model
```

### 5.4 Question 3 -- independent recalculation

Recalculated from the existing OOF predictions without refitting:

```text
DVHM Fold 4 AUC:   0.6030531276432916
M7 Fold 4 AUC:     0.6038184234905546
```

The recalculated values reproduce the canonical CSV to machine precision.

### 5.5 Question 4 -- paired difference and interval

```text
exact difference: -0.0007652958472630367
rounded display:  -0.001
95% interval:     -0.046 to +0.050
```

The interval is obtained by a paired row bootstrap of the fixed OOF predictions: 1,000 resamples with replacement, percentile 2.5% and 97.5% quantiles, seeded resampling. It does not refit either model and is not a posterior interval. Because both AUCs round to 0.603, the displayed `-0.001` looks inconsistent at first inspection; the unrounded values explain it.

### 5.6 Question 5 -- is Fold 4 the north-west?

Yes. This is established from the canonical fold assignments and coordinates, not from the manuscript narrative:

```text
Fold 4 easting:  282,675--504,594 m
Fold 4 northing: 8,756,582--8,859,512 m
domains: NRB_3a 452, MMSB 16
```

These are the western/north-western cells of the frame. Fold 1 is the south-east (easting 538,121--715,465; northing 8,384,711--8,589,373).

### 5.7 Question 6 -- genuine or error?

**Genuine.** The identical rounded AUCs are not caused by:

- a fold-mapping error: the fold vectors match exactly;
- a mismatched prediction subset: both have 468 Fold 4 predictions and 61 positives;
- a stale output: both files are the V14-suffixed current artifacts;
- incorrect column selection: the DVHM uses `prob_v14`; the global model uses `M7_All_Global_Prediction`. These are distinct columns with distinct values.

The near-identity is a coincidence at this sector: the DVHM's extra flexibility does not improve ranking in the north-west, and the global model is marginally better by 0.0008 AUC.

### 5.8 Question 7 -- conflicting values

| Source | Fold 4 value | Fold 4 label | Status |
|---|---|---|---|
| `phase7_spatial_validation_v14_centroid.csv` | DVHM 0.603; M7 0.603 | no label | canonical |
| `main.tex` Table 3 | DVHM 0.603 | north-west | correct |
| `main.tex` Section 4.1 | delta -0.001 | north-west | correct |
| `stage_04_results.md` | DVHM 0.605 | "Southern" | **stale/wrong** |
| `README.md` | V11 0.605 | no label | **stale** |

The `stage_04_results.md` "Fold 1 (Northern)" / "Fold 4 (Southern)" designations are the reverse of the canonical geometry. They are stale and must not be used.

### 5.9 Verdict

**VERIFIED WITH QUALIFICATIONS.**

The approximately identical north-west AUCs are a genuine result. The qualification is that the matched-global artifact lacks a stable ID column, so linkage is by row order, labels, folds and centroids rather than a validated ID join. The manuscript should describe the result as "the DVHM is marginally below the global model by about 0.0008 AUC, with an interval including zero", not as a meaningful advantage or disadvantage.

---

## 6. Hierarchical parameter uncertainty and a corrected attribution

### 6.1 Audit correction

An earlier version of this audit reported a table of `mu_f`, `sigma_f`, `mu_l`,
`sigma_l`, `mu_g` and `sigma_g` as if it described the V14 DVHM. That attribution
was incorrect. The table is the summary of
`figures/audit/phase4_domain_heterogeneity_trace_v14_centroid.nc`, generated by
`src/phase_4b.py`. Phase 4B is a separate, full-data descriptive model in which
gravity also varies by domain. It is not the out-of-fold DVHM and is not the
model compared with M7.

The actual DVHM model code fixes `beta_grav` as one global scalar. Its
domain-varying hierarchy is represented in the four fold traces by:

```text
alpha_sigma
sigma_f_lin
sigma_f_sq
sigma_l_lin
sigma_l_sq
```

The raw traces preserve the posterior distribution of these scales. No
dedicated summary table of them was found. A summary may be produced later from
the existing traces without a refit, but none is claimed here.

### 6.2 Consequence for interpretation

The Phase-4B `sigma_g` values must not be used as evidence that gravity has a
domain-varying effect in the DVHM: in V14 it does not. The Phase-4B output is
historical descriptive evidence only. The V14 claim of between-domain
variation rests on the distance-coefficient scales in the trace and remains
weakly identified because only NRB_3a and NRB_3b contain positive cells.

---

## 7. Posterior-object classification

| Object | Model version | Inputs / configuration | Required for historical results? | Equivalent summary exists? | Regenerable? | Classification |
|---|---|---|---|---|---|---|
| `figures/v14_centroid_contact/v14_centroid_fold_{1..4}_trace.nc` | V14 | V14 table; seed 42; 2 chains x 1,500 | yes | no | expensive refit | **historical-essential** |
| `v14_centroid_oof_predictions.csv` | V14 | traces + V14 table | yes | partial (fold metrics) | cheap from traces | **historical-essential** |
| `phase4_oof_predictions_v14_centroid.csv` | V14 / M7 | V14 table; fold-wise global logistic | yes | partial | cheap refit | **historical-essential** |
| `phase7_spatial_validation_v14_centroid.csv` | V14 / M5 / M7 | OOF predictions | yes | partial | cheap | **historical-essential** |
| `phase4_heterogeneity_hyperparameters_v14_centroid.csv` | Phase 4B descriptive model | V14 table; full-data fit | yes for Phase 4B only | no | cheap from Phase-4B trace | **historical-essential for Phase 4B; not V14 DVHM evidence** |
| `phase4_domain_heterogeneity_trace_v14_centroid.nc` | Phase 4B descriptive model | V14 table; full-data fit | yes for Phase 4B only | no | expensive within Phase 4B | **historical-essential for Phase 4B; not V14 DVHM evidence** |
| `v14_centroid_geological_stability_matrix.csv` | V14 | traces | yes | no | cheap from traces | **historical-essential** |
| `v14_model_output_validation_post_refit_20261021.json` | V14 | traces + table | yes | no | cheap | **historical-essential** |
| V11 `.pkl` traces (`figures/v11_fold_{1..4}_trace.pkl`) | V11 | V13 table | yes for V11 | no | expensive refit | **historical-essential** |
| V11 OOF and uncertainty | V11 | V11 traces | yes for V11 | no | cheap from traces | **historical-essential** |
| V13 audit outputs | V13 | V13 table | historical | partial | cheap | **historical** |
| Non-suffixed duplicates of V14 phase CSVs | V14 | same as V14 | no | exact duplicate | cheap | **historical-redundant** |
| `v14_centroid_contact_failed_run_manifest_20261020T204228.json` | failed run | pre-refit | no | no | no | **historical-redundant** |
| `figures/audit/beta_diagnostics/` (non-V14) | V13 | V13 traces | historical | partial | cheap | **historical** |
| `.runtime_cache/`, `pytensor_cache/` | runtime | PyTensor | no | no | yes | **reproducible-cache** |
| `.trash/` PyTensor leaks | runtime | flattened paths | no | no | yes | **reproducible-cache** |

No historical-essential object should be deleted merely to reduce repository size.

---

## 8. Reproduction recipe (not executed)

To reproduce V14 from the preserved assets:

1. Restore `data/copperbelt_training_v13_litho_contact.csv` and `data/derived/litho_contact_distance_v13.csv` from the archived pre-V14 snapshot.
2. Run `python src/rebuild_litho_contact_centroid_features.py` with the contact geometry in `C:\Project\data for projects\internal_contacts_only.gpkg`. Verify the output hash against `656be...` (manifest) and `161ad...` (training table).
3. Run `python src/v14_centroid_spatial_stability.py`. This refits four models and is expensive; it should not be run as a routine check.
4. Run `python src/extract_v14_oof.py` and verify `f1713d20...`.
5. Run the phase scripts in order and compare against the preserved CSVs.
6. Run `python src/validate_v14_model_outputs.py` and require `PASS`.

The upstream V5 table and the raw occurrence/gravity inputs are not reproducible from the repository alone. See `docs/audits/data_provenance_register.md`.

---

## 9. Do-not-modify list

The following must not be modified, moved, renamed, regenerated or overwritten during cleanup:

```text
data/copperbelt_training_v14_centroid_contact.csv
data/derived/canonical_training_manifest_v14_centroid_contact.json
data/derived/litho_contact_centroid_distance_v14.csv/.gpkg
figures/v14_centroid_contact/*
figures/audit/phase*_v14_centroid*
figures/audit/v14_model_output_validation_post_refit_20261021.json
archive/litho_contact_repair_2026-10-18/**
archive/centroid_contact_repair_2026-10-20/**
paper_1/manuscript/main.tex
paper_1/manuscript/main.pdf
paper_1/manuscript/images/fig1_study_area_context_annotated.pdf/.png
paper_1/copperbelt_exploration_figure1_recovered.qgz
tests/test_manuscript_numerical_consistency.py
```

The six uncommitted working-tree files listed in `docs/audits/repository_inventory.md` are protected in full.

---

## 10. Preservation verdict

The V14 numerical benchmark is internally consistent, independently reproducible from its OOF artifacts, and diagnostically sound. The historical-reported values agree with the reproduced values except for the Moran's I, calibration and prior block-bootstrap summaries, whose artifacts are missing.

The benchmark is **not durably preserved** until the ignored training table, traces and OOF predictions are archived or backed up under an approved policy. The north-west Fold 4 result is **verified with the ID-linkage qualification**.
