# Obsolete, Duplicate and Inconsistent Assets

**Audit date:** 2026-11-05
**Scope:** files that appear redundant, obsolete, temporary or inconsistent, with the evidence for each classification.
**No file was deleted, moved or renamed.**

Classification follows the mandate:

- **KEEP** -- active research, source provenance, reproducibility, validated processing or manuscript production.
- **ARCHIVE** -- no longer active but scientifically or historically important.
- **REMOVE** -- demonstrated redundant, temporary, reproducible cache, or obsolete duplicate.
- **REVIEW** -- purpose, dependency or provenance not yet established.

The cleanup plan in `docs/audits/cleanup_plan.md` converts these findings into an executable file-level plan. This report does not authorise execution.

---

## 1. Exact duplicate groups

Only exact SHA-256 matches are called duplicates. Similar filenames are not enough.

### 1.1 Root and manuscript images

| Group | Files | SHA-256 prefix | Size |
|---|---|---|---|
| G1 | `daly_draft_layout_validation.png`; `paper_1/manuscript/images/fig1_study_area_context.png` | `59709D3EA932` | 455,557 B |
| G2 | `final_single_boundary_layout_validation.png`; `tract_focused_closure_QA.png`; `paper_1/manuscript/images/fig1_study_area_context_current.png` | `688CC1630F2D` | 735,615 B |

Both groups are **superseded Figure 1 drafts**. The active Figure 1 is `fig1_study_area_context_annotated.pdf` / `.png`. The duplicates are not needed for the current manuscript, but one copy of each historical layout may be retained for the figure-revision trail.

### 1.2 `figures/audit` exact duplicates

Ten V14-suffixed files are byte-identical to their non-suffixed counterparts:

| V14 file | Non-suffixed file | SHA-256 prefix |
|---|---|---|
| `domain_summary_v14_centroid.csv` | `domain_summary.csv` | `0980ac6e36` |
| `domain_training_support_by_fold_v14_centroid.csv` | `domain_training_support_by_fold.csv` | `c086a6a1bc` |
| `fold_domain_heatmap_visual_v14_centroid.csv` | `fold_domain_heatmap_visual.csv` | `25d9364a47` |
| `fold_domain_overlap_v14_centroid.csv` | `fold_domain_overlap.csv` | `4647e02e31` |
| `fold_summary_v14_centroid.csv` | `fold_summary.csv` | `e2982913e6` |
| `phase6_feature_observation_support_v14_centroid.csv` | `phase6_feature_observation_support.csv` | `a75c6bfe7d` |
| `phase6_missingness_by_domain_v14_centroid.csv` | `phase6_missingness_by_domain.csv` | `debb93510e` |
| `phase6_missingness_by_fold_v14_centroid.csv` | `phase6_missingness_by_fold.csv` | `cdcd9e9b08` |
| `phase6_missingness_by_target_v14_centroid.csv` | `phase6_missingness_by_target.csv` | `fb7535279f` |
| `phase6_missingness_fold_domain_v14_centroid.csv` | `phase6_missingness_fold_domain.csv` | `704d4cbd9c` |

These are exact duplicates. The V14-suffixed copy is the canonical one because it names the dataset version; the non-suffixed copy is **historical-redundant** and a REMOVE candidate after approval.

### 1.3 Parallel but different V13/V14 files

Seventeen pairs have the same stem but different content. These are **not duplicates**; they are V13/V14 versions:

| File pair | Evidence of difference |
|---|---|
| `domain_stratified_auc_v14_centroid.csv` vs `domain_stratified_auc.csv` | different hash and 2-byte size |
| `feature_by_target_summary_*` | 5,171 vs 5,067 B |
| `feature_support_*` | 923 vs 921 B |
| `phase3_ablation_fold_results_*` | 3,485 vs 3,493 B |
| `phase3_ablation_summary_*` | 959 vs 960 B |
| `phase4_domain_coefficients_*` | 1,087 vs 1,046 B |
| `phase4_feature_by_daly_domain_*` | 1,787 vs 1,795 B |
| `phase4_heterogeneity_hyperparameters_*` | 440 vs 442 B |
| `phase4_oof_predictions_*` | 329,795 vs 329,824 B |
| `phase5_distance_response_*` | **72,001 vs 17,985 B** |
| `phase5_turning_points_*` | **1,722 vs 536 B** |
| `phase7_6_axis_a_training_info_*` | 789 vs 788 B |
| `phase7_6_axis_b_test_positives_*` | 534 vs 533 B |
| `phase7_final_validation_*` | 681 vs 682 B |
| `phase7_multiscale_robustness_*` | 1,479 vs 1,478 B |
| `phase7_spatial_validation_*` | 836 vs 837 B |

The V13/non-suffixed files must be retained as historical evidence, but must not be used for current claims.

---

## 2. Stale documentation

### 2.1 `README.md` is contradicted by the current manuscript

`README.md` describes the V11 model and the V13 polygon-to-contact pipeline. It reports:

```text
fold ROC-AUC 0.846, 0.709, 0.687, 0.605 (mean 0.712)
pooled OOF ROC-AUC 0.707 vs 0.559 for the compact global baseline
pooled PR-AUC 0.134 vs 0.113
```

The current manuscript reports:

```text
fold ROC-AUC 0.875, 0.736, 0.685, 0.603
pooled OOF ROC-AUC 0.716 vs 0.674 for the matched global reference
pooled average precision 0.137 vs 0.108
```

`README.md` is therefore **stale and materially wrong for V14**. It should be rewritten only after the cleanup plan is approved.

### 2.2 Stage reports

| File | Problem | Evidence |
|---|---|---|
| `paper_1/stages/stage_03_methodology.md` | wrong domain expansions | lines 22--25: `CRZ: Congo River Zone`, `NKB: North Kundelungu Basin`, `SRB: South Roan Basin`, `MMSB: Mwembeshi Shear Belt` |
| `paper_1/stages/stage_04_results.md` | V11 metrics, not V14 | pooled 0.707 vs 0.559; fold AUC 0.846/0.709/0.687/0.605 |
| `paper_1/stages/stage_04_results.md` | reversed geographic labels | `Fold 1 (Northern)`, `Fold 2 (North-Central)`, `Fold 3 (Central-South)`, `Fold 4 (Southern)` |
| `paper_1/stages/stage_05_interpretation.md` | V11/M5 interpretation | states V11 and 0.707 vs 0.559 |
| `paper_1/stages/stage_06_discussion.md` | V11/M5 comparison | states V11--M5 and 0.707 vs 0.559 |

Canonical fold geometry from the V14 OOF file:

| Fold | Easting range (m) | Northing range (m) | Sector | Domains |
|---|---|---|---|---|
| 1 | 538,121--715,465 | 8,384,711--8,589,373 | south-east | NRB_3b 310, CRZ 146, NKB 12 |
| 2 | 529,383--685,637 | 8,491,951--8,720,125 | central | NRB_3b 240, NRB_3a 207, NKB 12, CRZ 8, SRB 1 |
| 3 | 409,933--605,600 | 8,685,082--8,830,461 | central-north | NRB_3a 468 |
| 4 | 282,675--504,594 | 8,756,582--8,859,512 | north-west | NRB_3a 452, MMSB 16 |

The stage report's labels are **opposite** to the canonical geometry. The manuscript's Figure 2 caption and Table 3 are correct.

### 2.3 `main.tex.20261029-audit.bak`

This tracked backup predates the current caption edit. It is historical, not active. ARCHIVE.

---

## 3. Manuscript and reporting consistency audit

This section addresses Section 15.3 of the mandate. Each item is reported, not fixed.

### 3.1 Incorrect Daly-domain expansions

`paper_1/manuscript/main.tex:140` and `paper_1/stages/stage_03_methodology.md:22--25` state:

```text
CRZ  = Congo River Zone
NKB  = North Kundelungu Basin
MMSB = Mwembeshi Shear Belt
SRB  = South Roan Basin
```

The documented expansions required by the mandate are:

```text
CRZ  = Central Rift Zone and later orogen
NKB  = Northeastern Katangan Basin
MMSB = Mwashia marginal sub-basins
SRB  = Southern Rift Basin
```

`Mwembeshi Fault Zone` is a separate structure and must not be conflated with `MMSB`.

**Proposed correction:** rewrite the domain paragraph and the stage-03 glossary against Daly et al. (2025) before submission. This is a separately reviewable manuscript change.

### 3.2 Figure 1 domain labels

The lithology legend is now correct (names, no codes). The tectonic-domain labels on the map are not:

```text
Inverted Salt Province NRB 3a
Kafue NRB 3b
HP/HT Orogen 4a CRZ
N Katangan Basin 1 NKB
2 MMSB
Mumbwa Sub Basin 6 MSB
Inverted Basid 5 SRB
LP/LT CRZ 4b
```

Problems:

- `Basid` is a typo for `Basin`.
- `MSB` is not the manuscript's `MMSB`.
- `2 MMSB` is a code with no name.
- The labels are raw V5 domain strings, not the manuscript's six canonical labels.
- The map does not show the full expansions required in the text.

**Proposed correction:** regenerate the Figure 1 annotation from a single canonical domain-label table, with one name per domain, and add a map-legend entry for each. Do not edit the QGIS project until the correction is approved.

### 3.3 Figure 4 reference and duplication

Verified:

- `fig4_oof_prospectivity.pdf` is referenced exactly once (`main.tex:408`).
- `fig4_oof_uncertainty.pdf` is referenced exactly once (`main.tex:415`).
- The prospectivity figure appears before the uncertainty figure.
- The files have different SHA-256 hashes.
- Visual inspection confirms different titles, colour bars, scales and content.
- Neither title mentions V14 or any internal script.

**Verdict:** no duplication, no swap, no missing reference.

### 3.4 Posterior mean versus credible-interval width

The distinction is correct in the manuscript:

- Figure 4a shades the **posterior mean conditional probability**.
- Figure 4b shades the **95% credible-interval width**.
- The uncertainty caption states: "Larger widths denote greater posterior uncertainty (less certainty), not higher prospectivity."
- The uncertainty colour bar runs 0--1.0; the prospectivity colour bar runs approximately 0--0.65.

This is a correct and important distinction. No change is required.

### 3.5 Claims about geological layers not evaluated

The previously criticised sentence beginning "Layers not used. An isostatic gravity grid, an upward-continued magnetic grid, and mining-footprint counts..." is **not present** in the current `main.tex`. A search for `Layers not used`, `isostatic`, `emag`, `upward`, `magnetic` and `mining` returns no visible manuscript occurrence.

The V14 model uses four predictors only: fault distance, lithology-contact distance, Bouguer gravity and surficial lithology. Isostatic gravity, magnetic and mining footprints are **not** V14 predictors. The current manuscript does not claim otherwise.

**Residual issue:** The data register still contains `isostatic`, `emag` and mining-footprint tables from earlier work. They should be archived as unused, not described as evaluated model layers.

### 3.6 Table 5 and Figure 5 use V11 shares, not V14

This is the most consequential reporting inconsistency found in this audit.

The manuscript Table 5 (`main.tex:442--445`) and Figure 5 TikZ use:

```text
Fold 1: fault 86.6, contact 11.0, gravity 2.0, rocks 0.4
Fold 2: fault 46.8, contact 28.0, gravity 5.6, rocks 19.6
Fold 3: fault 30.5, contact 32.8, gravity 14.4, rocks 22.4
Fold 4: fault 33.2, contact 23.2, gravity 28.7, rocks 14.9
```

These values match **exactly** `figures/audit/v11_population_reconstruction/fold_predictor_summary.csv`, not the current V14 reconstruction.

The current V14 file `figures/audit/v14_centroid_population_reconstruction/fold_predictor_summary.csv` gives:

```text
Fold 1: fault 86.52, contact 10.94, gravity 2.13, rocks 0.42
Fold 2: fault 45.08, contact 29.94, gravity 5.80, rocks 19.18
Fold 3: fault 30.40, contact 32.90, gravity 14.31, rocks 22.39
Fold 4: fault 33.48, contact 22.76, gravity 28.78, rocks 14.98
```

Differences are small but real. The largest are Fold 2 fault (46.8 vs 45.1), Fold 2 contact (28.0 vs 29.9), and Fold 4 fault (33.2 vs 33.5). The qualitative claims -- fault dominates the south-east, gravity rises toward the north-west, contact and lithology peak in Fold 3 -- remain true in both versions.

The manuscript test suite does **not** check Table 5 or Figure 5 values. The stale values therefore pass silently.

**Proposed correction:** regenerate Table 5 and the Figure 5 TikZ from the V14 summary, and add a test that compares the printed values to `fold_predictor_summary.csv`. This is a separately reviewable correction.

### 3.7 Table 2 is family-matched but not term-matched

The table reports `Predictors (n) = 4` for all four models. That is a count of predictor families. The quadratic models contain one extra distance term per distance predictor.

The caption now states this explicitly and says A--C and B--D are not complexity-matched. That is honest. It does not satisfy a strict "same number of predictors/terms" requirement.

**Proposed correction:** either (a) retain the current design and describe it as a representation-sensitivity design with unequal basis dimension, or (b) authorise a separately reported term-matched refit. Do not present the four models as complexity-matched.

### 3.8 Fold 4 and the "0.603" coincidence

The manuscript reports both models at 0.603 in the north-west and a delta of `-0.001` with a 95% interval of `[-0.046, +0.050]`.

This is **genuine**, not a fold-mapping or column-selection error:

```text
DVHM Fold 4 AUC:   0.6030531276432916
Global Fold 4 AUC: 0.6038184234905546
Exact difference: -0.0007652958472630367
```

Both round to 0.603. The displayed `-0.001` is the rounded exact difference. The interval is a paired row bootstrap of fixed OOF predictions (1,000 resamples, percentile interval, seeded), not a posterior interval and not a refit.

**Qualification:** the matched-global OOF file has no stable `id` column. Linkage is by row position, identical label vector, identical fold vector and coincident centroids. This is adequate for the audit but weaker than an ID-based join.

### 3.9 Spatial-support count discrepancy

The manuscript says 253 full-area cells and 1,646 clipped cells. Independent measurement says 254 cells at or above 24.999 km2 and 1,645 below. The difference is one cell and depends on the area threshold.

**Proposed correction:** define the threshold explicitly (for example, `>= 24.999 km2` or `== 25.0 km2`) and regenerate both numbers from the collapsed grid layer. Do not edit one number without the other.

### 3.10 Reference-author verification

`references.bib:291--292` has:

```bibtex
@article{Matthieu2026,
  author = {Matthieu, Tshanga Matthieu and Ncube, Lindani and Thamaga, Kgabo Humphrey},
```

This renders as "Tshanga Matthieu Matthieu". The surname `Matthieu` is duplicated. Verify the publisher metadata before submission. Other author blocks inspected do not show an obvious duplication.

### 3.11 Build and PDF consistency

The existing `main.pdf` is 23 pages and 5,222,747 bytes, newer than `main.tex`. The build log shows:

- no undefined references or citations;
- Figures 4a and 4b used on pages 14 and 15;
- one overfull `\hbox` of 10.42587 pt at lines 78--119.

No fresh compile was performed in this audit. The PDF is therefore **consistent with the current source to the extent the log shows**, but not independently rebuilt.

### 3.12 Supplementary material absent

`main.tex` cites Supplementary Sections S1 and Table S1, including reconstruction details and confirmation status. No supplementary PDF or TeX file exists in the repository. This is a **submission blocker** independent of the model.

---

## 4. Temporary, cache and environment assets

| Path | Size / count | Evidence | Classification |
|---|---:|---|---|
| `node_modules/` | 180.91 MB / 3,315 files | no `package.json`; largest tracked files are Node binaries | REMOVE from Git; REVIEW on disk |
| `.venv/` | 825.87 MB / 29,212 files | interpreter path points to a missing Windows Store Python | REMOVE/REBUILD |
| `.venv-1/` | 11.54 MB / 885 files | ignored duplicate environment | REMOVE after confirmation |
| `.venv312/` | 5.10 MB / 419 files | QGIS-compatible fragment | REVIEW |
| `.runtime_vendor/` | 26.82 MB / 1,489 files | vendored runtime | REMOVE/REBUILD |
| `.runtime_cache/` | 16.77 MB / 337 files | PyTensor compiled cache | REMOVE (reproducible) |
| `pytensor_cache/` | 8.62 MB / 224 files | compiled cache | REMOVE (reproducible) |
| `.trash/` | 30.13 MB / 974 files | local Trash, `latex-stale`, flattened PyTensor paths | REMOVE after confirmation |
| `__pycache__/`, `src/__pycache__/`, `tests/__pycache__/` | small | Python bytecode | REMOVE |
| `.pytest_cache/` | 5 files | test cache | REMOVE |
| `paper_1/manuscript/main.aux`, `.bbl`, `.blg`, `.fdb_latexmk`, `.fls`, `.log`, `.out`, `.synctex.gz` | approximately 0.1 MB | LaTeX build artifacts | REMOVE/REGENERATE |
| `figures/audit - Shortcut.lnk` | 1,392 B | Windows shortcut | REMOVE |
| `.git/objects/ca/tmp_obj_iLlger` | 13,065 B | stray temporary Git object | REVIEW; do not delete without integrity check |

---

## 5. Obsolete or superseded scripts

These are **not** proposed for immediate deletion. They are candidates for ARCHIVE because they implement superseded pipeline stages.

| Script | Evidence of supersession | Recommended |
|---|---|---|
| `src/ablation_suite.py` | reads V13 table; no V14 references | ARCHIVE |
| `src/compare_validation_strategies.py` | reads V13 table; outputs old comparison | ARCHIVE |
| `src/compute_fold_counts.py` | reads V13 table; fold counts now in V14 metrics | ARCHIVE |
| `src/feature_x_dom_x_target.py` | ad hoc V5/V13 summaries; not in V14 chain | ARCHIVE |
| `src/hierachical_spatial_cv.py` | reads V13 table; superseded by V14 wrapper | ARCHIVE |
| `src/phase_4b.py` | V14 phase 4b exploratory artifact; not cited in manuscript | ARCHIVE |
| `src/v9_posterior_extraction.py` | V9-era extraction | ARCHIVE |
| `src/v10_spatail_heterogeneity_upgraded.py` | V10-era model | ARCHIVE |
| `src/extract_v11_oof.py` | V11-only extraction | ARCHIVE |
| `src/audit_v11_trace_provenance.py` | V11-only audit | ARCHIVE |
| `src/rebuild_litho_contact_features.py` | contains placeholder path `C:/path/to/Clipped.shp`; not the V14 rebuild | ARCHIVE |
| `tools/build_submission_readiness.py` | hard-codes V13/V11 metrics | ARCHIVE or REWRITE |

`src/v11b_spatial_support_audit.py`, `src/v11c_beta_diagnostics.py` and `src/v11c_independent_check.py` are V14-era diagnostics despite their V11 names; keep as historical audit evidence.

---

## 6. What is genuinely obsolete versus merely old

The following tests were applied before calling anything obsolete:

1. Does the file feed the active V14 chain?
2. Is it cited by the manuscript, a stage report, or a test?
3. Is it an exact duplicate of a current file?
4. Is it a reproducible cache or build artifact?
5. Does it contain placeholder paths or old metrics that would mislead a reader?

A file is **not** obsolete merely because it is old. V11 pickles, V13 tables and the V5 archive are historical evidence and must be preserved.

---

## 7. Summary counts

| Category | Count / size |
|---|---:|
| Exact duplicate image groups | 2 (5 files) |
| Exact duplicate audit CSV pairs | 10 |
| Parallel but different V13/V14 pairs | 17 |
| Stale documentation files | 6 (`README.md`, `stage_03`--`stage_06`, `main.tex.bak`) |
| Root QA PNGs | 6 |
| Tracked Node files | 3,315 / 180.91 MB |
| Local environment/cache files | approximately 31,000 / 890 MB |
| Manuscript consistency findings | 12 (Section 3) |

---

## 8. Verdict

The repository contains real duplicates, real stale documentation and real reporting inconsistencies. It also contains historical assets that look redundant by filename but are not duplicates by content. The correct response is **selective archiving**, not wholesale deletion.

The four highest-value corrections, each requiring separate review, are:

1. Replace the V11 predictor-share values in Table 5 and Figure 5 with V14 values.
2. Correct the Daly-domain expansions and Figure 1 domain labels.
3. Resolve the one-cell spatial-support count discrepancy.
4. Add the missing supplementary material and verify the Matthieu reference.
