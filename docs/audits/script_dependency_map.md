# Script and Dependency Map

**Audit date:** 2026-11-05
**Scope:** every executable Python module in `src/`, `tools/`, `tests/` and `.mcp/`, plus root-level servers.
**Method:** static inspection of module docstrings, imports, string literals, input/output paths and references to manuscript components. No script was executed.

This map distinguishes the active V14 chain from historical V11/V13 code. It also records where a manuscript figure or table has no reproducible generator.

---

## 1. End-to-end V14 chain

```text
EXTERNAL / ARCHIVED INPUTS
  USGS CACB deposits/prospects (H: drive, currently inaccessible)
  Bouguer + isostatic gravity grids (absent from repository)
  Selley et al. (2005) fault compilation (CACB_Structures.gpkg)
  Africa surficial lithology raster (Africa_Surface_Lithology.tif)
  internal_contacts_only.gpkg / all_lithology_lines.gpkg / outer_boundary_line.gpkg
  archive/litho_contact_repair_2026-10-18/data/copperbelt_training_v5_with_tectonic_domain.csv
        |
        |  (no repository script creates the V5 table)
        v
build_corrected_training_table.py
  -> data/copperbelt_training_v13_litho_contact.csv
        |
        v
rebuild_litho_contact_centroid_features.py
  -> data/copperbelt_training_v14_centroid_contact.csv
  -> data/derived/litho_contact_centroid_distance_v14.csv/.gpkg
  -> data/derived/canonical_training_manifest_v14_centroid_contact.json
        |
        v
v14_centroid_spatial_stability.py   (executes frozen v11_spatial_stability_final.py)
  -> figures/v14_centroid_contact/v14_centroid_fold_{1..4}_trace.nc
  -> v14_centroid_fold_metrics.csv
  -> v14_centroid_geological_stability_matrix.csv
  -> run_manifest.json
        |
        +--> extract_v14_oof.py
        |      -> figures/v14_centroid_contact/v14_centroid_oof_predictions.csv
        |      -> ..._manifest.json
        |
        +--> phase3_ablation_suite.py
        |      -> phase3_ablation_fold_results_v14_centroid.csv
        |      -> phase3_ablation_summary_v14_centroid.csv
        |
        +--> phase4_tectonic_heterogeneity.py
        |      -> phase4_feature_by_daly_domain_v14_centroid.csv
        |      -> phase4_oof_predictions_v14_centroid.csv   (M7 matched-global reference)
        |
        +--> phase_4b.py                                  (historical descriptive branch)
               -> phase4_domain_heterogeneity_trace_v14_centroid.nc
               -> phase4_domain_coefficients_v14_centroid.csv
               -> phase4_heterogeneity_hyperparameters_v14_centroid.csv
        |
        +--> phase5_response_curves.py
        |      -> phase5_distance_response_v14_centroid.csv
        |      -> phase5_turning_points_v14_centroid.csv
        |
        +--> phase6_audit.py
        |      -> phase6_missingness_*.csv, phase6_feature_observation_support.csv
        |
        +--> phase7_spatial_validation.py
        |      -> phase7_spatial_validation_v14_centroid.csv
        |
        +--> phase7_final_validation.py
        |      -> phase7_final_validation_v14_centroid.csv
        |
        +--> phase7_multiscale_bootstrap.py
        |      -> phase7_multiscale_robustness_v14_centroid.csv  (M5 comparator)
        |
        +--> phase7_6_sensitivity_analysis.py
        |      -> phase7_6_axis_a_training_info_v14_centroid.csv
        |      -> phase7_6_axis_b_test_positives_v14_centroid.csv
        |
        +--> v11_domain_stratified_oof.py
               -> domain_stratified_auc_v14_centroid.csv

FIGURE GENERATORS
  generate_figure2_spatial_validation.py -> Figure 2 PNG/PDF + cell GPKG + verification JSON
  generate_oof_posterior_uncertainty.py  -> Figure 4 prospectivity + uncertainty PDF/PNG
                                            + v14_centroid_oof_posterior_uncertainty.csv/.gpkg
  run_distance_sensitivity.py + distance_sensitivity_models.py
                                         -> sensitivity_summary.csv, robustness_assessment.csv,
                                            response_curves_*.csv, support_quantiles.csv,
                                            tail_sensitivity_summary.csv
  run_v11_population_reconstruction.py   -> fold_predictor_contributions.csv/.png,
                                            fold_response_*.png, population_response_*.csv/.png
```

---

## 2. Script-by-script dependency table

### 2.1 Active V14 chain

| Script | Inputs | Outputs | Downstream use | Status |
|---|---|---|---|---|
| `src/rebuild_litho_contact_centroid_features.py` | V13 training table; `internal_contacts_only.gpkg`; `all_lithology_lines.gpkg`; `outer_boundary_line.gpkg`; V13 distance CSV | V14 training CSV; V14 distance CSV/GPKG; V14 manifest | defines the V14 modelling table | ACTIVE |
| `src/v14_centroid_spatial_stability.py` | V14 training CSV; canonical manifest; frozen V11 model module | four V14 traces; fold metrics; stability matrix; run manifest | all V14 inference | ACTIVE |
| `src/v11_spatial_stability_final.py` | training fold; domain labels; predictors | posterior draws | frozen model specification invoked by V14 wrapper | FROZEN |
| `src/extract_v14_oof.py` | V14 traces; V14 training table | OOF predictions + manifest | all pooled/fold metrics; Figures 4--6 | ACTIVE |
| `src/phase3_ablation_suite.py` | V14 table; traces | ablation fold results and summary | model-comparison evidence | ACTIVE |
| `src/phase4_tectonic_heterogeneity.py` | V14 table | domain summaries; M7 OOF predictions | Table 4; matched-global 0.674 reference | ACTIVE |
| `src/phase5_response_curves.py` | V14 table; traces | distance-response and turning-point tables | D* discussion | ACTIVE |
| `src/phase6_audit.py` | V14 table | missingness/support tables | limitations evidence | ACTIVE |
| `src/phase7_spatial_validation.py` | V14 table; OOF predictions | fold AUC/AP/Brier and deltas | Table 3/4; manuscript fold claims | ACTIVE |
| `src/phase7_final_validation.py` | V14 table; OOF predictions | final validation summary | manuscript consistency | ACTIVE |
| `src/phase7_multiscale_bootstrap.py` | V14 table; OOF predictions | block-bootstrap robustness | sensitivity evidence; M5 comparator only | ACTIVE |
| `src/phase7_6_sensitivity_analysis.py` | V14 table; OOF predictions | training-support and test-positive axes | limitation evidence | ACTIVE |
| `src/v11_domain_stratified_oof.py` | V14 table; OOF predictions | domain-stratified AUC | NRB_3a/NRB_3b paragraph | ACTIVE |
| `src/generate_figure2_spatial_validation.py` | V14 table; fold geometry | Figure 2 PNG/PDF; cell GPKG; verification JSON | Figure 2 | ACTIVE |
| `src/generate_oof_posterior_uncertainty.py` | V14 table; V14 traces; OOF predictions; Figure 2 cells | Figure 4 prospectivity/uncertainty PDF/PNG; uncertainty CSV/GPKG | Figures 4a/4b | ACTIVE |
| `src/run_distance_sensitivity.py` | V14 traces; V14 table | sensitivity summaries, response curves, robustness table | Table 2; Figure 6 | ACTIVE |
| `src/distance_sensitivity_models.py` | V14 table | model-fitting helpers for the sensitivity runner | Table 2 | ACTIVE |
| `src/run_v11_population_reconstruction.py` | V14 table; V14 traces | `fold_predictor_contributions.csv/.png`; response curves; D* summaries | Table 5; Figure 5 values | ACTIVE but historically named |

### 2.2 Support modules

| Module | Role |
|---|---|
| `src/numerics.py` | NumPy/SciPy fallbacks for `StandardScaler` and `roc_auc_score` under the QGIS runtime |
| `src/runtime_shim.py` | import-time bootstrap for the QGIS CPython 3.12 interpreter |
| `src/trace_io.py` | NetCDF-first, pickle-fallback trace persistence |
| `src/oof_provenance.py` | validated, ID-based linkage of OOF predictions to the modelling table |
| `src/validation_strategies.py` | fold construction helpers |

### 2.3 Tooling

| Script | Role | Status |
|---|---|---|
| `tools/check_v14_runner.py` | preflight check that the V14 wrapper builds the frozen V11 runner | ACTIVE |
| `tools/check_fold_compatibility.py` | compares active folds with the frozen HEAD implementation | ACTIVE |
| `tools/run_pytest_qgis.py` | runs pytest under the QGIS interpreter | ACTIVE |
| `tools/build_submission_readiness.py` | hard-codes old V13/V11 metrics and figure checks | HISTORICAL/UNSAFE as evidence |

---

## 3. Figure provenance

| Displayed figure | Source file | Generator | Reproducibility |
|---|---|---|---|
| Figure 1 | `fig1_study_area_context_annotated.pdf` | QGIS project `paper_1/copperbelt_exploration_figure1_recovered.qgz`; external layers; manual annotation | **Partial**: QGIS project exists, but source layers are outside the repository |
| Figure 2 | `fig2_spatial_validation_design.png` | `src/generate_figure2_spatial_validation.py` | **Full** given V14 table and fold geometry |
| Figure 3 | `fig3_modelling_workflow.tikz` | manually authored TikZ | **Manual**: no generator |
| Figure 4a | `fig4_oof_prospectivity.pdf` | `src/generate_oof_posterior_uncertainty.py` | **Full** given V14 traces and table |
| Figure 4b | `fig4_oof_uncertainty.pdf` | `src/generate_oof_posterior_uncertainty.py` | **Full** given V14 traces and table |
| Figure 5 | `fig5_predictor_contributions.tikz` | underlying values from `run_v11_population_reconstruction.py`; TikZ manually populated | **Partial**: values reproducible, TikZ not generated |
| Figure 6 | `fig6_distance_sensitivity.tikz` | underlying values from `run_distance_sensitivity.py`; TikZ manually populated | **Partial**: values reproducible, TikZ not generated |

### 3.1 Figure 4 duplication audit

The two Figure 4 files are **not duplicates and not swapped**.

| Check | Prospectivity | Uncertainty |
|---|---|---|
| File | `fig4_oof_prospectivity.pdf` | `fig4_oof_uncertainty.pdf` |
| SHA-256 | `485855D7AC72080C1E15264EFF4B8876EC8235A2B155B821D35EFE87728E7C4D` | `08EDF5B08D2277C3120D3F3E3FCAA07B699C7F19E8E0C5970EC0264C5B806DF3` |
| Visible title | `Out-of-fold prospectivity` | `Posterior uncertainty in conditional probability` |
| Colour bar | `Posterior mean conditional occurrence probability` (approximately 0--0.65) | `95% credible-interval width` (0--1.0) |
| Meaning | posterior mean conditional probability | posterior credible-interval width; larger = more uncertain |
| Red crosses | 138 positive-cell centroids | none |

Visual inspection in this audit confirmed the two maps differ in title, colour bar, scale and content. Both are referenced exactly once in `main.tex` (lines 408 and 415). The shared `fig4_` basename is a naming convention, not evidence of duplication.

### 3.2 Figure 1 label audit

Visual inspection confirmed the lithology legend now uses names, not numeric codes:

```text
Metasedimentary, Silicic, Metaigneous, Alluvium-Fluvial, Alluvium-Saline, Water
```

No lithology class code (4, 6, 7, 14, 16, 20) is visible in the legend. The map title does not mention V14 or any internal script.

However, the tectonic-domain labels on Figure 1 are inconsistent with the manuscript:

| Figure 1 label | Manuscript label | Issue |
|---|---|---|
| `Inverted Salt Province NRB 3a` | NRB_3a | descriptive label, not the Daly expansion |
| `Kafue NRB 3b` | NRB_3b | descriptive label |
| `HP/HT Orogen 4a CRZ` | CRZ_4b | manuscript expansion is also wrong |
| `N Katangan Basin 1 NKB` | NKB | manuscript says `North Kundelungu Basin` |
| `2 MMSB` (map margins) | MMSB | code only, no name |
| `Mumbwa Sub Basin 6 MSB` | MMSB | spelling/code mismatch (`MSB` vs `MMSB`) |
| `Inverted Basid 5 SRB` | SRB | typo `Basid` for `Basin` |
| `LP/LT CRZ 4b` | CRZ_4b | sub-label |

These are reporting-consistency defects. They must be corrected in a separately reviewable manuscript/figure revision, not silently during cleanup.

---

## 4. Table provenance

| Manuscript table | Values | Generator | Status |
|---|---|---|---|
| Table 1 (predictor sources) | source, native support, transformation | manual synthesis from manifests and Supplementary S1 | **Manual** |
| Table 2 (distance-representation design) | Model A/B/C/D mean AUC and AP | `run_distance_sensitivity.py` -> `sensitivity_summary.csv`; TikZ hard-coded | **Partial** |
| Table 3 (fold metrics) | 468 cells; positives 9/31/37/61; AUC 0.875/0.736/0.685/0.603 | `v14_centroid_fold_metrics.csv` | **Full** |
| Table 4 (comparator metrics) | DVHM 0.716/0.137/0.069; global 0.674/0.108/0.079 | `phase4_oof_predictions_v14_centroid.csv`; `phase7_spatial_validation_v14_centroid.csv` | **Full** |
| Table 5 (variance shares) | 86.6/46.8/30.5/33.2 percent fault shares, etc. | `run_v11_population_reconstruction.py` -> `fold_predictor_contributions.csv` | **Full given V14 traces** |

### 4.1 Table 2 qualification

All four specifications use the same four predictor **families**: fault distance, lithology-contact distance, Bouguer gravity and surficial lithology. The `Predictors (n) = 4` column is therefore a count of families, not design-matrix columns.

The linear specifications contain one distance term per distance predictor. The quadratic specifications add one curvature term per distance predictor. Models A and B are matched within the quadratic basis; C and D are matched within the linear basis; A--C and B--D are not complexity-matched. The manuscript caption and Section 3.4 now state this explicitly.

A genuinely term-matched four-model comparison would require a separately authorised refit with an equal number of distance terms in every specification. No such refit exists in the repository. This is recorded as an unresolved reporting/design limitation, not treated as a present result.

---

## 5. Historical V11/V13 scripts

| Script | Input | Output | Classification |
|---|---|---|---|
| `src/build_corrected_training_table.py` | archived V5 table; V13 distance CSV | V13 training table; V13 manifest | HISTORICAL-ESSENTIAL |
| `src/rebuild_litho_contact_features.py` | placeholder `C:/path/to/Clipped.shp`; V13 outputs | V13 distance CSV/GPKG | HISTORICAL; contains placeholder paths |
| `src/validate_litho_contact_pipeline.py` | V13 table; V13 distance; V11 uncertainty GPKG | validation JSON | HISTORICAL |
| `src/audit_contact_distance_definition.py` | V13 table; V13 distance | `contact_distance_definition_check.json` | HISTORICAL |
| `src/v11_spatial_stability_final.py` | V13 table (original); V14 table (via wrapper) | V11/V14 traces | FROZEN MODEL |
| `src/extract_v11_oof.py` | V11 traces | V11 OOF predictions | HISTORICAL |
| `src/audit_v11_trace_provenance.py` | V11 traces; V13 table | trace provenance manifest | HISTORICAL |
| `src/v11b_spatial_support_audit.py` | V14 table | domain/fold support tables | HISTORICAL AUDIT |
| `src/v11c_beta_diagnostics.py` | V14 traces; V14 table | beta diagnostics | HISTORICAL AUDIT |
| `src/v11c_independent_check.py` | V14 table; beta diagnostics CSV | independent check | HISTORICAL AUDIT |
| `src/ablation_suite.py` | V13 table | ablation outputs | OBSOLETE/ARCHIVE |
| `src/compare_validation_strategies.py` | V13 table | validation-strategy comparison | OBSOLETE/ARCHIVE |
| `src/compute_fold_counts.py` | V13 table | fold counts | OBSOLETE/ARCHIVE |
| `src/feature_x_dom_x_target.py` | V5/V13-style table | feature-by-domain summaries | OBSOLETE/ARCHIVE |
| `src/hierachical_spatial_cv.py` | V13 table | hierarchical CV | OBSOLETE/ARCHIVE |
| `src/phase_4b.py` | V14 table | full-data descriptive trace; domain coefficients; hierarchical hyperparameters | HISTORICAL DESCRIPTIVE; not the DVHM and not manuscript evidence |
| `src/v9_posterior_extraction.py` | V13 table | V9 feature-by-block matrix | OBSOLETE/ARCHIVE |
| `src/v10_spatail_heterogeneity_upgraded.py` | V13 table | V10 trace and summaries | OBSOLETE/ARCHIVE |

---

## 6. Provenance gaps

| Gap | Evidence | Consequence |
|---|---|---|
| No gravity-ingestion script | no `src` file reads a Bouguer or isostatic raster; the columns are inherited from the V5 table | gravity predictors cannot be regenerated from repository code |
| No occurrence-ingestion script | no `src` file reads the USGS shapefile; labels are inherited from the V5 table | occurrence labels and their 1,000 m positional uncertainty cannot be regenerated |
| No V5-table generator | `build_corrected_training_table.py` reads `archive/.../copperbelt_training_v5_with_tectonic_domain.csv` | the provenance chain begins at a table whose own construction is undocumented |
| Figure 5 TikZ is hard-coded | no `src` or `tools` file writes `fig5_predictor_contributions.tikz` | figure values can drift from the CSV without detection |
| Figure 6 TikZ is hard-coded | no `src` or `tools` file writes `fig6_distance_sensitivity.tikz` | same risk |
| Figure 3 is manual | `fig3_modelling_workflow.tikz` is hand-authored | no automated check that it matches the model |
| Supplementary material absent | `main.tex` cites Supplementary Sections S1 and Table S1; no supplement PDF/TeX exists | several reconstruction claims cannot be checked |
| `generate_figure2_spatial_validation.py` has no direct repository reference | only its outputs are referenced | generator can be orphaned by refactoring |
| `validate_v14_model_outputs.py` has no direct repository reference | validation JSON exists, but no pipeline entry point is defined | validation is not automatically rerun |
| `tools/build_submission_readiness.py` hard-codes old metrics | contains V13/V11 values | must not be used as current evidence |

---

## 7. Import and call relationships

Local module imports observed:

```text
src.oof_provenance      -> used by tests and manuscript checks
src.numerics            -> used by V14 extraction, phase analyses, distance sensitivity
src.runtime_shim        -> imported by conftest.py and QGIS runners
src.trace_io            -> used by V14 extraction, phase analyses, population reconstruction
src.validation_strategies -> used by fold construction and historical CV
src.distance_sensitivity_models -> imported by run_distance_sensitivity.py
src.v11_spatial_stability_final -> imported by v14_centroid_spatial_stability.py
```

External scientific imports across the codebase:

```text
numpy, pandas, scipy, scikit-learn, matplotlib
geopandas, shapely, pyogrio, pyproj, rasterio (where geospatial)
pymc, pytensor, arviz
qgis, PyQt5 (QGIS runtime only)
mcp, platformdirs (MCP servers only)
```

No `pyproject.toml`, `requirements.txt` or lock file defines these dependencies. `requirements-test.txt` contains only `pytest==8.4.2`.

---

## 8. Reproducibility classification

| Class | Components |
|---|---|
| **Fully reproducible** (given existing assets) | V14 contact-distance rebuild from contact geometry; OOF extraction from traces; phase3--phase7 summaries from table + traces; Figures 2 and 4 |
| **Reproducible with available assets but not tested** | V14 refit from the V14 table (traces already exist; refit is expensive); Table 5 values from `run_v11_population_reconstruction.py` |
| **Partially reproducible** | V14 table from V13 table (V13 depends on archived V5); Table 2 (values reproducible, TikZ manual); Figures 5 and 6 (values reproducible, TikZ manual) |
| **Not reproducible from the repository alone** | occurrence labels; fault distance (external `CACB_Structures.gpkg`; 227 cells differ from stored values); gravity sampling (source rasters absent); Figure 1 (QGIS project plus external layers); Supplementary S1 (absent) |

---

## 9. Script-map verdict

The active V14 scientific chain is traceable from the V14 table through the traces to the OOF predictions and the phase outputs. The weakest links are **upstream of the V14 table**: labels, gravity, fault distance, domains and centroids are inherited from an archived V5 table whose construction is not scripted. The second weakest link is **downstream presentation**: Figures 5 and 6 and Tables 1 and 2 are partly manual.

No script should be deleted on the grounds that it is "old". Historical V11/V13 scripts are evidence for the benchmark and for the provenance of the V14 table.
