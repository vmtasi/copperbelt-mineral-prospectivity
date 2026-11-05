# Repository Inventory

**Project:** Copperbelt Mineral Prospectivity (Central African Copperbelt, DRC--Zambia)
**Repository:** `C:\Users\vanmu\copperbelt-mineral-prospectivity`
**Audit date:** 2026-11-05
**Audit mode:** read-only. No file was moved, deleted, renamed, regenerated, refitted or committed.

This document is Phase I of the repository audit required by the research mandate. It records what the repository contains, what is tracked, what is protected, and what remains uncertain. It does not authorise cleanup.

---

## 1. Version-control state

| Item | Value |
|---|---|
| Current branch | `main` |
| HEAD | `2be0dbb` (`Add V14 centroid-contact pipeline and manuscript updates`) |
| Remote relation | ahead of `origin/main` by 1 commit; behind by 0 |
| Other local branch | `backup/mcp-repair-20261020` |
| Tags | none |
| Tracked files | 3,426 |
| Tracked files under `node_modules/` | 3,315 (96.8% of tracked files) |
| Tracked files under `src/` | 44 |
| Tracked files under `paper_1/` | 29 |
| Tracked files under `tests/` | 6 |
| Tracked files under `.mcp/` | 6 |
| Tracked files under `tools/` | 5 |
| Tracked files under `data/` | 4 |

### 1.1 Protected uncommitted changes

The following six files have uncommitted edits and must be preserved exactly. They are the current manuscript, its rendered figure, the QGIS project, and the tests that enforce the manuscript's numbers.

| Path | Change | Protected because |
|---|---|---|
| `paper_1/copperbelt_exploration_figure1_recovered.qgz` | binary, 155,441 -> 156,532 bytes | QGIS source of Figure 1 |
| `paper_1/manuscript/images/fig1_study_area_context_annotated.pdf` | binary, 1,148,673 -> 1,726,898 bytes | rendered Figure 1 |
| `paper_1/manuscript/images/fig1_study_area_context_annotated.png` | binary, 2,955,561 -> 2,621,547 bytes | rendered Figure 1 preview |
| `paper_1/manuscript/main.pdf` | binary, 4,641,498 -> 5,222,747 bytes | current 23-page manuscript build |
| `paper_1/manuscript/main.tex` | 1 line changed | lithology caption now uses class names rather than codes |
| `tests/test_manuscript_numerical_consistency.py` | 12 insertions, 11 deletions | enforces the current manuscript numbers and labels |

The only non-ignored untracked file is `.claude/settings.local.json`.

**No Git checkpoint, stash, reset, commit or push was performed.**

---

## 2. Size and composition

Directory sizes exclude `.git` unless stated. Sizes are decimal MB from the filesystem.

| Directory | MB | Files | Role | Status |
|---|---:|---:|---|---|
| `.venv` | 825.87 | 29,212 | broken local Python environment | REMOVE/REBUILD |
| `node_modules` | 180.91 | 3,315 | accidental Node dependency tree, no `package.json` in repo | REMOVE from Git; REVIEW on disk |
| `literature` | 148.08 | 10 | reference PDFs | ARCHIVE (ignored) |
| `figures` | 118.37 | 214 | generated figures, traces, audit outputs | ARCHIVE/KEEP selectively |
| `archive` | 108.26 | 188 | historical snapshots | KEEP (ignored) |
| `.mcp` | 57.08 | 10 | MCP server and tunnel binaries | REVIEW |
| `.trash` | 30.13 | 974 | local Trash, LaTeX-stale and PyTensor leaks | REMOVE after confirmation |
| `.runtime_vendor` | 26.82 | 1,489 | vendored runtime | REMOVE/REBUILD |
| `paper_1` | 21.29 | 37 | manuscript, stages, QGIS project | KEEP |
| `.runtime_cache` | 16.77 | 337 | PyTensor compiled cache | REMOVE (reproducible cache) |
| `.venv-1` | 11.54 | 885 | duplicate/broken environment | REMOVE after confirmation |
| `pytensor_cache` | 8.62 | 224 | compiled cache | REMOVE |
| `.venv312` | 5.10 | 419 | QGIS-compatible environment fragment | REVIEW |
| `data` | 3.23 | 9 | canonical tables, manifests, contact distances | KEEP |
| `src` | 1.17 | 122 | 44 Python modules plus caches | KEEP selectively |
| `tests` | 0.34 | 21 | 6 test modules plus caches | KEEP |
| `tools` | 0.06 | 8 | 5 utility modules plus runtime cache | KEEP selectively |
| `docs` | 0.02 | 1 | only `beta_interpretation_audit.md` existed before this audit | KEEP |
| `outputs` | 0.00 | 0 | empty | KEEP as target directory |
| `.git` | 484.92 | 1,411 | Git object store | KEEP |

Total working tree excluding `.git`: approximately **1,564 MB**. Including `.git`: approximately **2,049 MB**. The earlier handoff figure of approximately 2,054 MB is consistent with inclusion of `.git`.

### 2.1 Largest tracked files

| Path | MB |
|---|---:|
| `node_modules/@napi-rs/canvas-win32-x64-msvc/skia.win32-x64-msvc.node` | 26.03 |
| `node_modules/@oai/artifact-tool/node_modules/skia-canvas/lib/skia.node` | 23.00 |
| `node_modules/@img/sharp-win32-x64/lib/libvips-42.dll` | 17.75 |
| `node_modules/@napi-rs/canvas-win32-x64-msvc/icudtl.dat` | 9.98 |
| `paper_1/manuscript/main.pdf` | 4.98 |
| `node_modules/tesseract.js-core/tesseract-core-relaxedsimd.wasm.js` | 4.48 |
| `node_modules/tesseract.js-core/tesseract-core-simd.wasm.js` | 4.47 |
| `node_modules/tesseract.js-core/tesseract-core.wasm.js` | 4.47 |

Every one of the ten largest tracked files except `main.pdf` is an accidental Node binary. This is the single largest repository-hygiene defect.

---

## 3. Tracked-data inventory

The Git index tracks only four data files:

| Path | Purpose |
|---|---|
| `data/derived/canonical_training_manifest.json` | V13 manifest, historical |
| `data/derived/canonical_training_manifest_v14_centroid_contact.json` | V14 canonical manifest |
| `data/derived/litho_contact_centroid_distance_v14.csv` | V14 centroid-to-contact distances |
| `data/derived/litho_contact_centroid_distance_v14.gpkg` | V14 centroid-to-contact geometry |

The following canonical V14 assets are **ignored by `.gitignore` and therefore exist only on the local disk**:

```text
data/copperbelt_training_v14_centroid_contact.csv
figures/v14_centroid_contact/v14_centroid_fold_1_trace.nc
figures/v14_centroid_contact/v14_centroid_fold_2_trace.nc
figures/v14_centroid_contact/v14_centroid_fold_3_trace.nc
figures/v14_centroid_contact/v14_centroid_fold_4_trace.nc
figures/v14_centroid_contact/v14_centroid_oof_predictions.csv
figures/audit/phase*_v14_centroid*
figures/audit/v14_model_output_validation_post_refit_20261021.json
archive/**  (entire historical archive)
```

`figures/` and `/archive/` are explicitly ignored. This is a **reproducibility and preservation risk**, not a scientific error: the benchmark files exist locally and pass validation, but a fresh clone of the repository would not contain them. Section 15.1 of the mandate requires a complete benchmark record; that record is written in `docs/audits/v14_preservation_report.md`, but physical preservation still requires an approved archive/backup decision.

---

## 4. Root-level file inventory

| File | Size | Git | Assessment |
|---|---:|---|---|
| `.gitignore` | 1,040 B | tracked | KEEP, revise (see cleanup plan) |
| `README.md` | 8,049 B | tracked | STALE: describes V11/V13 polygon-to-contact pipeline and old metrics |
| `AI_PROJECT_CONTEXT.md` | 10,151 B | ignored | local planning note; ARCHIVE |
| `RECONCILIATION_REPORT.md` | 21,261 B | ignored | historical reconciliation note; ARCHIVE |
| `conftest.py` | 270 B | tracked | KEEP: installs QGIS runtime shim before tests |
| `requirements-test.txt` | 15 B | tracked | KEEP but insufficient: only `pytest==8.4.2` |
| `server2.py` | 54,827 B | tracked | MCP server; REVIEW, security-sensitive |
| `oauth_gateway.py` | 13,028 B | tracked | MCP OAuth gateway; REVIEW, security-sensitive |
| `assessment_focused_layout_validation.png` | 811,649 B | tracked | Figure QA artifact; REMOVE or ARCHIVE after approval |
| `daly_boundary_validation_overlay.png` | 122,458 B | tracked | Figure QA artifact; REMOVE or ARCHIVE |
| `daly_boundary_validation_overlay.pgw` | 106 B | tracked | world file for QA overlay; REMOVE with overlay |
| `daly_draft_layout_validation.png` | 455,557 B | tracked | **exact duplicate** of `paper_1/manuscript/images/fig1_study_area_context.png` (SHA-256 prefix `59709D3EA932`) |
| `domain_colours_labels_validation.png` | 856,284 B | tracked | Figure QA artifact; REMOVE or ARCHIVE |
| `final_single_boundary_layout_validation.png` | 735,615 B | tracked | **exact duplicate** of `fig1_study_area_context_current.png` and `tract_focused_closure_QA.png` (`688CC1630F2D`) |
| `tract_focused_2km_grid_QA.png` | 690,501 B | tracked | Figure QA artifact; REMOVE or ARCHIVE |
| `tract_focused_closure_QA.png` | 735,615 B | tracked | exact duplicate (above) |
| `tract_focused_layout_validation.png` | 852,868 B | tracked | Figure QA artifact; REMOVE or ARCHIVE |
| `tract_focused_single_boundary_validation.png` | 741,605 B | tracked | Figure QA artifact; REMOVE or ARCHIVE |

---

## 5. Source-code inventory

`src/` contains 44 Python modules. They fall into four groups; the full dependency map is in `docs/audits/script_dependency_map.md`.

### 5.1 Active V14 pipeline

```text
rebuild_litho_contact_centroid_features.py
  -> data/copperbelt_training_v14_centroid_contact.csv
v14_centroid_spatial_stability.py
  -> figures/v14_centroid_contact/v14_centroid_fold_{1..4}_trace.nc
extract_v14_oof.py
  -> v14_centroid_oof_predictions.csv
phase3_ablation_suite.py
phase4_tectonic_heterogeneity.py
phase5_response_curves.py
phase6_audit.py
phase7_spatial_validation.py
phase7_final_validation.py
phase7_multiscale_bootstrap.py
phase7_6_sensitivity_analysis.py
v11_domain_stratified_oof.py
```

Figure generators:

```text
generate_figure2_spatial_validation.py  -> Figure 2
generate_oof_posterior_uncertainty.py   -> Figure 4 prospectivity + uncertainty
run_distance_sensitivity.py             -> distance-sensitivity CSV outputs
```

Support modules: `numerics.py`, `runtime_shim.py`, `trace_io.py`, `oof_provenance.py`, `validation_strategies.py`.

Frozen model specification: `v11_spatial_stability_final.py` (SHA-256 `587b6496...`), executed against the V14 table by `v14_centroid_spatial_stability.py`.

### 5.2 Historical V11/V13 modules

`ablation_suite.py`, `compare_validation_strategies.py`, `compute_fold_counts.py`, `feature_x_dom_x_target.py`, `hierachical_spatial_cv.py`, `phase_4b.py`, `v9_posterior_extraction.py`, `v10_spatail_heterogeneity_upgraded.py`, `v11c_beta_diagnostics.py`, `v11c_independent_check.py`, `v11b_spatial_support_audit.py`, `extract_v11_oof.py`, `run_v11_population_reconstruction.py`, `audit_v11_trace_provenance.py`, `audit_contact_distance_definition.py`, `validate_litho_contact_pipeline.py`, `build_corrected_training_table.py`, `rebuild_litho_contact_features.py`.

These are historical evidence, not active V14 code. Several read the V13 table.

### 5.3 Tooling

`tools/build_submission_readiness.py` hard-codes old V13/V11 metrics and must not be used as current evidence. `tools/check_fold_compatibility.py` and `tools/check_v14_runner.py` are V14 preflight utilities. `tools/run_pytest_qgis.py` runs tests under the QGIS interpreter.

### 5.4 Encoding defect

Approximately 25 `src` files begin with a UTF-8 BOM. Python executes them, but `ast.parse(open(..., encoding="utf-8"))` fails with `invalid non-printable character U+FEFF`. Any tooling that parses source without `utf-8-sig` will misclassify these files. This is a software-quality defect, not a scientific one.

---

## 6. Test inventory

| Test | Purpose | Current state |
|---|---|---|
| `test_centroid_contact_pipeline.py` | V14 centroid contact regression | not executed in this audit; needs geo stack |
| `test_litho_contact_pipeline.py` | V13 contact pipeline | historical |
| `test_distance_sensitivity_math.py` | sensitivity mathematics | not executed |
| `test_manuscript_numerical_consistency.py` | manuscript vs frozen V14 numbers, labels, figures | not executed; source inspected |
| `test_mcp_server_policy.py` | MCP policy | not relevant to science |
| `test_numerics_compat.py` | QGIS numerical fallbacks | not executed |

The available Codex runtime has `pandas` and `numpy` but not `pytest`, `geopandas`, `shapely`, `pyogrio`, `arviz`, `pymc`, `scipy` or `matplotlib`. The QGIS interpreter is the intended test runtime, but it was not launched during this read-only audit. No claim of test success is made.

---

## 7. Figure inventory

`paper_1/manuscript/images/` contains 19 files. The manuscript uses seven displayed figures:

| Displayed | File | Status |
|---|---|---|
| Figure 1 | `fig1_study_area_context_annotated.pdf` | current; names in legend; uncommitted |
| Figure 2 | `fig2_spatial_validation_design.png` | current |
| Figure 3 | `fig3_modelling_workflow.tikz` | current, manually authored |
| Figure 4a | `fig4_oof_prospectivity.pdf` | current; posterior mean probability |
| Figure 4b | `fig4_oof_uncertainty.pdf` | current; 95% credible-interval width |
| Figure 5 | `fig5_predictor_contributions.tikz` | current; hard-coded values, no generator |
| Figure 6 | `fig6_distance_sensitivity.tikz` | current; hard-coded values, no generator |

Unused or superseded image files include `fig1_spatial_folds.png`, `fig1_study_area_context.png`, `fig1_study_area_context_current.png`, `fig2_prediction_map.png`, `fig3_predictor_contributions.png`, `fig4_response_fault.png`, and `fig5_response_lithology.png`. Several are exact duplicates of root QA PNGs (Section 4). They are **ARCHIVE/REMOVE candidates, not automatic deletions**.

The filename numbering is offset from the displayed numbering for the last two figures: `fig5_predictor_contributions.tikz` is displayed as Figure 5 only because no other file occupies that slot, and `fig6_distance_sensitivity.tikz` is displayed as Figure 6. This is a naming inconsistency, not a duplication.

---

## 8. Archive inventory

`archive/` is ignored by Git and contains three snapshots:

| Directory | Contents | Status |
|---|---|---|
| `archive/litho_contact_repair_2026-10-18/` | V5 base table, V12 provisional tables, contact features, old figures, hashes | historical-essential for the V5 -> V13 provenance chain |
| `archive/centroid_contact_repair_2026-10-20/` | pre-V14 snapshot of V13 table, V11 pickles, OOF, source geometry, scripts, SHA-256 sums | historical-essential |
| `archive/paper1_provenance_2026-10-20/` | manuscript provenance snapshot | historical |

The V5 base table is the only surviving source of labels, domains, centroids, fault distance, gravity and lithology class before the V13/V14 contact-distance corrections. It must not be discarded.

---

## 9. Git hygiene findings

1. **3,315 `node_modules` files are tracked**, approximately 180 MB, with no `package.json`. This is the highest-priority hygiene defect.
2. A stray Git object exists at `.git/objects/ca/tmp_obj_iLlger` (13,065 bytes, 2026-10-18). It is a temporary object, not a commit. Do not delete without a Git integrity check.
3. `.venv`, `.venv-1` and `.venv312` are ignored. `.gitignore` lists `.venv/` but not the other two; they are ignored through an inaccessible global ignore file (`C:\Users\vanmu/.config/git/ignore`, permission denied). The exact mechanism is unresolved.
4. LaTeX build artifacts (`main.aux`, `.bbl`, `.blg`, `.fdb_latexmk`, `.fls`, `.log`, `.out`, `.synctex.gz`) are ignored but currently present. They are reproducible.
5. `main.tex.20261029-audit.bak` is tracked. It is a historical backup, not an active source.
6. `figures/audit - Shortcut.lnk` is a Windows shortcut in the generated-figures directory. It has no scientific value.
7. `.trash/` contains 974 files, including LaTeX-stale files and PyTensor cache leaks with flattened absolute paths. It is ignored but consumes 30 MB.
8. `outputs/` is empty and is the natural target for future generated outputs.

---

## 10. What remains uncertain

| Question | Current answer |
|---|---|
| Is the global ignore file the reason `.venv-1` and `.venv312` are ignored? | Not established; the file is unreadable. |
| Can the V14 training CSV be hashed now? | No: it is locked by another process. The manifest records `161ad506...`; all other V14 hashes matched. |
| Is the USGS occurrence shapefile available? | No: `H:` is inaccessible in this session. Counts are carried forward from the prior verified audit. |
| Are the gravity source rasters present? | No gravity raster was found in the repository or `C:\Project\data for projects`. Only sampled values in the canonical table survive. |
| Is the supplementary material present? | No supplementary PDF or TeX file exists in the repository, although `main.tex` cites Supplementary Sections S1 and Table S1. |
| Do the tests pass? | Not executed; the scientific Python stack is absent from the active runtime. |
| Is `node_modules` used by the MCP tooling? | No `package.json` or import path was found. It appears accidental. |

---

## 11. Inventory verdict

The repository contains a **scientifically coherent but operationally untidy** V14 benchmark. The active V14 chain is identifiable and its key artefacts validate. The principal risks are:

1. Canonical V14 data and traces are local-only (ignored by Git).
2. `node_modules` dominates the tracked file count and repository size.
3. Stale V11/V13 documentation and stage reports coexist with current V14 results.
4. Several manuscript claims depend on hard-coded figure files with no generator.
5. The V5 -> V13 -> V14 provenance chain depends on an archived table whose own generator is not in the repository.

No cleanup should begin until the limitation register and cleanup plan have been reviewed.
