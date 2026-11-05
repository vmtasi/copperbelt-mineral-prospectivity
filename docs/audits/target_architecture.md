# Target Repository and Evidence Architecture

**Audit date:** 2026-11-05<br>
**Scope:** repository, data-lineage and evidence organisation required by the Copperbelt redesign. This document does not select a new statistical model and does not authorise a code or file move.<br>
**Current baseline:** V14 at Git HEAD `2be0dbb` with six protected uncommitted files.

---

## 1. Purpose

The redesign will be scientifically useful only if its evidence can be reconstructed, challenged and compared with V14. The current repository contains a coherent V14 analysis but not a durable evidence system: the numerical benchmark is largely ignored by Git, upstream preprocessing is partly undocumented, historical and current outputs coexist without authoritative labels, and several printed numbers are maintained by hand.

The target architecture therefore treats the repository as an **evidence and execution boundary**, not merely a collection of scripts. The design principle is:

> Any reported number, map or model decision must resolve to one identified input, one identified code path, one recorded configuration and one retained output, and must be distinguishable from the frozen V14 benchmark.

This is a repository architecture requirement. It does not presuppose the final model family.

---

## 2. Current-state defects that the target must eliminate

Ranked by consequence for the redesign:

| Rank | Defect | Current evidence | Consequence |
|---:|---|---|---|
| 1 | The benchmark is not durably preserved | `data/*.csv`, `figures/` and `archive/` are ignored; V14 traces and OOF predictions exist only locally | A fresh clone cannot reproduce or audit the historical benchmark |
| 2 | Provenance begins at the V5 table | No script creates `copperbelt_training_v5_with_tectonic_domain.csv`; gravity and occurrence ingestion are absent | Labels, domains, fault distance and gravity cannot be independently regenerated |
| 3 | Active and historical code are mixed | 44 `src` files span V9-V14, with old model names still present | A reviewer cannot tell which code is authoritative |
| 4 | Printed numbers are partly manual | Figures 5-6 and parts of Tables 1-2 are hard-coded TikZ or manual synthesis | Silent drift from the CSV outputs is possible |
| 5 | The runtime is not reproducible | `.venv` is broken; no `pyproject.toml`/lock file; QGIS and modelling Python are split | Re-running tests or outputs depends on undocumented local state |
| 6 | Repository contents are polluted | 3,315 tracked `node_modules` files; multiple cache/env directories; root QA images | Cleanup obscures scientific diffs and inflates clone size |
| 7 | Stale reports are not labelled | `README.md` and stages 03-06 describe earlier models | A reader can mistake V11/V13 values for current results |
| 8 | Reporting claims outrun evidence | Missing Supplementary S1; stale Table 5/Figure 5; wrong domain names | The submission package cannot support its own citations |

The architecture should fix the organisational causes. It should not hide the scientific limitations identified in `docs/limitation_register.md`.

---

## 3. Target directory shape

The shape below is a destination, not an instruction to move files now. Existing work should be migrated in reviewable steps, with V14 frozen first.

```text
copperbelt-mineral-prospectivity/
├── README.md                    # current status, quick start, benchmark pointer
├── pyproject.toml               # pinned modelling dependencies
├── requirements-qgis.txt        # separate QGIS/GDAL environment, if retained
├── configs/
│   ├── v14_benchmark.yaml       # immutable description of the historical run
│   └── v15_*.yaml               # future design configurations
├── data/
│   ├── README.md                # data contract and access instructions
│   ├── manifests/               # tracked, small provenance manifests
│   ├── derived/                 # generated tables; normally ignored
│   └── external_registry.md     # DOI/URL, licence, coverage, checksum, access date
├── benchmarks/
│   └── v14/
│       ├── README.md            # status, scope, do-not-modify warning
│       ├── metrics.csv          # historical-reported and reproduced columns
│       ├── fold_support.csv     # cells, positives, prevalence, domain composition
│       ├── sensitivity.csv      # A-D and D* diagnostic summaries
│       ├── checksums.sha256     # hashes of canonical local artefacts
│       └── provenance.md        # lineage and missing-artefact statement
├── src/
│   ├── copperbelt/              # optional later package boundary
│   ├── v14_*.py                 # frozen historical entry points only
│   └── ...                      # active redesign code
├── archive/
│   └── code/
│       ├── v9_v10/
│       ├── v11/
│       └── v13/
├── outputs/                     # ignored; all newly generated runs
│   └── <run_id>/
├── docs/
│   ├── audits/                  # this audit package
│   ├── limitations/             # register and disposition updates
│   └── methods/                 # data contracts, model notes, decisions
├── paper_1/                     # manuscript and reviewable figure sources
└── tests/
    ├── test_manuscript_*.py
    ├── test_benchmark_*.py
    └── test_data_contracts.py
```

The only new structural concepts are `configs/`, `benchmarks/` and a clearly separated `outputs/` run area. A full `src` package rewrite is not required before the scientific redesign is approved.

---

## 4. Authoritative artefact policy

### 4.1 One canonical version per evidence type

| Evidence type | Canonical location | Historical copies |
|---|---|---|
| V14 training table | `data/copperbelt_training_v14_centroid_contact.csv` | archived snapshots only |
| V14 model specification | `src/v11_spatial_stability_final.py` via frozen wrapper | unchanged |
| V14 traces | `figures/v14_centroid_contact/v14_centroid_fold_{1..4}_trace.nc` | none substituted |
| V14 OOF predictions | `figures/v14_centroid_contact/v14_centroid_oof_predictions.csv` | none substituted |
| M7 OOF predictions | `figures/audit/phase4_oof_predictions_v14_centroid.csv` | V13/non-suffixed copies are separate history |
| V14 fold metrics | `figures/v14_centroid_contact/v14_centroid_fold_metrics.csv` | none |
| Validation summary | `figures/audit/phase7_spatial_validation_v14_centroid.csv` | V13/non-suffixed copies are separate history |
| Sensitivity outputs | `figures/audit/*_v14_centroid.csv` | V13/non-suffixed copies are separate history |
| Phase-4B descriptive fit | `figures/audit/phase4_domain_heterogeneity_trace_v14_centroid.nc` | labelled Phase 4B, not DVHM |

The nonsuffixed files are not interchangeable with their V14 counterparts. Equal basenames do not imply equal content.

### 4.2 Immutability

V14 artefacts are frozen. Any regeneration must write to a new run identifier and must not overwrite a canonical path. The following are never edited in place:

```text
data/copperbelt_training_v14_centroid_contact.csv
data/derived/canonical_training_manifest_v14_centroid_contact.json
figures/v14_centroid_contact/**
figures/audit/*_v14_centroid*
archive/**
```

### 4.3 Durability

Because large V14 artefacts are ignored by Git, the approved cleanup must create a benchmark bundle with:

- a compressed copy of the training table, manifests, traces, OOF predictions, fold and phase summaries;
- SHA-256 checksums for every file;
- the exact Git commit and model-specification hash;
- a restoration instruction;
- storage outside the Git object database or via an explicitly approved large-file mechanism.

Acceptance is a fresh-copy hash comparison, not the existence of a folder.

---

## 5. Data and run contracts

### 5.1 Dataset manifest

Every new modelling table must have a machine-readable manifest containing:

```text
dataset_id, created_utc, source_files, source_checksums,
crs, grid_or_support, row_count, unique_id_count,
positive_count, complete_case_count, exclusion_reasons,
transform_code_and_sha256, output_checksum,
columns_with_semantics, missingness_summary,
licence_and_access_status
```

The manifest must distinguish raw, derived and modelling-ready data. A file without this record is not eligible for a canonical result.

### 5.2 Fold manifest

Each spatial partition must be written as a standalone table with stable cell IDs and a reason or rule for membership. It must not be recreated implicitly from row order inside an analysis script. Required fields:

```text
cell_id, fold, centroid_x, centroid_y, domain, deposit_present,
cell_area, boundary_distance, partition_rule_version
```

Buffered and alternative partitions must have their own identifiers and may not overwrite the historical four-fold assignment.

### 5.3 Run manifest

Every model run must record:

```text
run_id, model_id, code_commit, code_hashes, config_path, config_hash,
input_dataset_hash, fold_manifest_hash, random_seed,
chains, draws, tuning, target_accept, runtime_versions,
started_utc, finished_utc, output_checksums, status,
known_warnings, model_specification_changed
```

The V14 manifest is a good starting point. New runs should not copy its weaknesses: the wrapper should not need to mutate a historical model file, and the environment should be reproducible without a numerical compatibility shim.

### 5.4 Metric contract

Each evaluation output must contain one row per model, fold and metric, with explicit:

```text
model_id, fold, metric, value, n_cells, n_positive,
prevalence, interval_low, interval_high, interval_method,
bootstrap_unit, n_resamples, seed, comparator_id
```

This removes ambiguity between pooled, fold-mean, row-bootstrap and spatial-block estimates.

### 5.5 Figure and table contract

No numerical value in a manuscript figure or table may exist only as a TikZ literal. Each displayed number must come from a CSV/JSON contract or be checked against one. For each display, retain:

- source file and column;
- transformation or rounding rule;
- generator or validation script;
- test that compares printed values with the source.

Figures 5 and 6 are the immediate examples of why this is needed.

---

## 6. Active versus historical code

### 6.1 Active V14 chain

The active V14 chain is limited to:

```text
src/rebuild_litho_contact_centroid_features.py
src/v14_centroid_spatial_stability.py
src/v11_spatial_stability_final.py        # frozen model specification
src/extract_v14_oof.py
src/phase3_ablation_suite.py
src/phase4_tectonic_heterogeneity.py
src/phase5_response_curves.py
src/phase6_audit.py
src/phase7_spatial_validation.py
src/phase7_final_validation.py
src/phase7_multiscale_bootstrap.py
src/phase7_6_sensitivity_analysis.py
src/v11_domain_stratified_oof.py
src/generate_figure2_spatial_validation.py
src/generate_oof_posterior_uncertainty.py
src/run_distance_sensitivity.py
src/distance_sensitivity_models.py
src/run_v11_population_reconstruction.py
```

Support modules (`numerics`, `trace_io`, `validation_strategies`, `oof_provenance`, `runtime_shim`) remain with the active chain. `phase_4b.py` is a separate descriptive model and must be labelled as such, not treated as the DVHM.

### 6.2 Historical code

V9/V10 model scripts, the old V11 extraction and the superseded comparison utilities should eventually move under `archive/code/`. They should not be interleaved with the active pipeline. No script is removed solely because it is old: its provenance value must first be assessed.

### 6.3 Redesign code

New model code must live under a new versioned identifier and must not silently inherit the V14 label. The redesign is expected to make at least one of these structural changes:

- a new spatial-support definition;
- a new occurrence likelihood or observation component;
- updated geological predictors;
- a new spatial-heterogeneity structure.

The code should therefore be a new execution path, not a rename of the old one.

---

## 7. Environment architecture

The repository needs at least two clearly separated environments:

| Environment | Purpose | Required record |
|---|---|---|
| Modelling/runtime | PyMC, ArviZ, NumPy, pandas, scikit-learn, plotting | Python version, package lock, platform, compiler/runtime flags |
| QGIS/geospatial | geometry inspection, map rendering, GDAL/QGIS operations | QGIS version, GDAL/GEOS/PROJ versions, plugin dependencies |

The current `.venv` must not be treated as a reproducible environment. A broken environment is a local operational issue, not evidence about the model. The redesign should declare dependencies in `pyproject.toml` or an equivalent lock and include a short verification command that runs without MCMC.

Minimum environment acceptance:

1. A clean checkout plus restored benchmark data can run import and manifest checks.
2. Existing non-MCMC tests can be collected and run.
3. Stored V14 summaries can be reproduced from traces.
4. QGIS-dependent figure work can be identified without requiring the modelling environment.

---

## 8. Manuscript and reporting architecture

The manuscript is the final reporting layer, not a data store. Its numerical values must be downstream of `benchmarks/`, `outputs/` and the approved fold/metric contracts.

Required separation:

| Layer | Responsibility | Must not do |
|---|---|---|
| Data and manifests | define inputs and exclusions | contain manuscript-only claims |
| Benchmark | freeze historical evidence | absorb revised values silently |
| Model configuration | declare a testable design | be embedded only in code |
| Outputs | retain predictions and diagnostics | overwrite prior runs |
| Verification | test reconstruction and numerical consistency | pass because a string appears in the source |
| Manuscript | state bounded interpretation | serve as the only record of a number |

Reporting corrections must be separate from repository cleanup. The following are separate review units:

1. Domain names and Figure 1 labels.
2. Table 5 and Figure 5 V14 values.
3. Table 2 complexity qualification or term-matched refit.
4. Stage-report/README supersession.
5. Supplementary S1.
6. Reference and licence metadata.

No manuscript correction should be bundled into a cache cleanup commit.

---

## 9. Scientific decision records

The redesign will need a small number of explicit records, not a new document for every idea:

| Record | Question | Blocking evidence |
|---|---|---|
| DR-01 | Which spatial support is the primary estimand? | Complete-grid/hexagon/point-process comparison |
| DR-02 | Which occurrence likelihood is identifiable? | Presence-background or point-process feasibility |
| DR-03 | Which structural predictor is defensible? | Fault source, hierarchy and reproducibility |
| DR-04 | Which lithological representation is defensible? | Bedrock map coverage and harmonisation |
| DR-05 | How is spatial heterogeneity represented? | Domain ablation or continuous alternative |
| DR-06 | What validation partition is primary? | Buffered/alternative-fold results |
| DR-07 | What uncertainty is claimed? | Input, parameter, label and spatial components separated |

Each record should state the decision, rejected alternatives, evidence, limitations and the test that would overturn it. This is more useful than adding more models without a decision trail.

---

## 10. Migration sequence and approval gates

The target architecture should be reached without losing V14 evidence.

| Gate | Action | Approval requirement | Success test |
|---|---|---|---|
| A0 | Review and accept this audit package | user review | documents accepted |
| A1 | Preserve V14 benchmark and protected working-tree files | explicit approval | checksum bundle restored and verified |
| A2 | Remove accidental/tracked operational clutter | explicit approval | fresh clone has no `node_modules`; tests still collect |
| A3 | Archive historical code and stale outputs | explicit approval | active chain maps to `script_dependency_map.md` |
| A4 | Add benchmark and data contracts | explicit approval | manifests validate; V14 reproduction checks pass |
| A5 | Repair reporting defects | separate manuscript review | numerical and label tests pass |
| A6 | Establish clean environments | explicit approval | non-MCMC verification runs |
| A7 | Begin redesign implementation | scientific review | new run has manifest, fold and metric contracts |

No gate may be skipped by moving files into `archive/` with a different name.

---

## 11. Target-state acceptance criteria

The architecture is adequate when all of the following are true:

1. **Traceability:** every reported number resolves to a stored source artefact and a code path.
2. **Immutability:** V14 can be identified and reproduced without being overwritten by V15.
3. **Durability:** the benchmark survives a fresh clone or approved restore procedure.
4. **Separation:** active, historical and obsolete code are visibly distinct.
5. **Contractual data:** every modelling table has a source, hash, exclusion record and column semantics.
6. **Validation clarity:** pooled, fold, bootstrap and spatial-block intervals cannot be confused.
7. **Reproducibility:** the environment and test entry points are declared, not inferred from a local `.venv`.
8. **Reporting discipline:** figures and tables are generated or tested against canonical values.
9. **Limitation continuity:** every V14 limitation has a status and an acceptance test in `docs/limitation_register.md`.
10. **Minimal operational risk:** cleanup has a rollback path and does not touch protected manuscript files.

The architecture is not adequate if it merely reorganises files while leaving the benchmark ignored, the provenance gaps unrecorded, or the manuscript numbers detached from their sources.

---

## 12. Architecture verdict

The target can be reached with a small number of high-value changes: freeze and back up V14, separate active from historical code, add manifests and contracts, make figure/table values testable, and establish a clean environment. A wholesale rewrite is not justified before the geological and occurrence-data feasibility work is complete.

The immediate priority is evidence preservation, not cosmetic tidying.
