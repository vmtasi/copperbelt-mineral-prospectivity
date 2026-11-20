# Copperbelt Mineral Prospectivity — Project Progress

**Status date:** 18 November 2026 (Africa/Nairobi)  
**Repository:** `C:\Users\vanmu\copperbelt-mineral-prospectivity`  
**Branch:** `main`  
**Current HEAD:** `c5d53ce` — Add acquisition and geology foundation workflows  
**Working tree at review:** one modified tracked file (`.gitignore`) and three untracked paths (`.claude/`, `src/data_acquisition/process_alternative_geological_maps.py`, `src/data_acquisition/verify_alternative_geological_maps.py`). Preserve them.

> **Update 20 Nov 2026:** The heterogeneous multimodal-transfer programme (Research Phase v1 / Stage 3B) is **shelved**. See [MULTIMODAL_PROGRAMME_CLOSURE.md](MULTIMODAL_PROGRAMME_CLOSURE.md). The V14 benchmark is now tracked in Git with checksums; see [paper_1/benchmark/V14_BENCHMARK.md](../paper_1/benchmark/V14_BENCHMARK.md). The current priority is Paper 1 submission.

This dashboard separates research streams that have different outcomes and denominators. It deliberately does **not** calculate one project-wide percentage by averaging unrelated work.

## Progress dashboard

| Workstream | Progress | Evidence-based status |
|---|---:|---|
| RDC OCR corpus | **100%** | 101 documents, 3,188 pages, 3,663,039 characters; zero page errors and zero remaining documents/pages (status file generated 8 Nov). |
| RDC map extraction / Phase 6 | **Build complete; CRS resolution open** | RDC_3863: 70 features; RDC_3854: 47; RDC_4055: 19. The first two remain in map-grid metres with datum unresolved; RDC_4055 remains in source pixels with easting-only calibration and unresolved northing. |
| Phase 7 evidence integration | **100% validated** | 136 geological features, 147 RDC occurrence observations, 113 candidate site entities, 21 exploration-history records; 32/32 validation checks passed. No occurrence identity is confirmed and no RDC row is admitted as a model positive. |
| Tract-supported occurrence-intensity workflow | **80% of defined stages** | Stage 4, Stage 5A, 5B and 5C are built and validated; Stage 5D is the next stated stage and has not been implemented. This workflow models compiled-record intensity per tract-supported area, not detection-free mineralisation probability. |
| Geological foundation / structural predictors | **3/3 foundation build stages verified** | Daly domains, named major structures v2 and 27 structural candidate features on 2,193 equal-area hexagons are built; independent checks pass (32/32 for structure reconstruction; 23/23 for structural predictors). No model has been fitted on these 27 features. |
| Daly-union model option (not adopted) | **Exploratory specification only; not an approved project direction** | Phase 7 assessed predictors under a hypothetical full-union frame. The study-area memo and Nov 12 evidence report explicitly do not adopt the Daly union as the study-area boundary. Do not treat this as the chosen next model. |
| Paper 1 / V14 submission | **Draft and benchmark exist; current submission readiness not established** | V14 benchmark is scientifically preserved locally, but key artefacts are ignored by Git. The Oct 20 submission-readiness report is based on the earlier V13/polygon-contact state and must not be treated as a current V14 readiness verdict. |
| Repository cleanup / reproducibility | **Plan written; execution not started** | Cleanup plan is explicitly proposed only. V14 benchmark durability, runtime/lockfile, historical-code separation and documentation freshness remain open. |

### Stage-level progress bars

- RDC OCR: `██████████` 100%
- Phase 7 integration validation: `██████████` 100% (32/32 checks)
- Tract occurrence-intensity workflow: `████████░░` 80% (4 of 5 defined stages; Stage 5D outstanding)
- Geological predictor construction: `██████████` 100% (3 of 3 construction stages; modelling is a separate next phase)
- Daly-union model option: not approved; no implementation commitment (comparison/sensitivity frame only unless a separate study-area decision is made)

These bars describe named deliverables only. They do not imply that unresolved source, coordinate, scientific or publication limitations have been solved.

## What is complete and what it means

### 1. RDC source corpus and evidence integration

- OCR queue is exhausted: 101/101 documents and 3,188/3,188 pages, with zero page errors.
- Phase 6 generated map-derived layers for RDC_3854, RDC_3863 and RDC_4055. Their coordinate systems are **not** established as EPSG:32735 or WGS84. RDC_4055 has no northing control.
- Phase 7 integrated 136 features and passed 32/32 validation checks.
- The RDC occurrence register preserves 147 source observations and 113 candidate site entities. Crosswalk classifications remain 76 no-candidate, 51 unverified site-level candidates and 20 ambiguous many-to-one. Confirmed identity matches: zero.
- The 21 exploration-history records have no spatial footprints; only five are dated. They establish documentary presence/chronology, not spatial survey effort.

### 2. Tract-supported occurrence-intensity analysis

- Stage 4 data build and model build both passed 14/14 validation checks.
- Stage 5A inventory audit passed 14/14 checks.
- Stage 5B support sensitivity passed 21/21 checks.
- Stage 5C residual spatial diagnostics passed 23/23 checks.
- `VORONOI_ALL` remains the accepted reference support; Stage 5B showed that strict 80%/99% hexagon subsets change the retained population and estimand.
- Stage 5C found residual spatial structure, but the fixed spatial comparators did not improve ranking, probability accuracy and residual dependence together. None was promoted.
- The next documented step is Stage 5D: test only bounded observation-process scenarios that are estimable from evidence. Current candidates are S0 (baseline), S2 (document-source effect), S5 (spatial-precision restriction) and S6 (entity-duplication sensitivity). S1, S3 and S4 are not estimable from the present corpus. Do not fit a free detection surface or call the response a mineralisation probability.

### 3. Geological foundation and candidate modelling frames

**Important study-area status:** no final boundary has been adopted. The study-area decision memo recommends a two-tier direction—central Katangan Copperbelt as the scientific study area, Roan/Mines district cluster as initial modelling support—but marks the polygon decision as pending. The Daly union is a comparison frame, not an adopted boundary. The Phase 7 full-union model-variable decision is an exploratory assessment under that alternative frame, not a superseding study-area decision.

There are two distinct structural-feature tracks; keep their supports and meanings separate:

- The **2,193-point equal-area hexagon** structural-predictor build has 27 features and a 23/23 independent validation pass. It uses the named v2 structure set (47 high/medium named segments) and has not been used in a fitted model. Support choice (full support versus overlap subsets) remains a modelling decision.
- The **16,492-cell Daly-union Phase 7** evidence integration supports a different, broader mapped major-structure measure: 127 high/medium extraction-confidence segments, 6,518 km of linework, with about 2–5 km positional uncertainty. The model-variable decision makes this distance the primary geological term. Structure density at 50 km and named-fault distance are secondary; other terms are sensitivity/context/excluded.

The Phase 7 outcome support is concentrated in NRB 3a/3b: 186 USGS deposits inside the Daly union, with 149 in 3a and 37 in 3b. The other seven reconstructed domain polygons have zero occurrence support. Therefore:
- estimate only in NRB 3a/3b;
- predict across the full Daly union with an explicit extrapolation flag outside NRB;
- do not fit or interpret a nine-domain effect;
- report country and fold composition transparently;
- preserve the four occurrence-supported fold counts [46, 48, 46, 46];
- retain source/precision/entity sensitivities and never infer barren geology from missing records.

### 4. Historical Zambian evidence acquisition

The 9 Nov handoff identifies these outstanding acquisitions/actions:
1. Thieme & Johnson (1981) 1:1,000,000 map and explanation/memoir.
2. GSD 1:250,000 Copperbelt sheets and memoirs, prioritising Konkola/Chililabombwe, Chingola/Nchanga, Kitwe/Nkana/Mufulira and Luanshya/Chambishi.
3. GSD Reports 60 and 66 with their 1:100,000 maps.
4. Drysdall et al. (1974), Thieme Report 26 (1969) and Report 32 (1971).
5. Economic Geology 2005 supplementary appendices/CD-ROM.
6. Documented-versus-inferred datum/grid register for RDC_3854, RDC_3863 and RDC_4055.
7. Zambian occurrence extraction from the 12 acquired theses and other independent sources, with source-page lineage.

Access-controlled GSD files must be obtained through authorised account/purchase or library routes; do not bypass access controls. A preview or catalogue index is not a substitute for the original map.

## Ordered remaining work

### P0 — Resolve project-state and preservation questions before risky execution

1. Reconcile the MCP frozen-baseline warning. The current server reports drift in `AI_PROJECT_CONTEXT.md` and `src/validation_strategies.py`; the Phase 7 validation note says the earlier warning could not be reproduced and that `src/validation_strategies.py` had no working-tree diff. Investigate the baseline history and classify the drift; do **not** recapture the baseline as a shortcut.
2. Preserve current Git state. Do not reset, clean, stage, commit or push the existing changes as part of this status update.
3. Create an approved durable V14 benchmark bundle with checksums and restoration instructions. The V14 training table, traces, OOF predictions, fold metrics and several audit outputs are ignored by Git; a fresh clone does not reproduce the benchmark.
4. Verify the two untracked alternative-geological-map scripts and their outputs. They were structurally inspected but not executed or independently validated in this review.

### P1 — Complete the evidence-to-model chain

5. Implement Stage 5D using only S0/S2/S5/S6, with explicit estimands, assumptions, validation and interpretation boundaries. Do not manufacture missing campaign footprints or a detection surface.
6. Revisit the study-area decision before committing to a new regional model. The current two-tier recommendation is central Katangan Copperbelt as scientific extent and Roan/Mines district cluster as initial modelling support, with Tenke-Fungurume held out as an independent test district. Complete the decision gates: Roan/Mines subcrop, stratigraphic crosswalk, structural connectivity, and sampling-gap assessment. Do not silently substitute the Daly union.
7. Separately decide whether and how to model the 2,193-cell structural-predictor table. Freeze support and a compact predictor/ablation set before outcome comparisons; use fold-training-only transformations; retain HIGH/ALL_NAMED and bridge variants as sensitivity only.
8. Keep the Daly-union Phase 7 model-variable assessment as a comparison/sensitivity branch unless a separate, explicit study-area decision adopts it. If revisited, retain NRB 3a/3b estimation limits and flag predictions elsewhere as extrapolations.
9. Acquire authoritative bedrock/stratigraphic and Zambian map sources. Until then, do not relabel surficial classes as bedrock or force unresolved local-grid map features into the basin CRS.
10. Resolve map coordinate evidence: RDC_3854/3863 datum and projection; RDC_4055 northing and full coordinate transform. Only then can those map-derived geometries be considered for regional spatial use.

### P2 — Publication and repository release

10. Re-audit Paper 1 against the current V14 artefacts and manuscript, not the older Oct 20 V13 readiness report. Verify citation metadata, supplement requirements and numerical/figure consistency; choose a target journal before journal-specific formatting.
11. Update the root README and agent context to describe the current V14 and redesign workstreams without replacing historical results with newer but different estimands.
12. Execute repository cleanup only as a separate approved operation. The cleanup plan protects six manuscript/QGIS/test files and requires per-file provenance/recovery checks. Do not delete or move ignored research data, archives, environments or MCP assets merely because they are large.
13. Establish a reproducible environment and lockfile, separate the active V14 benchmark from redesign code, and make every reported number/map traceable to input hashes, code, config and retained output.

## Current Git and server notes

- MCP project server responds and points to the expected repository root.
- Current Git status: `main...origin/main [ahead 1]`; modified `.gitignore`; untracked `.claude/` and the two alternative-geological-map scripts.
- The latest local commit is `c5d53ce`, which adds acquisition and geology-foundation workflows.
- Writable MCP locations include `docs/`, `src/`, `tests/`, `figures/audit/`, `paper_1/stages/` and `README.md`. The raw/processed data trees are not writable through the project MCP interface.
- Frozen research files and V14 evidence must remain unchanged unless a separate, explicit research decision authorises a new versioned output.

## Sources of project status

- `data/raw/rdc_rocks/derived/ocr/status_v1.json`
- `data/raw/rdc_rocks/derived/georef/phase6/phase6_status_v1.md`
- `data/raw/rdc_rocks/derived/integration/phase7/phase7_report_v1.md`
- `data/raw/rdc_rocks/derived/integration/phase7/phase7_validation_v1.md`
- `data/raw/rdc_rocks/derived/integration/phase7/supplementary/model_variable_decision_v1.md`
- `data/processed/stage4/stage4_data_validation_v1.json`
- `data/processed/stage4/stage4_model_validation_v1.json`
- `data/processed/stage5/stage5_inventory_validation_v1.json`
- `data/processed/stage5b/stage5b_validation_v1.json`
- `data/processed/stage5c/stage5c_validation_v1.json`
- `docs/geology_foundation/structural_predictors_v1.md`
- `data/processed/structural_predictors/structural_predictor_validation_v1.json`
- `data/derived/eg100_2005/historical_source_gaps_and_acquisition_priorities_v1.md`
- `docs/audits/v14_preservation_report.md`
- `docs/audits/cleanup_plan.md`
- `docs/limitation_register.md`

---
