# Project Progress - Next Research Phase (v1)

**Status date:** 18 November 2026  
**Repository:** `C:\Users\vanmu\copperbelt-mineral-prospectivity`  
**Branch:** `main`  
**HEAD at review:** `c5d53ce`

This is a new progress record for the heterogeneous multimodal learning phase. It does not replace `docs/PROJECT_PROGRESS.md` or any historical project record.

## Work completed

| Stage | Status | Output |
|---|---|---|
| 1. Baseline inspection | Complete | Atlas, Phase 7, evidence-harvest and register state confirmed; atlas left untouched |
| 2. Evidence lineage audit | Complete | 568-value overlap resolved; independence register written |
| 3. Spatial evidence recovery assessment | Complete (assessment only) | L1-L5 classification of 147 RDC records; recovery priority queue |
| 4. Regional modality architecture | Complete | Region-by-modality matrix and spatial-support taxonomy |
| 5. Multimodal learning design | Complete (design only) | Architecture options and transfer-bridge cases |
| 6. Regional validation design | Complete (design only) | Spatial block, leave-one-region-out and leakage controls |
| 7. Decision memorandum | Complete | Recommendation: recovery before fitting |

## Key findings

1. **The 568 geochemical values and 568 drilling assay values are the same observations.** `spatial_atlas_v1/tools/build_evidence.py` reads `geochemical_measurements_v1.csv` in both `geochemistry()` and `drillholes()`. The drilling layer adds collar geometry, not a second chemical modality.
2. **The drilling register is defective.** `assay_interval_count` and `assayed_analytes` are populated only for `ZORO0009`, despite six assayed holes and 568 assay rows.
3. **One chemical dataset, not two.** 568 values = 71 intervals x 8 analytes in six holes. The 17 collars include 11 unassayed holes.
4. **No located chemistry outside six collars.** 3,056 determinations from 181 samples, zero coordinates. `RDC_4374` is the only spatialisable chemistry, with a provisional CRS.
5. **No located structure.** 2,945 observations from `RDC_1388`/`RDC_1525` plus 81 from `RDC_4374`; all mine-scale.
6. **No located geophysics or radiometry.** Five catalogued maps, 24 radiometric summaries, no footprints or coordinates.
7. **Recovery classes for 147 RDC records:** 0 L1, 27 L2, 60 L3, 58 L4, 2 L5. No coordinate was computed.
8. **Two OCR-damaged records are cheap recoveries.** `OCC-3399-007` and `OCC-3399-008` contain coordinates in the source text; page-image inspection should convert them to L1.
9. **The 5 km grid is a comparison frame.** Chemistry support is 0.1-1 m and spans 3 km x 0.5 km; locality uncertainty reaches 20-30 km.
10. **Only mapping-plus-occurrence transfer between A and B is currently defensible.** Chemistry and geophysics transfer is not identifiable.

## Decisions recorded

- No model fitting is authorised in this phase.
- The atlas, frozen V14, Stages 3-5, Phase 6 and Phase 7 are unchanged.
- No occurrence identity was created or merged.
- No coordinate, bearing or CRS was invented.
- No study-area boundary was adopted.
- The website was not modified.
- Nothing was staged, committed or pushed.

## New outputs

Under `data/raw/rdc_rocks/derived/research_phase_v1/`:

- `README_v1.md`
- `reports/A_evidence_lineage_and_independence_v1.md`
- `reports/B_spatial_evidence_recovery_v1.md`
- `reports/C_modality_architecture_v1.md`
- `reports/D_multimodal_learning_and_validation_design_v1.md`
- `reports/E_research_decision_memo_v1.md`
- `reports/validation_v1.md`
- `tables/evidence_lineage_568_v1.csv`
- `tables/evidence_lineage_568_summary_v1.csv`
- `tables/evidence_independence_register_v1.csv`
- `tables/spatial_recovery_candidates_v1.csv`
- `tables/region_modality_matrix_v1.csv`
- `tables/validation_manifest_v1.json`
- `tools/build_research_phase_tables.py`

This progress record is at `docs/PROJECT_PROGRESS_RESEARCH_PHASE_V1.md`.

## Outstanding work

### P0 - before any model fitting

1. Confirm the `RDC_4374` collar CRS and document the evidence for or against EPSG:32735.
2. Digitise the 81 `RDC_1388` station coordinates.
3. Georeference `RDC_0205` (magnetics) and `RDC_0208` (Bouguer) with control points and residuals.
4. Fix the label definition for USGS deposits, prospects and groups; de-duplicate the 80 group components.
5. Recover the two OCR-damaged coordinates in `RDC_3399`.
6. Repair the drilling-register assay metadata in a separately authorised pass.

### P1 - evidence recovery

7. Build sample-locality registers for `RDC_0088`, `RDC_0100`, `RDC_3217` and `RDC_4190`.
8. Resolve Phase 6 datums for `RDC_3854`/`RDC_3863` and the northing for `RDC_4055`.
9. Recover the `Lsw-1215` and `Lsw-1216` collars.
10. Extract the four single-page maps and cross-sections among the 50 unextracted occurrence documents.
11. Complete the stratigraphic crosswalk where the sources permit proven equivalence.

### P2 - modelling, only after P0

12. Pre-register E1/E2 (within-A and within-B mapping-plus-occurrence) with spatial blocks, baselines and stopping rules.
13. Run E3/E4 (A-to-B and B-to-A) only after E1/E2 establish within-region baselines.
14. Defer E5 and all chemistry/geophysics transfer experiments.

## Dependencies

- Coordinate recovery depends on page images and original map sheets, not on new acquisition.
- Geophysics depends on control points and projection metadata not currently recorded.
- Any chemistry transfer depends on locating the existing 149 samples.
- Any Case 3 bridge depends on proven stratigraphic equivalence.

## Notes on repository state

- Pre-existing uncommitted work was preserved: `.gitignore`, `README.md`, `docs/PROJECT_PROGRESS.md`, and the two alternative-geological-map scripts.
- `data/raw/` is ignored by Git, so the new research-phase outputs do not appear in Git status.
- This phase added one tracked-area file: `docs/PROJECT_PROGRESS_RESEARCH_PHASE_V1.md`.

