# RDC.ROCKS Acquisition Handoff v4

Generated: 2026-11-08 12:56 Africa/Nairobi
Project: `C:\Users\vanmu\copperbelt-mineral-prospectivity`
Supersedes: `ACQUISITION_HANDOFF_v3.md` (left untouched).

## 1. Current state

Phase 6 is complete and preserved. Phase 7 integration and the Daly-based supplementary study-area work are complete. The next model has a written variable decision; no model has been fitted.

Do not repeat the OCR breadth pass, the OCR deep pass, the Phase 6 digitisation, or any completed download.

## 2. OCR state

- Deep-pass worker PID 15004 has exited.
- `status_v1.json` at 09:28 UTC: **101 documents, 3,188 pages, 3,663,039 characters, 0 page errors, 0 documents / 0 pages remaining.**
- Final QC `qc_v1.json` at 09:30 UTC: 101 documents, 3,188 pages, 81 empty pages, one flagged document (`2527`, mostly empty). This is a review flag, not a lost page.
- The OCR corpus remains local under `data/raw/rdc_rocks/derived/ocr/`. No OCR text was sent as a bulk model payload.

## 3. Phase 7 outputs

Canonical Phase 6 package and integrated evidence:

- `data/raw/rdc_rocks/derived/integration/phase7/canonical_phase6/` (8 artefacts, original/canonical SHA-256 match)
- `rdc_geological_evidence_v1.gpkg` (136 features: 70 RDC_3863, 47 RDC_3854, 19 RDC_4055)
- `geological_evidence_v1.csv`
- `geological_evidence_schema_v1.json` / `.md`
- `phase6_artifact_manifest_v1.csv`
- `phase7_provenance_v1.csv`

Occurrence entity system (source tables, not merged):

- `data/raw/rdc_rocks/derived/occurrences/entities/site_entities_v1.csv` (113 candidate sites)
- `occurrence_observations_v1.csv` (147 source observations)
- `occurrence_entity_crosswalk_v1.csv` (147 links)
- `occurrence_reconciliation_v1.csv` (review view; 0 confirmed identities, 0 model positives)

Exploration and observation process:

- `exploration_evidence_register_v1.csv` (21 records; 0 footprints; 5 dated)
- `predictor_eligibility_v1.csv` (30 Phase 6 candidates)
- `observation_scenarios_v1.csv` (7 bounded scenarios; 4 estimable)
- `observation_process_assessment_v1.md`

Daly study area, support and folds:

- `supplementary/daly_study_area_v1.gpkg` (union of 9 reconstructed domains, 398,428 km2)
- `supplementary/daly_support_5km_v1.gpkg` (16,492 cells; 15,440 full-support; 1,052 edge)
- `supplementary/daly_geographic_folds_v1.gpkg` (4-fold primary; 5-fold alternative)
- `supplementary/daly_fold_support_summary_v1.csv`
- `supplementary/daly_support_5km_summary_v1.csv`
- `supplementary/spatial_support_assessment_v1.md`
- `supplementary/usgs_deposit_overlay_audit_v1.csv` (per-record union, tract, domain, country and fold audit)
- `supplementary/usgs_deposit_overlay_summary_v1.csv` (reconciled counts)
- `supplementary/phase7_supplement_provenance_v1.csv` (source, script and output hashes)

Model-variable decision:

- `supplementary/model_variable_decision_v1.md`
- `supplementary/model_variable_decision_v1.csv` (18 decisions)
- `phase7_validation_v1.md` (32/32 checks passed; run with `src/data_acquisition/rdc_rocks/validate_phase7.py`)

Reproducible builders: `src/data_acquisition/rdc_rocks/build_phase7_supplement.py`, `audit_usgs_overlay.py`, `build_phase7_supplement_provenance.py` and `validate_phase7.py`.

## 4. The model-variable answer

One regional geological variable is sufficiently supported as a primary term: **distance to high/medium-confidence mapped Daly major structures** (127 of 248 segments, 6,518 km, 92% of mapped structure length; distance defined for all 16,492 cells; 71% within 50 km).

Secondary or sensitivity only: 50 km major-structure density, named-fault distance, named-fault density, bridged named-fault distance, USGS tract membership.

Not eligible for the basin-wide model: the 136 Phase 6 structural/stratigraphic features (unresolved datum and northing), bedrock lithology/stratigraphy (no admitted source), cover/exposure (not available), fault intersections (not observed), and all occurrence-derived distances or densities.

Daly domain membership is **not** estimable as a nine-domain effect. All 186 USGS deposits inside the Daly union fall in the NRB domain: 149 in subdomain 3a and 37 in 3b. The other seven domain polygons have zero occurrence support. Estimation must be confined to the NRB; the full Daly union remains the prediction frame with explicit extrapolation flags.

The exact overlay reconciliation is: 187 USGS inventory records; 186 inside the Daly union and 184 inside the permissive tracts; three outside the tracts (Kanaronga, Luishia and Nchanga Nose); one record outside the union but inside a tract (Sebembere). Source-country totals are 147 DRC and 40 Zambia; inside the union they are 147 DRC and 39 Zambia. Natural Earth assigns the two Zambian Mokambo records to its DRC polygon, so source country is retained for occurrence reporting. The four folds contain 46, 48, 46 and 46 deposits; fold 4 contains 7 DRC and all 39 Zambian records inside the union.

## 5. The observation-effort answer

There is no independent spatial surface for historical observation effort. The 21 exploration records have zero footprints; 13 are place-name only, 2 describe regional extents, 2 carry local-grid borehole coordinates with unresolved datum, and 5 are dated. They establish documentary presence and chronology, not coverage or intensity.

Only three observation-process controls are estimable:

- Source-document effect: 3 documents, with 110 of 147 RDC observations from RDC_3731.
- Spatial-precision restriction: 25 of 147 records carry numeric coordinates or bounding boxes.
- Entity-duplication sensitivity: 147 observations resolve to 113 candidate sites.

No detection/reporting-intensity surface, no temporal reporting trend and no campaign-footprint proximity can be estimated from the current corpus.

## 6. Next concrete operations

1. **Datum and northing recovery.** Search the original publications, map marginalia and survey reports for RDC_3854, RDC_4055 and RDC_3863. The current affine fits are not datum evidence. For RDC_4055, recover northing before assigning any projected CRS.
2. **Zambian geological coverage.** Acquire Thieme and Johnson (1981), Zambia Geological Survey 1:250,000 maps, memoirs and structural/stratigraphic descriptions for Konkola, Chililabombwe, Chingola, Nchanga, Kitwe, Mufulira and Nkana.
3. **Bedrock stratigraphy.** Acquire an authoritative cross-border bedrock/stratigraphic polygon source before creating host-unit predictors.
4. **Survey and concession footprints.** Seek boundaries for the 14,800 km2 airborne survey (1969-1970) and the 500 x 50 km Lufilian Arc concession. Do not buffer place names.
5. **Occurrence entity review.** Review the 71 candidate-match rows (51 site-level candidates, 20 ambiguous many-to-one). Keep ambiguous matches unresolved; do not merge source records or alter the USGS inventory.
6. **Model specification.** Fit only after the above evidence decisions are recorded. The current specification is in `supplementary/model_variable_decision_v1.md`.

## 7. Integrity

- Raw PDFs, TIFFs, OCR outputs, catalogues and manifests are unchanged.
- Phase 6 canonical copies match their originals (8/8 hash matches).
- Phase 7 validation: 32/32 checks passed (counts, referential integrity, hashes, empty geometries, support, folds, USGS overlay, decision-table schema and provenance).
- USGS baseline: 12 files, 0 hash mismatches at the recorded snapshot.
- `AI_PROJECT_CONTEXT.md` and `src/validation_strategies.py` match Git HEAD `d56afb1` with no working-tree diff. The earlier drift warning could not be reproduced.
- Caution: `frozen_baseline_hashes_v1.csv` is a recapture by `build_phase7_integration.py`, not a verification against an earlier baseline. Do not cite it as proof of frozen-file integrity.
- `model_variable_decision_v1.md` is a reviewed synthesis, not a generated output; its counts are checked against the generated support, fold and overlay tables.
- No Git commit, reset, clean, delete or bulk-add of untracked research directories was performed.
