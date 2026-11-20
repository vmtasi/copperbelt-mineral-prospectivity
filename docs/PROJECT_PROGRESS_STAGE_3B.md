# Project Progress - Stage 3B Located-Evidence Recovery

**Status date:** 19 November 2026  
**Repository:** `C:\Users\vanmu\copperbelt-mineral-prospectivity`  
**Branch:** `main`  
**HEAD at review:** `c5d53ce`

This is a separate Stage 3B progress record. It does not replace `docs/PROJECT_PROGRESS.md`, `docs/PROJECT_PROGRESS_RESEARCH_PHASE_V1.md` or any historical record.

## Status

Stage 3B is complete as an audit and protocol-preparation stage. No model was fitted. The recommendation is continued evidence recovery.

## Work completed

| Work package | Status | Output |
|---|---|---|
| Baseline and source inspection | Complete | Frozen inputs and prior research-phase reports re-checked |
| OCR-damaged occurrence recovery | Complete | `OCC-3399-007` and `OCC-3399-008` recovered from L5 to L2 |
| Station annex recovery | Complete, population corrected | 158 rows at 62 distinct coordinates, not 81 rows |
| `RDC_4374` collar CRS | Complete, unresolved datum | Zone 35S established; EPSG:32735 provisional |
| Geophysical georeferencing | Complete, negative | Neither map could be georeferenced; limitation documented |
| Geochemical sample localities | Complete | 90 samples and 2,199 determinations registered; zero sample coordinates |
| RDC gazetteer review | Complete | 15 exact, 10 token and 35 no-match names; zero identities |
| Drilling-register repair specification | Complete, not applied | Five row updates specified |
| Regional modality matrix | Complete | 35 rows, nine separate evidence fields |
| E1/E2 protocol | Complete, not executed | Deposit-only labels, group handling, data-driven blocks, stopping rules |
| E3/E4 protocol | Complete, conditional | A/B geological-context transfer only |
| Validation and decision memo | Complete | All ten gates reported |

## Key findings

1. **The 568 values remain one chemical dataset.** 71 intervals x 8 analytes in six assayed holes; drilling adds collar geometry, not a second chemical modality.
2. **The station population was wrong.** The old 81-row figure omitted one Annex VIII row and 76 Annex IX rows. The correct population is 158 rows at 62 distinct coordinate pairs.
3. **The station source was `RDC_4374`, not `RDC_1388`.** Annexes VIII-IX, PDF pp. 112-115.
4. **The collar CRS is zone 35S but not a confirmed EPSG.** Zones 34S and 36S are 650 km away. Zone 35S leaves a 3.3-4.1 km residual against the source corner centroid. The datum is unresolved.
5. **Neither priority geophysical map can be georeferenced.** `RDC_0205` and `RDC_0208` have no graticule, ticks, corner coordinates, projection or datum. No control points were invented.
6. **Ninety geochemical samples now have locality records, but zero have sample coordinates.** `RDC_0088` has conflicting locality labels; `RDC_3217` and `RDC_4190` have unkeyed survey coordinates.
7. **Two RDC records changed class.** L5 count falls from 2 to 0; L2 rises from 27 to 29. L1 remains 0.
8. **No occurrence identity was confirmed.** 147 observations, 113 provisional entities, 0 confirmed identities.
9. **A and B share mapped geology and occurrence labels only.** No shared located chemistry, geophysics or regional structure exists.
10. **The multimodal transfer programme remains unidentifiable.** E1/E2 are protocol-ready but should be treated as a mapping-plus-occurrence study, not as validation of multimodal transfer.

## Gate results

| Gate | Status |
|---|---|
| G1 - Evidence independence | Met |
| G2 - Collar CRS | Met as an explicit unresolved conclusion |
| G3 - Structural recovery | Met, with population correction |
| G4 - Geophysical recovery | Not met |
| G5 - Occurrence recovery | Partly met |
| G6 - Sample locality recovery | Partly met |
| G7 - Common labels | Met as protocol |
| G8 - Spatial support | Met as protocol |
| G9 - Transferability | Partly met |
| G10 - Pilot readiness | Met as protocol only |

## Decisions recorded

- No model fit or refit.
- No V14, Stage 3-5, Phase 6/7 or atlas modification.
- No original PDF or evidence-harvest input modification.
- No drilling-register modification.
- No occurrence identity created or merged.
- No study-area boundary adopted.
- No coordinate, bearing, control point or CRS invented.
- No website repository modification.
- No stage, commit, push, reset or clean.

## New outputs

Under `data/raw/rdc_rocks/derived/research_phase_v1/stage_3b/`:

- `README_stage_3b.md`
- `reports/` with ten reports and the validation report
- `tables/` with the recovery, gazetteer, station, CRS, sample-locality, repair and modality registers
- `tables/validation_manifest_stage_3b.json`
- `tools/` with deterministic builders and the validation script
- `images/` with the two RDC_3399 page crops

## Outstanding work

### P0 - before multimodal fitting

1. Obtain a datum/control-point statement for the `RDC_4374` collars.
2. Obtain control points or a georeferenced base for `RDC_0205` and `RDC_0208`.
3. Establish a sample-to-station key for `RDC_3217` and `RDC_4190`.
4. Resolve the `RDC_0088` Luiswishi/Luishia versus Mokambo locality conflict.
5. Complete an authoritative gazetteer review for the 60 L3 records.
6. Establish a map/grid origin for the 62 Shadirandzoro station coordinates.

### P1 - before any transfer claim

7. Determine whether the stratigraphic crosswalk contains any proven equivalence usable as a bridge.
8. Resolve Phase 6 local-grid and source-pixel coordinates.
9. Recover the Luiswishi borehole collars if they are needed for a second local radiometric site.

### P2 - separately authorised

10. E1/E2 mapping-plus-occurrence pilot.
11. E3/E4 conditional geological-context transfer.
12. No chemistry, geophysics or full multimodal transfer until a second located non-occurrence modality exists in at least two regions.

## Repository note

Pre-existing uncommitted work was preserved: `.gitignore`, `README.md`, `docs/PROJECT_PROGRESS.md`, `docs/PROJECT_PROGRESS_RESEARCH_PHASE_V1.md`, and the two alternative-geological-map scripts. This stage adds `docs/PROJECT_PROGRESS_STAGE_3B.md` and data outputs under the ignored `data/raw/` tree. Nothing was staged or committed.
