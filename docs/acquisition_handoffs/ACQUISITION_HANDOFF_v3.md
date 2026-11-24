# RDC.ROCKS Acquisition Handoff v3

Generated: 2026-11-08 10:37 Africa/Nairobi
Project: `C:\Users\vanmu\copperbelt-mineral-prospectivity`
Supersedes: `ACQUISITION_HANDOFF_v2.md` (left untouched).

## 1. Cause of the HTTP 413

The oversized request was an HTTP request-body limit, not the model token/context limit.
The failing thread had accumulated 50 turns, 2,782 command items and 210 image-view items,
including 250 inline `data:image` payloads totalling about 249 MB and about 20.6 MB of
command-output JSON. The body was buffered before inference, so retrying the same history
could never succeed.

## 2. Request-size rules

- Do not attach full PDFs, TIFFs, contact sheets or page collections.
- Python performs OCR, extraction, georeferencing, indexing and validation locally.
- Send only bounded text excerpts, metadata rows or one small image per decision.
- OCR runs one document at a time, ten pages per batch, checkpointing after every page.
- Do not retry a request that exceeds the provider payload limit.

## 3. Acquisition state

PDFs `data/raw/rdc_rocks/files/pdf/`
- 108 files, 4,021 pages. Manifest records 108/108 `acquired_verified` and 108 unique SHA-256 values.

TIFFs `data/raw/rdc_rocks/files/tiff/`
- 10 files. Catalogue records complete. `RDC_1711.tif` SHA-256 was verified previously.
- `0853` and `0866` share content and hash; they are two catalogue records, not two independent maps.

Manifest `data/raw/rdc_rocks/metadata/acquisition_manifest_v1.csv`
- 118 rows, all `acquired_verified`. Not rebuilt this session.

OCR `data/raw/rdc_rocks/derived/ocr/`
- Breadth pass complete: do not repeat it.
- Deep-pass worker PID 15004 was alive at 10:37 local and was not restarted or modified.
- Live state at 10:38 local: 101 documents, 1,764 pages, 1,958,919 characters.
- The queue refreshed `status_v1.json` at 07:35 UTC: 101 documents, 1,761 pages,
  1,958,381 characters, 0 page errors, 25 documents / 1,427 pages outstanding.
- Run the final QC only after PID 15004 exits.

## 4. Completed Phase 6 outputs

All three packages were loaded in QGIS; every layer returned zero null, empty or invalid geometries.
All use `srs_id=0` and an empty CRS. No datum or projection is asserted.

### RDC_3863 structural

- Package: `data/raw/rdc_rocks/derived/georef/phase6/3863/rdc_3863_structural_v1.gpkg`
- 70 features / 6 layers: faults 18, thrust sheets 7, fold axes 32, contacts 3, extrusions 4, excrescences 6.
- Transform RMSE 135.4 m, maximum 154.4 m.
- Feature table `rdc_3863_features_v1.csv`; controls `rdc_3863_control_points_v1.csv`;
  report `rdc_3863_digitising_report_v1.md`; overlay `work/3863_qa_overlay.png`.

### RDC_3854 stratigraphic

- Package: `data/raw/rdc_rocks/derived/georef/phase6/3854/rdc_3854_stratigraphy_v1.gpkg`
- 47 features / 6 layers: R.4 facies 10, absent zones 2, facies boundaries 4, isopachs 23, thrust faults 5, discordances 3.
- RMSE 83 m E / 131 m N; leave-one-out 143 m / 214 m. The 260 km interior label is excluded (1,054 m residual).
- Controls `rdc_3854_control_points_v1.csv`; features `rdc_3854_features_v1.csv`;
  report `rdc_3854_transformation_report_v1.md`; overlay `work/3854_qa_overlay.png`.

### RDC_4055 structural

- Package: `data/raw/rdc_rocks/derived/georef/phase6/4055/rdc_4055_structural_v1.gpkg`
- 19 features / 2 layers: carapaces 7, thrust-sheet windows 12.
- **Ten** numbered sheets are printed (8-17), not seventeen. The old count misread the last number.
- Signal Kapiri (11) has three separate windows, retained as three features with `sheet_no=11`.
- Easting-only fit from 7 bottom-border labels: 31.547774 px/km, RMSE 2.77 px (87.9 m), maximum 5.85 px (185.4 m).
- No northing labels exist on any margin. The package stays in source-pixel space; easting is carried as an attribute.
- Controls `rdc_4055_control_points_v1.csv`; features `rdc_4055_features_v1.csv`;
  report `rdc_4055_transformation_report_v1.md`; overlay `work/4055_qa_overlay.png`.

### Occurrence entities

`data/raw/rdc_rocks/derived/occurrences/entities/`
- `site_entities_v1.csv`: 113 entities.
- `occurrence_observations_v1.csv`: 147 observations, none dropped.
- `occurrence_entity_crosswalk_v1.csv`: 147 links; 76 `no_candidate`, 51 `site_level_candidate_unverified`, 20 `ambiguous_many_to_one`.
- No merges, no USGS edits, no model positives.

### Exploration history

`data/raw/rdc_rocks/derived/exploration/`
- `exploration_history_v1.csv`: 21 rows, seven evidence classes.
- Only 5 rows have a stated year. No polygons or negative observations invented.

### Validation

- `data/raw/rdc_rocks/derived/phase6_validation_v1.md`
- `data/raw/rdc_rocks/derived/georef/phase6/phase6_status_v1.md`

## 5. Integrity discrepancy

- `src/validation_strategies.py` is tracked and clean: working-tree blob equals the index
  (`b281dd37f615bd87d0fb217437eb10ef303723c9`).
- `AI_PROJECT_CONTEXT.md` is untracked and ignored at `.gitignore:68`.
- The baseline manifest behind the original drift warning was not found. Neither file was
  recaptured or overwritten. The discrepancy remains open.

## 6. Remaining tasks

1. Let PID 15004 finish; refresh OCR status and run QC. Do not duplicate completed pages.
2. Re-check RDC_3863 R.4/R.3 contacts against the original sheet; they are the weakest traces in that package.
3. If a north-margin label series for RDC_4055 is ever found, extend the transform; otherwise keep pixel space.
4. Review the 51 site-level candidates and 20 many-to-one rows manually. Keep ambiguous cases unresolved.
5. Recover missing campaign years only from source documents; do not infer them.
6. Assess scientific suitability before connecting any Phase 6 output to V14 or Stages 3-5.

## 7. Reproducible builders

- `src/data_acquisition/rdc_rocks/build_phase6_3863.py`
- `src/data_acquisition/rdc_rocks/build_phase6_3854.py` + `3854_features_v1.json`
- `src/data_acquisition/rdc_rocks/build_phase6_4055.py`
- `src/data_acquisition/rdc_rocks/build_phase6_registers.py`
- `src/data_acquisition/rdc_rocks/build_occurrence_entities.py`
- `src/data_acquisition/rdc_rocks/build_exploration_history.py`
- `src/data_acquisition/rdc_rocks/run_ocr_queue.py`, `qc_ocr.py`

## 8. Environment

- Use `C:\Users\vanmu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`.
- Do not use the broken `.venv`.
- GeoPackages are written without GDAL by `gpkg_writer.py`; QGIS is used read-only for validation.
- Do not commit, reset, clean or delete the untracked research directories.
