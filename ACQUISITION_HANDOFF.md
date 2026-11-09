# RDC.ROCKS Acquisition Handoff

Generated: 2026-11-07 19:10 Africa/Nairobi
Project: `C:\Users\vanmu\copperbelt-mineral-prospectivity`
Scope: RDC.ROCKS catalogue and document acquisition, SHA-256 verification, local OCR, georeferencing preparation.

## 1. Why the DeepSeek request exceeded the body limit

The failures were isolated to thread `01a18c61-d178-77e2-90a5-dade4e45c09a` (project `copperbelt-mineral-prospectivity`).

- 50 turns, 2,782 command-execution items, 2,217 reasoning items, 210 image-view items, 34 context compactions.
- 250 inline `data:image` payloads: 260,912,449 base64 characters, about **249 MB**.
- Command-output JSON: about **20.6 MB**; reasoning JSON about **4.4 MB**.
- Rollout file: **338.7 MB**; recorded usage: **432,683,588 tokens**.
- Error: `unexpected status 413 Payload Too Large: Failed to buffer the request body: length limit exceeded`.

This is an **HTTP request-body limit**, not the model context limit. The provider buffers the body before inference, so retrying the same history cannot succeed. The dominant cause was accumulated page renders/contact sheets, followed by accumulated command output and reasoning. The old thread must not be resumed with the same history.

## 2. Current acquisition state

Catalogue and selection:

- `data/raw/rdc_rocks/metadata/catalog_all_documents.csv`: **4,647** catalogue records.
- Selected acquisition set: **118** documents (108 PDF, 10 TIFF); 55 reports, 33 cross-sections, 30 maps.
- 118 per-document detail JSON files and 14 catalogue query files are preserved under `data/raw/rdc_rocks/metadata/`.

PDFs:

- `data/raw/rdc_rocks/files/pdf/`: **108 files, 3,359.8 MB, 4,021 pages**.
- Manifest status: **108/108 `acquired_verified`**.
- SHA-256: 108/108 recorded; **108 unique hashes, no duplicate PDF content**.
- All PDFs have matching byte counts, page counts and PDF signatures per `pdf_content_audit_v1.csv`.

TIFFs:

- `data/raw/rdc_rocks/files/tiff/`: **10 complete TIFFs**.
- `RDC_1711.tif`: **706,183,088 bytes**, TIFF magic `49492a00`, SHA-256 matches the manifest. Verified 2026-11-07.
- Independently re-hashed: all 10 files match their recorded SHA-256, byte count and TIFF magic bytes.
- Complete: `0596`, `0647`, `0698`, `0827`, `0848`, `0853`, `0866`, `0888`, `0893`, `1711`.
- Duplicate content: `0853` and `0866` have the same SHA-256 (`9ca7b7467fe6...`) and byte count. Keep both catalogue records; do not treat them as two independent sources.
- `acquisition_manifest_v1.csv` rebuilt: **118 records, 118 `acquired_verified`** (108 PDF, 10 TIFF), 118 unique IDs, no failures.
- Integrity audit: `data/raw/rdc_rocks/derived/integrity/rdc_rocks_integrity_v1.csv`.
- Rebuild verification: `data/raw/rdc_rocks/derived/integrity/manifest_rebuild_verification_v1.json`.
- Pre-rebuild manifest backup: `data/raw/rdc_rocks/metadata/backup/acquisition_manifest_v1_pre_tiff_rebuild_20261107.csv`.

No acquisition file was re-downloaded or overwritten during this work. `1711` completed in place and was verified against its recorded byte count and SHA-256. New outputs are limited to `data/raw/rdc_rocks/derived/{ocr,text,register,georef,integrity}/` and the local scripts listed below.

## 3. OCR progress

- **15 priority documents, 25 pages, 18,406 characters**, 0 errors.
- 10 one-page structural/stratigraphic/regional sheets OCR'd completely:
  `1043`, `1300`, `1317`, `2376`, `3129`, `3848`, `3854`, `3856`, `3863`, `4055`.
- 5 occurrence inventories OCR'd pages 1-3 each:
  `1919`, `2755`, `3399`, `3731`, `3461`.
- Full-body OCR completed later for `3731` (56 pp), `3399` (46 pp), `1919` (39 pp) and `3461` (41 pp); `2755` skipped as the duplicate of `1919`.
- OCR text: `data/raw/rdc_rocks/derived/ocr/text/RDC_XXXX_pNNN.txt`.
- State: `data/raw/rdc_rocks/derived/ocr/state_v1.json`.
- Page manifest: `data/raw/rdc_rocks/derived/ocr/ocr_manifest_v1.jsonl`.
- Rendering cache: `data/raw/rdc_rocks/derived/ocr/pages/`.
- French OCR is provided by the installed Windows OCR engine; pages are auto-rotated when the first pass scores poorly.

Embedded text extraction (no OCR needed):

- **7 PDFs, 833 pages, 1,117,787 characters**, 0 errors.
- IDs: `100`, `379`, `957`, `1177`, `1231`, `1388`, `1408`.
- Text: `data/raw/rdc_rocks/derived/text/RDC_XXXX.txt`.
- State: `data/raw/rdc_rocks/derived/text/text_layer_state_v1.json`.
- Manifest: `data/raw/rdc_rocks/derived/text/text_layer_manifest_v1.jsonl`.

Georeferencing: **candidate inventory complete, no derivatives produced**. There are still no GCPs, transforms or georeferenced rasters. Priority maps for control-point work are `3854`, `3863`, `4055`, then `1043`/`1317` and `2376`.

Phase B/C deliverables (independent acquisition workstream):

- `data/raw/rdc_rocks/derived/register/geological_evidence_register_v1.csv`: 21 priority documents, 22 fields. Direct observations and author interpretations are separate columns.
- `data/raw/rdc_rocks/derived/register/document_extraction_report_v1.csv` and `.md`: per-document pages, characters, lexical quality, confidence.
- `data/raw/rdc_rocks/derived/georef/georeferencing_candidates_v1.csv`: 9 candidates with scale, grid evidence, control points and evidence coverage.
- `data/raw/rdc_rocks/derived/register/research_gaps_v1.md`: missing evidence and ordered next steps.
- `data/raw/rdc_rocks/derived/register/rows/RDC_XXXX.json`: per-document checkpoints.
- Builder: `src/data_acquisition/rdc_rocks/build_evidence_register.py` (idempotent, rewrites the CSVs from the checked-in text).

## 4. Scripts and inputs

- `src/data_acquisition/rdc_rocks/download.py`: resumable downloads, byte/magic/SHA-256 checks, JSONL state.
- `src/data_acquisition/rdc_rocks/build_manifest.py`: joins probe, details, PDF audit and download states.
- `src/data_acquisition/rdc_rocks/inspect_pdfs.py`: PDF audit, contact sheets, page sheets, duplicate review.
- `src/data_acquisition/rdc_rocks/ocr_pages.py`: local page OCR with bounded batches and per-page checkpoints.
- `src/data_acquisition/rdc_rocks/extract_text_layers.py`: extracts embedded PDF text with per-document checkpoints.
- `src/data_acquisition/rdc_rocks/ocr_one.ps1`: Windows OCR wrapper (French/English).
- `src/data_acquisition/rdc_rocks/build_evidence_register.py`: builds the evidence register, extraction report, georeferencing inventory and gap report from local text only.
- `data/raw/rdc_rocks/metadata/`: catalogue, probe, document details, download states, manifests.
- `data/raw/rdc_rocks/derived/ocr/`: OCR text, page cache, state and manifest.

## 5. Next concrete operations

Status note (2026-11-07, later): items 3-7 below are superseded by section 7. Items 1-2 remain complete.

1. ~~Finish `1711`, verify it and rebuild the manifest.~~ Done: 10/10 TIFFs verified, 118/118 `acquired_verified`.
2. ~~Build the geological evidence register, extraction report, georeferencing inventory and gap report.~~ Done (see section 3).

```powershell
cd C:\Users\vanmu\copperbelt-mineral-prospectivity
.\.venv\Scripts\python.exe src\data_acquisition\rdc_rocks\build_manifest.py `
  --probe data\raw\rdc_rocks\metadata\file_probe_v2_sanitized.csv `
  --details data\raw\rdc_rocks\metadata\priority_document_details_v2.csv `
  --pdf-audit data\raw\rdc_rocks\metadata\pdf_content_audit_v1.csv `
  --state data\raw\rdc_rocks\metadata\download_state_pdf_v1.jsonl `
  --state data\raw\rdc_rocks\metadata\download_state_tiff_v1.jsonl `
  --files-root data\raw\rdc_rocks\files `
  --output data\raw\rdc_rocks\metadata\acquisition_manifest_v1.csv
```

Note: the project `.venv` interpreter points to a missing Python 3.10. Run these scripts with the bundled runtime instead:
`C:\Users\vanmu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`.

3. Continue occurrence OCR, pages 4-10. Note `1919` and `2755` are duplicate records of the same thesis: OCR `1919` only.

```powershell
.\.venv\Scripts\python.exe src\data_acquisition\rdc_rocks\ocr_pages.py `
  --ids 1919,3399,3731,3461 --max-docs 4 --max-pages-per-doc 10
```

4. OCR the remaining 86 image-only PDFs in batches of 5 documents / 10 pages, checkpointing after every page. The 7 text-layer PDFs are already extracted under `derived/text/`; do not OCR them.
5. Read the sheet margins of `3854`, `3863` and `4055` to establish the metric grid datum (Congo Belge Lambert vs UTM 35S) before publishing any coordinates.
6. Georeference `3854` and `3863` first, then `4055`; persist GCPs/transforms under `data/raw/rdc_rocks/derived/georef/`. Do not derive spatial layers until GCP residuals are documented.
7. Visually confirm whether the `379` PDF contains the 1:100,000 Mitwaba map plate; if yes it is the best-scale georeferencing candidate.

## 6. Request-size rules for the continuation

- Start from this handoff and a fresh thread; do not replay the 338.7 MB rollout.
- Never attach full PDFs, TIFFs, contact sheets or collections of rendered pages to a model request.
- Send only specific OCR text excerpts, catalogue metadata or one small image when a decision truly needs it.
- Process one document at a time; 5-10 pages per batch; persist state after each page.
- Reference local paths instead of pasting complete manifests or command outputs.

## 7. Visual validation, occurrence recovery and exploration evidence (completed 2026-11-07)

### Visual source validation

- All nine georeferencing candidates were inspected on page images, not OCR alone.
- Register: `data/raw/rdc_rocks/derived/visual/visual_source_validation_v1.csv` (9 rows, 16 fields).
- Evidence images: `derived/visual/overviews/`, `derived/visual/crops/`, `derived/visual/thumbs379/`.
- RDC_3863 is a 180-degree-rotated scan; upright copy at `derived/georef/render/RDC_3863_p001_upright.png`.
- RDC_379 contains only the notice; the 1:100,000 Mitwaba plate is absent. RDC_3129 is a depth section, not a map.

### Georeferencing

- `derived/georef/gcp_control_points_v1.csv`: 16 control points (RDC_3863 E1-E6/N1-N5; RDC_3856 E1-E3/N1-N2).
- `derived/georef/transform_comparison_v1.csv`, `transform_params_v1.json`, `georeferencing_report_v1.md`.
- RDC_3863: RMSE 135.4 m, max residual 154.4 m, fitted scale within 1% of nominal; defensible in grid coordinates.
- RDC_3856: two-point northing calibration; defensible but weaker.
- RDC_3854, RDC_4055, RDC_2376, RDC_1043/1317: insufficient automatic control points; manual digitising required.
- Datum unresolved on every sheet. No WGS84 assignment was made and no geological layer was digitised.
- Exception: `derived/georef/derived_raster_extents_v1.csv` records the raster corners of 3863 and 3856 in map-grid metres (extent only; datum unresolved, no geology digitised).
- Candidate inventory verdicts updated in `derived/georef/georeferencing_candidates_v1.csv`.

### Occurrence inventory

- RDC_3731 (56 pages), RDC_3399 (46 pages) and RDC_1919 (39 pages) fully OCR'd from the existing local cache.
- `derived/occurrences/rdc_occurrences_v1.csv`: 147 records (110 from 3731, 20 from 3399, 17 from 1919).
- `derived/occurrences/occurrence_usgs_reconciliation_v1.csv`: 71 candidate name matches (68 high, 3 medium), 76 unmatched; no merge and no model positive.
- `derived/occurrences/occurrence_extraction_report_v1.md`.
- Coordinate availability: 14 degree-minute ranges (including 2 OCR-damaged strings in 3399 that 1919 resolves), 11 indices share one regional box, 122 records have descriptive or place-name locations only.
- Builders: `src/data_acquisition/rdc_rocks/build_occurrences.py` (idempotent; rewrites both CSVs).

### Historical exploration evidence

- `derived/exploration/exploration_evidence_v1.csv`: 21 rows.
- Key evidence: 1913-1945 uranium discoveries during copper prospecting; 1955-1958 URAKAT airborne radiometry; 1969-1970 Hunting airborne radiometry and magnetics over 14,800 km2; 1982-1985 COGEMA-CGEA-GECAMINES radiometry, magnetics, VLF, detailed mapping and 126 cored boreholes (RDC_1408).
- Boreholes Lsw-1215 and Lsw-1216 have collar coordinates on an unnamed local grid (RDC_1408 pp. 25, 31).
- RDC_0100 documents hundreds of trenches/pits, numerous boreholes, whole-rock geochemistry and about 100 thin sections in the CHEMAF concessions (modern work, not historical tenure).
- RDC_0379 documents compilation of the 1:100,000 map from Van den Branden (1953) and 1:100,000 topographic maps.
- RDC_3461 documents a thesis-era ground scintillometer survey (Scintrex G15-5): 150 measurements in RSF averaging 570 cps, RSC pockets at 700-900 cps, CMN about 300 cps, and seven RAT lilas assays (3.8-4.3 ppm U).
- No tenure file, annual report, assay database, raw survey or drill-log archive is present in the corpus.

### Updated gap report

- `derived/register/research_gaps_v1.md` rewritten as v2 with the above status and ordered next steps.
- Document-level extraction report updated for all four newly completed OCR documents (3731, 3399, 1919, 3461).

### Next operations

1. Screen the 86 remaining image-only PDFs in bounded batches; no priority occurrence inventory remains un-OCR'd.
2. Manually digitise control points for RDC_3854, RDC_4055 and RDC_2376/1043; record residuals.
3. Resolve the grid datum from sheet margins or external metadata before publishing any coordinate.
4. Acquire tenure, annual exploration reports, drill logs and assay databases.
