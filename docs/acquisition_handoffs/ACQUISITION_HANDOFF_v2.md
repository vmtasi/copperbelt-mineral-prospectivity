# RDC.ROCKS Acquisition Handoff v2

Generated: 2026-11-08 07:51 Africa/Nairobi
Project: `C:\Users\vanmu\copperbelt-mineral-prospectivity`
Supersedes: `ACQUISITION_HANDOFF.md` (2026-11-07 19:10). That file is left untouched.

## 1. Cause of the HTTP 413

Isolated to thread `01a18c61-d178-77e2-90a5-dade4e45c09a`, project `copperbelt-mineral-prospectivity`.

- 50 turns, 2,782 command items, 210 image-view items, 34 context compactions.
- 250 inline `data:image` payloads: 260,912,449 base64 chars, about 249 MB.
- Command-output JSON about 20.6 MB; reasoning JSON about 4.4 MB.
- Rollout file 338.7 MB; recorded usage 432,683,588 tokens.
- Error: `unexpected status 413 Payload Too Large: Failed to buffer the request body`.

This is an HTTP request-body limit, not the model token/context limit: the body is
buffered before inference, so retrying the same history can never succeed. Rendered
pages and contact sheets dominated, followed by accumulated command output and reasoning.

## 2. What changed to reduce request size

- Continuation runs in a fresh thread from this report; the 338.7 MB rollout is not replayed.
- All PDFs, TIFFs, renders, OCR text, manifests stay on disk. Nothing is attached whole.
- Only short OCR excerpts, metadata rows or one small image go into a request.
- OCR runs one document at a time, 10 pages per batch, checkpointing after every page.
- No oversized request is retried.

## 3. Verified acquisition state (re-checked this session)

PDFs `data/raw/rdc_rocks/files/pdf/`
- 108 files, 3,359.8 MB, 4,021 pages.
- Manifest: 108/108 `acquired_verified`; 108 unique SHA-256, no duplicate content.

TIFFs `data/raw/rdc_rocks/files/tiff/`
- 10 files, 4,372.0 MB. All catalogue records complete.
- `RDC_1711.tif` finished 2026-11-07 19:10:53 at 706,183,088 bytes.
- Re-hashed 2026-11-08: SHA-256 `024ce2b8313527e6c90ea54ed9459f0ff023259e168475256a80160191b8cd8c`
  matches the manifest; TIFF magic `49-49-2A-00`.
- `0853` and `0866` share SHA-256 `9ca7b7467fe6...` and byte count (same content, two catalogue records).

Manifest `data/raw/rdc_rocks/metadata/acquisition_manifest_v1.csv`
- 118 rows, all `acquired_verified`, rebuilt 2026-11-07 19:19. No rebuild needed.

OCR `data/raw/rdc_rocks/derived/ocr/`
- Snapshot at 07:51: 32 documents, 356 pages, 414,275 characters, 0 errors.
- Live counts: `data/raw/rdc_rocks/derived/ocr/status_v1.json`.
- 20 documents at 10+ pages: 243, 417, 580, 1258, 1431, 1525, 1850, 1919, 2206, 2608,
  3061, 3399, 3411, 3461, 3731, 3952, 4041, 4042, 4343, 4374.
- 10 one-page sheets: 1043, 1300, 1317, 2376, 3129, 3848, 3854, 3856, 3863, 4055.
- `2755` pages 1-3.

Embedded text `data/raw/rdc_rocks/derived/text/`
- 7 PDFs, 833 pages, 1,117,787 characters: 100, 379, 957, 1177, 1231, 1388, 1408.

Georeferencing: not started. No GCPs or transforms yet. The 10 TIFF maps are held
out of the OCR queue (the OCR renderer uses `pdftoppm`).

## 4. Completed this session

Batch 1: all 8 stratigraphic reports and 4 structural studies, 10 pages each.
- 114 new pages, 113 ok, 1 empty (580 p6), 0 errors.
- IDs: 243, 580, 2206, 2608, 3061, 3411, 3952, 4343, 1525, 4041, 4042, 4374.

Batch 2: first 5 remaining occurrence inventories, 10 pages each.
- 41 new pages, all ok, 0 errors (1325 has a single page).
- IDs: 417, 1258, 1325, 1431, 1850.

## 5. Next concrete steps

1. OCR queue is now driven by `src/data_acquisition/rdc_rocks/run_ocr_queue.py` (see Architecture).
   Batches of 5 documents x 10 pages (pages already done are skipped automatically).

```powershell
cd C:\Users\vanmu\copperbelt-mineral-prospectivity
$ids = (Get-Content work\ocr_queue_v1.tsv | Select-Object -First 5 | ForEach-Object { ($_ -split "`t")[0] }) -join ','
C:\Users\vanmu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  src\data_acquisition\rdc_rocks\ocr_pages.py --ids $ids --max-docs 5 --max-pages-per-doc 10
```

2. Extend pages 11+ for the 12 priority documents where the text justifies it.
3. Rebuild `work/ocr_queue_v1.tsv` from manifest + OCR state after each run (same filter:
   `text_status != text_layer`, not in OCR state, `local_path` not `.tif`).
4. Build `data/raw/rdc_rocks/derived/maps/map_index_v1.csv` from OCR text and catalogue metadata.
5. Georeference priority maps (1043, 1317, 2376, 3863, 4055, 3854, 3856 plus TIFFs)
   and persist GCPs/transforms under `derived/georef/`.

## 6. Environment notes

- `.venv\Scripts\python.exe` is broken (points at a missing Store Python). Do not use it.
- Use `C:\Users\vanmu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`
  (3.12.14, Pillow 13.0.0). `pdftoppm` and the Windows OCR wrapper resolve correctly from `ocr_pages.py`.

## 7. Request-size rules for the continuation

- Never attach full PDFs, TIFFs, contact sheets or page collections.
- One document at a time, 10 pages per batch, state saved after every page.
- Reference local paths; do not paste complete manifests or command outputs.



## 8. Architecture

Codex orchestrates; Python processes documents locally. The model never receives a page,
a full OCR file or a full manifest.

- `src/data_acquisition/rdc_rocks/run_ocr_queue.py`: unattended queue driver, 5 documents
  x 10 pages per batch, checkpoint per page, stops cleanly and resumes where it left off.
- `data/raw/rdc_rocks/derived/ocr/status_v1.json`: small digest to read instead of logs.
- `work/ocr_runs.log`: per-page transcript, only opened when diagnosing a failure.
- Details and the model contract: `docs/acquisition_architecture.md`.

Started 2026-11-08 08:13 (Africa/Nairobi), 360-minute budget. First unattended batch
(`2509, 2519, 2527, 2747, 2847`) finished with 50 pages and 0 errors.

## 9. Progress since 07:51

- Breadth pass finished 08:59: all 101 image-only PDFs OCR'd at least once.
  671 pages, 720,101 characters, 0 errors (evaluated at that point).
- Deep pass started 08:59 (pid 15004): returns to the 44 documents with pages left,
  adding 10 pages per document per batch. 2,517 pages were outstanding.
- QC stage added: `src/data_acquisition/rdc_rocks/qc_ocr.py` writes
  `derived/ocr/qc_v1.json` with empty-page ratio, median text density and flags
  (`mostly_empty`, `low_text_density`, `first_pass_only`). Current run: 14 empty
  pages of 727, 0 page errors.

Live counts stay in `derived/ocr/status_v1.json`. Deep pass is bounded at 600 minutes
and checkpoints every page.
