# Acquisition architecture

Codex orchestrates; Python does the bulk processing locally. A model request must
never carry a document page, a full OCR text file, or a full manifest.

## Layers

| Layer | Owns | Never sends |
|---|---|---|
| Local repository `data/raw/rdc_rocks/` | PDFs, TIFFs, catalogue metadata, manifests, derived text, geospatial data | - |
| Python pipeline `src/data_acquisition/rdc_rocks/` | download, manifest build, PDF audit, OCR, text-layer extraction, indexing, georeferencing, validation | - |
| Codex / DeepSeek | reading small digests and evidence excerpts, research decisions | pages, full documents, full manifests |

## Data flow

1. `download.py` fetches catalogue objects with resumable state and SHA-256 checks.
2. `build_manifest.py` joins probe, details and audit into `acquisition_manifest_v1.csv`.
3. `ocr_pages.py` renders one page and OCRs it locally (Windows OCR engine), writing
   `derived/ocr/text/RDC_XXXX_pNNN.txt` and checkpointing after every page.
4. `run_ocr_queue.py` drives the whole queue unattended in bounded batches and writes
   `derived/ocr/status_v1.json`, a digest small enough to read directly.
5. `extract_text_layers.py` pulls embedded text when a PDF already has one.
6. Later stages: map index, georeferencing, validation, all writing to `derived/`.

## Model contract

- Ask the pipeline for a digest (`status_v1.json`) or a targeted excerpt, never for a document.
- Excerpt reads are line-ranged and capped; anything larger stays on disk.
- Requests carry paths and counts, not payloads.

## Run

```powershell
cd C:\Users\vanmu\copperbelt-mineral-prospectivity
$py = 'C:\Users\vanmu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py src\data_acquisition\rdc_rocks\run_ocr_queue.py --batch-docs 5 --batch-pages 10 --max-minutes 360
```

Stop and resume at any time: completed pages are skipped, and `status_v1.json` records
what remains. `work/ocr_runs.log` holds the per-page transcript.
