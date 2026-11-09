# Acquisition Handoff v5

Generated: 2026-11-09 (Africa/Nairobi)
Repository: `C:\Users\vanmu\copperbelt-mineral-prospectivity`
Git: `main` at `d56afb1`, aligned with `origin/main`; no commit, reset, clean or bulk add performed.

This handoff supersedes v4 for acquisition status only. `ACQUISITION_HANDOFF_v4.md` and all earlier handoffs remain intact. No Phase 6 or Phase 7 output, frozen V14 file, manuscript file or original source document was modified.

## 1. Verified acquisitions

All files below are local. No PDF, TIFF, contact sheet or rendered page collection was attached to a model request.

| Batch | Verified | Bytes | Pages | Local root |
|---|---:|---:|---:|---|
| Zambia National ETD Portal | 12 | 97,367,065 | 1,720 | `data/raw/zambia_acquisition/netd/files/` |
| RDC.ROCKS priority records | 12 | 887,110,879 | 925 | `data/raw/zambia_acquisition/rdc/files/` |
| **Total** | **24** | **984,477,944** | **2,645** | |

Manifest: `data/raw/zambia_acquisition/metadata/download_manifest_v1.csv`
Batch summary: `data/raw/zambia_acquisition/metadata/acquisition_batch_summary_v1.json`

### 1.1 Independent 1:250,000 map index

- Source: Zambia 1:250,000 geology FeatureServer, layer 0.
- URL: `https://services1.arcgis.com/AhXvNWFdL7hH4TjJ/ArcGIS/rest/services/Zambia_250k_geology/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=geojson`
- 62 sheet footprints, WGS 84 (EPSG:4326), bounds `21.000122, -19.0, 34.500122, -8.0`.
- Only 5 sheets are marked as owned in the source index.
- Outputs:
  - `data/raw/zambia_acquisition/metadata/reference/zambia_250k_geology.geojson`
  - `data/raw/zambia_acquisition/metadata/reference/zambia_250k_map_sheets_v1.csv`
  - `data/raw/zambia_acquisition/metadata/reference/zambia_250k_map_sheets_v1.gpkg`
  - `data/raw/zambia_acquisition/metadata/reference/zambia_250k_map_sheets_v1.json`
- The index records map coverage only. It is not geological evidence and does not contain map content.

### 1.2 GSD national map previews

Four 1:1,000,000 quarter-sheet previews were saved as `preview_only` evidence, not as full geological source data.

- `data/raw/zambia_acquisition/gsd/previews/`
- Manifest: `data/raw/zambia_acquisition/metadata/gsd_preview_manifest_v1.csv`
- NW 2000x1427, NE 2000x1401, SE 2000x1448, SW 2000x1422.

### 1.3 Zambian theses

Twelve public PDFs were downloaded from the National ETD Portal / UNZA repository and verified by PDF signature, byte count, SHA-256 and page count.

Failed with repository-side HTTP 500 on the only available bitstream; do not loop these URLs:

- `unza-123456789-1689` - Geophysical exploration for groundwater in limestone aquifers.
- `unza-123456789-6852` - Techno-economic potential for mining Zambia's low grade copper ore deposits: Kakosa South.

The portal's `/pdf` fallback and the RIMS handle page were checked. The fallback also returns HTTP 500; the handle page resolves but exposes the same failing bitstream.

### 1.4 RDC.ROCKS priority records

Twelve priority-1 records were downloaded and verified. Key identities confirmed from OCR title pages:

- `RDC-95` - Kipata Mwabanwa, PhD, KU Leuven, January 2013. Brittle tectonics in the Lufilian fold-and-thrust belt and its foreland (Katanga, DRC). 184 pp.
- `RDC-2995` - Mukonga Mikombe, Universite de Lubumbashi, 2004-2005. North flank of the Konkola Dome, Musoshi sector. 97 pp.
- `RDC-2812` - Kabamba wa Ndemba, Universite de Lubumbashi, 2004-2005. Geochemistry of detrital sediments, north flank of the Konkola Dome, Musoshi west. 62 pp.
- `RDC-4174` - Master, Robb and Armstrong, Journal of African Earth Sciences 42 (2005) 32-40. Geochronology of the Nchanga Granite and maximum age of the Katanga Supergroup. 55 pp.
- `RDC-2073` - Kazembe Kasongo, Universite de Lubumbashi, 1999-2000. Metalliferous deposits of the Katangan Copperbelt and geophysical prospecting methods. 90 pp.
- `RDC-4467` - Kasongo Kyayo Benjamin, Universite de Lubumbashi, 2024-2025. Microtectonics and metallogeny of the Mashi Cu-Co deposit, extension to Kinsevere. 129 pp.
- `RDC-732` and `RDC-1272` - Mulegalega, Universite de Lubumbashi, July 1991. Magmatism and mineralisation in the Zambian Copperbelt: geochemical constraints. 99 pp and 96 pp.
- `RDC-4558` and `RDC-4740` - Provisional Geological Map of Northern Rhodesia, compiled in the Geological Survey Office, Lusaka, 1956, after Guernsey (1951) and Bancroft (1952), with mapping to 1956; 1 inch = 2,000,000; 1 sheet each.
- `RDC-322` - Marcel Gysin, 1936. Geologie du Shaba meridional; legend includes Kundelungu, Mwashya/Roan, Muva, granite and copper deposits. 1 sheet.
- `RDC-61` - Geochemical exploration for copper-cobalt in the DRC, PR851 / Kibamba area. 110 pp; embedded digital text.

Duplicate review: `data/raw/zambia_acquisition/metadata/duplicate_review_v1.csv`

- `RDC-4558` / `RDC-4740`: same 1956 map edition, different scans and hashes. Both retained; count once as an intellectual source.
- `RDC-732` / `RDC-1272`: same July 1991 memoir, different scans and hashes. Both retained; count once as an intellectual source.
- `RDC-918`, `RDC-2271`, `RDC-3221` share the Frontier mine title and are not yet downloaded.

## 2. Local processing

Text extraction is in `data/raw/zambia_acquisition/derived/text/` with per-page files and checkpointing.

- 24 documents, 2,645 pages, 1,938,585 characters.
- 9 documents with usable embedded text; 15 image-only or sparse.
- Summary: `data/raw/zambia_acquisition/metadata/text_extraction_summary_v1.csv`

Selected local OCR is in `data/raw/zambia_acquisition/derived/ocr/` with per-page PNG and text checkpoints.

- 55 pages across 11 documents; 0 OCR errors.
- All three one-page maps were OCR'd at 300 dpi.
- First five pages were OCR'd for the six image-only reports; the Nchanga Granite article was OCR'd through page 9; the 1991 Zambian magmatism memoir was OCR'd through page 9.

Evidence index:

- `data/raw/zambia_acquisition/derived/evidence_index/page_hits_v1.csv`
- 332 page-level hits across 24 documents, with score, matched terms and a short snippet.
- Highest-ranked pages are from `RDC-61` (Kibamba / PR851) and the Nchanga underground mine thesis. These are candidate review pages, not a confirmed occurrence inventory.

## 3. Access restrictions and negative results

- GSD full files remain `HTTP 401 Unauthorized`; metadata and 2000 px previews are public. Do not bypass the account/purchase restriction.
- Thieme and Johnson (1981), Geological Map of Zambia: bibliographic identity is confirmed (OCLC 1056272404, 13797325, 1124366736; Stanford SearchWorks 513178; AfricaMuseum BE-RMCA-EARTHS-005233). No lawful full scan was found in the public GSD store or Internet Archive. The GSD `Thieme` search returned only an unrelated Musonda Falls map.
- The AfricaMuseum GeoNetwork hostname did not resolve from this environment.
- The Geological Society of Zambia resource links resolve back to the same page and are not usable downloads.
- `www.mmmd.gov.zm` states maps and reports are purchased from the Geological Survey Department.

Do not substitute the four 1:1,000,000 previews or the 1956 1:2,000,000 map for the 1981 Thieme-Johnson 1:250,000 series without recording the edition and scale difference.

## 4. Integrity

- Original acquired PDFs and TIFFs remain on disk and unmodified.
- 24 new PDFs were verified by signature, size and SHA-256; 2 failed repository downloads have no local partial file.
- Phase 7 validation remains 32/32; no model was fitted.
- Manuscript, V14, Stages 3-5, frozen geology foundation and the existing USGS occurrence table were not modified.
- Major research directories remain untracked; no destructive Git operation was performed.
- Request-size safeguard: all new source documents were streamed to disk, processed locally, and only small OCR excerpts and aggregate counts were read into the model context.

## 5. Next concrete operations

1. Build a reviewable Zambian occurrence-candidate table from `page_hits_v1.csv`, starting with the Nchanga, Uchi, Kakosa and Kansanshi/Lumwana mentions; keep every row as a source observation until independently verified.
2. OCR targeted pages only: occurrence tables, mine descriptions, maps and legends. Persist after every page batch.
3. Download the remaining RDC priority-1 records `RDC-918`, `RDC-2271`, `RDC-3221` and `RDC-501`, then hash-check the Frontier duplicate group.
4. Acquire lawful full copies of the GSD 1:250,000 sheets through an account or purchase; record the transaction/access path. Do not scrape around the 401.
5. Update the datum and coordinate register for RDC_3854, RDC_3863 and RDC_4055 from original map marginalia and companion publications. Do not assign WGS 84 or a projected CRS without documentary evidence.
6. Produce the Zambian/DRC coverage and gap assessment using the 62-sheet index, the acquired map extents and the Daly domain envelope. Distinguish catalogued, downloaded, verified and extracted evidence.
7. Build the independent mineral-system corridor only from admitted stratigraphic, structural and map evidence. Do not use occurrence buffers, USGS tract buffers or the predictors being evaluated.

## 6. Reproducible commands

```powershell
$py = 'C:\Users\vanmu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py src\data_acquisition\zambia\acquire.py index
& $py src\data_acquisition\zambia\acquire.py previews
& $py src\data_acquisition\zambia\acquire.py netd --max-priority 2
& $py src\data_acquisition\zambia\acquire.py rdc --max-priority 1 --ids 'RDC-95,RDC-2995,RDC-2812,RDC-61,RDC-4174,RDC-322,RDC-2073,RDC-4467,RDC-4558,RDC-4740,RDC-732,RDC-1272'
& $py src\data_acquisition\zambia\extract_text.py --platform NETD
& $py src\data_acquisition\zambia\extract_text.py --platform RDC
& $py src\data_acquisition\zambia\ocr_selected.py --ids 'RDC-95,RDC-2995,RDC-2812,RDC-4174,RDC-2073,RDC-4467' --first-pages 5
python src\data_acquisition\zambia\index_evidence.py
```
