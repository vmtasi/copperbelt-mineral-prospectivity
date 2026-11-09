# Acquisition Handoff v6

Generated: 2026-11-09 (Africa/Nairobi)
Repository: `C:\Users\vanmu\copperbelt-mineral-prospectivity`
Git: `main` at `d56afb1`, aligned with `origin/main`; no commit, reset, clean or
bulk add performed.

This handoff supersedes v5 for the historical-source work only.
`ACQUISITION_HANDOFF_v5.md` and all earlier handoffs remain intact. No Phase 6 or
Phase 7 output, frozen V14 file, manuscript file, USGS occurrence table or
original source document was modified.

## 1. New verified inputs

### Economic Geology 100th Anniversary Volume (2005)

- Local file: `literature/Economic geology * hundredth anniversary volume *.pdf`
- 1,158 pages; 134,671,897 bytes
- SHA-256: `483310b91b855c4feea8a71ad81f735465eed3935d3c7ee5b68c875d7ce22576`
- Matches `data/derived/eg100_2005/chapter_extraction_manifest_v1.json`.

### Daly et al. (2025)

- Local file (glob): `literature/Tectonics - 2025 - Daly*.pdf`
- 45 pages; 11,662,562 bytes
- SHA-256: `43d9c94152987e56ce0a68cb2bf7988f5cf700e8803a3f10092927041c3a569a`
- DOI: `10.1029/2025TC008955`

No PDF, TIFF, contact sheet or rendered page collection was attached to a model
request. Extraction and verification were local.

## 2. New outputs

| Output | Contents | Count |
|---|---|---:|
| `data/derived/eg100_2005/stratigraphic_correlations_v1.csv` | Documented stratigraphic units, positions, thickness/age, correlations and lineage | 20 |
| `data/derived/eg100_2005/structural_interpretations_v1.csv` | Named structures, interpretations, confidence and provenance | 18 |
| `data/derived/eg100_2005/geological_map_reproduction_register_v1.csv` | Original target, derivative maps, Eg 100 figures, RDC scans and GSD catalogue/preview records | 15 |
| `data/derived/eg100_2005/historical_source_registers_manifest_v1.json` | Source hash, row counts and output hashes | 1 |
| `data/derived/eg100_2005/thieme_johnson_source_trace_v1.md` | Bibliographic trace and evidence hierarchy | 1 |
| `data/derived/eg100_2005/economic_geology_2005_review_v1.md` | Chapter review, figure lineage, stratigraphy, structure and geochronology | 1 |
| `data/derived/eg100_2005/historical_source_gaps_and_acquisition_priorities_v1.md` | Prioritised acquisition queue | 1 |
| `data/derived/eg100_2005/historical_sources_validation_v1.md` | Concrete validation counts and limitations | 1 |

Output SHA-256 values:

| File | SHA-256 |
|---|---|
| `stratigraphic_correlations_v1.csv` | `f58d526da38f9e3b25a5558c42dda5bbfb509656ad6bb5d4c920633e76ef1f8b` |
| `structural_interpretations_v1.csv` | `d86fe552bbdcd6b848ec7cdb3b354706cf976dd286789c0b8c0f9338f73042dc` |
| `geological_map_reproduction_register_v1.csv` | `bd63f47e7d76a904c7ce1b3285ab6c85d30d9f753c531eacc44dfe06c9c36e78` |

Builder: `src/data_acquisition/zambia/build_historical_source_registers.py`.
Validator: `src/data_acquisition/zambia/validate_historical_sources.py`.
The builder is deterministic, checks the source hash, checks the chapter negative
result, checks for unique IDs, checks each local path and writes the manifest.

Supporting extraction artifacts (pre-existing, preserved):
`data/derived/eg100_2005/chapter_text_v1.txt`,
`data/derived/eg100_2005/chapter_pages/`,
`data/derived/eg100_2005/chapter_pages_layout/`,
`data/derived/eg100_2005/captions_v1.csv` and
`data/derived/eg100_2005/chapter_extraction_manifest_v1.json`.
The captions CSV is a heuristic extraction and is not authoritative.

## 3. Key findings

### 3.1 Scale correction

The Thieme & Johnson (1981) national map is **1:1,000,000**, not 1:250,000.
`ACQUISITION_HANDOFF_v5.md` used the wrong scale in its map-acquisition wording.
v5 is preserved unchanged; this handoff and the source trace record the correction.

The later Zambian 1:250,000 series is separate. Its local index contains 62 sheet
footprints, of which only 5 are marked as owned in the source index. The index is
coverage metadata, not map content.

### 3.2 The 1981 map remains a bibliographic target

- It is cited by Daly et al. (2025) Figure 3 (PDF p. 5) as the source of a modified
  map, and by Figures 12-13 (PDF pp. 24-25) as the GSD map used with 2016-2021
  field traverses.
- It is not cited or reproduced in the Economic Geology 2005 chapter.
- A full-volume scan of the 1,158-page anniversary volume found 2 `Thieme` hits,
  both `Thiemens` (a different author), and 0 `Drysdall` hits.
- No lawful full scan was located. GSD full files require an account or purchase;
  a full-file request returned HTTP 401. Do not bypass that control.

### 3.3 Daly figures are derivatives, not independent evidence

Daly Figures 3, 12 and 13 are later interpretations built on the 1981 GSD map and
new fieldwork. Daly's Figure 14 derives **six structural domains and a granitic
batholith**. The project's reconstructed framework has **nine domains**. Do not
conflate the two, and do not treat domain boundaries as faults or stratigraphic
contacts.

### 3.4 Eg 100 appendices are absent

The chapter cites Appendix Table A1, Table A2 and Figures A1, A2 and A6. They are
not in the local scan: the PDF has no attachments or outline, a sampled check of
PDF pages 1000-1040 found no appendix pages, and the supplement was not located.
Do not infer their content.

## 4. Intactness

- Phase 6 and Phase 7 outputs remain intact.
- Frozen V14, Stages 3-5, the manuscript, the frozen geology foundation and the
  USGS occurrence table were not modified.
- Original PDFs and TIFFs were not overwritten.
- The new work is confined to `data/derived/eg100_2005/`, one builder in
  `src/data_acquisition/zambia/`, and this handoff.
- Existing untracked research directories were not committed, reset, cleaned,
  deleted or reorganised.

## 5. Remaining acquisition tasks

1. Acquire the original Thieme & Johnson (1981) map and its explanation/memoir.
2. Acquire GSD 1:250,000 Copperbelt sheets and memoirs, starting with
   Konkola/Chililabombwe, Chingola/Nchanga, Kitwe/Nkana/Mufulira and
   Luanshya/Chambishi.
3. Acquire GSD Report No. 60 (Kitwe-Mufulira) and Report No. 66 (Chingola) with
   their 1:100,000 maps.
4. Acquire Drysdall et al. (1974), Thieme (1969) Report 26 and Thieme (1971)
   Report 32.
5. Locate the Eg 100 supplementary appendices or CD-ROM.
6. Build the documented-versus-inferred datum/grid register for RDC_3854,
   RDC_3863 and RDC_4055.
7. Extract Zambian occurrence records from the 12 acquired theses and other
   independent sources, preserving source page and observation status.

## 6. Exact next operation

Completed successfully: built the three registers (20 stratigraphic, 18
structural and 15 map records) and wrote the source trace, chapter review and
gap register.

Next: request the Thieme & Johnson (1981) map and explanation through the GSD
account/purchase route, or through a library/archive holding if GSD cannot supply
it. If that is blocked, request GSD Report No. 66 and the Chingola 1:100,000
sheet.

## 7. Validation

See `data/derived/eg100_2005/historical_sources_validation_v1.md`. The Phase 7
validator remains an independent check; these historical-source files are outside
the Phase 7 output directories.
