# Paper 1 — V14 benchmark bundle

This directory makes the Paper 1 (V14 centroid-contact DVHM) benchmark
reproducible from a fresh clone. It records which artefacts are tracked,
which are archived externally, and how to verify them.

## What is tracked in Git

`SHA256SUMS` lists 61 files (about 19.7 MB). Together they are every
numerical input and output that the manuscript reports:

| Group | Paths |
|---|---|
| Modelling table (1,899 cells; 1,872 complete cases) | `data/copperbelt_training_v14_centroid_contact.csv` |
| Contact-distance feature and manifest | `data/derived/litho_contact_centroid_distance_v14.*`, `data/derived/canonical_training_manifest_v14_centroid_contact.json` (already tracked) |
| DVHM fold traces, OOF predictions, fold metrics, run manifest | `figures/v14_centroid_contact/` |
| OOF posterior uncertainty (Figure 4) | `figures/v14_centroid_oof_posterior_uncertainty.{csv,gpkg}` |
| Matched-global M7, ablation, spatial-validation and Phase 4–7 summaries | `figures/audit/*_v14_centroid.{csv,nc}` |
| Coefficient (beta) diagnostics | `figures/audit/beta_diagnostics_v14_centroid/*.csv` |
| Distance-representation sensitivity summaries | `figures/audit/distance_representation_sensitivity_v14_centroid/*.csv` |
| Population reconstruction (D*, predictor contributions) | `figures/audit/v14_centroid_population_reconstruction/*.csv` |
| Post-refit output validation | `figures/audit/v14_model_output_validation_post_refit_20261021.json` |

Diagnostic PNGs are not tracked; the manuscript figures are tracked
under `paper_1/manuscript/images/`.

`.gitattributes` disables line-ending conversion for these paths, so the
recorded SHA-256 values hold on Windows, macOS and Linux checkouts.

## What is archived externally

`SHA256SUMS_external` lists the 20 per-fold traces from the
distance-representation sensitivity analysis (Models A-tail, B, B-tail, C
and D; about 35 MB). They are too large for this repository. The summary
CSVs derived from them are tracked.

**Action before submission:** deposit these traces, together with a
snapshot of this repository, in a DOI-issuing archive such as Zenodo.
Then record the DOI here and in the manuscript's data-availability
section. A local copy of the full preserved bundle is in
`archive/v14_benchmark_preservation_2026-11-05/`, which Git ignores.

## Verify

```bash
python -m pytest -q tests/test_v14_benchmark_integrity.py
# or, without Python:
sha256sum -c paper_1/benchmark/SHA256SUMS
```

The external traces are checked automatically once they are restored to
their paths. Until then their tests are skipped.

## Provenance

- Model: frozen V11 specification (`src/v11_spatial_stability_final.py`),
  run through the I/O-only wrapper `src/v14_centroid_spatial_stability.py`.
  It used 4 folds, 2 chains × 1,500 draws, 2,500 tuning steps,
  target_accept 0.99 and seed 42.
- Every tracked hash was checked against
  `docs/audits/v14_preservation_report.md` (5 Nov 2026) on 20 Nov 2026,
  and all matched.
- These artefacts are frozen. Any refit or new analysis must write new,
  versioned outputs and must not overwrite the files listed here.

## Third-party data

The modelling table contains per-cell values derived from the USGS Central
African Copperbelt assessment (occurrences, permissive tracts), the
WGM2012 complete Bouguer anomaly (BGI) and the Africa surficial lithology
raster (USGS). The source products are not redistributed; cite them as in
the manuscript.
