# Copperbelt Mineral Prospectivity: Geographic Transfer

Code and outputs for Paper 1, *Rankings Transfer but Base Rates Do Not:
Leave-One-Sector-Out Evaluation of a Prospectivity Model along the Central
African Copperbelt* (manuscript in `paper_1/manuscript/`).

## The question

A prospectivity model is trained where occurrences are known and used to rank
ground where they are not. This study measures how well a simple Bayesian
prospectivity model transfers between four contiguous sectors of the Central
African Copperbelt, and why transfer fails where it does.

## Data and model

- **Frame:** 1,872 nominal 5 km cells clipped to the USGS permissive tracts.
  138 cells contain at least one compiled USGS occurrence record.
- **Predictors:** distance to mapped faults, distance to lithological contacts
  and Bouguer gravity.
- **Model:** complementary log-log link with the clipped cell area as an
  offset; fitted by MCMC (PyMC).
- **Validation:** four along-belt sectors. Each sector is predicted by a model
  trained on the other three. Sector-to-sector transfer, random
  cross-validation and two alternative partitions serve as comparisons.
- **Pre-specification:** the analysis plan was written and hashed before any
  fit. It is in `paper_1/transfer/specification.json`, with a versioned
  amendment in `specification_v2.json`.

## Main results

- **Ranking transfers moderately.** Pooled held-out ROC-AUC is 0.721
  (0.687–0.755), and sector values are 0.727–0.834.
- **Probabilities do not transfer.** The south-eastern (Zambian) sector is
  over-predicted and the north-western sector under-predicted, under every
  model variant and partition.
- **Most of the loss in the worst sector is a base-rate shift.** An oracle
  intercept correction recovers +0.098 log-score units per cell there.
- **Predictors differ in how well they transfer.** Contact distance transfers,
  Bouguer gravity is informative only in the south-east, and the
  fault-distance association depends on uneven map coverage.
- **Random cross-validation overstates transfer** by 0.031 log-score units per
  cell and 0.06 in ROC-AUC.

The full report is `paper_1/transfer/reports/transfer_report.md`.

## Repository layout

**Paper 1**

| Path | Contents |
|---|---|
| `src/transfer/` | Transfer analysis: partitions, model, MCMC fitting, analyses, report, figures, supplement tables |
| `paper_1/transfer/` | Specification, predictions, coefficients, metrics, diagnostics, figures, report, checksums |
| `paper_1/manuscript/` | Manuscript (`main.tex`, Springer Nature template), supplement, figures, references |
| `paper_1/v15/`, `src/v15/`, `src/phase2b_two_domain.py` | Earlier area-aware domain models, summarized in Supplementary Section S7 |
| `paper_1/benchmark/`, `figures/` | Frozen earlier benchmark that the modelling frame and domain models depend on (`V14_BENCHMARK.md`) |
| `data/` | Modelling table and the contact-distance feature |
| `src/*.py` (root) | Data provenance and shared helpers: table and contact-distance builds, lithology audit, numerics |
| `tests/` | Integrity, leakage and manuscript-consistency tests |
| `tools/zenodo/` | Packaging and upload of the archived posterior samples |
| `requirements-paper1.txt` | Exact package versions used for every fit |

**Groundwork for Paper 2** (bedrock geology, structure, occurrence intensity)

| Path | Contents |
|---|---|
| `src/geology_foundation/`, `docs/geology_foundation/` | Geological domains, major structures and structural predictors |
| `src/data_acquisition/`, `docs/acquisition_*` | Acquisition and georeferencing of Zambian and DRC geological sources |
| `src/modeling/` | Tract-supported occurrence-intensity models (stages 4–5) |
| `configs/` | Configuration for the above |

Superseded scripts, notes and figures from earlier versions remain in the Git
history.

## Reproduce

The fits used Python 3.12.15 with PyMC 5.25.1; `requirements-paper1.txt` pins
every package. Commands run from the repository root:

```powershell
# Re-derive every metric, table and the report from stored predictions (no fitting)
python -m src.transfer analyze-mcmc
python -m src.transfer report-v2
python -m src.transfer.supplement_tables
python -m src.transfer.paper_figures      # needs geopandas and matplotlib

# Refit (hours): primary fits, then each batch of further fits; completed fits are skipped
python -m src.transfer fit-mcmc
python -m src.transfer fit-amendment --batch a2   # also a3, a6, a8, sensitivities

# Verify outputs and run the tests
cd paper_1/transfer; sha256sum -c checksums/SHA256SUMS; cd ../..
python -m pytest -q tests
```

Run MCMC fits in a visible terminal, not as background jobs.

## Data availability

Posterior samples (`*.nc`) are too large for Git and are archived at Zenodo
(https://doi.org/10.5281/zenodo.24498271). Until they are restored, their checksum entries report as
missing. The 11.6 MB concatenated prediction file is also excluded; the
analysis command above regenerates it.

The source products (USGS occurrence compilation, WGM2012 gravity, Africa
surficial lithology) are not redistributed; see the manuscript for citations.

## Licences

Code is under the MIT License ([`LICENSE`](LICENSE)). Derived data, outputs and
figures are under CC BY 4.0, subject to the terms of the third-party sources
([`LICENSE-DATA.md`](LICENSE-DATA.md)). The manuscript sources are excluded
from both licences until publication.
