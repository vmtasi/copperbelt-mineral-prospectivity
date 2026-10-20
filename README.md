# Copperbelt Mineral Prospectivity Mapping

## Overview

This project develops a spatially structured Bayesian framework for mineral-prospectivity mapping in the Central African Copperbelt. The completed Paper 1 analysis addresses a central challenge in spatial geoscientific machine learning: predictive relationships can appear stronger when spatial leakage or locally specific feature definitions are allowed to substitute for transferable geological information. The project therefore combines exogenous geological and geophysical predictors with geographically separated out-of-fold evaluation.

## Modeling Approach

The final V11 model is a hierarchical Bayesian logistic regression with partial pooling across Daly geological domains. Its three principal continuous predictors are **distance to fault**, **distance to lithology contact**, and **Bouguer gravity**, together with retained lithological-class terms. The intercept, fault-distance linear and quadratic coefficients, and lithology-contact-distance linear and quadratic coefficients are domain-varying under hierarchical partial pooling. Bouguer gravity enters through a **global linear coefficient**. Distance predictors are standardized using the training data before their quadratic terms are constructed, and inference uses MCMC with the NUTS sampler.

The two distance predictors therefore permit nonlinear fitted response shapes; Bouguer gravity does not have a quadratic term in V11 and consequently has no corresponding D*. This is a property of the model specification, not an indication that gravity is unimportant.

## Lithology-Contact Data Provenance

The active lithology-contact predictor is the minimum Euclidean distance from each grid-cell centroid to the nearest mapped **internal** lithological contact in EPSG:32735; the outer study-area perimeter is excluded. The validated reconstruction is `data/derived/litho_contact_distance_v13.csv` / `.gpkg`, and the canonical modelling table is `data/copperbelt_training_v13_litho_contact.csv` with exactly one row per grid cell. The archived surficial-lithology class is retained only as a secondary categorical attribute and was not reconstructed from the nearest-contact geometry. The superseded many-to-many candidate table and its first-row-per-cell columns are preserved in the read-only archive at `archive/litho_contact_repair_2026-10-18/` and are not read by the active pipeline.

The research datasets and historical archive are intentionally kept out of
GitHub. The canonical CSV is covered by the existing `data/*.csv` ignore rule;
the V13 distance CSV/GeoPackage and the full `archive/` directory are also
local-only. The tracked manifest records expected versions and hashes, but
rebuilding or validating the pipeline requires the corresponding local data
and archive files. The pipeline test module is skipped when those assets are
not available.

## Current Validation Status (2026-10-18)

The corrected distance has been propagated through the full pipeline. The canonical table contains 1,899 cells (1,872 complete-case; 138 positive; 1,734 background), with four folds of 468 cells and positive counts `[9, 31, 37, 61]`. The V11 refit gives fold ROC-AUC values of 0.846, 0.709, 0.687, and 0.605 (mean 0.712). Pooled OOF ROC-AUC is 0.707 (95% interval 0.667--0.744) versus 0.559 (0.509--0.615) for the compact global baseline, a gain of +0.147 (0.073--0.217); pooled PR-AUC is 0.135 versus 0.114.

Validation entry points:

```powershell
.runtime_vendor\run_qgis_python.bat src\validate_litho_contact_pipeline.py
.runtime_vendor\run_qgis_python.bat tests\test_litho_contact_pipeline.py
.runtime_vendor\run_qgis_python.bat src\run_distance_sensitivity.py --aggregate-only
```

The aggregate-only sensitivity mode regenerates its tables from the refitted traces without sampling. Ordinary pipeline stages read only the canonical V13 table and never the historical archive.

The repository's pytest dependency is pinned in `requirements-test.txt`. From
PowerShell, install it into the project environment and run the reviewed tests:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv\Scripts\python.exe -m pytest -q tests\test_distance_sensitivity_math.py tests\test_litho_contact_pipeline.py tests\test_mcp_server_policy.py
```

The lithology-contact validator requires an explicit new report path so the
existing audit report is never replaced:

```powershell
.\.venv\Scripts\python.exe src\validate_litho_contact_pipeline.py --report figures\audit\litho_contact_pipeline_validation_recheck.json
```

## Spatial and Nonlinear Analysis

The primary predictive validation is **four-fold along-belt spatial out-of-fold (OOF) validation**, using geographic separation to assess predictive transferability rather than relying on random cross-validation. The Daly-domain results are a secondary stratification of those frozen spatial OOF predictions, not a separate domain-refitting or independent domain-validation exercise.

For the two quadratic distance predictors, D* is used as a mathematical diagnostic of an algebraic stationary point. For a standardized distance response, the stationary point follows from the fitted quadratic coefficients. Curvature, existence of a stationary point, its classification as a minimum or maximum, empirical predictor support, and posterior concentration are distinct quantities. An in-support D* is less extrapolative than one outside observed training support, but empirical support is not geological validation, and a finite D* does not by itself establish a geological optimum.

## Development History

The feature set evolved from an initial deposit-distance feature that introduced proximal bias, through a regional tract-boundary feature that showed poor spatial transferability, toward exogenous geological/geophysical predictors with exact spatial coordinates removed. This history documents how leakage and spatially arbitrary feature definitions were addressed during development; it is not the complete scientific identity of the final model.

## Key Findings

- Mineral-prospectivity relationships vary spatially within the modeled Copperbelt framework.
- Fault distance and lithology-contact distance exhibit nonlinear fitted relationships in relevant settings, with evidence varying across domains and spatial folds.
- **NRB_3a lithology-contact distance provides particularly strong combined evidence for a supported interior nonlinear response**, while fault-distance nonlinear evidence is more heterogeneous.
- Bouguer gravity remains an integral third predictor of the multivariate V11 model despite having no quadratic turning-point diagnostic.
- Spatial OOF prediction shows heterogeneous geographic transferability rather than uniform predictive behavior across the study area.
- The completed analysis provides qualified evidence consistent with aspects of Daly's proposition, but does not establish a causal geological mechanism or a universal geological distance.

## Scope

The project should be interpreted as a spatially structured Bayesian prospectivity framework with geographically held-out evaluation and heterogeneous predictor relationships. Its results do not establish commercial drilling performance, universal transferability across the entire Copperbelt or every unexplored frontier zone, geological causation, or a universal distance relationship. Paper 1 contains the detailed methodology, results, interpretation, and discussion.
