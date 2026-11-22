# V15 independent review checklist

## Repository and safeguards

- [ ] Confirm branch is `paper1-completion` and HEAD is `74b436c` unless the reviewer has intentionally advanced it.
- [ ] Confirm no V14 benchmark, trace, OOF prediction, modelling table, fold file, manifest, manuscript result table or figure was modified.
- [ ] Run `pytest -q tests/test_v14_benchmark_integrity.py`; expected 82 passed.
- [ ] Inspect `git status --short`; V15 files should be untracked or new, with no staged or committed changes.
- [ ] Verify the frozen V14 benchmark checksums independently.
- [ ] Confirm `src/validation_strategies.py` matches HEAD (`44a43dad...`) and that the V14 manifest hash (`dee9c6...`) is an uncommitted intermediate version absent from Git history. This is a source-provenance gap, not a working-tree change.
- [ ] Confirm `AI_PROJECT_CONTEXT.md` is untracked and ignored, so its baseline warning is not a Git modification.
- [ ] Verify the V15 specification hash is `7caae51d39c6bc5f842d3637b7cd5df3a5f8fa82b809e9289d1d9aa616b05156`; the earlier `2427e5e1...` value is superseded by the 18:06:50Z regeneration.

## Data and outcome

- [ ] Verify 1,899 rows and 1,899 unique grid IDs in `data/copperbelt_training_v14_centroid_contact.csv`.
- [ ] Verify 139 positive rows before complete-case filtering and 138 after excluding missing-domain rows.
- [ ] Verify the lost positive is grid 7074.
- [ ] Verify 1,872 cells, four folds of 468, and fold positives 9, 31, 37 and 61.
- [ ] Verify the outcome means at least one compiled USGS occurrence, not a deposit count.
- [ ] Verify 187 USGS records, 186 in complete-case cells, 184 in positive cells, two in negative cells and one in excluded cell 7074.
- [ ] Verify 147 RDC.ROCKS observations are absent from the modelling inventory and are never described as model positives.

## Area and exposure

- [ ] Reproduce area in EPSG:32735, not in degrees.
- [ ] Verify the area definition is the 5 km cell polygon intersected with the union of the five permissive tracts.
- [ ] Verify 21,790.3558 km2 total clipped area.
- [ ] Verify 205 cells below 1 km2 and zero positives among them.
- [ ] Verify projected area-alone AUCs: 0.4103, 0.6095, 0.6745, 0.7031; pooled 0.5834 (corrected 22 Nov 2026; earlier 0.5591 values came from a clipping bug in `roc_auc`).
- [ ] Verify `roc_auc` ranks raw covariates without clipping (`tests/test_v15_models.py`).
- [ ] Verify gravity fold AUCs with low gravity scored positive: 0.850, 0.783, 0.566, 0.509.
- [ ] Check the area-covariate model against the offset model; the covariate has better held-out log score.

## Models

- [ ] Confirm M0-M3 all include fault and contact linear/quadratic terms, gravity, categorical lithology and the same complete cases and folds.
- [ ] Confirm M0 has a global intercept; M1 domain intercepts with shared slopes; M2 domain-varying fault/contact slopes; M3 additionally domain-varying gravity slopes.
- [ ] Confirm the primary cloglog models use fixed `log(area_km2)` with coefficient one.
- [ ] Confirm the V14 control uses logit, no area treatment, shared gravity and domain-varying distance slopes.
- [ ] Confirm M2 and M3 strict refits have four chains, 1,500 draws, 2,500 tuning steps, target acceptance 0.995 and zero divergences.
- [ ] Confirm the non-converged initial M2/M3 runs are archived and not used for inference.
- [ ] Confirm M0 and M1 used the earlier two-chain sampler and converged.
- [ ] Recompute all convergence diagnostics with `python -m src.v15.sampling_diagnostics`; do not trust the shared `fit_status.json`, which was written concurrently by multiple runs.

## Metrics

- [ ] Verify held-out mean log score is computed from OOF probabilities, not in-sample likelihood.
- [ ] Verify paired differences use identical cells and models' common `id` set.
- [ ] Verify central comparisons are oriented as upper minus lower.
- [ ] Verify fold-level paired differences are reported.
- [ ] Verify spatial-block bootstrap uses 25 km blocks and 2,000 replicates.
- [ ] Check calibration intercept and slope in the pooled table.
- [ ] Confirm four folds are treated as four spatial units, not as a precise sampling distribution.

## Key numerical checks

- [ ] M0 pooled AUC 0.6882, mean log score -0.2725.
- [ ] M1 pooled AUC 0.7087, mean log score -0.2633.
- [ ] M2 pooled AUC 0.7340, mean log score -0.2571.
- [ ] M3 pooled AUC 0.8020, mean log score -0.2267.
- [ ] Frozen V14 pooled AUC 0.7155, mean log score -0.2510.
- [ ] V14 control AUC 0.7094, mean log score -0.2526.
- [ ] M1 minus M0 mean log-score gain 0.00918; bootstrap interval includes zero.
- [ ] M2 minus M1 mean log-score gain 0.00622; bootstrap interval includes zero.
- [ ] M3 minus M2 mean log-score gain 0.03038; bootstrap interval excludes zero.
- [ ] Random forest AUC 0.7528, mean log score -0.2352.
- [ ] M3 versus random forest log-score difference 0.00849; interval includes zero.

## Sensitivity checks

- [ ] Confirm `M2_cloglog_no_litho` changes only the categorical lithology field.
- [ ] Confirm categorical lithology removal has negligible effect: AUC change below 0.001 and log-score change about 0.00002.
- [ ] Confirm no raster-centroid lithology model was promoted.
- [ ] Inspect fault-distance non-reproducibility: 227 cells and 26 positives, with maximum difference 33.1 km.
- [ ] Inspect label perturbation: mean 40.07 changed cells, 500 simulations, 1 km radius.
- [ ] Inspect Moran's I: all models retain positive residual autocorrelation at p = 0.001.
- [ ] Inspect buffered validation: 5 km removes 22/36/30/21 training cells and 0/2/7/3 positives; 10 km removes 43/69/65/43 cells and 1/3/13/4 positives. Pooled M2 AUC is 0.7267 at 5 km and 0.6974 at 10 km, against 0.7340 canonically.
- [ ] Confirm three buffered M2 fits fail the strict tail-ESS criterion: 5 km Fold 4 (tail ESS 136), 10 km Fold 1 (tail ESS 140), 10 km Fold 4 (three divergences, tail ESS 171). Treat them as limited-convergence sensitivity results.
- [ ] Confirm the 10 km Fold 1 M2 AUC of 0.4713 is uninformative: nine held-out positives and a heavily altered training neighbourhood.
- [ ] Inspect prior-width fits: prior 0.5 gives AUC 0.7378 and mean log score -0.2480; prior 1.0 gives 0.7340 and -0.2571; prior 2.0 gives 0.7274 and -0.2680.
- [ ] Confirm the between-domain scale posteriors track prior width: `sigma_beta_f_lin` mean 0.762 at prior 0.5, 1.176 at prior 1.0 and 1.790 at prior 2.0.
- [ ] Confirm no prior variant is promoted to a primary model; the narrower prior is not selected post hoc.
- [ ] Verify the random forest has training-only tuning and preprocessing.
- [ ] Verify no comparator uses held-out information.

## Interpretation audit

- [ ] Do not claim varying slopes are convincingly superior: M2 versus M1 spatial-block uncertainty includes zero.
- [ ] Do not claim domain intercepts explain all improvement: M2 adds a further descriptive gain.
- [ ] Do not call M3 a stable gravity law: gains are fold-dependent and strongest in the nine-positive first fold.
- [ ] Do not describe area correction as exploration-effort correction.
- [ ] Do not treat 184 USGS records as 184 independent positives.
- [ ] Do not treat 147 RDC.ROCKS observations as model positives.
- [ ] Do not describe residual independence where Moran's I is positive.
- [ ] Distinguish relative predictive performance from causal geological inference.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m src.v15.build_data
.\.venv\Scripts\python.exe -m src.v15.sampling_diagnostics
.\.venv\Scripts\python.exe -m src.v15.evaluate
.\.venv\Scripts\python.exe -m src.v15.gravity
.\.venv\Scripts\python.exe -m src.v15.moran
.\.venv\Scripts\python.exe -m src.v15.fault_sensitivity
.\.venv\Scripts\python.exe -m src.v15.perturbation
.\.venv\Scripts\python.exe -m src.v15.prior_sensitivity
# Do not rerun write_spec.py: it rewrites created_utc and invalidates the frozen specification hash.
.\.venv\Scripts\python.exe -m pytest -q tests/test_v15_models.py
.\.venv\Scripts\python.exe -m pytest -q tests/test_v14_benchmark_integrity.py
.\.venv\Scripts\python.exe -m src.v15.checksums
.\.venv\Scripts\python.exe -m src.v15.checksums --verify
```
