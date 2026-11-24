# Task: Geographic transfer analysis for Paper 1

Repository: `C:\Users\vanmu\copperbelt-mineral-prospectivity`.
Python: `.\.venv\Scripts\python.exe`. Work on the current branch.

## The research question

Paper 1 is about **geographic knowledge transfer along the Central African
Copperbelt**. The question is how well a prospectivity model trained on some
along-belt sectors predicts a sector it has not seen, and why transfer
succeeds or fails. The four contiguous along-belt folds are the sectors.

The Daly-domain work (V14 hierarchy, V15 ladder, Phase 2b fixed effects) is
finished. It enters the paper only as one negative section: adding domain
structure does not repair transfer. Do **not** refit any domain model.

## What is already established (use; do not re-derive)

- Frame: 1,872 complete-case cells, 138 positives. The outcome is the presence
  of at least one compiled USGS occurrence in the cell, not a deposit count or
  verified absence.
- Folds are fixed: the `fold` column (0–3) of `paper_1/v15/data/v15_model_frame.csv`.
  Fold 0 = Sector 1 (south-east, 9 positives); Fold 3 = Sector 4 (north-west,
  61 positives). Prevalence by sector: 1.9%, 6.6%, 7.9%, 13.0%.
- Cell area matters. The likelihood is complementary log-log with `log_area_km2`
  as a fixed offset (see `paper_1/v15/reports/master_results_report.md` §5).
- The categorical lithology field is a nearest-contact attribute, not the cell
  lithology, and it contributes nothing. It is excluded.
- Fault distance has two versions, neither verifiable (see "Fault-distance
  provenance" below). Stored = `distance_to_fault` in the V15 frame; rebuilt =
  `distance_to_fault_km` in `paper_1/v15/phase2b_two_domain/data/primary_frame.csv`
  (km; the stored column is in metres).
- Geography and geology are confounded. Sector 1 is NRB_3b, Sectors 3–4 are
  NRB_3a, and Sector 2 is mixed. Domain may be used to *describe* sectors,
  never as a model term in the primary analysis.
- Within-sector marginal AUCs of gravity (low gravity scored positive) are
  0.850, 0.783, 0.566 and 0.509 (`paper_1/v15/diagnostics/gravity_fold_summary.csv`).
- `src/v15/metrics.py` provides log score, AUC (ranks raw scores, fixed
  22 Nov 2026), AP, Brier, calibration and the 25-km spatial-block bootstrap.
  Reuse it.

### Fault-distance provenance
The 126-line layer is attributed to Selley et al. (2005) Fig. 1b, a regional
map with a 200 km scale bar. Features 120–125 were added later in the
Zambian Copperbelt core. They cannot be verified in the source, but a
leakage test was negative. Report the main results under both versions.

## Hard rules

1. **Pre-register before fitting.** Write `paper_1/transfer/specification.json`
   (model, predictors, estimation, metrics, the analyses below, and which
   results are primary) and its SHA-256 before any fit. Do not change it after
   seeing results. If a change is unavoidable, version it (`specification_v2.json`)
   and record why.
2. Do not modify frozen artefacts. Verify these before and after:
   `paper_1/benchmark/SHA256SUMS`, `SHA256SUMS_external`,
   `paper_1/v15/checksums/SHA256SUMS_v15` (`python -m src.v15.checksums --verify`),
   and `paper_1/v15/phase2b_two_domain/SHA256SUMS`.
3. New code goes in `src/transfer/`, new outputs in `paper_1/transfer/`, and
   tests in `tests/test_transfer.py`.
4. Do not run MCMC in the background. Run it in a visible terminal, or give the
   user the exact commands.
5. Do not select any model, variant or threshold by its held-out performance.
6. Do not edit the manuscript. Do not commit or push. Report back first.

## Primary model (pre-specified)

**T0, global transfer model:** cloglog likelihood, `log_area_km2` offset,
predictors fault, fault², contact, contact², gravity. Use training-only
standardization (squares formed after standardization), Normal(0, 1) priors on
all coefficients, and an intercept prior centred on the log training occurrence
rate per km². Use the **stored** fault distance.

Pre-specified sensitivities, reported but not used for conclusions:
- T0-rebuilt: rebuilt fault distance.
- T0-linear: no squared terms.
- T0-nrb: T0 plus a fixed outside-NRB indicator.

Estimation:
- Leave-one-sector-out fits (the 4 canonical folds): full Bayesian MCMC
  (4 chains, ≥1,000 draws, ≥1,500 tuning, target acceptance ≥0.95; V15
  convergence criteria).
- Everything else (transfer matrix, random-CV repeats, decomposition refits):
  MAP estimation of the same model, with Laplace-approximation intervals.
- Validate MAP against MCMC on the four leave-one-sector-out fits. Compare
  held-out probabilities (report correlation and maximum absolute difference)
  and coefficients. If the pooled held-out log scores differ by more than
  0.002 per cell, stop and report before using MAP elsewhere.

## Analyses

### A1. Leave-one-sector-out transfer (the main result)
For T0, report per held-out sector and pooled:
- mean log score;
- ROC-AUC with class-stratified bootstrap 95% CI;
- AP with its prevalence baseline;
- Brier skill vs training prevalence;
- calibration-in-the-large and calibration slope.

Add a pooled spatial-block bootstrap. Show the V14 benchmark per-sector
values beside T0 for reference only (`figures/v14_centroid_contact/v14_centroid_fold_metrics.csv`).

### A2. Sector-to-sector transfer matrix
Train on each single sector and test on each of the other three (12 cells).
Report log score and AUC.

Also train on each pair of sectors and test on each remaining sector, and add
the leave-one-out results (three-sector training) for comparison. This shows
whether transfer improves with more, or nearer, training data.

Describe the matrix in terms of:
- symmetry, i.e. whether A→B equals B→A;
- dependence on sector separation, using the distance between sector centroids
  along the belt axis;
- the best training sector.

Single-sector training has few positives (Sector 1 has 9). Flag any fit with
fewer than 15 training positives and do not interpret it alone.

### A3. Why transfer fails: decomposing the loss
For each held-out sector *k*, compute four log scores on the same cells:
1. `L_transfer`: T0 trained on the other sectors.
2. `L_base`: the same model with only the intercept re-estimated on sector *k*.
   This is an oracle base-rate correction.
3. `L_recal`: intercept and a single slope multiplier re-estimated on sector *k*.
   This is an oracle calibration.
4. `L_local`: T0 trained within sector *k* by inner spatial cross-validation.
   Use 25-km blocks within the sector, assign cells to inner folds by block,
   and never put a cell in training and test at once.

Report the gaps:
- `L_base − L_transfer`: loss due to the base-rate shift;
- `L_recal − L_base`: loss due to miscalibration of slopes;
- `L_local − L_recal`: loss due to different relationships.

State clearly that 2–4 are oracles that use test-sector labels. They are
diagnostic decompositions, not deployable models. Give spatial-block
bootstrap intervals for each gap.

### A4. Covariate shift
For each held-out sector, measure how far its cells lie outside the training
predictor space:
- univariate: the share of cells outside the training 1st–99th percentile, for
  each predictor;
- multivariate: the dissimilarity index and area of applicability of Meyer &
  Pebesma (2021) on standardized T0 predictors, with the threshold derived from
  training data only, as in that paper.

Relate held-out errors to dissimilarity: the cell-level log score inside versus
outside the area of applicability, per sector. Map the area of applicability.

### A5. Which relationships transfer
- **Coefficients:** T0 coefficients (posterior median and 95% interval) from
  the four leave-one-sector-out fits, and MAP and Laplace coefficients from the
  four single-sector fits. Report a sign-agreement table per predictor.
- **Within-sector marginal AUCs:** for fault, contact and gravity (closer or
  lower scored positive; state the orientation), with class-stratified
  bootstrap CIs. Use `src.v15.metrics.roc_auc`, not a clipped version.
- **Conclusion per predictor:** transfers (stable sign and useful within-sector
  AUC in ≥3 sectors), partly transfers, or does not transfer. Fix this rule in
  the specification.

### A6. The optimism of non-spatial validation
Fit T0 under repeated random 4-fold cross-validation: 20 repeats, stratified by
outcome, same estimation. Compare its pooled log score and AUC, with the spread
across repeats, against leave-one-sector-out.

Add the existing buffered spatial results for the V15 global model as a third
point (`paper_1/v15/metrics/buffered_validation_metrics.csv`, M0 rows),
labelled as a different model (it includes lithology).

### A8. Partition sensitivity: does transfer depend on where sector boundaries fall?
The primary partition is the canonical equal-cell folds. Two alternatives are
pre-specified now, before any transfer result exists. Both use the same
principal axis as `src/validation_strategies.get_along_belt_folds`. Orient
the axis so that 0 km is the south-eastern end, measure positions in km from
the minimum projection, and use complete-case positive cells only.

- **P-GAP (district hold-out):** cut at the midpoints of the largest gaps
  between consecutive positive cells along the axis, subject to every sector
  being at least 100 km long, with K = 4. Expected cuts are about 172.5, 300.0
  and 423.5 km, giving sectors with 533 / 384 / 351 / 604 cells and
  10 / 30 / 25 / 73 positives. Recompute the cuts with the rule and stop if
  they differ from these by more than 2 km. A tie between two 15.1 km gaps is
  resolved by the 100 km minimum length.
- **P-POS (equal positives):** with K = 4, cut so that each sector holds 34–35
  positive cells (expected 786 / 550 / 234 / 302 cells). Sectors 3 and 4 are
  only about 49 and 72 km long and their boundaries cross dense deposit
  clusters. Report this as a known source of optimism.

For each partition, report:
- **The partition itself:** sector lengths, cell and positive counts, domain
  composition, and the number of positives within 10 km of each boundary.
  Write the fold assignments to `paper_1/transfer/partitions/`.
- **A1 for T0 under the partition:** MAP is acceptable after the A1 MAP–MCMC
  validation passes.
- **The A2 single-sector transfer matrix under P-POS:** every training sector
  then has about 34 positives. This is the better-powered version of A2.
- **A comparison of conclusions:** whether the per-predictor verdicts (A5),
  the sign of the decomposition gaps (A3, primary partition only unless
  cheap), and the optimism gap (A6) hold under each partition.

State the rule: transfer conclusions are robust to the partition if the
A5 verdicts and the direction of the main A1 findings agree under all three.

### A7. Domain structure does not repair transfer (no new fits)
Assemble a summary table from existing outputs:
- V15 ladder: `paper_1/v15/metrics/`;
- Phase 2b fixed-effect ladder: `paper_1/v15/phase2b_two_domain/metrics/`;
- Phase 2c, the same ladder on stored fault distances:
  `paper_1/v15/phase2c_stored_faults/metrics/fault_variant_2x2.csv`.

Report the gravity-gain gradient. The gain is +0.002 with fixed effects and
rebuilt faults, +0.014 with fixed effects and stored faults (interval
includes zero), and +0.030 with the six-domain hierarchy and stored faults
(interval excludes zero). The gain arose only under the hierarchy, shrank
with corrected faults, and does not survive the identifiable specification.

Report per-sector log scores for the global and domain models and the paired
comparisons with their intervals. Quote the established verdicts: domain-varying
distance terms are not supported (both ladders); the domain gravity term is not
supported in the fixed-effect test; and the V15 hierarchical gain was
prior-dependent and data-suggested.

## Deliverables

- `paper_1/transfer/reports/transfer_report.md`, containing:
  - the specification hash and any deviations;
  - convergence results and the MAP–MCMC validation;
  - A1–A7 with tables;
  - figures: transfer-matrix heatmap, decomposition bars per sector,
    area-of-applicability map, and coefficient and marginal-AUC stability plot;
  - a plain-language summary of what transfers, what does not and why;
  - limitations: 4 sectors, Sector 1 has 9 positives, arbitrary equal-count
    fold boundaries, prevalence differences, residual autocorrelation across
    boundaries, geology–geography confounding, and transfer along one belt
    is not frontier transfer.
- Machine-readable outputs: predictions, metrics, coefficients, and
  `paper_1/transfer/checksums/SHA256SUMS`. Keep `.nc` traces out of Git.
- `tests/test_transfer.py`, covering at least:
  - fold integrity;
  - training-only standardization;
  - no cell in both training and test, including inner spatial CV;
  - correct log-area offset;
  - MAP–MCMC agreement within tolerance;
  - correct decomposition arithmetic;
  - AUC orientation;
  - partition rules reproduce the expected P-GAP and P-POS cuts and counts.
- Final message:
  - the five most important numbers;
  - the per-predictor transfer verdicts;
  - the size of the decomposition gaps per sector;
  - anything skipped or failed;
  - confirmation that all frozen checksums still verify.

---

## Amendment 1 (after the MAP–MCMC gate failed) — resume from here

The pre-registered gate failed: MAP minus MCMC was −0.00456 per cell against a
0.002 tolerance, driven by Sector 1 (−0.0177; 9 positives). The cause is the
plug-in MAP prediction, which ignores posterior uncertainty and is
overconfident where training support is thin. MAP is therefore retired.
Do not attempt another approximation, such as Laplace predictive draws.

1. **Version the specification.** Write `specification_v2.json` with its
   SHA-256. Record the gate result and this amendment as the reason, and do
   not change anything else in the analysis plan. Keep `specification.json`
   unchanged.
2. **Use full MCMC for every fit.** The four leave-one-sector-out fits keep
   their existing settings and results; do not refit them. All other fits
   (A2, A3, A6, A8, sensitivities and single-sector coefficients) use
   2 chains, 1,000 draws, 1,500 tuning steps, target acceptance 0.97,
   recorded seeds, and the same convergence criteria. Refit any failure with
   more tuning; never drop a fit.
3. **Compute budget.** Each global fit takes about 1–4 min. Reduce A6 to
   **10** repeats of random 4-fold CV (40 fits); this changes compute only.
   Run fits in a visible terminal in resumable batches, skipping completed
   fits, and estimate the total runtime before starting.
4. **Add to A1 (reporting only, no new fits).** Raw log scores cannot be
   compared across sectors with different prevalence. For each sector report:
   - log-score skill against the training-partition prevalence baseline
     (difference per cell and relative skill);
   - mean predicted probability against the observed rate, which is the
     base-rate transfer error.
   Rank sectors by skill, not by raw log score.
5. **Add to A5 (interpretation note).** In Sector 1, positive cells lie far
   from mapped faults (oriented AUC 0.24). The fault layer is a regional
   digitization that is dense in the DRC and sparse in the Zambian
   Copperbelt, so treat this as a likely map-coverage effect and report it as
   such, not as a geological relationship.
6. **Environment.** Record the interpreter and package versions used for every
   fit. `.venv` (Python 3.10.11) works in the user's shell. If
   `.venv-pymc-full` (Python 3.12) is used instead, use it consistently for
   all new fits, and verify that it reproduces the four existing
   leave-one-sector-out held-out probabilities (max |Δp| < 0.01) before
   proceeding. If it does not, stop and report.

Everything else in the hard rules and deliverables still applies.
