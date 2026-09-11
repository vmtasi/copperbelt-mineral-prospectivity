# Copperbelt Mineral Prospectivity — AI Project Context

## Purpose
Persistent working context for AI agents connected to this repository. This
records established research logic and methodological decisions so agents can
work from the project itself rather than a long chat history.

The repository is authoritative for exact implementation details. Do not infer
missing model definitions, coefficients, fold assignments, or transformations.

## Research objective
The project concerns spatially transferable Cu–Co mineral prospectivity in the
Copperbelt, with attention to tectonic/geological domain heterogeneity.

The project optimizes jointly for:
1. useful predictive performance;
2. geological/statistical validity;
3. rigorous spatial validation;
4. credible uncertainty;
5. reproducibility and auditability.

The current dataset contains 1,800+ observations.

## Core validation principle
The principal validation framework uses four fixed along-belt spatial folds,
B1–B4, generated through `src/validation_strategies.py` and
`get_along_belt_folds(...)`.

Non-negotiable rules:
- Do not replace B1–B4 with random cross-validation.
- No information from a held-out spatial region may enter training.
- Learned preprocessing must be fitted on training data only.
- Model comparisons should use identical spatial folds.
- Fold 4 must be investigated rather than simply dismissed.
- Improvements must remain geologically and statistically defensible.

## Model-development context
The project includes M1–M5 and later nonlinear/hierarchical variants.

IMPORTANT: Exact definitions of M1–M5 must be recovered from the repository.
Do not infer them from their names.

Approximate benchmark context previously discussed:
- M5 Gravity + Lithology: ~0.699 AUC
- V9 Global nonlinear: ~0.693 AUC
- V11 Hierarchical nonlinear: ~0.686 AUC

These are benchmark context, not universal values for every fold/rerun.

The working principle is: prefer the simplest model that demonstrates a
meaningful, spatially valid, geologically defensible improvement.

## PyMC / beta audit
A major methodological priority is verifying that PyMC beta coefficients and
the corresponding analysis scripts are mathematically and semantically
consistent.

For every beta, trace:

raw feature -> transformation -> scaling -> design-matrix column
-> PyMC parameter -> posterior summary -> interpretation

Verify:
- sign convention;
- units;
- transformations;
- standardisation;
- nonlinear terms;
- design-matrix column/beta correspondence;
- posterior summaries;
- odds-ratio scale;
- comparability across models;
- hierarchical/domain effects;
- agreement between code and thesis wording.

Never interpret a beta merely from its variable name.

## Phase 7 — sensitivity analysis
Phase 7 tests whether predictive performance is sensitive to training
information and to the number of positive deposits in a held-out test region.

The four spatial folds remain fixed.

### Axis A
Fixed test region, varying number of positive training deposits. Negative samples
are drawn to approximately preserve the observed negative/positive ratio.
Repeated seeded sampling reports median AUC and a 2.5–97.5 percentile interval.

### Axis B
Fixed full training model, varying number of positive deposits in the held-out
test region. Test negatives remain included. Repeated seeded sampling reports
median AUC and a 2.5–97.5 percentile interval.

Current implementation uses:
`np.random.default_rng(42)`
and `500` iterations.

Interpretation:
- stable AUC across reduced sample sizes supports robustness to sample
  composition;
- wide intervals or strong degradation indicate sample-size sensitivity;
- test-positive uncertainty must not be confused with generalisation ability.

## Phase 7 guardrails
1. Freeze B1–B4.
2. Preserve fold-trained preprocessing.
3. Avoid leakage.
4. Compare models on identical spatial splits.
5. Investigate Fold 4.
6. Screen scientifically justified predictors before adding complexity.
7. Use cheap spatial screening where appropriate.
8. Bayesian-confirm serious finalists.
9. Evaluate AUC and PR-AUC where appropriate.
10. Inspect calibration and paired fold differences.
11. Preserve interpretability.
12. Stop adding predictors when extra complexity no longer gives defensible
    spatial improvement.

## Spatial-support audits — V10 / V11 / V11b
The spatial-support audit examines whether held-out regions are represented
within training feature support.

Core features:
- `distance_to_fault`
- `distance_to_lithology_contact`
- `bouguer`

Domains are mapped to:
- `NRB_3a`
- `NRB_3b`
- `CRZ`
- `SRB`
- `NKB`
- `MMSB`

Unknown domains are excluded from this audit.

Outputs:
- `fold_summary.csv`
- `domain_summary.csv`
- `feature_by_target_summary.csv`
- `domain_training_support_by_fold.csv`
- `fold_domain_overlap.csv`
- `fold_domain_heatmap_visual.csv`
- `feature_support.csv`

The audit records region/domain sizes, deposit/non-deposit counts and rates,
feature distributions by target, domain support in training for each held-out
fold, fold-domain overlap, and feature extrapolation.

Feature-support diagnostics include:
- Wasserstein distance;
- Wasserstein distance divided by training SD;
- percentage of test values outside training min/max;
- percentage outside training 1st–99th percentile.

These are support diagnostics, not performance metrics.

## Interpretation philosophy
A high AUC under strong feature extrapolation should not be interpreted the
same way as a high AUC within well-supported feature space.

Poor AUC may reflect:
- genuine lack of transferability;
- domain shift;
- feature-support mismatch;
- sparse positive deposits;
- geological heterogeneity;
- model misspecification;
- data/measurement limitations.

The audits exist to distinguish these mechanisms.

## Important files
- `src/validation_strategies.py` — spatial-fold strategy.
- `data/copperbelt_training_v5_with_tectonic_domain.csv` — dataset used by the
  current audit scripts.
- `.mcp/server.py` — MCP interface to the repository.
- `figures/audit/` — audit outputs.

Phase 7 outputs:
- `figures/audit/phase7_6_axis_a_training_info.csv`
- `figures/audit/phase7_6_axis_b_test_positives.csv`

Spatial-support outputs:
- `figures/audit/fold_summary.csv`
- `figures/audit/domain_summary.csv`
- `figures/audit/feature_by_target_summary.csv`
- `figures/audit/domain_training_support_by_fold.csv`
- `figures/audit/fold_domain_overlap.csv`
- `figures/audit/fold_domain_heatmap_visual.csv`
- `figures/audit/feature_support.csv`

## Agent operating rules
Before changing methodology:
- inspect the actual implementation;
- trace transformations;
- identify train/test boundaries;
- check leakage;
- determine whether the change alters the estimand;
- preserve reproducibility.

Before interpreting coefficients:
- inspect model code;
- inspect preprocessing;
- inspect the design matrix;
- map every coefficient to its predictor;
- verify units and transformations.

Before claiming an improvement, require evidence under fixed spatial
validation. Random-CV, in-sample, or single-fold improvement is insufficient.

When debugging, record what was wrong, why, how it was detected, what changed,
and whether previous results must be regenerated.

When uncertain, distinguish known facts, inference, and repository checks.
Never invent M1–M5 definitions, beta mappings, fold assignments, or feature
transformations.

## Current priority
Make the project auditable end-to-end:
1. verify `src/validation_strategies.py`;
2. verify the PyMC model/design matrix and beta mapping;
3. verify Phase 7 sensitivity scripts;
4. verify V10/V11/V11b spatial-support logic;
5. reconcile code outputs with thesis interpretations;
6. then make further model-development decisions.

The objective is not to make results look better. It is to make the final
prospectivity conclusions difficult to challenge methodologically while still
pursuing scientifically defensible predictive performance.

## Context maintenance
Update this file after major methodological decisions. Do not make it a chat
transcript. Record decision, rationale, affected files, validation consequence,
and whether old results must be regenerated.

The repository is the source of truth for executable behaviour.
