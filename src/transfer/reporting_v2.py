"""Render the Amendment 1 full-MCMC transfer report."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pandas as pd

from src.transfer import config
from src.transfer.data import ensure_output_path
from src.transfer.reporting import _read, _table


def write_report_v2() -> str:
    spec_path = config.OUTPUT / "specification_v2.json"
    spec_hash = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    v1_hash = hashlib.sha256(
        (config.OUTPUT / "specification.json").read_bytes()
    ).hexdigest()
    gate = json.loads(
        (config.OUTPUT / "diagnostics" / "map_mcmc_validation.json").read_text(
            encoding="utf-8"
        )
    )
    reproduction = json.loads(
        (config.OUTPUT / "diagnostics" / "environment_reproduction.json").read_text(
            encoding="utf-8"
        )
    )
    status = pd.read_csv(config.OUTPUT / "diagnostics" / "amendment1_status.csv")
    a1 = _read("metrics/a1_transfer_metrics.csv")
    a2 = _read("metrics/a2_transfer_matrix_long.csv")
    a2_wide = _read("metrics/a2_single_sector_matrices.csv")
    a2_descriptor = json.loads(
        (config.OUTPUT / "metrics" / "a2_descriptor.json").read_text(
            encoding="utf-8"
        )
    )
    a3 = _read("metrics/a3_decomposition.csv")
    a4 = _read("metrics/a4_aoa_summary.csv")
    a5 = _read("metrics/a5_predictor_verdicts.csv")
    a5_auc = _read("metrics/a5_marginal_auc.csv")
    a6 = _read("metrics/a6_optimism_comparison.csv")
    sensitivity = _read("metrics/sensitivity_metrics.csv")
    a7 = _read("metrics/a7_domain_summary.csv")
    a7_pairs = _read("metrics/a7_paired_comparisons.csv")
    gravity = _read("metrics/a7_gravity_gradient.csv")
    a8 = _read("metrics/a8_partition_metrics.csv")
    a8_verdict = _read("metrics/a8_partition_verdicts.csv")
    a8_opt = _read("metrics/a8_optimism_gap.csv")
    a8_dir = _read("metrics/a8_partition_direction.csv")
    p_gap = _read("partitions/p_gap_summary.csv")
    p_pos = _read("partitions/p_pos_summary.csv")
    a8_conclusion = json.loads(
        (config.OUTPUT / "metrics" / "a8_conclusion.json").read_text(
            encoding="utf-8"
        )
    )
    pooled = a1.loc[a1["sample"] == "pooled"].iloc[0]
    sectors = a1.loc[a1["sample"].str.startswith("sector_")].set_index("sample")
    verdict_text = "; ".join(
        f"{row.predictor}: {row.verdict}" for row in a5.itertuples(index=False)
    )
    verdict_pivot = a8_verdict.pivot(
        index="predictor", columns="partition", values="verdict"
    )
    verdict_changes = [
        f"{predictor} ({verdicts['primary']} under the primary partition, "
        f"{verdicts[name]} under {name.replace('_', '-')})"
        for name in ("P_GAP", "P_POS")
        for predictor, verdicts in verdict_pivot.iterrows()
        if verdicts[name] != verdicts["primary"]
    ]
    verdict_change_text = (
        "The verdicts that change are: " + "; ".join(verdict_changes) + "."
        if verdict_changes
        else "No predictor verdict changes between partitions."
    )
    local_sector_1 = status.loc[
        (status["batch"] == "a3_local") & (status["target_sector"] == 1),
        "positive_train",
    ]
    a3_rows = a3.set_index("sector")
    a6_random = a6.loc[a6["validation"].str.startswith("10 repeats")].iloc[0]
    sensitivity_sector_1 = sensitivity.loc[
        sensitivity["sample"] == "fold_1"
    ].set_index("model")["auc"]
    sensitivity_pooled = sensitivity.loc[
        sensitivity["sample"] == "pooled"
    ].set_index("model")
    gaps = "; ".join(
        f"Sector {int(row.sector)}: {row.base_minus_transfer:+.4f}, "
        f"{row.recal_minus_base:+.4f}, {row.local_minus_recal:+.4f}"
        for row in a3.itertuples(index=False)
    )
    lines = [
        "# Geographic transfer analysis report — Amendment 1",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}",
        "",
        f"Original specification SHA-256: `{v1_hash}`",
        "",
        f"Amendment 1 specification SHA-256: `{spec_hash}`",
        "",
        (
            "The original specification remains unchanged. Version 2 records "
            "the failed MAP-MCMC gate and retires MAP, replacing all remaining "
            "fits with full MCMC. Ten random-CV repeats replace twenty as a "
            "compute-only amendment."
        ),
        "",
        "## Estimator decision and environment",
        "",
        (
            f"Original MAP-Laplace mean log score was "
            f"{gate['pooled_log_score_map']:.6f}, versus "
            f"{gate['pooled_log_score_mcmc']:.6f} for MCMC; the "
            f"{gate['mean_log_score_difference_map_minus_mcmc']:+.6f} per-cell "
            f"difference exceeded the {gate['tolerance']:.3f} tolerance. MAP "
            "and all further predictive approximations were retired."
        ),
        "",
        (
            f"All fits used `{reproduction['interpreter']}`. Recomputing the four "
            "existing MCMC predictions from their traces gave maximum absolute "
            f"probability difference {reproduction['max_abs_delta']:.3e}, below "
            f"the {reproduction['threshold']:.2f} check."
        ),
        "",
        (
            f"Amendment 1 fits completed: {len(status)}; convergence failures: "
            f"{int((~status['converged'].astype(bool)).sum())}. Maximum R-hat: "
            f"{status['max_rhat'].max():.4f}."
        ),
        "",
        "## A1. Leave-one-sector-out transfer",
        "",
        _table(
            a1,
            [
                "sample",
                "n",
                "positives",
                "mean_log_score",
                "log_score_skill_vs_prevalence",
                "relative_log_score_skill",
                "skill_rank",
                "mean_minus_observed_rate",
                "auc",
                "auc_ci_lower",
                "auc_ci_upper",
                "pr_auc",
                "pr_auc_baseline",
                "brier_skill",
                "calibration_in_the_large",
                "calibration_intercept",
                "calibration_slope",
                "v14_benchmark_auc",
            ],
        ),
        "",
        (
            "`calibration_in_the_large` is the pre-specified intercept with the "
            "slope fixed at 1; negative values mean predictions are too high. "
            "`calibration_intercept` and `calibration_slope` come from the joint "
            "logistic recalibration. A slope below 1 means the predictions are "
            "too spread out."
        ),
        "",
        (
            f"Pooled MCMC transfer log score is **{pooled['mean_log_score']:.5f}**, "
            f"with prevalence-baseline skill "
            f"**{pooled['log_score_skill_vs_prevalence']:+.5f}** per cell. "
            f"Pooled AUC is **{pooled['auc']:.4f}** "
            f"[{pooled['auc_ci_lower']:.4f}, {pooled['auc_ci_upper']:.4f}]."
        ),
        "",
        "## A2. Sector-to-sector transfer",
        "",
        _table(
            a2_wide,
            ["metric", "source_sector", "target_1", "target_2", "target_3", "target_4"],
        ),
        "",
        _table(
            a2,
            [
                "source_type",
                "source_sector",
                "source_sector_a",
                "source_sector_b",
                "target_sector",
                "training_cells",
                "training_positives",
                "mean_log_score",
                "auc",
                "sector_separation_km",
                "few_training_positives",
            ],
        ),
        "",
        (
            "The single-sector and pair matrices show whether transfer changes "
            "with direction, sector separation, and the amount of training "
            "support. Entries with fewer than 15 training positives are flagged "
            "and are not interpreted alone; in particular, the apparent best "
            "source for Sector 2 is the nine-positive Sector 1 and must not be "
            "treated as a stable recommendation."
        ),
        "",
        (
            "Mean absolute A-to-B versus B-to-A log-score difference is "
            f"**{a2_descriptor['mean_absolute_directional_log_score_difference']:.4f}**. "
            "The correlation between sector separation and the mean directional "
            f"log score is **{a2_descriptor['separation_vs_mean_directional_score_correlation']:.4f}**. "
            "Best single training sectors by target: "
            + ", ".join(
                f"Sector {target} from Sector {source}"
                for target, source in a2_descriptor[
                    "best_single_training_sector_by_target"
                ].items()
            )
            + "."
        ),
        "",
        (
            "These directional scores are raw log scores. Sector separation is "
            "also correlated with prevalence, so this correlation must not be "
            "read as an isolated geographic-distance effect."
        ),
        "",
        "## A3. Oracle decomposition",
        "",
        _table(
            a3,
            [
                "sector",
                "n",
                "positives",
                "L_transfer",
                "L_base",
                "L_recal",
                "L_local",
                "base_minus_transfer",
                "base_minus_transfer_ci_lower",
                "base_minus_transfer_ci_upper",
                "recal_minus_base",
                "recal_minus_base_ci_lower",
                "recal_minus_base_ci_upper",
                "local_minus_recal",
                "local_minus_recal_ci_lower",
                "local_minus_recal_ci_upper",
            ],
        ),
        "",
        (
            f"Gap results: {gaps}. Positive values indicate improvement in "
            "held-out log score. Base-rate, recalibration and local steps use "
            "held-out labels and are diagnostic oracles, not deployable models. "
            "Base and recalibration re-estimate the intercept and one slope "
            "multiplier directly on the posterior-mean T0 intensity predictor; "
            "they do not reuse a MAP or posterior-median slope vector."
        ),
        "",
        (
            "The local step confounds relationship differences with training "
            "size: each inner fit uses about three quarters of one sector. In "
            f"Sector 1 the inner training folds hold only "
            f"{int(local_sector_1.min())}–{int(local_sector_1.max())} positives, "
            "so its local gap falls under the same fewer-than-15-positives rule "
            "as A2 and is not interpreted alone."
        ),
        "",
        "## A4. Covariate shift and area of applicability",
        "",
        _table(
            a4,
            ["sector", "aoa_status", "cells", "area_km2", "positives", "mean_log_score"],
        ),
        "",
        "![Area of applicability](../figures/area_of_applicability.png)",
        "",
        "## A5. Which relationships transfer",
        "",
        _table(
            a5,
            ["predictor", "coefficient_sign_stable", "useful_auc_sectors", "auc_condition", "verdict"],
        ),
        "",
        _table(
            a5_auc,
            [
                "sector",
                "predictor",
                "orientation",
                "auc",
                "auc_ci_lower",
                "auc_ci_upper",
            ],
        ),
        "",
        (
            f"Predictor verdicts: **{verdict_text}**. Sector 1 fault-distance "
            "AUC is 0.24 with close distances scored positive, so the fault "
            "relationship fails the useful-AUC condition there. This is "
            "interpreted primarily as a likely map-coverage effect: the regional "
            "fault digitization is dense in DRC and sparse in the Zambian "
            "Copperbelt, not as evidence for a geological relationship."
        ),
        "",
        "![Coefficient and marginal-AUC stability](../figures/coefficient_marginal_auc_stability.png)",
        "",
        "## A6. Optimism of non-spatial validation",
        "",
        _table(
            a6,
            [
                "validation",
                "mean_log_score",
                "log_score_sd",
                "log_score_q2_5",
                "log_score_q97_5",
                "auc",
                "auc_sd",
                "auc_q2_5",
                "auc_q97_5",
            ],
        ),
        "",
        (
            "The 10-repeat random-CV result uses the same full MCMC estimator. "
            "The buffered rows are a separate V15 model that includes lithology."
        ),
        "",
        "### Pre-specified model sensitivities",
        "",
        _table(
            sensitivity,
            [
                "model",
                "sample",
                "n",
                "positives",
                "mean_log_score",
                "auc",
                "brier_skill",
                "calibration_slope",
            ],
        ),
        "",
        (
            "Sector 1 discrimination depends strongly on specification: held-out "
            f"AUC is {sectors.loc['sector_1', 'auc']:.3f} for T0, "
            f"{sensitivity_sector_1['T0_linear']:.3f} for T0-linear, "
            f"{sensitivity_sector_1['T0_rebuilt']:.3f} for T0-rebuilt and "
            f"{sensitivity_sector_1['T0_nrb']:.3f} for T0-nrb (9 positives). "
            "Pooled AUC is less sensitive "
            f"({sensitivity_pooled['auc'].min():.3f}–{sensitivity_pooled['auc'].max():.3f} "
            f"across the sensitivities, {pooled['auc']:.3f} for T0). The Sector 1 "
            "ranking result should therefore not be quoted without this range. "
            "T0-rebuilt has a better pooled log score than T0 "
            f"({sensitivity_pooled.loc['T0_rebuilt', 'mean_log_score']:.4f} versus "
            f"{pooled['mean_log_score']:.4f}); stored faults remain primary by "
            "pre-specification, not by performance."
        ),
        "",
        "## A7. Domain structure does not repair transfer",
        "",
        _table(
            gravity,
            [
                "row_type",
                "fault_variant",
                "comparison",
                "mean_log_score_gain",
                "ci_lower",
                "ci_upper",
                "positive_folds",
                "verdict",
            ],
        ),
        "",
        (
            "The gravity gain is approximately +0.002 with fixed effects and "
            "rebuilt faults, +0.014 with fixed effects and stored faults but "
            "with an interval including zero, and +0.030 only under the "
            "six-domain hierarchy with an interval excluding zero. Domain-varying "
            "distance terms are not supported in either fixed-effect ladder. The "
            "V15 hierarchical gravity gain was data-suggested and prior-dependent."
        ),
        "",
        _table(a7_pairs, ["source", "comparison", "mean_delta_log_score", "ci_lower", "ci_upper", "verdict"]),
        "",
        _table(
            a7,
            ["source", "model", "sample", "n", "positives", "auc", "mean_log_score"],
        ),
        "",
        "## A8. Partition sensitivity",
        "",
        "### P-GAP",
        "",
        _table(
            p_gap,
            [
                "sector",
                "start_km",
                "end_km",
                "length_km",
                "cells",
                "positives",
                "prevalence",
                "positives_within_10km_boundaries",
                "domain_composition",
            ],
            3,
        ),
        "",
        "### P-POS",
        "",
        _table(
            p_pos,
            [
                "sector",
                "start_km",
                "end_km",
                "length_km",
                "cells",
                "positives",
                "prevalence",
                "positives_within_10km_boundaries",
                "domain_composition",
            ],
            3,
        ),
        "",
        _table(
            a8,
            [
                "partition",
                "sample",
                "n",
                "positives",
                "mean_log_score",
                "auc",
                "auc_ci_lower",
                "auc_ci_upper",
            ],
        ),
        "",
        _table(a8_verdict, ["partition", "predictor", "coefficient_sign_stable", "useful_auc_sectors", "verdict"]),
        "",
        _table(
            a8_opt,
            ["partition", "random_cv_mean_log_score", "spatial_loo_mean_log_score", "optimism_gap_random_minus_spatial"],
        ),
        "",
        _table(a8_dir, ["partition", "sector_log_score_correlation_with_primary", "best_sector_primary", "best_sector_alternate"]),
        "",
        (
            f"Partition-robust conclusion: **{'yes' if a8_conclusion['partition_robust'] else 'no'}**. "
            "A8 uses full MCMC for both leave-one-sector-out and single-sector "
            f"fits. {verdict_change_text} Because the pre-specified rule requires "
            "the A5 verdicts to agree under all three partitions, this change "
            "alone makes the conclusions not partition-robust. The "
            "sector-profile correlations are descriptive only: sectors cover "
            "different ground under each partition, and raw log scores mainly "
            "track sector prevalence. The decomposition remains a "
            "primary-partition diagnostic."
        ),
        "",
        "## Figures",
        "",
        "![Transfer matrix](../figures/transfer_matrix.png)",
        "",
        "![Decomposition gaps](../figures/decomposition_gaps.png)",
        "",
        "## Plain-language summary",
        "",
        (
            "- **Ranking transfers moderately.** A model trained on three sectors "
            f"ranks cells in the fourth with pooled AUC {pooled['auc']:.3f} "
            f"[{pooled['auc_ci_lower']:.3f}, {pooled['auc_ci_upper']:.3f}] and "
            f"sector AUCs of {sectors['auc'].min():.3f}–{sectors['auc'].max():.3f}. "
            "Sector 1's value is fragile across specifications (see "
            "'Pre-specified model sensitivities').\n"
            "- **Probabilities do not transfer.** Pooled calibration slope is "
            f"{pooled['calibration_slope']:.2f}. Calibration-in-the-large is "
            + ", ".join(
                f"{sectors.loc[f'sector_{k}', 'calibration_in_the_large']:+.2f}"
                for k in range(1, 5)
            )
            + " in Sectors 1–4 (negative: predicted too high). In Sector 1 the mean "
            "predicted probability exceeds the observed rate by "
            f"{sectors.loc['sector_1', 'mean_minus_observed_rate']:.3f}, and the "
            "model scores worse than the training-prevalence baseline "
            f"(skill {sectors.loc['sector_1', 'log_score_skill_vs_prevalence']:+.3f} "
            "per cell). Skill is "
            f"{sectors.loc['sector_2', 'log_score_skill_vs_prevalence']:+.3f}, "
            f"{sectors.loc['sector_3', 'log_score_skill_vs_prevalence']:+.3f} and "
            f"{sectors.loc['sector_4', 'log_score_skill_vs_prevalence']:+.3f} in "
            "Sectors 2–4.\n"
            "- **Why (oracle decomposition).** In Sector 1 an intercept-only "
            "correction recovers "
            f"{a3_rows.loc[1, 'base_minus_transfer']:+.3f} per cell "
            f"[{a3_rows.loc[1, 'base_minus_transfer_ci_lower']:+.3f}, "
            f"{a3_rows.loc[1, 'base_minus_transfer_ci_upper']:+.3f}], so the loss "
            "there is mainly a base-rate shift. In Sectors 2–4 base-rate and "
            "slope corrections are small. A local model beats the recalibrated "
            "transfer model only in the mixed Sector 2 "
            f"({a3_rows.loc[2, 'local_minus_recal']:+.3f} "
            f"[{a3_rows.loc[2, 'local_minus_recal_ci_lower']:+.3f}, "
            f"{a3_rows.loc[2, 'local_minus_recal_ci_upper']:+.3f}]). In Sector 4 "
            "it is worse "
            f"({a3_rows.loc[4, 'local_minus_recal']:+.3f} "
            f"[{a3_rows.loc[4, 'local_minus_recal_ci_lower']:+.3f}, "
            f"{a3_rows.loc[4, 'local_minus_recal_ci_upper']:+.3f}]), so other "
            "sectors' data help there more than local data do. These are "
            "oracle diagnostics, not deployable corrections.\n"
            f"- **Predictors.** {verdict_text}. Gravity is useful within the "
            "south-eastern sectors only. The Sector 1 fault reversal is treated "
            "as a likely map-coverage effect.\n"
            "- **Random cross-validation is optimistic.** Random 4-fold CV gives "
            f"log score {a6_random['mean_log_score']:.4f} and AUC "
            f"{a6_random['auc']:.3f}, against {pooled['mean_log_score']:.4f} and "
            f"{pooled['auc']:.3f} for leave-one-sector-out. That is an optimism "
            f"of {a6_random['mean_log_score'] - pooled['mean_log_score']:.3f} per "
            "cell, rising to "
            f"{a8_opt.loc[a8_opt['partition'] != 'primary', 'optimism_gap_random_minus_spatial'].min():.3f}–"
            f"{a8_opt.loc[a8_opt['partition'] != 'primary', 'optimism_gap_random_minus_spatial'].max():.3f} "
            "under the alternative partitions.\n"
            "- **Domain structure does not repair transfer** (A7).\n"
            "- **Not partition-robust under the pre-specified rule.** "
            f"{verdict_change_text}"
        ),
        "",
        "## Limitations",
        "",
        (
            "- Four sectors provide four spatial units, not a precise sampling distribution.\n"
            "- Sector 1 has only nine positives.\n"
            "- Canonical equal-cell fold boundaries are arbitrary relative to deposit clusters.\n"
            "- Prevalence ranges from 1.9% to 13.0%.\n"
            "- Residual autocorrelation crosses sector boundaries.\n"
            "- Geology and geography are strongly confounded.\n"
            "- Fault-distance coverage differs between DRC and Zambia.\n"
            "- Transfer along one belt is not frontier transfer."
        ),
        "",
    ]
    path = ensure_output_path(config.OUTPUT / "reports" / "transfer_report.md")
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)

