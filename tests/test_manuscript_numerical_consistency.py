"""Numerical and structural checks between the manuscript and frozen V14 outputs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.numerics import average_precision_score, brier_score_loss, roc_auc_score
from src.oof_provenance import load_v14_oof

ROOT = Path(__file__).resolve().parents[1]
TEX_PATH = ROOT / "paper_1" / "manuscript" / "main.tex"
GLOBAL_OOF_PATH = ROOT / "figures" / "audit" / "phase4_oof_predictions_v14_centroid.csv"
SENSITIVITY_PATH = (
    ROOT
    / "figures"
    / "audit"
    / "distance_representation_sensitivity_v14_centroid"
    / "sensitivity_summary.csv"
)
FIG6_PATH = (
    ROOT / "paper_1" / "manuscript" / "images" / "fig6_distance_sensitivity.tikz"
)
V14_MANIFEST_PATH = (
    ROOT / "data" / "derived" / "canonical_training_manifest_v14_centroid_contact.json"
)
V14_DATA_PATH = ROOT / "data" / "copperbelt_training_v14_centroid_contact.csv"


def _manuscript() -> str:
    assert TEX_PATH.exists(), "main.tex must exist."
    return TEX_PATH.read_text(encoding="utf-8")


def _visible_tex(text: str) -> str:
    # Strip LaTeX comments but keep escaped percent signs (\%).
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in text.splitlines())


def _predictions() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    dvhm = load_v14_oof()
    global_model = pd.read_csv(GLOBAL_OOF_PATH)
    assert len(dvhm) == len(global_model) == 1_872
    y = dvhm["deposit_present"].to_numpy()
    assert np.array_equal(y, global_model["deposit_present"].to_numpy())
    assert np.array_equal(dvhm["fold"].to_numpy(), global_model["spatial_block"].to_numpy())
    return dvhm, global_model, y


def test_manuscript_key_numbers_match_frozen_v14_outputs():
    tex = _manuscript()
    dvhm, global_model, y = _predictions()

    dvhm_probability = dvhm["prob_v14"].to_numpy()
    global_probability = global_model["M7_All_Global_Prediction"].to_numpy()

    dvhm_auc = roc_auc_score(y, dvhm_probability)
    global_auc = roc_auc_score(y, global_probability)
    dvhm_ap = average_precision_score(y, dvhm_probability)
    global_ap = average_precision_score(y, global_probability)
    dvhm_brier = brier_score_loss(y, dvhm_probability)
    global_brier = brier_score_loss(y, global_probability)

    assert f"{dvhm_auc:.3f}" == "0.716"
    assert f"{global_auc:.3f}" == "0.674"
    assert f"{dvhm_ap:.3f}" == "0.137"
    assert f"{global_ap:.3f}" == "0.108"
    assert f"{dvhm_brier:.3f}" == "0.069"
    assert f"{global_brier:.3f}" == "0.079"

    assert "DVHM ROC-AUC is 0.716 (95\\% row-bootstrap CI 0.680--0.751)" in tex
    assert (
        "A predictor-matched global logistic model with the same four predictors "
        "provides the reference at 0.674"
    ) in tex
    assert "Pooled average precision is 0.137 for the DVHM and 0.108" in tex
    assert "Brier scores are 0.069 and 0.079" in tex
    assert "DVHM & 4 & Domain-varying & 0.716 & 0.137 & 0.069" in tex
    assert (
        "Global logistic reference & 4 & Single global set & 0.674 & 0.108 & 0.079"
    ) in tex

    fold_dvhm = []
    fold_global = []
    for fold in range(4):
        mask = dvhm["fold"].to_numpy() == fold
        fold_dvhm.append(roc_auc_score(y[mask], dvhm_probability[mask]))
        fold_global.append(roc_auc_score(y[mask], global_probability[mask]))

    assert f"{np.mean(fold_dvhm):.3f}" == "0.725"
    assert f"{np.mean(fold_global):.3f}" == "0.691"
    differences = [
        f"{value:+.3f}".replace("-", "-") for value in np.subtract(fold_dvhm, fold_global)
    ]
    assert differences == ["+0.071", "+0.048", "+0.015", "-0.001"]
    assert (
        "Relative to the reference model, fold-level differences vary along the belt. "
        "The DVHM-minus-reference fold differences in ROC-AUC are "
        "$+0.071$ $[-0.015,+0.162]$, $+0.048$ $[-0.049,+0.142]$, "
        "$+0.015$ $[-0.005,+0.033]$, and $-0.001$ $[-0.046,+0.050]$"
    ) in tex

    domain_values = {}
    for domain in ("NRB_3a", "NRB_3b"):
        mask = dvhm["daly_domain"].to_numpy() == domain
        domain_values[domain] = (
            roc_auc_score(y[mask], dvhm_probability[mask]),
            roc_auc_score(y[mask], global_probability[mask]),
        )
    assert f"{domain_values['NRB_3a'][0]:.3f}" == "0.679"
    assert f"{domain_values['NRB_3a'][1]:.3f}" == "0.675"
    assert f"{domain_values['NRB_3b'][0]:.3f}" == "0.667"
    assert f"{domain_values['NRB_3b'][1]:.3f}" == "0.718"
    assert "NRB\\_3a (ROC-AUC $0.679$ versus $0.675$)" in tex
    assert "NRB\\_3b ($0.667$ versus $0.718$)" in tex

    sensitivity = pd.read_csv(SENSITIVITY_PATH)
    means = sensitivity.groupby("model")[["auc", "pr_auc"]].mean()
    expected = {
        "A": (0.725, 0.126),
        "B": (0.707, 0.154),
        "C": (0.687, 0.178),
        "D": (0.722, 0.200),
    }
    for model, (auc, ap) in expected.items():
        assert f"{means.loc[model, 'auc']:.3f}" == f"{auc:.3f}"
        assert f"{means.loc[model, 'pr_auc']:.3f}" == f"{ap:.3f}"
    assert (
        "Mean ROC-AUC across folds is 0.725 (Model A), 0.707 (B), "
        "0.687 (C), and 0.722 (D)"
    ) in tex
    assert "mean average precision is 0.126, 0.154, 0.178, and 0.200" in tex

    tikz = FIG6_PATH.read_text(encoding="utf-8")
    assert "4.85/A/0.725/0.126,4.35/B/0.707/0.154" in tikz
    assert "3.85/C/0.687/0.178,3.35/D/0.722/0.200" in tikz


def test_all_reported_model_comparisons_are_predictor_matched():
    tex = _manuscript()
    visible = _visible_tex(tex)

    forbidden = [
        "CGLB",
        "compact global logistic",
        "compact CGLB",
        "two-predictor",
        "omits two predictors",
        "only gravity and contact distance",
    ]
    for phrase in forbidden:
        assert phrase.lower() not in visible.lower(), f"Unequal baseline remains: {phrase}"
    assert "0.557" not in visible, "Two-predictor comparator score remains in the paper."

    table_start = tex.index(r"\label{tab:sensitivity_design}")
    table_end = tex.index(r"\end{table}", table_start)
    sensitivity_table = tex[table_start:table_end]
    model_rows = [
        line
        for line in sensitivity_table.splitlines()
        if re.match(r"\s*(A \(DVHM\)|B|C|D)\s*&", line)
    ]
    assert len(model_rows) == 4
    for line in model_rows:
        assert re.search(r"&\s*4\s*&", line), f"Predictor count is not four: {line}"

    comparison_start = tex.index(r"\label{tab:comparator_metrics}")
    comparison_end = tex.index(r"\end{table}", comparison_start)
    comparison_table = tex[comparison_start:comparison_end]
    comparison_rows = [
        line
        for line in comparison_table.splitlines()
        if re.match(r"\s*(DVHM|Global logistic reference)\s*&", line)
    ]
    assert len(comparison_rows) == 2
    for line in comparison_rows:
        assert re.search(r"&\s*4\s*&", line), f"Predictor count is not four: {line}"

    assert (
        "All four models therefore use the same four input predictors and the "
        "same partially pooled domain hierarchy."
    ) in tex
    assert "No contrast changes the predictor set or the hierarchical structure:" in tex
    assert "The reference uses the same four predictors" in tex
    assert "All four specifications are hierarchical" in tex
    assert "same partially pooled domain hierarchy" in tex
    assert "Hierarchical quadratic specifications" in tex
    assert "Hierarchical linear specifications" in tex


def test_manuscript_has_no_internal_run_labels_or_unequal_versions():
    visible = _visible_tex(_manuscript())
    assert "The held-out sector is the primary inferential axis" in visible
    assert "fold transfer remains the primary analysis" in visible
    assert "The scientific comparison is therefore among held-out sectors" in visible
    for token in ("V11", "V14", "M5", "M7", "M8"):
        assert not re.search(rf"\b{token}\b", visible), f"Internal label {token} is visible."
    assert "0.707 (95" not in visible
    assert "0.744" not in visible


def test_contact_distance_wording_matches_canonical_v14_geometry():
    tex = _manuscript()
    manifest = json.loads(V14_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["dataset_version"] == "V14_CENTROID_CONTACT"
    assert "grid-cell centroid" in manifest["distance_definition"].lower()

    data = pd.read_csv(V14_DATA_PATH, usecols=["distance_to_lithology_contact"])
    distances = data["distance_to_lithology_contact"].dropna().to_numpy()
    assert len(distances) == 1_899
    assert np.all(distances > 0.0)
    assert f"{distances.min():.0f}" == "11"
    assert f"{distances.mean() / 1000.0:.1f}" == "16.3"
    assert f"{np.median(distances) / 1000.0:.1f}" == "12.0"

    assert (
        "minimum planar distance from each cell centroid to the nearest internal contact line"
        in tex
    )
    assert "The fitted table contains no zero contact distances" in tex
    assert "centroid-to-contact distance" in tex
    for stale in (
        "polygon-to-contact",
        "clipped cell polygon",
        "zero where a contact intersects",
        "contact predictor is measured from the cell polygon",
    ):
        assert stale not in tex, f"Stale contact-distance wording: {stale}"


def test_two_oof_maps_are_distinct_and_domain_named():
    tex = _manuscript()
    assert tex.count("images/fig4_oof_prospectivity.pdf") == 1
    assert tex.count("images/fig4_oof_uncertainty.pdf") == 1

    prospectivity_path = tex.index("images/fig4_oof_prospectivity.pdf")
    uncertainty_path = tex.index("images/fig4_oof_uncertainty.pdf")
    assert prospectivity_path < uncertainty_path

    prospectivity_start = tex.rfind(r"\begin{figure}", 0, prospectivity_path)
    prospectivity_end = tex.index(r"\end{figure}", prospectivity_path)
    uncertainty_start = tex.rfind(r"\begin{figure}", 0, uncertainty_path)
    uncertainty_end = tex.index(r"\end{figure}", uncertainty_path)
    prospectivity_caption = tex[prospectivity_start:prospectivity_end]
    uncertainty_caption = tex[uncertainty_start:uncertainty_end]

    assert "posterior mean conditional probability" in prospectivity_caption.lower()
    assert "credible-interval width" in uncertainty_caption.lower()
    assert "Larger widths denote greater posterior uncertainty" in uncertainty_caption
    for domain in ("NRB\\_3a", "NRB\\_3b", "CRZ\\_4b", "NKB", "MMSB", "SRB"):
        assert domain in prospectivity_caption
        assert domain in uncertainty_caption


def test_regional_context_figure_names_all_lithology_classes():
    tex = _manuscript()
    figure_start = tex.index("images/fig1_study_area_context_annotated.pdf")
    caption_start = tex.index(r"\caption{", figure_start)
    caption_end = tex.index(r"\label{", caption_start)
    caption = tex[caption_start:caption_end]
    expected = (
        "metasedimentary",
        "silicic",
        "metaigneous",
        "alluvium-fluvial",
        "alluvium-saline",
        "water",
    )
    for name in expected:
        assert name in caption.lower()
    assert "class code" not in caption.lower()


# --------------------------------------------------------------------------- #
# Passages that previously carried V13 numbers after the V14 refit.
BETA_V14_PATH = (
    ROOT / "figures" / "audit" / "beta_diagnostics_v14_centroid"
    / "beta_diagnostics_coefficients_by_fold.csv"
)
POPULATION_V14 = ROOT / "figures" / "audit" / "v14_centroid_population_reconstruction"
ROBUSTNESS_V14 = (
    ROOT / "figures" / "audit" / "distance_representation_sensitivity_v14_centroid"
    / "robustness_assessment.csv"
)
FIG5_PATH = ROOT / "paper_1" / "manuscript" / "images" / "fig5_predictor_contributions.tikz"


def test_coefficient_passage_uses_v14_traces():
    text = _visible_tex(_manuscript())
    beta = pd.read_csv(BETA_V14_PATH)
    nrb3a_contact = beta[(beta.Domain == "NRB_3a") & (beta.Variable == "lithology")]
    nrb3a_fault = beta[(beta.Domain == "NRB_3a") & (beta.Variable == "fault")]
    nrb3b_contact = beta[(beta.Domain == "NRB_3b") & (beta.Variable == "lithology")]
    expected = [
        f"${nrb3a_contact.beta_lin_median.min():.3f}$ to ${nrb3a_contact.beta_lin_median.max():.3f}$",
        f"${nrb3a_fault.beta_lin_median.min():.3f}$ to ${nrb3a_fault.beta_lin_median.max():.3f}$",
        f"{nrb3b_contact['P(beta_sq>0)'].min():.3f}--{nrb3b_contact['P(beta_sq>0)'].max():.3f}",
    ]
    for token in expected:
        assert token in text, token
    # V13 values that must not reappear.
    for stale in ("-0.977", "-2.622", "0.766--0.896", "+1.435", "+1.627"):
        assert stale not in text, stale


def test_share_table_and_figure5_use_v14_reconstruction():
    shares = pd.read_csv(POPULATION_V14 / "fold_predictor_summary.csv")
    text = _visible_tex(_manuscript())
    fig5 = FIG5_PATH.read_text(encoding="utf-8")
    for row in shares.itertuples():
        values = [row.fault_share_pct, row.lith_share_pct, row.gravity_share_pct, row.rocks_share_pct]
        table_row = f"{row.fold} & " + " & ".join(f"{v:.1f}\%" for v in values) + r" \\"
        assert table_row in text, table_row
        fig_row = f"{row.fold}/{row.fold}/" + "/".join(f"{v:.1f}" for v in values)
        assert fig_row in fig5, fig_row


def test_population_curvature_and_dstar_use_v14():
    d = pd.read_csv(POPULATION_V14 / "population_d_star_summary.csv")
    text = _visible_tex(_manuscript())
    fault = d[d.predictor == "distance_to_fault"]
    contact = d[d.predictor == "distance_to_lithology_contact"]
    assert f"from {fault.p_beta_sq_pos.min():.2f} to {fault.p_beta_sq_pos.max():.2f}" in text
    assert f"from {contact.p_beta_sq_pos.min():.2f} to {contact.p_beta_sq_pos.max():.2f}" in text
    assert f"[{fault.d_star_ci_lower_km.min():.0f}, {fault.d_star_ci_upper_km.max():.0f}]" in text
    assert f"[{contact.d_star_ci_lower_km.min():.0f}, {contact.d_star_ci_upper_km.max():.0f}]" in text


def test_robustness_counts_and_case_use_v14():
    r = pd.read_csv(ROBUSTNESS_V14)
    text = _visible_tex(_manuscript())
    assert (
        f"the curvature criterion was met in {int(r.crit1_pass.sum())}, the $D^*$ agreement "
        f"criterion in {int(r.crit2_pass.sum())}, and the training-support criterion in "
        f"{int(r.crit3_both_in_support.sum())}"
    ) in text
    robust = r[r.robust_all_criteria]
    assert len(robust) == 1
    case = robust.iloc[0]
    assert f"{case.domain.replace('_', chr(92) + '_')} lithological contact in the fit with Fold~{case.fold} held out" in text


def test_domain_names_follow_daly_2025():
    text = _visible_tex(_manuscript())
    for wrong in ("Northern Roan Basin", "Congo River Zone", "North Kundelungu Basin",
                  "Mwembeshi Shear Belt", "South Roan Basin"):
        assert wrong not in text, wrong
    for right in ("Northeastern Katangan Basin", "Mwashia marginal sub-basins",
                  "Southern Rift Basin", "Northern Rift Basin", "Central Rift Zone"):
        assert right in text, right


# --------------------------------------------------------------------------- #
# Area-aware ablation ladder (paper_1/v15): every reported number is file-backed.
V15 = ROOT / "paper_1" / "v15"


def test_ladder_table_matches_v15_metrics():
    text = _visible_tex(_manuscript())
    summary = pd.read_csv(V15 / "metrics" / "model_summary.csv").set_index("model")
    for label, model in (("M0", "M0_global"), ("M1", "M1_domain_intercept"),
                         ("M2", "M2_domain_distance"), ("M3$^{*}$", "M3_domain_distance_gravity")):
        r = summary.loc[model]
        cells = (f"{r.auc:.3f} & {r.pr_auc:.3f} & ${r.mean_log_score:.4f}$ & "
                 f"${r.brier_skill:+.3f}$ & {r.calibration_slope:.2f}")
        assert cells in text, (label, cells)
    rf = summary.loc["RF_comparator"]
    assert f"Random forest & -- & {rf.auc:.3f} & {rf.pr_auc:.3f} & ${rf.mean_log_score:.4f}$" in text


def test_ladder_comparisons_match_spatial_block_bootstrap():
    text = _visible_tex(_manuscript())
    b = pd.read_csv(V15 / "metrics" / "paired_bootstrap.csv")

    def row(left, right):
        r = b[(b.left == left) & (b.right == right)].iloc[0]
        return r

    # bootstrap stores left minus right; the manuscript reports upper minus lower.
    for lower, upper in (("M0_global", "M1_domain_intercept"),
                         ("M1_domain_intercept", "M2_domain_distance"),
                         ("M2_domain_distance", "M3_domain_distance_gravity")):
        r = row(lower, upper)
        token = (f"${-r.mean_delta_log_score:+.4f}$ & [${-r.ci_upper:+.4f}$, ${-r.ci_lower:+.4f}$]")
        assert token in text, token
    r = row("M3_domain_distance_gravity", "RF_comparator")
    assert f"${r.mean_delta_log_score:+.4f}$ & [${r.ci_lower:+.4f}$, ${r.ci_upper:+.4f}$]" in text


def test_area_and_gravity_auc_sentences_match_corrected_diagnostics():
    text = _visible_tex(_manuscript())
    area = pd.read_csv(V15 / "diagnostics" / "area_auc_by_fold.csv").set_index("fold")
    assert f"pooled ROC-AUC of {area.loc['pooled', 'auc_area']:.3f}" in text
    assert f"from {area.loc['0', 'auc_area']:.3f} in Fold~1 to {area.loc['3', 'auc_area']:.3f} in Fold~4" in text
    g = pd.read_csv(V15 / "diagnostics" / "gravity_fold_summary.csv")
    values = ", ".join(f"{v:.3f}" for v in g.gravity_auc_low_gravity_positive[:3])
    assert f"ROC-AUC {values}, and {g.gravity_auc_low_gravity_positive.iloc[3]:.3f}" in text
