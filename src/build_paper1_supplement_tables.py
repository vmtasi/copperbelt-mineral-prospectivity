"""Render Paper 1 supplementary tables from frozen and derived V14 outputs.

Writes LaTeX table bodies to paper_1/manuscript/supplement/tables/, which
supplementary.tex includes. Every number comes from a file, so the
supplement cannot drift from the benchmark by hand transcription.

Run after src.paper1_supplement_summaries and src.audit_lithology_class_assignment:
    python -m src.build_paper1_supplement_tables
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUPP = ROOT / "figures" / "audit" / "paper1_supplement_v14"
BETA = ROOT / "figures" / "audit" / "beta_diagnostics_v14_centroid" / "beta_diagnostics_coefficients_by_fold.csv"
OUT = ROOT / "paper_1" / "manuscript" / "supplement" / "tables"
DOMAIN_ORDER = ["NRB_3a", "NRB_3b", "CRZ", "NKB", "MMSB", "SRB"]
TEX_DOMAIN = {d: d.replace("_", r"\_") for d in DOMAIN_ORDER}
TEX_DOMAIN["CRZ"] = r"CRZ\_4b"


def num(x: float, digits: int = 2, sign: bool = True) -> str:
    text = f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"
    return f"${text}$"


def interval(med: float, lo: float, hi: float, digits: int = 2) -> str:
    return f"{num(med, digits)} [{num(lo, digits)}, {num(hi, digits)}]"


def write(name: str, lines: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_lithology() -> None:
    audit = json.loads((SUPP / "lithology_class_assignment.json").read_text(encoding="utf-8"))
    labels = {
        "H1_centroid_raster_class": "Raster class at the cell centroid",
        "H2_majority_class_in_cell": "Majority raster class within the cell",
        "H3_nearest_contact_to_centroid": "Class of the contact line nearest the centroid",
        "H4_any_tied_nearest_contact_to_polygon": "Class of any contact line at the minimum polygon-to-contact distance",
    }
    lines = [r"\begin{tabular}{p{0.62\textwidth}rrr}", r"\toprule",
             r"Candidate rule & Cells & Matches & Agreement \\", r"\midrule"]
    for key, label in labels.items():
        r = audit["rules"][key]
        lines.append(f"{label} & {r['cells']:,} & {r['matches']:,} & {100 * r['share']:.1f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("s2_lithology_rules.tex", lines)

    retained = pd.read_csv(SUPP / "lithology_retained_by_fold.csv")
    lines = [r"\begin{tabular}{llll}", r"\toprule",
             r"Fold & Reference class & Retained indicators & Pooled with reference \\", r"\midrule"]
    for row in retained.itertuples():
        lines.append(f"{row.fold} & {row.reference_class} & {row.retained_classes} & "
                     f"{row.excluded_classes_pooled_with_reference} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("s2_lithology_retained.tex", lines)


def table_coefficients() -> None:
    beta = pd.read_csv(BETA)
    beta["order"] = beta["Domain"].map({d: i for i, d in enumerate(DOMAIN_ORDER)})
    beta = beta.sort_values(["Variable", "order", "Fold"])
    lines = [r"\begin{tabular}{lllllr}", r"\toprule",
             r"Predictor & Domain & Fold & $\beta_{\mathrm{lin}}$ median [95\% CI] & "
             r"$\beta_{\mathrm{sq}}$ median [95\% CI] & $P(\beta_{\mathrm{sq}}>0)$ \\", r"\midrule"]
    previous = None
    for row in beta.itertuples():
        variable = "Fault" if row.Variable == "fault" else "Contact"
        if previous is not None and variable != previous:
            lines.append(r"\midrule")
        previous = variable
        lin = interval(row.beta_lin_median, getattr(row, "_5"), getattr(row, "_6"))
        sq = interval(row.beta_sq_median, getattr(row, "_8"), getattr(row, "_9"))
        p = getattr(row, "_10")
        lines.append(f"{variable} & {TEX_DOMAIN[row.Domain]} & {row.Fold} & {lin} & {sq} & {p:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("s3_coefficients.tex", lines)


def table_direction() -> None:
    d = pd.read_csv(SUPP / "direction_by_representation.csv")
    pivot = d.pivot_table(index=["domain", "predictor", "model"], columns="fold",
                          values="p_beta_lin_neg").reset_index()
    names = {"A": "A (raw, quadratic)", "B": "B (log, quadratic)",
             "C": "C (raw, linear)", "D": "D (log, linear)"}
    lines = [r"\begin{tabular}{lllrrrr}", r"\toprule",
             r"Domain & Predictor & Model & Fold 1 & Fold 2 & Fold 3 & Fold 4 \\", r"\midrule"]
    previous = None
    for row in pivot.itertuples(index=False):
        key = (row.domain, row.predictor)
        if previous is not None and key != previous:
            lines.append(r"\addlinespace")
        previous = key
        values = " & ".join(f"{getattr(row, f'_{k}'):.3f}" for k in range(3, 7))
        lines.append(f"{TEX_DOMAIN[row.domain]} & {row.predictor.capitalize()} & {names[row.model]} & {values} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("s4_direction.tex", lines)


def table_accuracy() -> None:
    a = json.loads((SUPP / "probability_accuracy.json").read_text(encoding="utf-8"))
    rows = [
        ("DVHM Brier score", f"{a['brier_dvhm']:.4f}"),
        ("Baseline: training-partition prevalence (out-of-fold)", f"{a['brier_baseline_oof_training_prevalence']:.4f}"),
        ("Baseline: pooled prevalence", f"{a['brier_baseline_pooled_prevalence']:.4f}"),
        ("Brier skill vs.\\ training-partition prevalence", num(a["brier_skill_vs_oof_baseline"], 3)),
        ("Brier skill vs.\\ pooled prevalence", num(a["brier_skill_vs_pooled_prevalence"], 3)),
        ("Calibration slope (logistic, on logit $\\bar p$)", f"{a['calibration_slope']:.3f}"),
        ("Calibration intercept (with slope)", num(a["calibration_slope_intercept"], 3)),
        ("Calibration-in-the-large (logit offset)", num(a["calibration_intercept_in_the_large"], 3)),
        ("Mean predicted probability / observed prevalence",
         f"{a['mean_predicted']:.4f} / {a['pooled_prevalence']:.4f}"),
    ]
    lines = [r"\begin{tabular}{lr}", r"\toprule", r"Quantity & Value \\", r"\midrule"]
    lines += [f"{k} & {v} \\\\" for k, v in rows]
    lines += [r"\midrule", r"\multicolumn{2}{l}{\emph{By held-out fold: Brier (baseline) and skill}} \\"]
    for fold, f in a["per_fold"].items():
        lines.append(f"Fold {fold} & {f['brier']:.4f} ({f['brier_baseline']:.4f}); {num(f['brier_skill'], 3)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("s5_accuracy.tex", lines)

    rel = pd.read_csv(SUPP / "reliability_bins.csv")
    lines = [r"\begin{tabular}{rrrrr}", r"\toprule",
             r"Decile & Cells & Mean predicted & Observed rate & Positive cells \\", r"\midrule"]
    for i, row in enumerate(rel.itertuples(), start=1):
        lines.append(f"{i} & {row.n} & {row.mean_predicted:.3f} & {row.observed_rate:.3f} & {int(row.positives)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("s5_reliability.tex", lines)


def table_uncertainty() -> None:
    u = pd.read_csv(SUPP / "uncertainty_by_domain.csv")
    dom = u[u["grouping"] == "domain"].copy()
    dom["order"] = dom["daly_domain"].map({d: i for i, d in enumerate(DOMAIN_ORDER)})
    dom = dom.sort_values("order")
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Group & Cells & Median & 90th pct. & Max & $>0.2$ & $>0.6$ \\", r"\midrule"]

    def row_text(label: str, r) -> str:
        return (f"{label} & {int(r.cells)} & {r.median_width:.3f} & {r.q90_width:.3f} & "
                f"{r.max_width:.3f} & {100 * r.share_width_gt_0_2:.1f}\\% & {100 * r.share_width_gt_0_6:.1f}\\% \\\\")

    for r in dom.itertuples():
        label = TEX_DOMAIN[r.daly_domain] + ("" if str(r.domain_has_positives) in ("True", "1", "1.0") else r"$^{\dagger}$")
        lines.append(row_text(label, r))
    lines.append(r"\midrule")
    status = u[u["grouping"] == "training_support"]
    labels = {
        (False, False): "Domain absent from training",
        (True, False): "In training, no positive cells",
        (True, True): "In training, with positive cells",
    }
    for r in status.itertuples():
        key = (str(r.domain_in_training) in ("True", "1", "1.0"), str(r.domain_has_positives) in ("True", "1", "1.0"))
        lines.append(row_text(labels[key], r))
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("s6_uncertainty.tex", lines)


def table_scales() -> None:
    s = pd.read_csv(SUPP / "scale_posteriors.csv")
    names = {
        "alpha_mu": r"$\mu_\alpha$", "alpha_sigma": r"$\sigma_\alpha$",
        "mu_f_lin": r"$\mu_{f,\mathrm{lin}}$", "sigma_f_lin": r"$\sigma_{f,\mathrm{lin}}$",
        "mu_f_sq": r"$\mu_{f,\mathrm{sq}}$", "sigma_f_sq": r"$\sigma_{f,\mathrm{sq}}$",
        "mu_l_lin": r"$\mu_{l,\mathrm{lin}}$", "sigma_l_lin": r"$\sigma_{l,\mathrm{lin}}$",
        "mu_l_sq": r"$\mu_{l,\mathrm{sq}}$", "sigma_l_sq": r"$\sigma_{l,\mathrm{sq}}$",
        "beta_grav": r"$\beta_g$",
    }
    lines = [r"\begin{tabular}{lllll}", r"\toprule",
             r"Parameter & Fold 1 & Fold 2 & Fold 3 & Fold 4 \\", r"\midrule"]
    for parameter, label in names.items():
        part = s[s["parameter"] == parameter].sort_values("fold")
        cells = " & ".join(interval(r.median, r.q025, r.q975) for r in part.itertuples())
        lines.append(f"{label} & {cells} \\\\")
    lines += [r"\midrule",
              f"\\multicolumn{{5}}{{l}}{{Minimum bulk ESS {s['ess_bulk'].min():,.0f}; "
              f"maximum $\\hat R$ {s['r_hat'].max():.3f} across these parameters and folds.}} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    write("s7_scales.tex", lines)


def table_dstar() -> None:
    j = json.loads((SUPP / "dstar_informative_subset.json").read_text(encoding="utf-8"))
    mark = {True: r"\checkmark", False: "--"}
    lines = [r"\begin{tabular}{lllrrcrrccc}", r"\toprule",
             r"Fold & Domain & Predictor & $P_A$ & $P_B$ & C1 & $D^*_A$ & $D^*_B$ & C2 & C3 & All \\",
             r"\midrule"]
    for c in sorted(j["informative_cases"], key=lambda c: (c["predictor"], c["domain"], c["fold"])):
        lines.append(
            f"{c['fold']} & {TEX_DOMAIN[c['domain']]} & {c['predictor'].replace('lith', 'contact').capitalize()} & "
            f"{c['p_beta_sq_pos_A']:.2f} & {c['p_beta_sq_pos_B']:.2f} & {mark[bool(c['crit1_pass'])]} & "
            f"{c['d_star_median_A_km']:.1f} & {c['d_star_median_B_km']:.1f} & {mark[bool(c['crit2_pass'])]} & "
            f"{mark[bool(c['crit3_both_in_support'])]} & {mark[bool(c['robust_all_criteria'])]} \\\\")
    a, i = j["all_48"], j["informative_16"]
    lines += [r"\midrule",
              f"\\multicolumn{{11}}{{l}}{{All 48 cases: C1 {a['criterion_1_curvature']}, C2 {a['criterion_2_dstar']}, "
              f"C3 {a['criterion_3_support']}, all three {a['all_three']}. "
              f"Informative 16: C1 {i['criterion_1_curvature']}, C2 {i['criterion_2_dstar']}, "
              f"C3 {i['criterion_3_support']}, all three {i['all_three']}.}} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    write("s8_dstar.tex", lines)


def main() -> None:
    table_lithology()
    table_coefficients()
    table_direction()
    table_accuracy()
    table_uncertainty()
    table_scales()
    table_dstar()
    print(f"wrote {len(list(OUT.glob('*.tex')))} tables to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
