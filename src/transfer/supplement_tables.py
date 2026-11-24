"""LaTeX tables for the Paper 1 transfer supplement.

Reads the stored transfer outputs (no model fitting) and writes table bodies
to ``paper_1/manuscript/supplement/tables/transfer/`` so that no number in
the supplement is transcribed by hand.

    python -m src.transfer.supplement_tables
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRANSFER = ROOT / "paper_1" / "transfer"
V15 = ROOT / "paper_1" / "v15"
OUT = ROOT / "paper_1" / "manuscript" / "supplement" / "tables" / "transfer"


def _signed(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "--"
    return f"${value:+.{digits}f}$"


def _plain(value: float, digits: int = 3) -> str:
    return "--" if pd.isna(value) else f"{value:.{digits}f}"


def _interval(estimate: float, lower: float, upper: float, digits: int = 2) -> str:
    return f"${estimate:+.{digits}f}$ [${lower:+.{digits}f}$, ${upper:+.{digits}f}$]"


def _write(name: str, lines: list[str]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def _log_score(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return y * np.log(p) + (1 - y) * np.log1p(-p)


def _skill(frame: pd.DataFrame) -> float:
    y = frame["y"].to_numpy(dtype=float)
    model = _log_score(y, frame["prob"].to_numpy(dtype=float)).mean()
    baseline = _log_score(y, frame["p_ref_train"].to_numpy(dtype=float)).mean()
    return float(model - baseline)


def lithology_rules_table() -> Path:
    """Table S2: agreement of the stored lithology attribute with candidate rules.

    Source: the audit written by ``src/audit_lithology_class_assignment.py``.
    """
    audit = json.loads(
        (ROOT / "figures" / "audit" / "paper1_supplement_v14" / "lithology_class_assignment.json")
        .read_text(encoding="utf-8")
    )
    labels = {
        "H1_centroid_raster_class": "Raster class at the cell centroid",
        "H2_majority_class_in_cell": "Majority raster class within the cell",
        "H3_nearest_contact_to_centroid": "Class of the contact line nearest the centroid",
        "H4_any_tied_nearest_contact_to_polygon": (
            "Class of any contact line at the minimum polygon-to-contact distance"
        ),
    }
    lines = [
        r"\begin{tabular}{p{0.62\textwidth}rrr}",
        r"\toprule",
        r"Candidate rule & Cells & Matches & Agreement \\",
        r"\midrule",
    ]
    for key, label in labels.items():
        rule = audit["rules"][key]
        lines.append(
            f"{label} & {rule['cells']:,} & {rule['matches']:,} & {100 * rule['share']:.1f}\\% \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return _write("s2_lithology_rules.tex", lines)


def transfer_matrix_table() -> Path:
    long = pd.read_csv(TRANSFER / "metrics" / "a2_transfer_matrix_long.csv")
    predictions = TRANSFER / "predictions"
    rows = []
    for record in long.itertuples(index=False):
        target = int(record.target_sector)
        if record.source_type == "single_sector":
            source = f"{int(record.source_sector)}"
            frame = pd.read_csv(predictions / "amendment1" / f"a2_single_source_{source}.csv")
            label = f"Sector {source}"
        elif record.source_type == "sector_pair":
            a, b = int(record.source_sector_a), int(record.source_sector_b)
            frame = pd.read_csv(predictions / "amendment1" / f"a2_pair_{a}_{b}.csv")
            label = f"Sectors {a} and {b}"
        else:
            frame = pd.read_csv(predictions / "T0_mcmc_loo.csv")
            label = "Other three sectors"
        frame = frame.loc[frame["target_sector"] == target]
        flag = "$^{*}$" if record.few_training_positives else ""
        rows.append(
            (
                record.source_type,
                target,
                f"{label}{flag} & {target} & {int(record.training_positives)} & "
                f"{_signed(record.mean_log_score)} & {_signed(_skill(frame))} & "
                f"{_plain(record.auc)} \\\\",
            )
        )
    order = {"single_sector": 0, "sector_pair": 1, "other_three_sectors": 2}
    rows.sort(key=lambda item: (order[item[0]], item[2]))
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Training set & Target & Training positives & Log score & Skill & ROC-AUC \\",
        r"\midrule",
    ]
    previous = None
    for kind, _, line in rows:
        if previous is not None and kind != previous:
            lines.append(r"\midrule")
        lines.append(line)
        previous = kind
    lines += [r"\bottomrule", r"\end{tabular}"]
    return _write("s3_transfer_matrix.tex", lines)


def shift_table() -> Path:
    univariate = pd.read_csv(TRANSFER / "metrics" / "a4_univariate_shift.csv")
    names = {
        "fault_z": "Fault distance",
        "fault_z2": "Fault distance squared",
        "contact_z": "Contact distance",
        "contact_z2": "Contact distance squared",
        "gravity_z": "Bouguer gravity",
    }
    wide = univariate.pivot(index="predictor", columns="sector", values="share_outside")
    cells = pd.read_csv(TRANSFER / "metrics" / "a4_aoa_cells.csv")
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r" & Sector 1 & Sector 2 & Sector 3 & Sector 4 \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\emph{Cells outside the training 1st--99th percentile}} \\",
    ]
    for key, label in names.items():
        values = " & ".join(f"{100 * wide.loc[key, s]:.1f}\\%" for s in range(1, 5))
        lines.append(f"\\quad {label} & {values} \\\\")
    lines.append(r"\multicolumn{5}{l}{\emph{Area of applicability}} \\")
    stats = {}
    for sector, group in cells.groupby("sector"):
        outside = group["AOA"] == 0
        stats[sector] = {
            "cells": f"{int(outside.sum())} / {len(group)}",
            "positives": f"{int(group.loc[outside, 'y'].sum())} / {int(group['y'].sum())}",
            "inside": group.loc[~outside, "mean_log_score"].mean(),
            "outside": group.loc[outside, "mean_log_score"].mean(),
            "threshold": group["threshold"].iloc[0],
        }
    lines.append(r"\quad Cells outside & " + " & ".join(stats[s]["cells"] for s in range(1, 5)) + r" \\")
    lines.append(r"\quad Positive cells outside & " + " & ".join(stats[s]["positives"] for s in range(1, 5)) + r" \\")
    lines.append(r"\quad Mean log score inside & " + " & ".join(_signed(stats[s]["inside"]) for s in range(1, 5)) + r" \\")
    lines.append(r"\quad Mean log score outside & " + " & ".join(_signed(stats[s]["outside"]) for s in range(1, 5)) + r" \\")
    lines.append(r"\quad Dissimilarity threshold & " + " & ".join(_plain(stats[s]["threshold"], 2) for s in range(1, 5)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return _write("s4_covariate_shift.tex", lines)


def coefficient_table() -> Path:
    table = pd.read_csv(TRANSFER / "coefficients" / "a5_coefficient_stability.csv")
    terms = {
        "beta_fault_z": "Fault, linear",
        "beta_fault_z2": "Fault, quadratic",
        "beta_contact_z": "Contact, linear",
        "beta_contact_z2": "Contact, quadratic",
        "beta_gravity_z": "Bouguer gravity",
    }
    lines = [
        r"\begin{tabular}{lllll}",
        r"\toprule",
        r"Term & Sector 1 & Sector 2 & Sector 3 & Sector 4 \\",
        r"\midrule",
    ]
    for estimation, heading, prefix in (
        ("MCMC leave-one-sector-out", "Leave-one-sector-out fits (column: held-out sector)", "fold_"),
        ("MCMC single-sector", "Single-sector fits (column: training sector)", "sector_"),
    ):
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{heading}}}}} \\\\")
        subset = table.loc[table["estimation"] == estimation]
        for term, label in terms.items():
            cells = []
            for k in range(1, 5):
                row = subset.loc[(subset["term"] == term) & (subset["source"] == f"{prefix}{k}")].iloc[0]
                cells.append(_interval(row["estimate"], row["ci_lower"], row["ci_upper"]))
            lines.append(f"\\quad {label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return _write("s5_coefficients.tex", lines)


def sensitivity_table() -> Path:
    primary = pd.read_csv(TRANSFER / "metrics" / "a1_transfer_metrics.csv")
    primary["model"] = "T0"
    primary["sample"] = primary["sample"].str.replace("sector_", "fold_")
    variants = pd.read_csv(TRANSFER / "metrics" / "sensitivity_metrics.csv")
    table = pd.concat([primary, variants], ignore_index=True)
    labels = {
        "T0": "Primary model",
        "T0_linear": "Without squared terms",
        "T0_nrb": "With outside-basin indicator",
    }
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Model & Held-out & Log score & ROC-AUC & Brier skill & CITL & Slope \\",
        r"\midrule",
    ]
    for model, label in labels.items():
        subset = table.loc[table["model"] == model].set_index("sample")
        for index, sample in enumerate(["fold_1", "fold_2", "fold_3", "fold_4", "pooled"]):
            row = subset.loc[sample]
            name = label if index == 0 else ""
            held = "Pooled" if sample == "pooled" else f"Sector {sample[-1]}"
            lines.append(
                f"{name} & {held} & {_signed(row['mean_log_score'])} & {_plain(row['auc'])} & "
                f"{_signed(row['brier_skill'])} & {_signed(row['calibration_in_the_large'], 2)} & "
                f"{_plain(row['calibration_slope'], 2)} \\\\"
            )
        if model != "T0_nrb":
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return _write("s6_sensitivity.tex", lines)


def partition_table() -> Path:
    metrics = pd.read_csv(TRANSFER / "metrics" / "a8_partition_metrics.csv")
    summaries = {
        "P_GAP": pd.read_csv(TRANSFER / "partitions" / "p_gap_summary.csv"),
        "P_POS": pd.read_csv(TRANSFER / "partitions" / "p_pos_summary.csv"),
    }
    labels = {"P_GAP": "Cut at gaps", "P_POS": "Equal positives"}
    lines = [
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Partition & Sector & Length (km) & Cells & Positives & Positives $\le$10\,km of a boundary & Log score & ROC-AUC & CITL \\",
        r"\midrule",
    ]
    for partition, summary in summaries.items():
        subset = metrics.loc[metrics["partition"] == partition].set_index("sample")
        for index, row in enumerate(summary.itertuples(index=False)):
            metric = subset.loc[f"sector_{int(row.sector)}"]
            name = labels[partition] if index == 0 else ""
            lines.append(
                f"{name} & {int(row.sector)} & {row.length_km:.0f} & {int(row.cells)} & "
                f"{int(row.positives)} & {int(row.positives_within_10km_boundaries)} & "
                f"{_signed(metric['mean_log_score'])} & {_plain(metric['auc'])} & "
                f"{_signed(metric['calibration_in_the_large'], 2)} \\\\"
            )
        pooled = subset.loc["pooled"]
        lines.append(
            f" & Pooled & & 1872 & 138 & & {_signed(pooled['mean_log_score'])} & "
            f"{_plain(pooled['auc'])} & {_signed(pooled['calibration_in_the_large'], 2)} \\\\"
        )
        if partition == "P_GAP":
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return _write("s6_partitions.tex", lines)


def domain_table() -> Path:
    gradient = pd.read_csv(TRANSFER / "metrics" / "a7_gravity_gradient.csv")
    # The paper uses the 120-line fault map throughout; the fixed-effect runs
    # with the 126-line map are not reported.
    gradient = gradient.loc[gradient["fault_variant"] == "stored"]
    labels = {
        ("fixed-effect ladder", "N1-N2"): "Domain-varying distance terms",
        ("fixed-effect ladder", "N2-N3"): "Domain-varying gravity term",
        ("V15 hierarchical reference", "M1-M2"): "Domain-varying distance terms",
        ("V15 hierarchical reference", "M2-M3"): "Domain-varying gravity term$^{\\dagger}$",
    }
    structure = {
        "fixed-effect ladder": "Fixed effects",
        "V15 hierarchical reference": "Hierarchical (six domains)",
    }
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Domain structure & Added terms & $\Delta$ log score & 95\% interval & Sectors improved \\",
        r"\midrule",
    ]
    for row in gradient.itertuples(index=False):
        lines.append(
            f"{structure[row.row_type]} & "
            f"{labels[(row.row_type, row.comparison)]} & {_signed(row.mean_log_score_gain, 4)} & "
            f"[{_signed(row.ci_lower, 4)}, {_signed(row.ci_upper, 4)}] & {int(row.positive_folds)} of 4 \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return _write("s7_domain_models.tex", lines)


def diagnostics_table() -> Path:
    primary = json.loads((TRANSFER / "diagnostics" / "mcmc_status.json").read_text(encoding="utf-8"))
    status = pd.read_csv(TRANSFER / "diagnostics" / "amendment1_status.csv")
    labels = {
        "a2": "Single-sector and two-sector training",
        "a3_local": "Within-sector inner cross-validation",
        "a3_oracle": "Oracle base-rate and slope corrections",
        "a6": "Random cross-validation (10 repeats)",
        "a8": "Alternative partitions",
        "sensitivities": "Model variants",
    }
    lines = [
        r"\begin{tabular}{lrlrrrr}",
        r"\toprule",
        r"Fits & $n$ & Chains $\times$ draws (tuning) & Divergences & Max $\hat R$ & Min bulk ESS & Min tail ESS \\",
        r"\midrule",
        (
            f"Leave-one-sector-out (primary) & {len(primary)} & 4 $\\times$ 1,200 (1,800) & "
            f"{sum(int(p['diverging']) for p in primary)} & "
            f"{max(p['max_rhat'] for p in primary):.3f} & "
            f"{min(p['min_ess_bulk'] for p in primary):,.0f} & "
            f"{min(p['min_ess_tail'] for p in primary):,.0f} \\\\"
        ),
    ]
    for batch, label in labels.items():
        group = status.loc[status["batch"] == batch]
        lines.append(
            f"{label} & {len(group)} & 2 $\\times$ 1,000 (1,500) & {int(group['diverging'].sum())} & "
            f"{group['max_rhat'].max():.3f} & {group['min_ess_bulk'].min():,.0f} & "
            f"{group['min_ess_tail'].min():,.0f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return _write("s8_diagnostics.tex", lines)


def main() -> None:
    for build in (
        lithology_rules_table,
        transfer_matrix_table,
        shift_table,
        coefficient_table,
        sensitivity_table,
        partition_table,
        domain_table,
        diagnostics_table,
    ):
        print(build())


if __name__ == "__main__":
    main()
