"""Numerical and structural checks between the Paper 1 manuscript and the
geographic transfer outputs in ``paper_1/transfer/``.

Each table row and key sentence is rebuilt from the stored metrics and must
appear verbatim in ``main.tex``, so a change in the outputs or a typing error
in the manuscript fails here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "paper_1" / "manuscript"
TEX_PATH = MANUSCRIPT / "main.tex"
SUPPLEMENT_PATH = MANUSCRIPT / "supplementary.tex"
TRANSFER = ROOT / "paper_1" / "transfer"
METRICS = TRANSFER / "metrics"
FRAME = ROOT / "paper_1" / "v15" / "data" / "v15_model_frame.csv"


def _manuscript() -> str:
    assert TEX_PATH.exists(), "main.tex must exist."
    return TEX_PATH.read_text(encoding="utf-8")


def _visible_tex(text: str) -> str:
    # Strip LaTeX comments but keep escaped percent signs (\%).
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in text.splitlines())


def _signed(value: float, digits: int = 3) -> str:
    return f"${value:+.{digits}f}$"


def _a1() -> pd.DataFrame:
    return pd.read_csv(METRICS / "a1_transfer_metrics.csv").set_index("sample")


SECTOR_LABELS = {1: "1 (SE)", 2: "2", 3: "3", 4: "4 (NW)"}


def test_sector_table_matches_frame_and_partition():
    tex = _manuscript()
    frame = pd.read_csv(FRAME, usecols=["fold", "deposit_present"])
    summary = pd.read_csv(TRANSFER / "partitions" / "primary_summary.csv").set_index("sector")
    assert len(frame) == 1872 and int(frame["deposit_present"].sum()) == 138
    for fold, group in frame.groupby("fold"):
        sector = int(fold) + 1
        positives = int(group["deposit_present"].sum())
        row = (
            f"{SECTOR_LABELS[sector]} & {summary.loc[sector, 'length_km']:.0f} & "
            f"{len(group)} & {positives} & {100 * positives / len(group):.1f}\\% &"
        )
        assert row in tex, row


def test_country_split_matches_border_layer():
    countries = (
        ROOT / "data/raw/rdc_rocks/derived/integration/phase7/supplementary"
        / "daly_country_context_v1.gpkg"
    )
    if not countries.exists():
        pytest.skip("Country layer is not distributed with the repository")
    gpd = pytest.importorskip("geopandas")
    tex = _manuscript()
    frame = pd.read_csv(FRAME, usecols=["fold", "deposit_present", "centroid_x", "centroid_y"])
    layer = gpd.read_file(countries, layer="countries")
    points = gpd.GeoDataFrame(
        frame, geometry=gpd.points_from_xy(frame.centroid_x, frame.centroid_y), crs=layer.crs
    )
    joined = gpd.sjoin(points, layer[["iso_a3", "geometry"]], how="left", predicate="within")
    counts = joined.groupby(["fold", "iso_a3"]).size().unstack(fill_value=0)
    for fold, row in counts.iterrows():
        assert f"& {int(row.get('ZMB', 0))} / {int(row.get('COD', 0))} \\\\" in tex
    totals = joined.groupby("iso_a3").agg(cells=("id", "size") if "id" in joined else ("fold", "size"),
                                          positives=("deposit_present", "sum"))
    zmb, cod = totals.loc["ZMB"], totals.loc["COD"]
    assert (
        f"{int(zmb.positives)} of the {int(zmb.cells)} cells in Zambia "
        f"({100 * zmb.positives / zmb.cells:.1f}\\%)" in tex
    )
    assert (
        f"{int(cod.positives)} of the {int(cod.cells):,} cells in the DRC "
        f"({100 * cod.positives / cod.cells:.1f}\\%)" in tex
    )


def test_leave_one_sector_out_table_matches_metrics():
    tex = _manuscript()
    a1 = _a1()
    for sample, label in [
        ("sector_1", SECTOR_LABELS[1]),
        ("sector_2", SECTOR_LABELS[2]),
        ("sector_3", SECTOR_LABELS[3]),
        ("sector_4", SECTOR_LABELS[4]),
        ("pooled", "Pooled"),
    ]:
        r = a1.loc[sample]
        row = (
            f"{label} & {r.auc:.3f} ({r.auc_ci_lower:.3f}--{r.auc_ci_upper:.3f}) & "
            f"{r.pr_auc:.3f} ({r.pr_auc_baseline:.3f}) & {_signed(r.mean_log_score)} & "
            f"{_signed(r.log_score_skill_vs_prevalence)} & "
            f"{r.mean_probability:.3f} / {r.observed_rate:.3f} & "
            f"{_signed(r.calibration_in_the_large, 2)} & {r.calibration_slope:.2f} \\\\"
        )
        assert row in tex, row


def test_abstract_and_results_sentences_match_metrics():
    tex = _manuscript()
    a1 = _a1()
    pooled = a1.loc["pooled"]
    sectors = a1.loc[a1.index.str.startswith("sector_")]
    random_cv = pd.read_csv(METRICS / "a6_optimism_comparison.csv").iloc[0]
    a3 = pd.read_csv(METRICS / "a3_decomposition.csv").set_index("sector")
    expected = [
        f"pooled ROC-AUC of {pooled.auc:.3f} (95\\% interval "
        f"{pooled.auc_ci_lower:.3f}--{pooled.auc_ci_upper:.3f})",
        f"sector values of {sectors.auc.min():.3f}--{sectors.auc.max():.3f}",
        f"against {random_cv.auc:.3f} under random cross-validation",
        f"mean predicted probability {sectors.loc['sector_1', 'mean_probability']:.3f} "
        f"against an observed rate of {sectors.loc['sector_1', 'observed_rate']:.3f}",
        f"({sectors.loc['sector_4', 'mean_probability']:.3f} against "
        f"{sectors.loc['sector_4', 'observed_rate']:.3f})",
        f"(+{a3.loc[1, 'base_minus_transfer']:.3f} log-score units per cell)",
        f"Pooled log-score skill was ${pooled.log_score_skill_vs_prevalence:+.3f}$ per cell",
        f"Calibration-in-the-large was ${sectors.loc['sector_1', 'calibration_in_the_large']:+.2f}$ "
        f"in Sector~1 and ${sectors.loc['sector_4', 'calibration_in_the_large']:+.2f}$ in Sector~4",
        f"mean log score of ${random_cv.mean_log_score:+.3f}$",
        f"ROC-AUC of {random_cv.auc:.3f} ({random_cv.auc_q2_5:.3f}--{random_cv.auc_q97_5:.3f})",
        f"The optimism of {random_cv.mean_log_score - pooled.mean_log_score:.3f} per cell",
    ]
    for text in expected:
        assert text in tex, text


def test_decomposition_table_matches_metrics():
    tex = _manuscript()
    a3 = pd.read_csv(METRICS / "a3_decomposition.csv").set_index("sector")

    gaps = ["base_minus_transfer", "recal_minus_base", "local_minus_recal"]
    for sector, row in a3.iterrows():
        # Estimates on one line, 95% intervals on the next.
        estimates = (
            f"{SECTOR_LABELS[int(sector)]} & {_signed(row.L_transfer)} & "
            + " & ".join(_signed(row[name]) for name in gaps)
        )
        intervals = " & & " + " & ".join(
            f"[{_signed(row[name + '_ci_lower'])}, {_signed(row[name + '_ci_upper'])}]"
            for name in gaps
        ) + " \\\\"
        assert estimates in tex, estimates
        assert estimates + ("$^{*}$" if sector == 1 else "") + " \\\\\n" + intervals in tex, intervals


def test_single_sector_skill_sentences_match_predictions():
    tex = _manuscript()
    skill = pd.read_csv(METRICS / "a2_single_sector_skill.csv").set_index(
        ["source_sector", "target_sector"]
    )["log_score_skill"]
    positive = skill[skill > 0]
    assert sorted(positive.index.tolist()) == [(1, 2), (2, 1), (3, 4)]
    for (source, target), value in positive.items():
        assert f"Sector~{source} to~{target} (${value:+.3f}$" in tex
    assert f"to ${skill[(1, 4)]:.2f}$ and ${skill[(4, 1)]:.2f}$ per cell" in tex
    descriptor = pd.read_json(METRICS / "a2_descriptor.json", typ="series")
    assert (
        f"was {descriptor['mean_absolute_directional_log_score_difference']:.3f} per cell" in tex
    )


def test_covariate_shift_sentences_match_area_of_applicability():
    tex = _manuscript()
    cells = pd.read_csv(METRICS / "a4_aoa_cells.csv")
    for sector, group in cells.groupby("sector"):
        outside = group["AOA"] == 0
        positives_outside = int(group.loc[outside, "y"].sum())
        assert re.search(rf"\b{positives_outside} of (the )?{int(group['y'].sum())}\b", tex)
    shares = cells.groupby("sector")["AOA"].apply(lambda s: round(100 * (s == 0).mean()))
    assert shares.loc[1] == shares.loc[2] == 64
    assert (
        f"{shares.loc[1]}\\% of cells in Sectors~1 and~2 lay outside the area of "
        f"applicability, against {shares.loc[3]}\\% in Sector~3 and {shares.loc[4]}\\% in Sector~4"
    ) in tex


def test_predictor_table_matches_marginal_auc_and_verdicts():
    tex = _manuscript()
    marginal = pd.read_csv(METRICS / "a5_marginal_auc.csv")
    verdicts = pd.read_csv(METRICS / "a8_partition_verdicts.csv")
    short = {"transfers": "transfers", "partly transfers": "partly", "does not transfer": "no"}
    # Columns are the predictors in this order; rows are sectors and verdicts.
    order = ["contact", "gravity", "fault"]
    row_labels = {1: "Sector 1 (SE)", 2: "Sector 2", 3: "Sector 3", 4: "Sector 4 (NW)"}
    for sector, label in row_labels.items():
        cells = []
        for predictor in order:
            r = marginal.loc[(marginal["predictor"] == predictor) & (marginal["sector"] == sector)].iloc[0]
            cells.append(f"{r.auc:.2f} ({r.auc_ci_lower:.2f}--{r.auc_ci_upper:.2f})")
        expected = f"{label} & " + " & ".join(cells) + " \\\\"
        assert expected in tex, expected
    table = verdicts.set_index(["predictor", "partition"])
    stable = ["yes" if table.loc[(p, "primary"), "coefficient_sign_stable"] else "no" for p in order]
    assert "Sign stable & " + " & ".join(stable) + " \\\\" in tex
    for partition, label in [("primary", "Primary partition"), ("P_GAP", "Cut at gaps"),
                             ("P_POS", "Equal positives")]:
        values = [short[table.loc[(p, partition), "verdict"]] for p in order]
        expected = f"{label} & " + " & ".join(values) + " \\\\"
        assert expected in tex, expected


def test_robustness_table_matches_metrics():
    tex = _manuscript()
    a1 = _a1()
    sensitivity = pd.read_csv(METRICS / "sensitivity_metrics.csv").set_index(["model", "sample"])
    partitions = pd.read_csv(METRICS / "a8_partition_metrics.csv").set_index(["partition", "sample"])
    random_cv = pd.read_csv(METRICS / "a6_optimism_comparison.csv").iloc[0]
    assert f"Random cross-validation & {_signed(random_cv.mean_log_score)} & {random_cv.auc:.3f}" in tex

    def row(label: str, pooled: pd.Series, s1: pd.Series, s4: pd.Series) -> str:
        return (
            f"{label} & {_signed(pooled.mean_log_score)} & {pooled.auc:.3f} & {s1.auc:.3f} & "
            f"{_signed(s1.calibration_in_the_large, 2)} & {_signed(s4.calibration_in_the_large, 2)} \\\\"
        )

    expected = [
        row("Primary model and partition", a1.loc["pooled"], a1.loc["sector_1"], a1.loc["sector_4"]),
    ]
    for model, label in [
        ("T0_linear", "\\quad Without squared terms"),
        ("T0_nrb", "\\quad With outside-basin indicator"),
    ]:
        expected.append(
            row(label, sensitivity.loc[(model, "pooled")], sensitivity.loc[(model, "fold_1")],
                sensitivity.loc[(model, "fold_4")])
        )
    for partition, label in [
        ("P_GAP", "\\quad Cut at gaps between positives"),
        ("P_POS", "\\quad Equal positives per sector"),
    ]:
        expected.append(
            row(label, partitions.loc[(partition, "pooled")], partitions.loc[(partition, "sector_1")],
                partitions.loc[(partition, "sector_4")])
        )
    for text in expected:
        assert text in tex, text


def test_convergence_and_gate_numbers_match_diagnostics():
    tex = _manuscript()
    primary = pd.read_json(TRANSFER / "diagnostics" / "mcmc_status.json")
    status = pd.read_csv(TRANSFER / "diagnostics" / "amendment1_status.csv")
    # The paper does not report the 126-line fault variant (4 fits).
    status = status.loc[status["model"] != "T0_rebuilt"]
    gate = pd.read_json(TRANSFER / "diagnostics" / "map_mcmc_validation.json", typ="series")
    assert len(primary) + len(status) == 102
    assert f"The {len(status)} further fits used two chains" in tex
    assert f"All {len(primary) + len(status)} fits met these criteria" in tex
    assert status["converged"].astype(bool).all() and (status["diverging"] == 0).all()
    assert (
        f"the maximum $\\hat R$ was {primary['max_rhat'].max():.3f} and the minimum bulk "
        f"effective sample size {primary['min_ess_bulk'].min():,.0f}"
    ) in tex
    assert (
        f"these were {status['max_rhat'].max():.3f} and {status['min_ess_bulk'].min():.0f}"
    ) in tex
    assert f"It differed by ${gate['mean_log_score_difference_map_minus_mcmc']:.4f}$ per cell" in tex


def test_manuscript_has_no_internal_labels():
    visible = _visible_tex(_manuscript())
    for token in ("V11", "V13", "V14", "V15", "DVHM", "T0", "M0", "M1", "M2", "M3",
                  "N1", "N2", "N3", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8",
                  "P-GAP", "P-POS", "P\\_GAP", "P\\_POS"):
        assert not re.search(rf"(?<![\w\\-]){re.escape(token)}(?![\w-])", visible), token
    for phrase in ("NRB\\_3a", "NRB\\_3b", "Amendment 1", "Phase 2", "ablation ladder", "MAP--MCMC"):
        assert phrase not in visible, phrase


def test_every_figure_and_supplement_table_exists():
    for path_text in (_manuscript(), SUPPLEMENT_PATH.read_text(encoding="utf-8")):
        for match in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", path_text):
            assert (MANUSCRIPT / match).exists(), match
        for match in re.findall(r"\\input\{([^}]+)\}", path_text):
            assert (MANUSCRIPT / match).exists(), match


def test_supplement_transfer_matrix_matches_metrics():
    table = (MANUSCRIPT / "supplement" / "tables" / "transfer" / "s3_transfer_matrix.tex").read_text(
        encoding="utf-8"
    )
    long = pd.read_csv(METRICS / "a2_transfer_matrix_long.csv")
    loo = long.loc[long["source_type"] == "other_three_sectors"]
    for row in loo.itertuples(index=False):
        assert (
            f"Other three sectors & {int(row.target_sector)} & {int(row.training_positives)} & "
            f"{_signed(row.mean_log_score)} &" in table
        )


def test_regional_context_figure_names_all_lithology_classes():
    tex = _manuscript()
    figure_start = tex.index("images/fig1_study_area_context_annotated.pdf")
    caption_start = tex.index(r"\caption{", figure_start)
    caption_end = tex.index(r"\label{", caption_start)
    caption = tex[caption_start:caption_end].lower()
    for name in ("metasedimentary", "silicic", "metaigneous", "alluvium-fluvial",
                 "alluvium-saline", "water"):
        assert name in caption
    assert "context only" in caption
