"""Publication figures for the Paper 1 transfer manuscript.

Reads only the stored transfer outputs (no model fitting) and writes vector
PDFs to ``paper_1/manuscript/images/transfer/``. Run with an environment
that has geopandas and matplotlib:

    python -m src.transfer.paper_figures [--preview DIR]

Colour use: sectors are ordered along the belt and are drawn with a
single-hue ordinal ramp, labelled on the map; categorical series use the
first three slots of a palette validated for all pairwise comparisons;
magnitude uses one sequential hue, and signed skill a blue-grey-red
diverging scale with grey at zero.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.colors import (  # noqa: E402
    LinearSegmentedColormap,
    ListedColormap,
    PowerNorm,
    TwoSlopeNorm,
)
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TRANSFER = ROOT / "paper_1" / "transfer"
OUT = ROOT / "paper_1" / "manuscript" / "images" / "transfer"
CELLS = ROOT / "figures" / "v14_centroid_oof_posterior_uncertainty.gpkg"
COUNTRIES = (
    ROOT
    / "data/raw/rdc_rocks/derived/integration/phase7/supplementary"
    / "daly_country_context_v1.gpkg"
)

# Validated palette (see the module docstring).
SECTOR_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SEQUENTIAL = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
DIVERGING_NEG = "#d03b3b"
DIVERGING_MID = "#f0efec"
DIVERGING_POS = "#1c5cab"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d9d8d4"

MM = 1 / 25.4
FULL_WIDTH = 129 * MM  # Springer full-width figure size

plt.rcParams.update(
    {
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)

HALO = [patheffects.withStroke(linewidth=2.0, foreground="white")]


def _cells() -> gpd.GeoDataFrame:
    cells = gpd.read_file(CELLS)[["id", "centroid_x", "centroid_y", "geometry"]]
    partitions = pd.read_csv(TRANSFER / "partitions" / "transfer_partitions.csv")
    frame = cells.merge(
        partitions[["id", "deposit_present", "fold_primary", "fold_p_gap", "fold_p_pos"]],
        on="id",
        how="inner",
        validate="one_to_one",
    )
    if len(frame) != 1872 or int(frame["deposit_present"].sum()) != 138:
        raise ValueError("Cell polygons do not match the 1,872-cell frame")
    return frame


def _border(frame: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """Zambia-DRC border segment within the frame extent."""
    countries = gpd.read_file(COUNTRIES, layer="countries").to_crs(frame.crs)
    zambia = countries.loc[countries["iso_a3"] == "ZMB", "geometry"].union_all()
    drc = countries.loc[countries["iso_a3"] == "COD", "geometry"].union_all()
    from shapely.geometry import box

    line = zambia.boundary.intersection(drc.buffer(50))
    xmin, ymin, xmax, ymax = frame.total_bounds
    clip = box(xmin - 20000, ymin - 20000, xmax + 20000, ymax + 20000)
    return gpd.GeoSeries([line.intersection(clip)], crs=frame.crs)


def _map_axes(ax: plt.Axes, frame: gpd.GeoDataFrame) -> None:
    xmin, ymin, xmax, ymax = frame.total_bounds
    pad = 15000
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID)


def _scale_bar(ax: plt.Axes, frame: gpd.GeoDataFrame, length_m: float = 100000) -> None:
    xmin, ymin, _, _ = frame.total_bounds
    x0, y0 = xmin, ymin - 5000
    ax.plot([x0, x0 + length_m], [y0, y0], color=INK, linewidth=1.2, solid_capstyle="butt")
    ax.text(x0 + length_m / 2, y0 + 6000, f"{length_m / 1000:.0f} km", ha="center",
            va="bottom", fontsize=6.5, color=INK)


def _positives(ax: plt.Axes, frame: gpd.GeoDataFrame, size: float = 3.0) -> None:
    positive = frame.loc[frame["deposit_present"] == 1]
    ax.scatter(positive["centroid_x"], positive["centroid_y"], s=size, c=INK,
               marker="o", linewidths=0, zorder=4)


def figure_partitions(frame: gpd.GeoDataFrame, border: gpd.GeoSeries) -> plt.Figure:
    specs = [
        ("fold_primary", "a  Equal cells (primary)"),
        ("fold_p_gap", "b  Cut at gaps"),
        ("fold_p_pos", "c  Equal positives"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 0.52 * FULL_WIDTH))
    cmap = ListedColormap(SECTOR_RAMP)
    for ax, (column, title) in zip(axes, specs):
        frame.plot(ax=ax, column=column, cmap=cmap, vmin=-0.5, vmax=3.5,
                   linewidth=0, edgecolor="none")
        dissolved = frame.dissolve(by=column)
        dissolved.boundary.plot(ax=ax, color="white", linewidth=0.5)
        border.plot(ax=ax, color=INK_2, linewidth=0.7, linestyle=(0, (3, 2)))
        _positives(ax, frame)
        for sector, row in dissolved.iterrows():
            point = row.geometry.representative_point()
            positives = int(frame.loc[frame[column] == sector, "deposit_present"].sum())
            ax.text(point.x, point.y, f"{int(sector) + 1}\n({positives})", ha="center",
                    va="center", fontsize=6.5, color=INK, fontweight="bold",
                    path_effects=HALO, zorder=5, linespacing=0.9)
        _map_axes(ax, frame)
        ax.set_title(title, loc="left", color=INK, fontsize=7.5)
    _scale_bar(axes[0], frame)
    handles = [Patch(facecolor=c, label=f"Sector {i + 1}") for i, c in enumerate(SECTOR_RAMP)]
    handles += [
        Line2D([], [], marker="o", linestyle="none", markersize=2.5, color=INK,
               label="Positive cell"),
        Line2D([], [], color=INK_2, linewidth=0.7, linestyle=(0, (3, 2)),
               label="Zambia–DRC border"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.10), handlelength=1.4, columnspacing=1.2,
               fontsize=6.5)
    fig.subplots_adjust(wspace=0.03, bottom=0.12, left=0.0, right=1.0)
    return fig


def figure_heldout_map(frame: gpd.GeoDataFrame, border: gpd.GeoSeries) -> plt.Figure:
    loo = pd.read_csv(TRANSFER / "predictions" / "T0_mcmc_loo.csv")[["id", "prob"]]
    aoa = pd.read_csv(TRANSFER / "metrics" / "a4_aoa_cells.csv")[["id", "AOA"]]
    data = frame.merge(loo, on="id", validate="one_to_one").merge(
        aoa, on="id", validate="one_to_one"
    )
    sectors = frame.dissolve(by="fold_primary")
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 0.62 * FULL_WIDTH))

    cmap = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=float(np.ceil(data["prob"].max() * 10) / 10))
    data.plot(ax=axes[0], column="prob", cmap=cmap, norm=norm, linewidth=0)
    axes[0].set_title("a  Held-out probability", loc="left")
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[0], orientation="horizontal",
        fraction=0.04, pad=0.02, aspect=35,
    )
    colorbar.set_label("Posterior mean probability", fontsize=6.5)
    colorbar.outline.set_edgecolor(GRID)

    inside = data["AOA"] == 1
    data.loc[inside].plot(ax=axes[1], color=SERIES[0], linewidth=0)
    data.loc[~inside].plot(ax=axes[1], color=SERIES[1], linewidth=0)
    axes[1].set_title("b  Area of applicability", loc="left")
    axes[1].legend(
        handles=[
            Patch(facecolor=SERIES[0], label="Inside"),
            Patch(facecolor=SERIES[1], label="Outside"),
            Line2D([], [], marker="o", linestyle="none", markersize=2.5, color=INK,
                   label="Positive cell"),
        ],
        loc="center left", bbox_to_anchor=(0.0, 0.42), frameon=False, fontsize=6.5,
    )
    for ax in axes:
        border.plot(ax=ax, color=INK_2, linewidth=0.7, linestyle=(0, (3, 2)))
        _positives(ax, data, size=2.5)
        for sector, row in sectors.iterrows():
            point = row.geometry.representative_point()
            ax.text(point.x, point.y, f"S{int(sector) + 1}", ha="center", va="center",
                    fontsize=7, fontweight="bold", color=INK, path_effects=HALO, zorder=5)
        _map_axes(ax, frame)
    _scale_bar(axes[0], frame)
    fig.subplots_adjust(wspace=0.05)
    return fig


def figure_sector_transfer() -> plt.Figure:
    a1 = pd.read_csv(TRANSFER / "metrics" / "a1_transfer_metrics.csv")
    sectors = a1.loc[a1["sample"].str.startswith("sector_")].copy()
    sectors["sector"] = sectors["sample"].str.replace("sector_", "").astype(int)
    pooled = a1.loc[a1["sample"] == "pooled"].iloc[0]
    x = sectors["sector"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 0.42 * FULL_WIDTH))
    ax = axes[0]
    ax.axhline(0.5, color=GRID, linewidth=0.8, zorder=0)
    ax.axhline(pooled["auc"], color=INK_2, linewidth=0.7, linestyle=(0, (3, 2)), zorder=0)
    ax.text(0.55, pooled["auc"] - 0.01, f"pooled\n{pooled['auc']:.3f}", fontsize=6,
            color=INK_2, va="top", ha="left")
    ax.errorbar(x, sectors["auc"], yerr=[sectors["auc"] - sectors["auc_ci_lower"],
                sectors["auc_ci_upper"] - sectors["auc"]], fmt="o", color=SERIES[0],
                ecolor=SERIES[0], elinewidth=1.0, capsize=0, markersize=4.5)
    for xi, value, upper in zip(x, sectors["auc"], sectors["auc_ci_upper"]):
        ax.text(xi, upper + 0.012, f"{value:.3f}", fontsize=6, color=INK, ha="center",
                va="bottom")
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("a  Ranking transfers", loc="left")

    ax = axes[1]
    for xi, observed, predicted in zip(x, sectors["observed_rate"], sectors["mean_probability"]):
        ax.plot([xi, xi], [observed, predicted], color=GRID, linewidth=1.5, zorder=1)
    ax.scatter(x, sectors["observed_rate"], s=20, color=INK, zorder=3, label="Observed")
    ax.scatter(x, sectors["mean_probability"], s=20, color=SERIES[1], zorder=3,
               label="Predicted (mean)")
    ax.set_ylim(0, 0.16)
    ax.set_ylabel("Proportion of cells")
    ax.set_title("b  Base rates do not transfer", loc="left")
    ax.legend(frameon=False, loc="center", bbox_to_anchor=(0.52, 0.72), fontsize=6.5,
              handletextpad=0.2)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([f"{i}" + ("\n(SE)" if i == 1 else "\n(NW)" if i == 4 else "")
                            for i in x])
        ax.set_xlabel("Held-out sector")
        ax.set_xlim(0.5, 4.95)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.subplots_adjust(wspace=0.28)
    return fig


def single_sector_skill() -> pd.DataFrame:
    """Log-score skill of single-sector transfer against the source prevalence."""
    rows = []
    for path in sorted((TRANSFER / "predictions" / "amendment1").glob("a2_single_source_*.csv")):
        table = pd.read_csv(path)
        for target, group in table.groupby("target_sector"):
            y = group["y"].to_numpy(dtype=float)
            p = np.clip(group["prob"].to_numpy(dtype=float), 1e-12, 1 - 1e-12)
            ref = np.clip(group["p_ref_train"].to_numpy(dtype=float), 1e-12, 1 - 1e-12)
            model = np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))
            baseline = np.mean(y * np.log(ref) + (1 - y) * np.log1p(-ref))
            rows.append(
                {
                    "source_sector": int(group["source_sector"].iloc[0]),
                    "target_sector": int(target),
                    "mean_log_score": model,
                    "baseline_log_score": baseline,
                    "log_score_skill": model - baseline,
                }
            )
    return pd.DataFrame(rows)


def _skill_limit(matrix: pd.DataFrame) -> float:
    """Symmetric colour limit, so equal gains and losses get equal saturation."""
    return float(np.ceil(np.nanmax(np.abs(matrix.to_numpy(dtype=float))) * 10) / 10)


def figure_transfer_matrix() -> tuple[plt.Figure, pd.DataFrame]:
    long = pd.read_csv(TRANSFER / "metrics" / "a2_transfer_matrix_long.csv")
    single = long.loc[long["source_type"] == "single_sector"]
    auc = single.pivot(index="source_sector", columns="target_sector", values="auc")
    skill = single_sector_skill()
    skill_matrix = skill.pivot(index="source_sector", columns="target_sector",
                               values="log_score_skill")
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 0.42 * FULL_WIDTH))
    panels = [
        (axes[0], auc, LinearSegmentedColormap.from_list("seq", SEQUENTIAL),
         plt.Normalize(0.5, 1.0), "a  ROC-AUC", "{:.2f}", "ROC-AUC"),
        (axes[1], skill_matrix,
         LinearSegmentedColormap.from_list("div", [DIVERGING_NEG, DIVERGING_MID, DIVERGING_POS]),
         TwoSlopeNorm(vcenter=0.0, vmin=-_skill_limit(skill_matrix),
                      vmax=_skill_limit(skill_matrix)),
         "b  Log-score skill per cell", "{:+.2f}", None),
    ]
    for index, (ax, matrix, cmap, norm, title, fmt, label) in enumerate(panels):
        values = matrix.reindex(index=[1, 2, 3, 4], columns=[1, 2, 3, 4]).to_numpy(dtype=float)
        image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm)
        for i in range(4):
            for j in range(4):
                if i == j:
                    ax.text(j, i, "–", ha="center", va="center", color=INK_2)
                    continue
                value = values[i, j]
                rgba = cmap(norm(value))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=6.3,
                        color="white" if luminance < 0.5 else INK)
        ax.set_xticks(range(4), [f"{k}" for k in range(1, 5)])
        ax.set_yticks(range(4), [f"{k}" + ("*" if k == 1 else "") for k in range(1, 5)])
        ax.set_xlabel("Target sector")
        if index == 0:
            ax.set_ylabel("Training sector")
        ax.set_title(title, loc="left", fontsize=7.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.ax.tick_params(labelsize=6)
        colorbar.outline.set_edgecolor(GRID)
    fig.subplots_adjust(wspace=0.35)
    return fig, skill


def figure_decomposition() -> plt.Figure:
    a3 = pd.read_csv(TRANSFER / "metrics" / "a3_decomposition.csv")
    components = [
        ("base_minus_transfer", "Correct the base rate"),
        ("recal_minus_base", "Also rescale the slopes"),
        ("local_minus_recal", "Refit within the sector"),
    ]
    fig, ax = plt.subplots(figsize=(0.62 * FULL_WIDTH, 0.40 * FULL_WIDTH))
    width = 0.25
    x = a3["sector"].to_numpy(dtype=float)
    for k, (column, label) in enumerate(components):
        offset = (k - 1) * (width + 0.02)
        value = a3[column].to_numpy()
        lower = a3[f"{column}_ci_lower"].to_numpy()
        upper = a3[f"{column}_ci_upper"].to_numpy()
        bars = ax.bar(x + offset, value, width=width, color=SERIES[k], label=label,
                      edgecolor="white", linewidth=0.5)
        ax.errorbar(x + offset, value, yerr=[value - lower, upper - value], fmt="none",
                    ecolor=INK, elinewidth=0.7, capsize=0)
        if column == "local_minus_recal":
            bars[0].set_hatch("////")
            bars[0].set_edgecolor("white")
    ax.axhline(0, color=INK_2, linewidth=0.7)
    ax.set_xticks(x, [f"{int(s)}" + ("\n(SE)" if s == 1 else "\n(NW)" if s == 4 else "")
                      for s in x])
    ax.set_xlabel("Held-out sector")
    ax.set_ylabel("Log-score gain per cell")
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper right", title="Oracle step (uses the target labels)",
              title_fontsize=6.5)
    return fig


def figure_predictors() -> plt.Figure:
    marginal = pd.read_csv(TRANSFER / "metrics" / "a5_marginal_auc.csv")
    predictors = [
        ("contact", "Contact distance (closer)"),
        ("gravity", "Bouguer gravity (lower)"),
        ("fault", "Fault distance (closer)"),
    ]
    fig, ax = plt.subplots(figsize=(0.62 * FULL_WIDTH, 0.40 * FULL_WIDTH))
    ax.axhline(0.5, color=INK_2, linewidth=0.7)
    for k, (name, label) in enumerate(predictors):
        rows = marginal.loc[marginal["predictor"] == name].sort_values("sector")
        x = rows["sector"].to_numpy(dtype=float) + (k - 1) * 0.2
        ax.errorbar(x, rows["auc"], yerr=[rows["auc"] - rows["auc_ci_lower"],
                    rows["auc_ci_upper"] - rows["auc"]], fmt="o", color=SERIES[k],
                    ecolor=SERIES[k], elinewidth=1.0, capsize=0, markersize=4.5, label=label)
    ax.set_xticks([1, 2, 3, 4], ["1\n(SE)", "2", "3", "4\n(NW)"])
    ax.set_xlabel("Sector")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Within-sector ROC-AUC")
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=6.5)
    ax.text(4.45, 0.5, "no\nassociation", fontsize=6.3, color=INK_2, va="center")
    ax.set_xlim(0.6, 4.95)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", type=Path, help="also write PNG previews here")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    frame = _cells()
    border = _border(frame)
    figures = {
        "fig2_partitions": figure_partitions(frame, border),
        "fig3_heldout_map": figure_heldout_map(frame, border),
        "fig4_sector_transfer": figure_sector_transfer(),
        "fig6_decomposition": figure_decomposition(),
        "fig7_predictors": figure_predictors(),
    }
    matrix_figure, skill = figure_transfer_matrix()
    figures["fig5_transfer_matrix"] = matrix_figure
    skill.to_csv(TRANSFER / "metrics" / "a2_single_sector_skill.csv", index=False)
    for name, figure in figures.items():
        figure.savefig(OUT / f"{name}.pdf")
        if args.preview:
            args.preview.mkdir(parents=True, exist_ok=True)
            figure.savefig(args.preview / f"{name}.png", dpi=170)
        plt.close(figure)
        print(OUT / f"{name}.pdf")


if __name__ == "__main__":
    main()
