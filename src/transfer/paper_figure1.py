"""Build the study-area context map as the sole map in Figure 1.

The figure uses the stored modelling frame and source layers only. Tectonic
domains are shown as faint context lines; they do not define or colour the
modelling frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from shapely.geometry import box  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "paper_1" / "manuscript" / "images" / "fig1_study_area_context_annotated.pdf"

CELLS = ROOT / "figures" / "v14_centroid_oof_posterior_uncertainty.gpkg"
TRACT_LAYER = (
    ROOT
    / "data/raw/rdc_rocks/derived/spatial_atlas_v1/layers/usgs_permissive_tracts.gpkg"
)
OCCURRENCE_LAYER = (
    ROOT / "data/raw/rdc_rocks/derived/spatial_atlas_v1/layers/usgs_deposits.gpkg"
)
FAULT_LAYER = (
    ROOT / "data/raw/selley_2005/Selley_2005_faults_shear_zones_compilation.gpkg"
)
CONTACT_LAYER = ROOT / "data/raw/africa_surficial_lithology/internal_contacts_only.gpkg"
DOMAIN_LAYER = ROOT / "data/raw/rdc_rocks/derived/spatial_atlas_v1/layers/daly_domains.gpkg"
COUNTRY_LAYER = (
    ROOT
    / "data/raw/rdc_rocks/derived/integration/phase7/supplementary"
    / "daly_country_context_v1.gpkg"
)

MM = 1 / 25.4
WIDTH = 129 * MM
HEIGHT = 112 * MM

LAND = "#f8f5ee"
LAND_ZAMBIA = "#f7f0e3"
LAND_DRC = "#f3f2ed"
TRACT_FILL = "#ece4d4"
TRACT_EDGE = "#766a5b"
DOMAIN = "#a5a7a4"
CONTACT = "#9dbebb"
FAULT = "#944a3f"
BORDER = "#1e272c"
INK = "#101315"
INK_2 = "#51575c"
GRID = "#e8e3d9"

DEPOSIT = "#7a342d"
PROSPECT = "#c98a24"
GROUP = "#5d5684"


def _load_layers() -> dict[str, gpd.GeoDataFrame]:
    frame = gpd.read_file(CELLS)
    crs = frame.crs
    if len(frame) != 1872:
        raise ValueError(f"Expected the 1,872-cell modelling frame, found {len(frame)}")

    layers = {
        "frame": frame,
        "tracts": gpd.read_file(TRACT_LAYER, layer="usgs_permissive_tracts").to_crs(crs),
        "occurrences": gpd.read_file(OCCURRENCE_LAYER, layer="usgs_deposits").to_crs(crs),
        "faults": gpd.read_file(FAULT_LAYER, layer="faults_shear_zones").to_crs(crs),
        "contacts": gpd.read_file(CONTACT_LAYER).to_crs(crs),
        "domains": gpd.read_file(DOMAIN_LAYER, layer="daly_domains").to_crs(crs),
        "countries": gpd.read_file(COUNTRY_LAYER, layer="countries").to_crs(crs),
    }
    if len(layers["occurrences"]) != 187:
        raise ValueError("Occurrence layer does not contain the expected 187 records")
    if len(layers["faults"]) != 126:
        raise ValueError("Fault layer does not contain the expected 126 segments")
    if len(layers["tracts"]) != 5:
        raise ValueError("Expected five USGS permissive tracts")
    return layers


def _zambia_drc_border(countries: gpd.GeoDataFrame, clip_geometry) -> gpd.GeoSeries:
    zambia = countries.loc[countries["iso_a3"] == "ZMB", "geometry"].union_all()
    drc = countries.loc[countries["iso_a3"] == "COD", "geometry"].union_all()
    line = zambia.boundary.intersection(drc.buffer(50)).intersection(clip_geometry)
    return gpd.GeoSeries([line], crs=countries.crs)


def _map_axes(ax: plt.Axes, bounds: tuple[float, float, float, float]) -> None:
    xmin, ymin, xmax, ymax = bounds
    pad = 12000
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal")
    x0 = np.ceil((xmin - pad) / 100000) * 100000
    y0 = np.ceil((ymin - pad) / 100000) * 100000
    ax.set_xticks(np.arange(x0, xmax + pad, 100000))
    ax.set_yticks(np.arange(y0, ymax + pad, 100000))
    ax.set_xticklabels([f"{value / 1000:.0f}" for value in ax.get_xticks()], fontsize=6)
    ax.set_yticklabels([f"{value / 1000:.0f}" for value in ax.get_yticks()], fontsize=6)
    ax.tick_params(length=2.2, pad=2, color=INK_2)
    ax.set_xlabel("UTM easting (km)", labelpad=3)
    ax.set_ylabel("UTM northing (km)", labelpad=3)
    ax.set_facecolor(LAND)
    ax.grid(color=GRID, linewidth=0.45, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#aeb1ad")
        spine.set_linewidth(0.7)


def _scale_bar(ax: plt.Axes, bounds: tuple[float, float, float, float]) -> None:
    xmin, ymin, _, _ = bounds
    x0, y0 = xmin + 12000, ymin - 4000
    length = 100000
    ax.plot(
        [x0, x0 + length],
        [y0, y0],
        color=INK,
        linewidth=1.4,
        solid_capstyle="butt",
        zorder=20,
    )
    ax.plot(
        [x0, x0],
        [y0 - 2600, y0 + 2600],
        color=INK,
        linewidth=0.8,
        zorder=20,
    )
    ax.plot(
        [x0 + length, x0 + length],
        [y0 - 2600, y0 + 2600],
        color=INK,
        linewidth=0.8,
        zorder=20,
    )
    ax.text(
        x0 + length / 2,
        y0 + 4500,
        "100 km",
        ha="center",
        va="bottom",
        fontsize=6.3,
        color=INK,
        zorder=20,
    )


def _north_arrow(ax: plt.Axes) -> None:
    ax.annotate(
        "N",
        xy=(0.945, 0.635),
        xytext=(0.945, 0.535),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=6.2,
        color=INK,
        arrowprops={
            "arrowstyle": "-|>",
            "facecolor": INK,
            "edgecolor": INK,
            "linewidth": 0.75,
            "mutation_scale": 8,
        },
        zorder=25,
    )


def _occurrences(ax: plt.Axes, occurrences: gpd.GeoDataFrame) -> None:
    styles = {
        "deposit": dict(s=17, marker="o", color=DEPOSIT, linewidths=0.35),
        "prospect": dict(s=8, marker="o", color=PROSPECT, linewidths=0.28),
        "group": dict(s=13, marker="s", color=GROUP, linewidths=0.30),
    }
    for status, style in styles.items():
        subset = occurrences.loc[occurrences["site_status"] == status]
        ax.scatter(
            subset.geometry.x,
            subset.geometry.y,
            edgecolors="white",
            zorder=12,
            **style,
        )


def _locator(
    ax: plt.Axes,
    countries: gpd.GeoDataFrame,
    frame: gpd.GeoDataFrame,
) -> None:
    inset = ax.inset_axes([0.735, 0.705, 0.245, 0.255], zorder=30)
    inset.set_facecolor("#fbfaf6")
    geographic = countries.to_crs("EPSG:4326")
    geographic.plot(ax=inset, facecolor="#eeede7", edgecolor="#aaa9a3", linewidth=0.3)
    highlight = geographic.loc[geographic["iso_a3"].isin(["ZMB", "COD"])]
    highlight.plot(ax=inset, facecolor="#d8d1c2", edgecolor="#565b5d", linewidth=0.45)

    study_geographic = frame.dissolve().to_crs("EPSG:4326")
    study_geographic.boundary.plot(ax=inset, color=FAULT, linewidth=0.85)
    xmin, ymin, xmax, ymax = geographic.total_bounds
    inset.set_xlim(xmin - 0.8, xmax + 0.8)
    inset.set_ylim(ymin - 0.8, ymax + 0.8)
    inset.set_aspect("equal")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_xlabel("")
    inset.set_ylabel("")
    inset.set_title("Location", fontsize=5.5, color=INK, pad=1.4, fontweight="bold")
    for spine in inset.spines.values():
        spine.set_color("#9ca09f")
        spine.set_linewidth(0.55)


def build_figure() -> plt.Figure:
    layers = _load_layers()
    frame = layers["frame"]
    bounds = tuple(frame.total_bounds)
    xmin, ymin, xmax, ymax = bounds
    clip_geometry = box(xmin - 20000, ymin - 20000, xmax + 20000, ymax + 20000)

    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(4.25, 1.72),
        left=0.10,
        right=0.985,
        bottom=0.055,
        top=0.985,
        wspace=0.055,
    )
    ax = fig.add_subplot(grid[0, 0])
    key_ax = fig.add_subplot(grid[0, 1])
    key_ax.axis("off")

    countries = layers["countries"]
    visible_countries = gpd.clip(countries, clip_geometry)
    for code, color in (("ZMB", LAND_ZAMBIA), ("COD", LAND_DRC)):
        gpd.clip(countries.loc[countries["iso_a3"] == code], clip_geometry).plot(
            ax=ax,
            facecolor=color,
            edgecolor="#c5c2b9",
            linewidth=0.40,
            zorder=0,
        )
    other = visible_countries.loc[~visible_countries["iso_a3"].isin(["ZMB", "COD"])]
    if not other.empty:
        other.plot(
            ax=ax,
            facecolor=LAND,
            edgecolor="#c5c2b9",
            linewidth=0.40,
            zorder=0,
        )

    domains = gpd.clip(layers["domains"], clip_geometry)
    domains.boundary.plot(
        ax=ax,
        color=DOMAIN,
        linewidth=0.38,
        linestyle=(0, (1.2, 2.0)),
        alpha=0.72,
        zorder=1,
    )

    contacts = gpd.clip(layers["contacts"], clip_geometry)
    contacts.plot(ax=ax, color=CONTACT, linewidth=0.35, alpha=0.72, zorder=2)

    tracts = gpd.clip(layers["tracts"], clip_geometry)
    tracts.plot(
        ax=ax,
        facecolor=TRACT_FILL,
        edgecolor=TRACT_EDGE,
        linewidth=0.55,
        alpha=0.68,
        zorder=3,
    )
    tracts.boundary.plot(ax=ax, color=TRACT_EDGE, linewidth=0.65, zorder=4)
    frame.dissolve().boundary.plot(
        ax=ax,
        color="#343a3c",
        linewidth=0.75,
        linestyle=(0, (3.5, 1.5)),
        zorder=5,
    )

    faults = gpd.clip(layers["faults"], clip_geometry)
    faults.plot(ax=ax, color="white", linewidth=1.35, alpha=0.78, zorder=6)
    faults.plot(ax=ax, color=FAULT, linewidth=0.72, zorder=7)

    border = _zambia_drc_border(countries, clip_geometry)
    border.plot(ax=ax, color="white", linewidth=2.8, alpha=0.92, zorder=8)
    border.plot(
        ax=ax,
        color=BORDER,
        linewidth=1.05,
        linestyle=(0, (5.0, 2.2)),
        zorder=9,
    )

    _occurrences(ax, layers["occurrences"])

    halo = [patheffects.withStroke(linewidth=2.0, foreground="white")]
    ax.text(
        xmin + 0.40 * (xmax - xmin),
        ymin + 0.38 * (ymax - ymin),
        "ZAMBIA",
        fontsize=7.2,
        fontweight="bold",
        color=BORDER,
        ha="center",
        va="center",
        alpha=0.72,
        path_effects=halo,
        zorder=10,
    )
    ax.text(
        xmin + 0.62 * (xmax - xmin),
        ymin + 0.68 * (ymax - ymin),
        "DRC",
        fontsize=7.2,
        fontweight="bold",
        color=BORDER,
        ha="center",
        va="center",
        alpha=0.72,
        path_effects=halo,
        zorder=10,
    )

    _map_axes(ax, bounds)
    _scale_bar(ax, bounds)
    _locator(ax, countries, frame)
    _north_arrow(ax)

    handles = [
        Patch(
            facecolor=TRACT_FILL,
            edgecolor=TRACT_EDGE,
            linewidth=0.65,
            label="USGS permissive tracts",
        ),
        Line2D([], [], color=DOMAIN, linewidth=0.6, linestyle=(0, (1.2, 2.0)),
               label="Tectonic domain (context)"),
        Line2D([], [], color=CONTACT, linewidth=0.8, label="Lithology contact"),
        Line2D([], [], color=FAULT, linewidth=1.0, label="Mapped fault"),
        Line2D([], [], color=BORDER, linewidth=1.0, linestyle=(0, (5.0, 2.2)),
               label="Zambia–DRC border"),
        Line2D([], [], marker="o", linestyle="none", markersize=4.5,
               markerfacecolor=DEPOSIT, markeredgecolor="white", label="Deposit"),
        Line2D([], [], marker="o", linestyle="none", markersize=3.2,
               markerfacecolor=PROSPECT, markeredgecolor="white", label="Prospect"),
        Line2D([], [], marker="s", linestyle="none", markersize=4.0,
               markerfacecolor=GROUP, markeredgecolor="white", label="Grouped record"),
    ]
    key_ax.text(
        0.0,
        0.98,
        "MAP KEY",
        ha="left",
        va="top",
        fontsize=6.8,
        fontweight="bold",
        color=INK,
    )
    key_ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(-0.02, 0.92),
        frameon=False,
        fontsize=6.0,
        handlelength=1.35,
        handletextpad=0.5,
        labelspacing=0.92,
        borderaxespad=0,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", type=Path, help="also write a PNG preview")
    parser.add_argument("--output", type=Path, default=FIGURE)
    args = parser.parse_args()

    figure = build_figure()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)
    if args.preview:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.preview, dpi=190)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
