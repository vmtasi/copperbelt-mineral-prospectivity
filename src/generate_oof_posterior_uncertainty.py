"""
generate_oof_posterior_uncertainty.py
=====================================
Reconstruct cell-level posterior probability distributions from frozen V11
traces and export QGIS-ready geospatial data with uncertainty statistics.

This script:
1. Loads the same data and applies the identical preprocessing as the frozen
   V11 pipeline (v11_spatial_stability_final.py).
2. For each of 4 spatial folds, loads the frozen trace and reconstructs
   3,000 draw-wise conditional probabilities p_i^(s) = logistic(eta_i^(s))
   for every held-out cell.
3. Computes posterior summary statistics (mean, median, SD, 2.5%/97.5%
   quantiles, 95% interval width).
4. Validates that reconstructed prob_mean reproduces frozen prob_v11 within
   floating-point tolerance.
5. Validates and reuses the recorded 5 km cell polygons and exports a
   polygon GeoPackage, CSV, publication map, and a separate uncertainty map.

IMPORTANT: This script does NOT modify any frozen files, refit the model,
or generate Bernoulli outcome draws. The uncertainty statistics describe
the posterior distribution of the cell's conditional prospectivity
probability, not future binary outcomes.

The existing uncertainty outputs are preserved. New products are written to
separately named files, and existing output files are never overwritten.
"""

import sys
import hashlib
import numpy as np
import pandas as pd
import geopandas as gpd
import arviz as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from scipy.spatial import cKDTree

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation_strategies import get_along_belt_folds

DATA_DIR = ROOT / 'data'
FIG_DIR = ROOT / 'figures'

GRID_GPKG = FIG_DIR / 'figure2_spatial_validation_cells.gpkg'
GRID_LAYER = 'spatial_validation_cells'
OUT_GPKG = FIG_DIR / 'v11_oof_posterior_uncertainty_reconstructed.gpkg'
OUT_CSV = FIG_DIR / 'v11_oof_posterior_uncertainty_reconstructed.csv'
LAYER_NAME = 'v11_oof_probability_uncertainty'
MAP_DIR = ROOT / 'paper_1' / 'manuscript' / 'images'
MAP_PDF = MAP_DIR / 'fig4_oof_prospectivity.pdf'
MAP_PNG = MAP_DIR / 'fig4_oof_prospectivity.png'
UNCERTAINTY_MAP_PDF = MAP_DIR / 'fig4_oof_uncertainty.pdf'
UNCERTAINTY_MAP_PNG = MAP_DIR / 'fig4_oof_uncertainty.png'
REPRODUCTION_TOLERANCE = 1e-15

# -------------------------------------------------------------------------
# Frozen file integrity
# -------------------------------------------------------------------------
PROTECTED_FILES = [
    FIG_DIR / 'v11_fold_1_trace.nc',
    FIG_DIR / 'v11_fold_2_trace.nc',
    FIG_DIR / 'v11_fold_3_trace.nc',
    FIG_DIR / 'v11_fold_4_trace.nc',
    FIG_DIR / 'v11_oof_predictions.csv',
    ROOT / 'src' / 'v11_spatial_stability_final.py',
    ROOT / 'src' / 'validation_strategies.py',
    GRID_GPKG,
]


def hash_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

missing_protected_files = [str(path) for path in PROTECTED_FILES if not path.is_file()]
if missing_protected_files:
    raise FileNotFoundError(
        'Required frozen/source files are missing: '
        + ', '.join(missing_protected_files)
    )
INITIAL_HASHES = {str(f): hash_file(f) for f in PROTECTED_FILES}

def verify_integrity(stage_name):
    for f_str, init_hash in INITIAL_HASHES.items():
        curr_hash = hash_file(Path(f_str))
        if curr_hash != init_hash:
            raise RuntimeError(
                f"[ABORT] Protected file modified during {stage_name}: {f_str}"
            )
    print(f"  [Integrity PASS] All frozen files unchanged at: {stage_name}")


def match_source_cell_geometries(result_df):
    """Align predictions to the original clipped 5 km polygons by centroid."""
    cells = gpd.read_file(GRID_GPKG, layer=GRID_LAYER)
    if cells.crs is None or cells.crs.to_epsg() != 32735:
        raise ValueError(f"Unexpected modelling-grid CRS: {cells.crs}")
    if len(cells) != len(result_df):
        raise ValueError(
            f"Expected {len(result_df)} source cell polygons, found {len(cells)}"
        )
    if not cells.geometry.is_valid.all() or cells.geometry.is_empty.any():
        raise ValueError("The source grid contains invalid or empty geometries")

    points = result_df[['centroid_x', 'centroid_y']].to_numpy(dtype=float)
    centers = np.asarray([
        (geometry.centroid.x, geometry.centroid.y)
        for geometry in cells.geometry
    ])
    distances, indices = cKDTree(centers).query(points, k=1)
    if len(np.unique(indices)) != len(result_df):
        raise ValueError("Centroid-to-polygon matching is not one-to-one")
    max_distance = float(distances.max())
    if max_distance >= 0.01:
        raise ValueError(
            "Source cell geometries do not reproduce the modelling centroids "
            f"within 1 cm (maximum distance {max_distance:.6f} m)"
        )

    matched = cells.iloc[indices].reset_index(drop=True)
    if not np.array_equal(
        matched['spatial_block'].to_numpy(dtype=int),
        result_df['spatial_block'].to_numpy(dtype=int),
    ):
        raise ValueError("Source polygon fold assignments differ from the V11 folds")
    if not np.array_equal(
        matched['daly_domain'].astype(str).to_numpy(),
        result_df['daly_domain'].astype(str).to_numpy(),
    ):
        raise ValueError("Source polygon Daly-domain labels differ from V11")
    if not np.array_equal(
        matched['deposit_present'].to_numpy(dtype=int),
        result_df['deposit_present'].to_numpy(dtype=int),
    ):
        raise ValueError("Source polygon deposit labels differ from the model data")

    widths = np.asarray([
        geometry.bounds[2] - geometry.bounds[0]
        for geometry in matched.geometry
    ])
    heights = np.asarray([
        geometry.bounds[3] - geometry.bounds[1]
        for geometry in matched.geometry
    ])
    if np.any(widths > 5000.01) or np.any(heights > 5000.01):
        raise ValueError("A source grid polygon exceeds the 5 km modelling-cell size")

    return matched, cells.crs, max_distance


def _format_kilometres(value, _position):
    return f'{value / 1000:.0f}'


def _add_map_decorations(ax, gdf, show_deposits):
    domain_boundaries = gdf.dissolve(by='daly_domain').boundary
    domain_boundaries.plot(ax=ax, color='#30343b', linewidth=0.48, zorder=3)

    if show_deposits:
        deposits = gdf[gdf['deposit_present'] == 1]
        ax.scatter(
            deposits['centroid_x'], deposits['centroid_y'],
            marker='x', s=23, linewidths=1.15, color='#c62828',
            zorder=5,
        )

    bounds = gdf.total_bounds
    xspan = bounds[2] - bounds[0]
    yspan = bounds[3] - bounds[1]
    ax.set_xlim(bounds[0] - xspan * 0.025, bounds[2] + xspan * 0.025)
    ax.set_ylim(bounds[1] - yspan * 0.025, bounds[3] + yspan * 0.025)
    ax.set_aspect('equal', adjustable='box')
    ax.xaxis.set_major_formatter(FuncFormatter(_format_kilometres))
    ax.yaxis.set_major_formatter(FuncFormatter(_format_kilometres))
    ax.set_xlabel('Easting (km; WGS 84 / UTM zone 35S)')
    ax.set_ylabel('Northing (km; WGS 84 / UTM zone 35S)')
    ax.grid(color='#777777', linewidth=0.35, alpha=0.28, zorder=0)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    scale_x = xmin + (xmax - xmin) * 0.055
    scale_y = ymin + (ymax - ymin) * 0.045
    scale_length = 100_000
    ax.plot(
        [scale_x, scale_x + scale_length], [scale_y, scale_y],
        color='white', linewidth=5, solid_capstyle='butt', zorder=6,
    )
    ax.plot(
        [scale_x, scale_x + scale_length], [scale_y, scale_y],
        color='#202124', linewidth=2.4, solid_capstyle='butt', zorder=7,
    )
    ax.text(
        scale_x + scale_length / 2, scale_y + (ymax - ymin) * 0.012,
        '100 km', ha='center', va='bottom', fontsize=8, color='#202124',
        bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.85, 'pad': 1.3},
        zorder=7,
    )
    ax.annotate(
        'N', xy=(0.955, 0.94), xytext=(0.955, 0.84),
        xycoords='axes fraction', textcoords='axes fraction',
        ha='center', va='center', fontsize=9, fontweight='bold',
        arrowprops={'arrowstyle': '-|>', 'color': '#202124', 'lw': 1.1},
        zorder=8,
    )

    handles = [
        Patch(facecolor='none', edgecolor='#30343b', linewidth=0.8,
              label='Daly-domain edges (cell labels)'),
    ]
    if show_deposits:
        handles.insert(0, Line2D(
            [], [], marker='x', linestyle='none', markersize=6,
            markeredgewidth=1.2, color='#c62828',
            label=(
                'USGS-record-positive cell centroids '
                f'(centroid markers; n={int(gdf.deposit_present.sum())})'
            ),
        ))
    ax.legend(
        handles=handles, loc='upper right', frameon=True, framealpha=0.95,
        facecolor='white', edgecolor='#c5c7ca', fontsize=7.3,
    )


def render_map(gdf, value_column, title, colorbar_label, output_pdf, output_png,
               show_deposits):
    """Render a vector map of cell values and a manuscript-resolution PNG."""
    values_max = float(gdf[value_column].max())
    vmax = max(0.05, np.ceil(values_max * 20) / 20)
    fig, ax = plt.subplots(figsize=(7.6, 7.0), constrained_layout=True)
    gdf.plot(
        column=value_column, ax=ax, cmap='viridis', vmin=0, vmax=vmax,
        linewidth=0, edgecolor='none', zorder=1,
    )
    _add_map_decorations(ax, gdf, show_deposits)
    ax.set_title(title, fontsize=12, pad=9, fontweight='semibold')
    ax.text(
        0.015, 0.985,
        f'{len(gdf):,} held-out 5 km cells',
        transform=ax.transAxes, ha='left', va='top', fontsize=7.4,
        color='#202124',
        bbox={'facecolor': 'white', 'edgecolor': '#d5d7da', 'alpha': 0.93, 'pad': 3},
        zorder=8,
    )
    norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
    scalar_mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap='viridis')
    colorbar = fig.colorbar(
        scalar_mappable, ax=ax, fraction=0.043, pad=0.035, shrink=0.88,
    )
    colorbar.set_label(colorbar_label, fontsize=8.5)
    colorbar.ax.tick_params(labelsize=7.5, length=2)
    fig.savefig(output_pdf, bbox_inches='tight', metadata={'Title': title})
    fig.savefig(output_png, dpi=350, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# -------------------------------------------------------------------------
# Data loading — identical to v11_spatial_stability_final.py
# -------------------------------------------------------------------------
def load_prepared_data():
    data_path = DATA_DIR / 'copperbelt_training_v5_with_tectonic_domain.csv'
    df = pd.read_csv(data_path)

    lithology_col = 'litho_contact_litho_class'
    dist_to_lith_col = 'distance_to_lithology_contact'
    dist_to_fault_col = 'distance_to_fault'
    gravity_col = 'bouguer'
    continuous_features = [dist_to_lith_col, dist_to_fault_col, gravity_col]

    df = df.dropna(
        subset=['centroid_x', 'centroid_y', lithology_col, 'domain']
        + continuous_features
    ).copy()

    def map_daly_domain(x):
        x_str = str(x).lower()
        if '3a' in x_str: return 'NRB_3a'
        elif '3b' in x_str: return 'NRB_3b'
        elif 'crz' in x_str: return 'CRZ'
        elif 'srb' in x_str: return 'SRB'
        elif 'nkb' in x_str: return 'NKB'
        elif 'mmsb' in x_str: return 'MMSB'
        else: return 'Unknown'

    df['daly_domain'] = df['domain'].apply(map_daly_domain)
    df = df[df['daly_domain'] != 'Unknown'].copy()
    unique_domains = sorted(df['daly_domain'].unique())
    domain_to_idx = {dom: i for i, dom in enumerate(unique_domains)}
    df['domain_idx'] = df['daly_domain'].map(domain_to_idx)
    df['spatial_block'] = get_along_belt_folds(df, n_folds=4)

    df_encoded = pd.get_dummies(
        df, columns=[lithology_col], drop_first=True, dtype=float
    )
    all_rock_features = [
        col for col in df_encoded.columns
        if col.startswith(f'{lithology_col}_')
    ]

    return df_encoded, unique_domains, all_rock_features


# -------------------------------------------------------------------------
# Main reconstruction
# -------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("V11 OOF POSTERIOR RECONSTRUCTION AND FIGURE 4")
    print("=" * 70)

    # Pre-flight checks
    output_paths = [
        OUT_GPKG, OUT_CSV, MAP_PDF, MAP_PNG,
        UNCERTAINTY_MAP_PDF, UNCERTAINTY_MAP_PNG,
    ]
    for p in output_paths:
        if p.exists():
            raise FileExistsError(
                f"Output already exists — will not overwrite: {p}"
            )

    verify_integrity("Start")

    # Load data
    print("\n--- Loading data and applying V11 preprocessing ---")
    df, unique_domains, all_rock_features = load_prepared_data()
    n_folds = 4
    n_domains = len(unique_domains)
    if len(df) != 1872:
        raise ValueError(f"Expected 1,872 complete-case cells, found {len(df)}")
    if df.duplicated(['centroid_x', 'centroid_y']).any():
        raise ValueError("Complete-case model coordinates are not unique")
    print(f"  Total cells: {len(df)}")
    print(f"  Domains ({n_domains}): {unique_domains}")

    # Load frozen OOF predictions for validation
    oof_frozen = pd.read_csv(FIG_DIR / 'v11_oof_predictions.csv')
    assert len(oof_frozen) == 1872, f"Expected 1872 frozen OOF rows, got {len(oof_frozen)}"

    # Collect results across folds
    all_rows = []

    for fold in range(n_folds):
        fold_num = fold + 1
        print(f"\n--- Fold {fold_num} ---")

        train_df = df[df['spatial_block'] != fold].copy()
        test_df = df[df['spatial_block'] == fold].copy()
        y_train = train_df['deposit_present'].values.astype(np.int32)

        print(f"  Train: {len(train_df)} cells, Test: {len(test_df)} cells")

        # Reconstruct training scalers — identical to V11
        scaler_fault = StandardScaler().fit(train_df[['distance_to_fault']])
        scaler_lith = StandardScaler().fit(
            train_df[['distance_to_lithology_contact']]
        )
        scaler_grav = StandardScaler().fit(train_df[['bouguer']])

        # Transform features for train and test
        for d in [train_df, test_df]:
            d['fault_z'] = scaler_fault.transform(d[['distance_to_fault']])
            d['lith_z'] = scaler_lith.transform(
                d[['distance_to_lithology_contact']]
            )
            d['grav_z'] = scaler_grav.transform(d[['bouguer']])
            d['fault_z_sq'] = d['fault_z'] ** 2
            d['lith_z_sq'] = d['lith_z'] ** 2

        # Filter valid rocks — identical logic
        valid_rocks = [
            col for col in all_rock_features
            if ((train_df[col] == 1) & (y_train == 1)).sum() > 0
            and ((train_df[col] == 1) & (y_train == 0)).sum() > 0
        ]
        print(f"  Valid rock features: {len(valid_rocks)}")

        # Load frozen trace
        trace_path = FIG_DIR / f'v11_fold_{fold_num}_trace.nc'
        trace = az.from_netcdf(trace_path)
        trace_draws = int(
            trace.posterior.sizes['chain'] * trace.posterior.sizes['draw']
        )
        if trace_draws != 3000:
            raise ValueError(
                f"Fold {fold_num} trace has {trace_draws} draws; expected 3,000"
            )

        # Extract posterior draws — identical to reconstruction script
        alpha_dom = trace.posterior['alpha_dom'].values.reshape(-1, n_domains)
        bf_lin = trace.posterior['beta_f_lin'].values.reshape(-1, n_domains)
        bf_sq = trace.posterior['beta_f_sq'].values.reshape(-1, n_domains)
        bl_lin = trace.posterior['beta_l_lin'].values.reshape(-1, n_domains)
        bl_sq = trace.posterior['beta_l_sq'].values.reshape(-1, n_domains)
        beta_grav = trace.posterior['beta_grav'].values.flatten()
        beta_rocks = trace.posterior['beta_rocks'].values.reshape(
            -1, len(valid_rocks)
        )

        n_draws = len(beta_grav)
        if n_draws != 3000:
            raise ValueError(
                f"Fold {fold_num} coefficient arrays contain {n_draws} draws"
            )
        print(f"  Posterior draws: {n_draws}")

        # Build test feature arrays
        test_dom_idx = test_df['domain_idx'].values
        X_f = test_df['fault_z'].values
        X_f_sq = test_df['fault_z_sq'].values
        X_l = test_df['lith_z'].values
        X_l_sq = test_df['lith_z_sq'].values
        X_g = test_df['grav_z'].values
        X_r = test_df[valid_rocks].values
        n_test = len(test_df)

        # Reconstruct draw-wise linear predictor for ALL held-out cells
        # Shape: (n_draws, n_test)
        eta = (
            alpha_dom[:, test_dom_idx]
            + bf_lin[:, test_dom_idx] * X_f
            + bf_sq[:, test_dom_idx] * X_f_sq
            + bl_lin[:, test_dom_idx] * X_l
            + bl_sq[:, test_dom_idx] * X_l_sq
            + beta_grav[:, None] * X_g
            + np.dot(beta_rocks, X_r.T)
        )

        # Draw-wise conditional probabilities
        p_draws = 1 / (1 + np.exp(-eta))  # Match the frozen extraction arithmetic.

        # Verify shape
        assert p_draws.shape == (n_draws, n_test), (
            f"Shape mismatch: {p_draws.shape} != ({n_draws}, {n_test})"
        )

        # Check for NaN / infinite
        assert not np.any(np.isnan(p_draws)), "NaN found in p_draws"
        assert not np.any(np.isinf(p_draws)), "Inf found in p_draws"

        # Compute posterior summary statistics
        prob_mean = np.mean(p_draws, axis=0)
        prob_median = np.median(p_draws, axis=0)
        prob_sd = np.std(p_draws, axis=0, ddof=0)
        prob_q025 = np.percentile(p_draws, 2.5, axis=0)
        prob_q975 = np.percentile(p_draws, 97.5, axis=0)
        prob_iw95 = prob_q975 - prob_q025

        # Collect per-cell rows
        cx = test_df['centroid_x'].values
        cy = test_df['centroid_y'].values
        domains = test_df['daly_domain'].values

        for i in range(n_test):
            all_rows.append({
                'centroid_x': cx[i],
                'centroid_y': cy[i],
                'spatial_block': fold,
                'daly_domain': domains[i],
                'deposit_present': int(test_df['deposit_present'].iloc[i]),
                'prob_mean': prob_mean[i],
                'prob_median': prob_median[i],
                'prob_sd': prob_sd[i],
                'prob_q025': prob_q025[i],
                'prob_q975': prob_q975[i],
                'prob_interval_width95': prob_iw95[i],
            })

        print(f"  Cells processed: {n_test}")
        print(f"  prob_mean range: [{prob_mean.min():.6f}, {prob_mean.max():.6f}]")
        print(f"  prob_sd range:   [{prob_sd.min():.6f}, {prob_sd.max():.6f}]")
        print(f"  IW95 range:      [{prob_iw95.min():.6f}, {prob_iw95.max():.6f}]")

    # =====================================================================
    # QUALITY CONTROL
    # =====================================================================
    print("\n" + "=" * 70)
    print("QUALITY CONTROL")
    print("=" * 70)

    result_df = pd.DataFrame(all_rows)

    # QC1: Exactly 1,872 cells
    assert len(result_df) == 1872, f"Expected 1872 cells, got {len(result_df)}"
    print(f"  [QC1 PASS] Exactly {len(result_df)} cells")

    # QC2: Coordinate and row ordering match frozen OOF
    # Sort both by coordinates for matching (fold order may differ from
    # frozen CSV row order)
    result_sorted = result_df.sort_values(
        ['centroid_x', 'centroid_y']
    ).reset_index(drop=True)
    frozen_sorted = oof_frozen.sort_values(
        ['centroid_x', 'centroid_y']
    ).reset_index(drop=True)

    coord_diff_x = np.abs(
        result_sorted['centroid_x'].values - frozen_sorted['centroid_x'].values
    )
    coord_diff_y = np.abs(
        result_sorted['centroid_y'].values - frozen_sorted['centroid_y'].values
    )
    assert coord_diff_x.max() < 1e-3, (
        f"X coordinate mismatch: max diff = {coord_diff_x.max()}"
    )
    assert coord_diff_y.max() < 1e-3, (
        f"Y coordinate mismatch: max diff = {coord_diff_y.max()}"
    )
    print(f"  [QC2 PASS] Coordinates match frozen OOF (max diff X={coord_diff_x.max():.2e}, Y={coord_diff_y.max():.2e})")

    # QC3: Reconstructed prob_mean reproduces prob_v11
    prob_diff = np.abs(
        result_sorted['prob_mean'].values - frozen_sorted['prob_v11'].values
    )
    max_diff = prob_diff.max()
    mean_diff = prob_diff.mean()
    print(f"  [QC3] prob_mean vs prob_v11: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
    assert max_diff <= REPRODUCTION_TOLERANCE, (
        "Reconstruction mismatch exceeds the floating-point tolerance: "
        f"max_diff={max_diff:.17g}, tolerance={REPRODUCTION_TOLERANCE:.1e}"
    )
    print(
        "  [QC3 PASS] Reconstructed prob_mean reproduces prob_v11: "
        f"max_diff={max_diff:.3e}, mean_diff={mean_diff:.3e}"
    )

    # QC4: All 3,000 draws available (verified per fold above)
    print(f"  [QC4 PASS] All draws available (verified during fold processing)")

    # QC5: No NaN/Inf
    for col in ['prob_mean', 'prob_median', 'prob_sd', 'prob_q025',
                'prob_q975', 'prob_interval_width95']:
        assert not result_df[col].isna().any(), f"NaN found in {col}"
        assert not np.isinf(result_df[col].values).any(), f"Inf found in {col}"
    print(f"  [QC5 PASS] No NaN or Inf values in any output field")

    # =====================================================================
    # EXPORT
    # =====================================================================
    print("\n" + "=" * 70)
    print("EXPORTING OUTPUTS")
    print("=" * 70)

    # Reuse the exact clipped 5 km source polygons; do not turn points into
    # assumed cell-sized rectangles.
    matched_cells, source_crs, max_centroid_distance = (
        match_source_cell_geometries(result_df)
    )
    gdf = gpd.GeoDataFrame(
        result_df.copy(),
        geometry=matched_cells.geometry.to_numpy(),
        crs=source_crs,
    )
    print(
        "  [QC6 PASS] Source polygons align one-to-one with all OOF cells; "
        f"maximum centroid offset={max_centroid_distance:.6f} m, CRS={source_crs}"
    )

    # Export GeoPackage
    gdf.to_file(OUT_GPKG, layer=LAYER_NAME, driver='GPKG')
    print(f"  [+] GeoPackage: {OUT_GPKG}")
    print(f"      Layer name: {LAYER_NAME}")

    # Export CSV
    result_df.to_csv(OUT_CSV, index=False)
    print(f"  [+] CSV: {OUT_CSV}")

    render_map(
        gdf, 'prob_mean', 'V11 out-of-fold prospectivity',
        'Posterior mean conditional occurrence probability',
        MAP_PDF, MAP_PNG, show_deposits=True,
    )
    render_map(
        gdf, 'prob_interval_width95',
        'V11 posterior uncertainty in conditional probability',
        '95% credible-interval width',
        UNCERTAINTY_MAP_PDF, UNCERTAINTY_MAP_PNG, show_deposits=False,
    )
    print(f"  [+] Prospectivity map (vector PDF): {MAP_PDF}")
    print(f"  [+] Prospectivity map (PNG): {MAP_PNG}")
    print(f"  [+] Uncertainty map (vector PDF): {UNCERTAINTY_MAP_PDF}")
    print(f"  [+] Uncertainty map (PNG): {UNCERTAINTY_MAP_PNG}")

    # =====================================================================
    # FINAL REPORT
    # =====================================================================
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(f"  Output GeoPackage:     {OUT_GPKG}")
    print(f"  Output CSV:            {OUT_CSV}")
    print(f"  Layer name:            {LAYER_NAME}")
    print(f"  Number of cells:       {len(gdf)}")
    print(f"  Posterior draws/cell:  {n_draws}")
    print(f"  CRS:                   {gdf.crs}")
    print("  Geometry type:         Original clipped 5 km modelling-cell polygons")
    print(f"  Max centroid offset:   {max_centroid_distance:.6f} m")
    print(f"  Fields:                {list(result_df.columns)}")
    print(f"  Validation:            prob_mean reproduces prob_v11 (max diff {max_diff:.3e})")
    print(f"  Probability range:     [{gdf.prob_mean.min():.6f}, {gdf.prob_mean.max():.6f}]")
    print()
    print("  Summary statistics:")
    print(result_df[['prob_mean', 'prob_median', 'prob_sd', 'prob_q025',
                      'prob_q975', 'prob_interval_width95']].describe().to_string())

    # Final integrity check
    verify_integrity("End of script")

    print("\n[SUCCESS] OOF posterior uncertainty dataset generated.")
    print("  The GPKG can be opened directly in QGIS for:")
    print("    1. Prospectivity heat map:  symbolize by prob_mean")
    print("    2. Uncertainty map:         symbolize by prob_interval_width95")


if __name__ == '__main__':
    main()
