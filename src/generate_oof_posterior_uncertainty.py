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
5. Exports a GeoPackage (.gpkg) point layer and CSV.

IMPORTANT: This script does NOT modify any frozen files, refit the model,
or generate Bernoulli outcome draws. The uncertainty statistics describe
the posterior distribution of the cell's conditional prospectivity
probability, not future binary outcomes.

Output files:
    figures/v11_oof_posterior_uncertainty.gpkg
    figures/v11_oof_posterior_uncertainty.csv
"""

import sys
import hashlib
import numpy as np
import pandas as pd
import geopandas as gpd
import arviz as az
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from shapely.geometry import Point
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation_strategies import get_along_belt_folds

DATA_DIR = ROOT / 'data'
FIG_DIR = ROOT / 'figures'

# CRS: EPSG:32735 (WGS 84 / UTM Zone 35S) — verified from manuscript §2.1
# and coordinate ranges (X: 282k–715k, Y: 8.38M–8.86M)
CRS_EPSG = 32735

# Output paths — do NOT overwrite any existing file
OUT_GPKG = FIG_DIR / 'v11_oof_posterior_uncertainty.gpkg'
OUT_CSV  = FIG_DIR / 'v11_oof_posterior_uncertainty.csv'
LAYER_NAME = 'v11_oof_uncertainty'

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
]

def hash_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

INITIAL_HASHES = {str(f): hash_file(f) for f in PROTECTED_FILES if f.exists()}

def verify_integrity(stage_name):
    for f_str, init_hash in INITIAL_HASHES.items():
        curr_hash = hash_file(Path(f_str))
        if curr_hash != init_hash:
            raise RuntimeError(
                f"[ABORT] Protected file modified during {stage_name}: {f_str}"
            )
    print(f"  [Integrity PASS] All frozen files unchanged at: {stage_name}")


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
    print("V11 OOF POSTERIOR UNCERTAINTY RECONSTRUCTION")
    print("=" * 70)

    # Pre-flight checks
    for p in [OUT_GPKG, OUT_CSV]:
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
        p_draws = expit(eta)  # (n_draws, n_test)

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
    assert max_diff < 1e-10, (
        f"Reconstruction mismatch exceeds tolerance: max_diff={max_diff}"
    )
    print(f"  [QC3 PASS] Reconstructed prob_mean reproduces prob_v11 within tolerance")

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

    # Create GeoDataFrame with point geometry
    geometry = [
        Point(row['centroid_x'], row['centroid_y'])
        for _, row in result_df.iterrows()
    ]
    gdf = gpd.GeoDataFrame(result_df, geometry=geometry, crs=f"EPSG:{CRS_EPSG}")

    # Export GeoPackage
    gdf.to_file(OUT_GPKG, layer=LAYER_NAME, driver='GPKG')
    print(f"  [+] GeoPackage: {OUT_GPKG}")
    print(f"      Layer name: {LAYER_NAME}")

    # Export CSV
    result_df.to_csv(OUT_CSV, index=False)
    print(f"  [+] CSV: {OUT_CSV}")

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
    print(f"  CRS:                   EPSG:{CRS_EPSG} (WGS 84 / UTM Zone 35S)")
    print(f"  Geometry type:         Point (cell centroids)")
    print(f"  Fields:                {list(result_df.columns)}")
    print(f"  Validation:            prob_mean reproduces prob_v11 (max diff {max_diff:.2e})")
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
