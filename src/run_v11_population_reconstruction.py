"""
run_v11_population_reconstruction.py
====================================
PHASE 2: V11 Posterior Reconstruction & Fold-Level Heterogeneity Analysis

This module performs the complete supplementary analysis on the FROZEN V11 posterior traces:
1. Reconstructs population-level response curves (raw km) for Fault and Lithology.
2. Derives population-level D* stationary point diagnostics with full posterior distributions.
3. Evaluates fold-level linear predictor contributions across the 4 spatial folds.
4. Compares functional behavior and D* stability across spatial folds.
5. Exports all required CSVs and publication-grade figures to figures/audit/v11_population_reconstruction/.
6. Performs SHA-256 integrity verification of all protected V11 baseline files.
"""

import os
import sys
import hashlib
import numpy as np
import pandas as pd
import arviz as az
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation_strategies import get_along_belt_folds

DATA_DIR = ROOT / 'data'
FIG_DIR = ROOT / 'figures'
OUT_DIR = FIG_DIR / 'audit' / 'v11_population_reconstruction'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# 1. SHA-256 Integrity Verification
# -------------------------------------------------------------------------
PROTECTED_FILES = [
    ROOT / 'src' / 'v11_spatial_stability_final.py',
    ROOT / 'src' / 'validation_strategies.py',
    FIG_DIR / 'v11_fold_1_trace.nc',
    FIG_DIR / 'v11_fold_2_trace.nc',
    FIG_DIR / 'v11_fold_3_trace.nc',
    FIG_DIR / 'v11_fold_4_trace.nc',
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
            raise RuntimeError(f"[ABORT] Protected file modified during {stage_name}: {f_str}")
    print(f"[Integrity Check PASS] All protected V11 files unchanged at {stage_name}.")


# -------------------------------------------------------------------------
# 2. Data Loading & Spatial Partitioning
# -------------------------------------------------------------------------
def load_prepared_data():
    data_path = DATA_DIR / 'copperbelt_training_v5_with_tectonic_domain.csv'
    df = pd.read_csv(data_path)

    lithology_col = 'litho_contact_litho_class'
    dist_to_lith_col = 'distance_to_lithology_contact'
    dist_to_fault_col = 'distance_to_fault'
    gravity_col = 'bouguer'  
    continuous_features = [dist_to_lith_col, dist_to_fault_col, gravity_col]

    df = df.dropna(subset=['centroid_x', 'centroid_y', lithology_col, 'domain'] + continuous_features).copy()

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

    df_encoded = pd.get_dummies(df, columns=[lithology_col], drop_first=True, dtype=float)
    all_rock_features = [col for col in df_encoded.columns if col.startswith(f'{lithology_col}_')]

    return df_encoded, unique_domains, all_rock_features


# -------------------------------------------------------------------------
# 3. Master Analysis Execution
# -------------------------------------------------------------------------
def run_reconstruction():
    print("=" * 70)
    print("PHASE 2: V11 POSTERIOR RECONSTRUCTION & FOLD-LEVEL HETEROGENEITY")
    print("=" * 70)
    verify_integrity("Start of Phase 2")

    df, unique_domains, all_rock_features = load_prepared_data()
    n_folds = 4
    n_domains = len(unique_domains)
    grid_km = np.linspace(0, 70, 200)

    # Containers for results
    pop_resp_fault_rows = []
    pop_resp_lith_rows = []
    pop_d_star_rows = []
    fold_d_star_rows = []
    fold_contrib_rows = []
    fold_pred_summary_rows = []

    # D* sample containers for distribution plotting
    d_star_samples_dict = {
        'pop_fault': {},
        'pop_lith': {},
        'nrb3a_fault': {},
        'nrb3a_lith': {}
    }

    # Trace containers
    traces = {}
    scalers = {}

    for fold in range(n_folds):
        fold_num = fold + 1
        print(f"\n--- Processing Fold {fold_num} Trace ---")

        train_df = df[df['spatial_block'] != fold].copy()
        test_df = df[df['spatial_block'] == fold].copy()
        y_train = train_df['deposit_present'].values.astype(np.int32)
        y_test = test_df['deposit_present'].values.astype(np.int32)

        # Reconstruct Training Scalers exactly as V11 did
        scaler_fault = StandardScaler().fit(train_df[['distance_to_fault']])
        scaler_lith = StandardScaler().fit(train_df[['distance_to_lithology_contact']])
        scaler_grav = StandardScaler().fit(train_df[['bouguer']])

        mu_f, std_f = scaler_fault.mean_[0], scaler_fault.scale_[0]
        mu_l, std_l = scaler_lith.mean_[0], scaler_lith.scale_[0]
        mu_g, std_g = scaler_grav.mean_[0], scaler_grav.scale_[0]

        scalers[fold_num] = {
            'fault': (mu_f, std_f),
            'lith': (mu_l, std_l),
            'grav': (mu_g, std_g),
            'max_f_train': train_df['distance_to_fault'].max() / 1000.0,
            'p95_f_train': np.percentile(train_df['distance_to_fault'] / 1000.0, 95),
            'max_l_train': train_df['distance_to_lithology_contact'].max() / 1000.0,
            'p95_l_train': np.percentile(train_df['distance_to_lithology_contact'] / 1000.0, 95),
        }

        # Transform features
        for d in [train_df, test_df]:
            d['fault_z'] = scaler_fault.transform(d[['distance_to_fault']])
            d['lith_z'] = scaler_lith.transform(d[['distance_to_lithology_contact']])
            d['grav_z'] = scaler_grav.transform(d[['bouguer']])
            d['fault_z_sq'] = d['fault_z'] ** 2
            d['lith_z_sq'] = d['lith_z'] ** 2

        # Filter valid rocks
        valid_rocks = [
            col for col in all_rock_features
            if ((train_df[col] == 1) & (y_train == 1)).sum() > 0 and ((train_df[col] == 1) & (y_train == 0)).sum() > 0
        ]

        # Load Frozen Trace
        trace_path = FIG_DIR / f'v11_fold_{fold_num}_trace.nc'
        trace = az.from_netcdf(trace_path)
        traces[fold_num] = trace

        # Extract posterior draws
        alpha_mu = trace.posterior['alpha_mu'].values.flatten()
        mu_f_lin = trace.posterior['mu_f_lin'].values.flatten()
        mu_f_sq = trace.posterior['mu_f_sq'].values.flatten()
        mu_l_lin = trace.posterior['mu_l_lin'].values.flatten()
        mu_l_sq = trace.posterior['mu_l_sq'].values.flatten()
        beta_grav = trace.posterior['beta_grav'].values.flatten()
        beta_rocks = trace.posterior['beta_rocks'].values.reshape(-1, len(valid_rocks))

        alpha_dom = trace.posterior['alpha_dom'].values.reshape(-1, n_domains)
        bf_lin = trace.posterior['beta_f_lin'].values.reshape(-1, n_domains)
        bf_sq = trace.posterior['beta_f_sq'].values.reshape(-1, n_domains)
        bl_lin = trace.posterior['beta_l_lin'].values.reshape(-1, n_domains)
        bl_sq = trace.posterior['beta_l_sq'].values.reshape(-1, n_domains)

        n_draws = len(alpha_mu)

        # -------------------------------------------------------------
        # A. Population Response Curves
        # -------------------------------------------------------------
        # Fault grid in Z-space
        z_grid_f = (grid_km * 1000.0 - mu_f) / std_f
        z_grid_l = (grid_km * 1000.0 - mu_l) / std_l

        # Logit: alpha_mu + mu_lin * z + mu_sq * z^2 (gravity and rocks at mean 0)
        logit_resp_f = alpha_mu[:, None] + mu_f_lin[:, None] * z_grid_f[None, :] + mu_f_sq[:, None] * (z_grid_f**2)[None, :]
        prob_resp_f = expit(logit_resp_f)

        logit_resp_l = alpha_mu[:, None] + mu_l_lin[:, None] * z_grid_l[None, :] + mu_l_sq[:, None] * (z_grid_l**2)[None, :]
        prob_resp_l = expit(logit_resp_l)

        for pt_idx, x_km in enumerate(grid_km):
            pop_resp_fault_rows.append({
                'fold': fold_num,
                'grid_km': x_km,
                'response_median': np.median(prob_resp_f[:, pt_idx]),
                'response_mean': np.mean(prob_resp_f[:, pt_idx]),
                'response_ci_lower': np.percentile(prob_resp_f[:, pt_idx], 2.5),
                'response_ci_upper': np.percentile(prob_resp_f[:, pt_idx], 97.5),
                'logit_mean': np.mean(logit_resp_f[:, pt_idx]),
                'logit_ci_lower': np.percentile(logit_resp_f[:, pt_idx], 2.5),
                'logit_ci_upper': np.percentile(logit_resp_f[:, pt_idx], 97.5)
            })
            pop_resp_lith_rows.append({
                'fold': fold_num,
                'grid_km': x_km,
                'response_median': np.median(prob_resp_l[:, pt_idx]),
                'response_mean': np.mean(prob_resp_l[:, pt_idx]),
                'response_ci_lower': np.percentile(prob_resp_l[:, pt_idx], 2.5),
                'response_ci_upper': np.percentile(prob_resp_l[:, pt_idx], 97.5),
                'logit_mean': np.mean(logit_resp_l[:, pt_idx]),
                'logit_ci_lower': np.percentile(logit_resp_l[:, pt_idx], 2.5),
                'logit_ci_upper': np.percentile(logit_resp_l[:, pt_idx], 97.5)
            })

        # -------------------------------------------------------------
        # B. Population D* Diagnostic
        # -------------------------------------------------------------
        # Fault Population D*
        valid_f = np.abs(mu_f_sq) > 1e-4
        z_star_f = -mu_f_lin[valid_f] / (2.0 * mu_f_sq[valid_f])
        d_star_f = (z_star_f * std_f + mu_f) / 1000.0
        d_star_samples_dict['pop_fault'][fold_num] = d_star_f

        p_b2_pos_f = np.mean(mu_f_sq > 0)
        p_b2_neg_f = np.mean(mu_f_sq < 0)
        in_max_f = np.mean((d_star_f >= 0) & (d_star_f <= scalers[fold_num]['max_f_train']))
        in_p95_f = np.mean((d_star_f >= 0) & (d_star_f <= scalers[fold_num]['p95_f_train']))

        # Lith Population D*
        valid_l = np.abs(mu_l_sq) > 1e-4
        z_star_l = -mu_l_lin[valid_l] / (2.0 * mu_l_sq[valid_l])
        d_star_l = (z_star_l * std_l + mu_l) / 1000.0
        d_star_samples_dict['pop_lith'][fold_num] = d_star_l

        p_b2_pos_l = np.mean(mu_l_sq > 0)
        p_b2_neg_l = np.mean(mu_l_sq < 0)
        in_max_l = np.mean((d_star_l >= 0) & (d_star_l <= scalers[fold_num]['max_l_train']))
        in_p95_l = np.mean((d_star_l >= 0) & (d_star_l <= scalers[fold_num]['p95_l_train']))

        pop_d_star_rows.append({
            'predictor': 'distance_to_fault',
            'fold': fold_num,
            'p_beta_sq_pos': p_b2_pos_f,
            'p_beta_sq_neg': p_b2_neg_f,
            'd_star_median_km': np.median(d_star_f),
            'd_star_mean_km': np.mean(d_star_f),
            'd_star_ci_lower_km': np.percentile(d_star_f, 2.5),
            'd_star_ci_upper_km': np.percentile(d_star_f, 97.5),
            'prop_in_support_max': in_max_f,
            'prop_in_support_p95': in_p95_f,
            'identifiability_status': 'Unidentifiable (Singularity / Wide CI)' if (np.percentile(d_star_f, 97.5) - np.percentile(d_star_f, 2.5) > 100) else 'Identifiable'
        })

        pop_d_star_rows.append({
            'predictor': 'distance_to_lithology_contact',
            'fold': fold_num,
            'p_beta_sq_pos': p_b2_pos_l,
            'p_beta_sq_neg': p_b2_neg_l,
            'd_star_median_km': np.median(d_star_l),
            'd_star_mean_km': np.mean(d_star_l),
            'd_star_ci_lower_km': np.percentile(d_star_l, 2.5),
            'd_star_ci_upper_km': np.percentile(d_star_l, 97.5),
            'prop_in_support_max': in_max_l,
            'prop_in_support_p95': in_p95_l,
            'identifiability_status': 'Unidentifiable (Singularity / Wide CI)' if (np.percentile(d_star_l, 97.5) - np.percentile(d_star_l, 2.5) > 100) else 'Identifiable'
        })

        # -------------------------------------------------------------
        # C. Domain-Level D* Extraction (for Comparison)
        # -------------------------------------------------------------
        for d_idx, dom_name in enumerate(unique_domains):
            # Domain Fault D*
            b1_f = bf_lin[:, d_idx]
            b2_f = bf_sq[:, d_idx]
            v_f = np.abs(b2_f) > 1e-4
            z_s_f = -b1_f[v_f] / (2.0 * b2_f[v_f])
            d_s_f = (z_s_f * std_f + mu_f) / 1000.0
            p_b2_f_dom = np.mean(b2_f > 0)
            in_max_f_dom = np.mean((d_s_f >= 0) & (d_s_f <= scalers[fold_num]['max_f_train']))

            status_f = 'Unsupported / Out of Range'
            if p_b2_f_dom < 0.60:
                status_f = 'Weak-Curvature / Unidentifiable'
            elif in_max_f_dom > 0.70 and (np.percentile(d_s_f, 97.5) - np.percentile(d_s_f, 2.5) < 80):
                status_f = 'Identifiable D*'
            elif in_max_f_dom > 0.40:
                status_f = 'Boundary D*'

            fold_d_star_rows.append({
                'fold': fold_num,
                'domain': dom_name,
                'predictor': 'distance_to_fault',
                'd_star_median_km': np.median(d_s_f),
                'd_star_ci_lower_km': np.percentile(d_s_f, 2.5),
                'd_star_ci_upper_km': np.percentile(d_s_f, 97.5),
                'p_beta_sq_pos': p_b2_f_dom,
                'support_max_km': scalers[fold_num]['max_f_train'],
                'prop_in_support': in_max_f_dom,
                'classification': status_f
            })

            if dom_name == 'NRB_3a':
                d_star_samples_dict['nrb3a_fault'][fold_num] = d_s_f

            # Domain Lith D*
            b1_l = bl_lin[:, d_idx]
            b2_l = bl_sq[:, d_idx]
            v_l = np.abs(b2_l) > 1e-4
            z_s_l = -b1_l[v_l] / (2.0 * b2_l[v_l])
            d_s_l = (z_s_l * std_l + mu_l) / 1000.0
            p_b2_l_dom = np.mean(b2_l > 0)
            in_max_l_dom = np.mean((d_s_l >= 0) & (d_s_l <= scalers[fold_num]['max_l_train']))

            status_l = 'Unsupported / Out of Range'
            if p_b2_l_dom < 0.60:
                status_l = 'Weak-Curvature / Unidentifiable'
            elif in_max_l_dom > 0.70 and (np.percentile(d_s_l, 97.5) - np.percentile(d_s_l, 2.5) < 80):
                status_l = 'Identifiable D*'
            elif in_max_l_dom > 0.40:
                status_l = 'Boundary D*'

            fold_d_star_rows.append({
                'fold': fold_num,
                'domain': dom_name,
                'predictor': 'distance_to_lithology_contact',
                'd_star_median_km': np.median(d_s_l),
                'd_star_ci_lower_km': np.percentile(d_s_l, 2.5),
                'd_star_ci_upper_km': np.percentile(d_s_l, 97.5),
                'p_beta_sq_pos': p_b2_l_dom,
                'support_max_km': scalers[fold_num]['max_l_train'],
                'prop_in_support': in_max_l_dom,
                'classification': status_l
            })

            if dom_name == 'NRB_3a':
                d_star_samples_dict['nrb3a_lith'][fold_num] = d_s_l

        # -------------------------------------------------------------
        # D. Fold-Level Predictor Contributions (OOF Test Set)
        # -------------------------------------------------------------
        test_dom_idx = test_df['domain_idx'].values
        X_f = test_df['fault_z'].values
        X_f_sq = test_df['fault_z_sq'].values
        X_l = test_df['lith_z'].values
        X_l_sq = test_df['lith_z_sq'].values
        X_g = test_df['grav_z'].values
        X_r = test_df[valid_rocks].values

        # Posterior mean contributions per test cell
        c_intercept = np.mean(alpha_dom[:, test_dom_idx], axis=0)
        c_fault = np.mean(bf_lin[:, test_dom_idx] * X_f + bf_sq[:, test_dom_idx] * X_f_sq, axis=0)
        c_lith = np.mean(bl_lin[:, test_dom_idx] * X_l + bl_sq[:, test_dom_idx] * X_l_sq, axis=0)
        c_grav = np.mean(beta_grav[:, None] * X_g, axis=0)
        c_rocks = np.mean(np.dot(beta_rocks, X_r.T), axis=0)
        c_total = c_intercept + c_fault + c_lith + c_grav + c_rocks
        p_pred = expit(c_total)

        for obs_idx in range(len(test_df)):
            fold_contrib_rows.append({
                'id': test_df.iloc[obs_idx]['id'] if 'id' in test_df.columns else obs_idx,
                'fold': fold_num,
                'domain': test_df.iloc[obs_idx]['daly_domain'],
                'deposit_present': y_test[obs_idx],
                'eta_intercept': c_intercept[obs_idx],
                'eta_fault': c_fault[obs_idx],
                'eta_lith': c_lith[obs_idx],
                'eta_grav': c_grav[obs_idx],
                'eta_rocks': c_rocks[obs_idx],
                'eta_total': c_total[obs_idx],
                'prob_pred': p_pred[obs_idx]
            })

        # Summary statistics per fold
        sd_f = np.std(c_fault)
        sd_l = np.std(c_lith)
        sd_g = np.std(c_grav)
        sd_r = np.std(c_rocks)
        tot_sd = sd_f + sd_l + sd_g + sd_r

        share_f = (sd_f / tot_sd) * 100.0
        share_l = (sd_l / tot_sd) * 100.0
        share_g = (sd_g / tot_sd) * 100.0
        share_r = (sd_r / tot_sd) * 100.0

        # Dominant predictor identification
        shares = {'Distance to Fault': share_f, 'Distance to Lithology': share_l, 'Bouguer Gravity': share_g, 'Lithology Classes': share_r}
        top_pred, top_share = sorted(shares.items(), key=lambda x: x[1], reverse=True)[0]
        second_pred, second_share = sorted(shares.items(), key=lambda x: x[1], reverse=True)[1]
        
        if top_share - second_share > 15.0:
            dom_status = f"{top_pred} ({top_share:.1f}%)"
        else:
            dom_status = f"Shared / Co-dominant ({top_pred} {top_share:.1f}%, {second_pred} {second_share:.1f}%)"

        fold_pred_summary_rows.append({
            'fold': fold_num,
            'test_cells': len(test_df),
            'test_deposits': int(y_test.sum()),
            'fault_variance_std': sd_f,
            'fault_share_pct': share_f,
            'fault_median_abs': np.median(np.abs(c_fault)),
            'lith_variance_std': sd_l,
            'lith_share_pct': share_l,
            'lith_median_abs': np.median(np.abs(c_lith)),
            'gravity_variance_std': sd_g,
            'gravity_share_pct': share_g,
            'gravity_median_abs': np.median(np.abs(c_grav)),
            'rocks_variance_std': sd_r,
            'rocks_share_pct': share_r,
            'rocks_median_abs': np.median(np.abs(c_rocks)),
            'dominant_predictor': dom_status
        })

    # Convert to DataFrames
    df_pop_resp_f = pd.DataFrame(pop_resp_fault_rows)
    df_pop_resp_l = pd.DataFrame(pop_resp_lith_rows)
    df_pop_d_star = pd.DataFrame(pop_d_star_rows)
    df_fold_d_star = pd.DataFrame(fold_d_star_rows)
    df_fold_contrib = pd.DataFrame(fold_contrib_rows)
    df_fold_summary = pd.DataFrame(fold_pred_summary_rows)

    # -------------------------------------------------------------
    # 4. Save CSV Deliverables
    # -------------------------------------------------------------
    print("\n--- Saving CSV Deliverables ---")
    df_pop_resp_f.to_csv(OUT_DIR / 'population_response_fault.csv', index=False)
    print("  [+] population_response_fault.csv")

    df_pop_resp_l.to_csv(OUT_DIR / 'population_response_lithology.csv', index=False)
    print("  [+] population_response_lithology.csv")

    df_pop_d_star.to_csv(OUT_DIR / 'population_d_star_summary.csv', index=False)
    print("  [+] population_d_star_summary.csv")

    df_fold_d_star.to_csv(OUT_DIR / 'fold_d_star_summary.csv', index=False)
    print("  [+] fold_d_star_summary.csv")

    df_fold_contrib.to_csv(OUT_DIR / 'fold_predictor_contributions.csv', index=False)
    print("  [+] fold_predictor_contributions.csv")

    df_fold_summary.to_csv(OUT_DIR / 'fold_predictor_summary.csv', index=False)
    print("  [+] fold_predictor_summary.csv")

    verify_integrity("Post-CSV generation")

    # -------------------------------------------------------------
    # 5. Plotting Deliverables
    # -------------------------------------------------------------
    print("\n--- Generating Visualization Figures ---")

    # Fig 7: Population Response Curve - Fault
    fig, ax = plt.subplots(figsize=(9, 6), dpi=200)
    grp_f = df_pop_resp_f.groupby('grid_km').agg({'response_mean': 'mean', 'response_ci_lower': 'mean', 'response_ci_upper': 'mean'}).reset_index()
    ax.plot(grp_f['grid_km'], grp_f['response_mean'], color='#1f77b4', linewidth=2.5, label='Population Response (Mean across Folds)')
    ax.fill_between(grp_f['grid_km'], grp_f['response_ci_lower'], grp_f['response_ci_upper'], color='#1f77b4', alpha=0.20, label='95% Credible Interval Envelope')
    for f_idx in range(1, 5):
        f_sub = df_pop_resp_f[df_pop_resp_f['fold'] == f_idx]
        ax.plot(f_sub['grid_km'], f_sub['response_mean'], linestyle='--', linewidth=1.2, alpha=0.6, label=f'Fold {f_idx} Population')
    ax.set_title('V11 Population-Level Distance to Fault Response Curve', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Distance to Major Fault (km)', fontsize=11)
    ax.set_ylabel('Population Probability P(Y=1)', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlim(0, 60)
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'population_response_fault.png', bbox_inches='tight')
    plt.close(fig)
    print("  [+] population_response_fault.png")

    # Fig 8: Population Response Curve - Lithology
    fig, ax = plt.subplots(figsize=(9, 6), dpi=200)
    grp_l = df_pop_resp_l.groupby('grid_km').agg({'response_mean': 'mean', 'response_ci_lower': 'mean', 'response_ci_upper': 'mean'}).reset_index()
    ax.plot(grp_l['grid_km'], grp_l['response_mean'], color='#2ca02c', linewidth=2.5, label='Population Response (Mean across Folds)')
    ax.fill_between(grp_l['grid_km'], grp_l['response_ci_lower'], grp_l['response_ci_upper'], color='#2ca02c', alpha=0.20, label='95% Credible Interval Envelope')
    for f_idx in range(1, 5):
        f_sub = df_pop_resp_l[df_pop_resp_l['fold'] == f_idx]
        ax.plot(f_sub['grid_km'], f_sub['response_mean'], linestyle='--', linewidth=1.2, alpha=0.6, label=f'Fold {f_idx} Population')
    ax.set_title('V11 Population-Level Distance to Lithology Contact Response Curve', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Distance to Lithology Contact (km)', fontsize=11)
    ax.set_ylabel('Population Probability P(Y=1)', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlim(0, 50)
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'population_response_lithology.png', bbox_inches='tight')
    plt.close(fig)
    print("  [+] population_response_lithology.png")

    # Fig 9: Fold Response - Fault
    fig, ax = plt.subplots(figsize=(9, 6), dpi=200)
    fold_colors = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728'}
    for f_idx in range(1, 5):
        f_sub = df_pop_resp_f[df_pop_resp_f['fold'] == f_idx]
        ax.plot(f_sub['grid_km'], f_sub['response_mean'], color=fold_colors[f_idx], linewidth=2.2, label=f'Fold {f_idx} Population Response')
        ax.fill_between(f_sub['grid_km'], f_sub['response_ci_lower'], f_sub['response_ci_upper'], color=fold_colors[f_idx], alpha=0.10)
    ax.set_title('Fault Distance Response across Spatial Folds (Population Parameterization)', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Distance to Major Fault (km)', fontsize=11)
    ax.set_ylabel('Probability P(Y=1)', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlim(0, 60)
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'fold_response_fault.png', bbox_inches='tight')
    plt.close(fig)
    print("  [+] fold_response_fault.png")

    # Fig 10: Fold Response - Lithology
    fig, ax = plt.subplots(figsize=(9, 6), dpi=200)
    for f_idx in range(1, 5):
        f_sub = df_pop_resp_l[df_pop_resp_l['fold'] == f_idx]
        ax.plot(f_sub['grid_km'], f_sub['response_mean'], color=fold_colors[f_idx], linewidth=2.2, label=f'Fold {f_idx} Population Response')
        ax.fill_between(f_sub['grid_km'], f_sub['response_ci_lower'], f_sub['response_ci_upper'], color=fold_colors[f_idx], alpha=0.10)
    ax.set_title('Lithology Contact Response across Spatial Folds (Population Parameterization)', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Distance to Lithology Contact (km)', fontsize=11)
    ax.set_ylabel('Probability P(Y=1)', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlim(0, 50)
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'fold_response_lithology.png', bbox_inches='tight')
    plt.close(fig)
    print("  [+] fold_response_lithology.png")

    # Fig 11: Fold Predictor Contributions (Share of Variance Bar Chart)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=200)
    folds_arr = np.array([1, 2, 3, 4])
    w = 0.55

    # Stacked Variance Share
    p_fault = df_fold_summary['fault_share_pct'].values
    p_lith = df_fold_summary['lith_share_pct'].values
    p_grav = df_fold_summary['gravity_share_pct'].values
    p_rocks = df_fold_summary['rocks_share_pct'].values

    ax1.bar(folds_arr, p_fault, width=w, label='Distance to Fault', color='#1f77b4')
    ax1.bar(folds_arr, p_lith, width=w, bottom=p_fault, label='Distance to Lithology', color='#2ca02c')
    ax1.bar(folds_arr, p_grav, width=w, bottom=p_fault + p_lith, label='Bouguer Gravity', color='#ff7f0e')
    ax1.bar(folds_arr, p_rocks, width=w, bottom=p_fault + p_lith + p_grav, label='Lithology Classes', color='#9467bd')
    ax1.set_xticks(folds_arr)
    ax1.set_xticklabels([f'Fold 1 (AUC=0.817)\n[N=9 ore cells]', f'Fold 2 (AUC=0.688)\n[N=31 ore cells]', f'Fold 3 (AUC=0.676)\n[N=37 ore cells]', f'Fold 4 (AUC=0.561)\n[N=61 ore cells]'], fontsize=9.5)
    ax1.set_ylabel('Predictor Variance Contribution Share (%)', fontsize=11)
    ax1.set_title('A. Relative Predictor Variance Share across Folds', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    # Median Absolute Contribution
    x_pos = np.arange(4)
    bw = 0.20
    ax2.bar(x_pos - 1.5*bw, df_fold_summary['fault_median_abs'], width=bw, label='Distance to Fault', color='#1f77b4')
    ax2.bar(x_pos - 0.5*bw, df_fold_summary['lith_median_abs'], width=bw, label='Distance to Lithology', color='#2ca02c')
    ax2.bar(x_pos + 0.5*bw, df_fold_summary['gravity_median_abs'], width=bw, label='Bouguer Gravity', color='#ff7f0e')
    ax2.bar(x_pos + 1.5*bw, df_fold_summary['rocks_median_abs'], width=bw, label='Lithology Classes', color='#9467bd')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4'], fontsize=10)
    ax2.set_ylabel('Median Absolute Linear Contribution |η|', fontsize=11)
    ax2.set_title('B. Median Absolute Predictor Contribution', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)

    fig.suptitle('Spatial Heterogeneity in Predictor Importance across Along-Belt Folds', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(OUT_DIR / 'fold_predictor_contributions.png', bbox_inches='tight')
    plt.close(fig)
    print("  [+] fold_predictor_contributions.png")

    # Fig 12: Population D* Distributions (Showing Extreme Dispersion / Denominator Singularity)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=200)
    
    # Fault Pop D*
    all_pop_dstar_f = np.concatenate([d_star_samples_dict['pop_fault'][f] for f in range(1, 5)])
    all_pop_dstar_f_trimmed = all_pop_dstar_f[(all_pop_dstar_f > -100) & (all_pop_dstar_f < 200)]
    ax1.hist(all_pop_dstar_f_trimmed, bins=60, color='#1f77b4', edgecolor='black', alpha=0.7, density=True)
    ax1.axvline(0, color='red', linestyle='--', label='Lower Bound (0 km)')
    ax1.axvline(43.6, color='darkgreen', linestyle='--', label='Empirical P95 Support (43.6 km)')
    ax1.set_title('Population-Level Fault D* Posterior Distribution\n(Truncated [-100, 200] km for visibility; 95% CI spans [-279, +366] km)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Quadratic Stationary Point D* (km)', fontsize=10)
    ax1.set_ylabel('Posterior Density', fontsize=10)
    ax1.legend(fontsize=8.5)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # Lith Pop D*
    all_pop_dstar_l = np.concatenate([d_star_samples_dict['pop_lith'][f] for f in range(1, 5)])
    all_pop_dstar_l_trimmed = all_pop_dstar_l[(all_pop_dstar_l > -100) & (all_pop_dstar_l < 200)]
    ax2.hist(all_pop_dstar_l_trimmed, bins=60, color='#2ca02c', edgecolor='black', alpha=0.7, density=True)
    ax2.axvline(0, color='red', linestyle='--', label='Lower Bound (0 km)')
    ax2.axvline(46.9, color='darkgreen', linestyle='--', label='Empirical P95 Support (46.9 km)')
    ax2.set_title('Population-Level Lithology D* Posterior Distribution\n(Truncated [-100, 200] km for visibility; 95% CI spans [-161, +209] km)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Quadratic Stationary Point D* (km)', fontsize=10)
    ax2.set_ylabel('Posterior Density', fontsize=10)
    ax2.legend(fontsize=8.5)
    ax2.grid(True, linestyle='--', alpha=0.4)

    fig.suptitle('Population-Level Quadratic D* Diagnostic: Extreme Denominator Dispersion', fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(OUT_DIR / 'population_d_star_distributions.png', bbox_inches='tight')
    plt.close(fig)
    print("  [+] population_d_star_distributions.png")

    # Fig 13: Fold D* Distributions Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=200)

    # NRB_3a Fault D* across Folds vs Population
    for f_idx in range(1, 5):
        s_dom = d_star_samples_dict['nrb3a_fault'][f_idx]
        s_dom_sub = s_dom[(s_dom > -20) & (s_dom < 120)]
        ax1.hist(s_dom_sub, bins=30, alpha=0.45, density=True, label=f'NRB_3a Fold {f_idx} (Median={np.median(s_dom):.1f} km)')
    ax1.axvline(0, color='black', linestyle=':')
    ax1.set_title('A. Domain NRB_3a Fault D* across Folds', fontsize=11, fontweight='bold')
    ax1.set_xlabel('D* (km)', fontsize=10)
    ax1.set_ylabel('Density', fontsize=10)
    ax1.legend(fontsize=8.5)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # NRB_3a Lith D* across Folds vs Population
    for f_idx in range(1, 5):
        s_dom_l = d_star_samples_dict['nrb3a_lith'][f_idx]
        s_dom_l_sub = s_dom_l[(s_dom_l > -20) & (s_dom_l < 100)]
        ax2.hist(s_dom_l_sub, bins=30, alpha=0.45, density=True, label=f'NRB_3a Fold {f_idx} (Median={np.median(s_dom_l):.1f} km)')
    ax2.axvline(0, color='black', linestyle=':')
    ax2.set_title('B. Domain NRB_3a Lithology D* across Folds', fontsize=11, fontweight='bold')
    ax2.set_xlabel('D* (km)', fontsize=10)
    ax2.set_ylabel('Density', fontsize=10)
    ax2.legend(fontsize=8.5)
    ax2.grid(True, linestyle='--', alpha=0.4)

    fig.suptitle('Fold-to-Fold Comparison of Domain D* Distributions', fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(OUT_DIR / 'fold_d_star_distributions.png', bbox_inches='tight')
    plt.close(fig)
    print("  [+] fold_d_star_distributions.png")

    verify_integrity("End of Phase 2 Execution")
    print("\n[SUCCESS] Phase 2 calculations and visualizations completed successfully!")


if __name__ == '__main__':
    run_reconstruction()

