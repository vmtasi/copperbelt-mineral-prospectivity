import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Allow PyTensor to use the native C++ compiler (Smart App Control is OFF)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation_strategies import get_along_belt_folds

def run_v11_spatial_cv(n_folds=4):
    print("--- 1. Loading Datasets ---")
    data_path = ROOT / 'data' / 'copperbelt_training_v5_with_tectonic_domain.csv'
    df = pd.read_csv(data_path)

    lithology_col = 'litho_contact_litho_class'
    dist_to_lith_col = 'distance_to_lithology_contact'
    dist_to_fault_col = 'distance_to_fault'
    gravity_col = 'bouguer'  
    continuous_features = [dist_to_lith_col, dist_to_fault_col, gravity_col]

    df = df.dropna(subset=['centroid_x', 'centroid_y', lithology_col, 'domain'] + continuous_features).copy()
    
    print("--- 2. Engineering Daly's Tectonic Domains (Index Mapping) ---")
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
    n_domains = len(unique_domains)

    print("--- 3. Preparing Spatial Folds ---")
    df['spatial_block'] = get_along_belt_folds(df, n_folds=n_folds)
    
    df = pd.get_dummies(df, columns=[lithology_col], drop_first=True, dtype=float)
    all_rock_features = [col for col in df.columns if col.startswith(f'{lithology_col}_')]

    fold_metrics = []
    posterior_stability_records = []
    
    output_dir = ROOT / 'figures'
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n--- 4. Executing V11 Spatial CV (Native C++ Backend) ---")
    for fold in range(n_folds):
        print(f"\n{'='*50}\nEvaluating Spatial Block {fold + 1} of {n_folds}\n{'='*50}")
        
        train_df = df[df['spatial_block'] != fold].copy()
        test_df = df[df['spatial_block'] == fold].copy()
        
        y_train = train_df['deposit_present'].values.astype(np.int32)
        y_test = test_df['deposit_present'].values.astype(np.int32)

        if sum(y_test) == 0 or sum(y_train) == 0:
            print(f"Skipping Block {fold+1}: Class imbalance anomaly.")
            continue

        # 1. Fit Scalers STRICTLY on Training data (Zero Leakage)
        scaler_fault = StandardScaler().fit(train_df[[dist_to_fault_col]])
        scaler_lith = StandardScaler().fit(train_df[[dist_to_lith_col]])
        scaler_grav = StandardScaler().fit(train_df[[gravity_col]])
        
        mu_fault = scaler_fault.mean_[0]
        std_fault = scaler_fault.scale_[0]

        # 2. Transform into Z-space
        for d in [train_df, test_df]:
            d['fault_z'] = scaler_fault.transform(d[[dist_to_fault_col]])
            d['lith_z'] = scaler_lith.transform(d[[dist_to_lith_col]])
            d['grav_z'] = scaler_grav.transform(d[[gravity_col]])
            # 3. Create Z^2 directly from Z-space
            d['fault_z_sq'] = d['fault_z'] ** 2
            d['lith_z_sq'] = d['lith_z'] ** 2

        # 4. Strict Train-Only valid rock filtering
        valid_rocks = []
        for col in all_rock_features:
            if ((train_df[col] == 1) & (y_train == 1)).sum() > 0 and ((train_df[col] == 1) & (y_train == 0)).sum() > 0:
                valid_rocks.append(col)

        # Build Inference Arrays
        X_train_grav = train_df['grav_z'].values
        X_train_f_lin = train_df['fault_z'].values
        X_train_f_sq = train_df['fault_z_sq'].values
        X_train_l_lin = train_df['lith_z'].values
        X_train_l_sq = train_df['lith_z_sq'].values
        X_train_rocks = train_df[valid_rocks].values
        train_domain_idx = train_df['domain_idx'].values
        
        X_test_grav = test_df['grav_z'].values
        X_test_f_lin = test_df['fault_z'].values
        X_test_f_sq = test_df['fault_z_sq'].values
        X_test_l_lin = test_df['lith_z'].values
        X_test_l_sq = test_df['lith_z_sq'].values
        X_test_rocks = test_df[valid_rocks].values
        test_domain_idx = test_df['domain_idx'].values

        logit_base_rate = float(np.log(y_train.sum() / (len(y_train) - y_train.sum())))

        with pm.Model() as prospectivity_model:
            # Non-Centered Hierarchical Architecture
            alpha_mu = pm.Normal('alpha_mu', mu=logit_base_rate, sigma=1.0)
            alpha_sigma = pm.HalfNormal('alpha_sigma', sigma=1.0)
            alpha_offset = pm.Normal('alpha_offset', mu=0.0, sigma=1.0, shape=n_domains)
            alpha_dom = pm.Deterministic('alpha_dom', alpha_mu + alpha_offset * alpha_sigma)

            mu_f_lin = pm.Normal('mu_f_lin', mu=0.0, sigma=1.0)
            sigma_f_lin = pm.HalfNormal('sigma_f_lin', sigma=1.0)
            offset_f_lin = pm.Normal('offset_f_lin', mu=0.0, sigma=1.0, shape=n_domains)
            beta_f_lin = pm.Deterministic('beta_f_lin', mu_f_lin + offset_f_lin * sigma_f_lin)

            mu_f_sq = pm.Normal('mu_f_sq', mu=0.0, sigma=1.0)
            sigma_f_sq = pm.HalfNormal('sigma_f_sq', sigma=1.0)
            offset_f_sq = pm.Normal('offset_f_sq', mu=0.0, sigma=1.0, shape=n_domains)
            beta_f_sq = pm.Deterministic('beta_f_sq', mu_f_sq + offset_f_sq * sigma_f_sq)

            mu_l_lin = pm.Normal('mu_l_lin', mu=0.0, sigma=1.0)
            sigma_l_lin = pm.HalfNormal('sigma_l_lin', sigma=1.0)
            offset_l_lin = pm.Normal('offset_l_lin', mu=0.0, sigma=1.0, shape=n_domains)
            beta_l_lin = pm.Deterministic('beta_l_lin', mu_l_lin + offset_l_lin * sigma_l_lin)

            mu_l_sq = pm.Normal('mu_l_sq', mu=0.0, sigma=1.0)
            sigma_l_sq = pm.HalfNormal('sigma_l_sq', sigma=1.0)
            offset_l_sq = pm.Normal('offset_l_sq', mu=0.0, sigma=1.0, shape=n_domains)
            beta_l_sq = pm.Deterministic('beta_l_sq', mu_l_sq + offset_l_sq * sigma_l_sq)

            beta_grav = pm.Normal('beta_grav', mu=0.0, sigma=1.0)
            beta_rocks = pm.Normal('beta_rocks', mu=0.0, sigma=1.0, shape=len(valid_rocks))

            mu = (
                alpha_dom[train_domain_idx] + 
                beta_f_lin[train_domain_idx] * X_train_f_lin + 
                beta_f_sq[train_domain_idx] * X_train_f_sq +
                beta_l_lin[train_domain_idx] * X_train_l_lin + 
                beta_l_sq[train_domain_idx] * X_train_l_sq +
                beta_grav * X_train_grav + 
                pm.math.dot(X_train_rocks, beta_rocks)
            )
            
            y_obs = pm.Bernoulli('y_obs', logit_p=mu, observed=y_train)
            trace = pm.sample(draws=1500, tune=2500, chains=2, cores=1, target_accept=0.99, progressbar=True, random_seed=42)
            
            trace_path = output_dir / f'v11_fold_{fold+1}_trace.nc'
            trace.to_netcdf(trace_path)
            print(f"  -> Fold {fold+1} Trace permanently saved to {trace_path}")

        # Out-Of-Sample Prediction Matrix
        alpha_samples = trace.posterior['alpha_dom'].values.reshape(-1, n_domains)
        b_f_lin_samples = trace.posterior['beta_f_lin'].values.reshape(-1, n_domains)
        b_f_sq_samples = trace.posterior['beta_f_sq'].values.reshape(-1, n_domains)
        b_l_lin_samples = trace.posterior['beta_l_lin'].values.reshape(-1, n_domains)
        b_l_sq_samples = trace.posterior['beta_l_sq'].values.reshape(-1, n_domains)
        b_grav_samples = trace.posterior['beta_grav'].values.flatten()
        b_rocks_samples = trace.posterior['beta_rocks'].values.reshape(-1, len(valid_rocks))
        
        logit_test = (
            alpha_samples[:, test_domain_idx] +
            b_f_lin_samples[:, test_domain_idx] * X_test_f_lin +
            b_f_sq_samples[:, test_domain_idx] * X_test_f_sq +
            b_l_lin_samples[:, test_domain_idx] * X_test_l_lin +
            b_l_sq_samples[:, test_domain_idx] * X_test_l_sq +
            b_grav_samples[:, None] * X_test_grav +
            np.dot(b_rocks_samples, X_test_rocks.T)
        )
        
        prob_test = 1 / (1 + np.exp(-logit_test))
        mean_probs = np.mean(prob_test, axis=0)

        fold_auc = roc_auc_score(y_test, mean_probs)
        fold_metrics.append(fold_auc)
        print(f"  -> V11 Block {fold + 1} Out-of-Sample AUC: {fold_auc:.3f}")

        # Fold-by-Fold Geological Stability Extraction
        print(f"  -> Extracting Spatial Stability Metrics for Block {fold + 1}...")
        for idx, dom_name in enumerate(unique_domains):
            b1_trace = b_f_lin_samples[:, idx]
            b2_trace = b_f_sq_samples[:, idx]
            
            prob_curvature = np.mean(b2_trace > 0)
            record = {
                'Fold': fold + 1,
                'Domain': dom_name,
                'P(b2 > 0)': prob_curvature,
                'Median_D_star_km': np.nan,
                'HDI_lower_km': np.nan,
                'HDI_upper_km': np.nan,
                'P(0 < D_star < 50)': np.nan
            }
            
            curvature_mask = b2_trace > 0.05
            if np.mean(curvature_mask) > 0.5:
                z_star_trace = -b1_trace[curvature_mask] / (2 * b2_trace[curvature_mask])
                d_star_km_trace = ((z_star_trace * std_fault) + mu_fault) / 1000.0
                
                record['Median_D_star_km'] = np.median(d_star_km_trace)
                record['HDI_lower_km'] = np.percentile(d_star_km_trace, 2.5)
                record['HDI_upper_km'] = np.percentile(d_star_km_trace, 97.5)
                record['P(0 < D_star < 50)'] = np.mean((d_star_km_trace >= 0) & (d_star_km_trace <= 50))
                
            posterior_stability_records.append(record)

    if fold_metrics:
        print("\n" + "="*50)
        print(f"V11 AGGREGATE OUT-OF-SAMPLE AUC (Mean): {np.mean(fold_metrics):.3f}")
        print("="*50)
        
        stability_df = pd.DataFrame(posterior_stability_records)
        stability_csv = output_dir / 'v11_geological_stability_matrix.csv'
        stability_df.to_csv(stability_csv, index=False)
        print(f"Geological stability tracked and saved to: {stability_csv}")

if __name__ == "__main__":
    run_v11_spatial_cv()