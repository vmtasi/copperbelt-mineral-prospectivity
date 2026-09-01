import os
import sys
from pathlib import Path

# --- THE BACKEND BYPASS ---
os.environ["PYTENSOR_FLAGS"] = "mode=NUMBA"
import pytensor

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation_strategies import get_along_belt_folds

def extract_v9_posteriors(n_folds=4):
    print("--- 1. Loading Datasets ---")
    data_path = ROOT / 'data' / 'copperbelt_training_v5_with_tectonic_domain.csv'
    df = pd.read_csv(data_path)

    lithology_col = 'litho_contact_litho_class'
    dist_to_lith_col = 'distance_to_lithology_contact'
    dist_to_fault_col = 'distance_to_fault'
    gravity_col = 'bouguer'  

    dist_to_lith_sq_col = 'distance_to_lithology_contact_sq'
    dist_to_fault_sq_col = 'distance_to_fault_sq'
    df[dist_to_lith_sq_col] = df[dist_to_lith_col] ** 2
    df[dist_to_fault_sq_col] = df[dist_to_fault_col] ** 2

    continuous_features = [dist_to_lith_col, dist_to_fault_col, gravity_col]
    continuous_sq_features = [dist_to_lith_sq_col, dist_to_fault_sq_col]
    all_continuous = continuous_features + continuous_sq_features

    df = df.dropna(subset=['centroid_x', 'centroid_y', lithology_col, 'domain'] + all_continuous).copy()
    
    print("--- 2. Engineering Daly's Tectonic Domains ---")
    df['domain'] = df['domain'].fillna('Unknown')
    
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
    
    df = pd.get_dummies(df, columns=['daly_domain'], drop_first=False, dtype=float)
    domain_cols = [c for c in df.columns if c.startswith('daly_domain_')]
    
    interaction_cols = [f'fault_x_{d}' for d in domain_cols]
    for int_col in interaction_cols:
        df[int_col] = 0.0

    print("--- 3. Engineering Lithological Features ---")
    df['spatial_block'] = get_along_belt_folds(df, n_folds=n_folds)

    df = pd.get_dummies(df, columns=[lithology_col], drop_first=True, dtype=float)
    rock_features = [col for col in df.columns if col.startswith(f'{lithology_col}_')]

    zero_deposit_dummies = [c for c in rock_features if df.loc[df[c] == 1, 'deposit_present'].sum() == 0]
    if zero_deposit_dummies:
        df = df.drop(columns=zero_deposit_dummies)
        rock_features = [c for c in rock_features if c not in zero_deposit_dummies]

    # Data structures to hold the extracted posterior matrices
    feature_matrix_raw = []
    feature_matrix_formatted = {}

    print("\n--- 4. Executing V9 Posterior Extraction (Fold by Fold) ---")
    for fold in range(n_folds):
        print(f"\n  Sampling Spatial Block {fold + 1} of {n_folds}...")
        
        train_df = df[df['spatial_block'] != fold].copy()
        y_train_temp = train_df['deposit_present'].values

        scaler = StandardScaler()
        train_df[all_continuous] = scaler.fit_transform(train_df[all_continuous])
        
        for d_col, int_col in zip(domain_cols, interaction_cols):
            train_df[int_col] = train_df[dist_to_fault_col] * train_df[d_col]

        model_continuous_features = [c for c in continuous_features if c != dist_to_fault_col]
        valid_features = model_continuous_features + continuous_sq_features + domain_cols + interaction_cols
        
        for col in rock_features:
            if ((train_df[col] == 1) & (y_train_temp == 1)).sum() > 0 and ((train_df[col] == 1) & (y_train_temp == 0)).sum() > 0:
                valid_features.append(col)

        X_train_scaled = train_df[valid_features].values.astype(float)
        y_train = train_df['deposit_present'].values.astype(np.int32)
        logit_base_rate = float(np.log(y_train.sum() / (len(y_train) - y_train.sum())))

        with pm.Model() as prospectivity_model:
            alpha = pm.Normal('alpha', mu=logit_base_rate, sigma=1, initval=logit_base_rate)
            
            mu_dom = pm.Normal('mu_dom', mu=0.0, sigma=1.0)
            sigma_dom = pm.HalfNormal('sigma_dom', sigma=1.0)
            mu_fault = pm.Normal('mu_fault', mu=0.0, sigma=1.0)
            sigma_fault = pm.HalfNormal('sigma_fault', sigma=1.0)
            
            beta_coefficients = []
            for idx, feat_name in enumerate(valid_features):
                if feat_name in domain_cols:
                    offset = pm.Normal(f'offset_{feat_name}', mu=0.0, sigma=1.0)
                    b = pm.Deterministic(f'beta_{feat_name}', mu_dom + offset * sigma_dom)
                elif feat_name in interaction_cols:
                    offset = pm.Normal(f'offset_{feat_name}', mu=0.0, sigma=1.0)
                    b = pm.Deterministic(f'beta_{feat_name}', mu_fault + offset * sigma_fault)
                else:
                    b = pm.Normal(f'beta_{feat_name}', mu=0.0, sigma=1.0, initval=0.0)
                beta_coefficients.append(b)
            
            beta_vector = pm.math.stack(beta_coefficients)
            mu = alpha + pm.math.dot(X_train_scaled, beta_vector)
            y_obs = pm.Bernoulli('y_obs', logit_p=mu, observed=y_train)
            
            trace = pm.sample(draws=1000, tune=1500, chains=2, cores=1, target_accept=0.99, progressbar=False, random_seed=42)

        # Extract posterior logic for the supervisor's matrix
        summary_df = az.summary(trace, var_names=[f'beta_{f}' for f in valid_features], hdi_prob=0.95)
        
        for feat in valid_features:
            samples = trace.posterior[f'beta_{feat}'].values.flatten()
            mean_val = np.mean(samples)
            hdi_2_5 = np.percentile(samples, 2.5)
            hdi_97_5 = np.percentile(samples, 97.5)
            p_neg = np.mean(samples < 0)
            
            # Formatted text cell for CSV: "Mean [2.5%, 97.5%] (P<0: X%)"
            cell_text = f"{mean_val:.2f} [{hdi_2_5:.2f}, {hdi_97_5:.2f}] (P<0: {p_neg:.2f})"
            
            if feat not in feature_matrix_formatted:
                feature_matrix_formatted[feat] = {}
            feature_matrix_formatted[feat][f'Block {fold+1}'] = cell_text
            
            feature_matrix_raw.append({
                'Feature': feat,
                'Block': f'Block {fold+1}',
                'Mean': mean_val,
                'HDI_lower': hdi_2_5,
                'HDI_upper': hdi_97_5,
                'P_neg': p_neg
            })

    print("\n--- 5. Generating Matrices & Forest Plots ---")
    output_dir = ROOT / 'figures'
    os.makedirs(output_dir, exist_ok=True)

    # 1. The Supervisor's Formatted Feature x Block Matrix
    formatted_df = pd.DataFrame.from_dict(feature_matrix_formatted, orient='index')
    formatted_df.index.name = 'Feature'
    formatted_df.to_csv(output_dir / 'v9_feature_x_block_matrix.csv')
    print(f"  -> Saved text matrix to: {output_dir / 'v9_feature_x_block_matrix.csv'}")

    # 2. Forest Plots for Spatial Heterogeneity
    raw_df = pd.DataFrame(feature_matrix_raw)
    
    # We will plot the fault interactions across blocks to visually prove heterogeneity
    fault_features = [f for f in raw_df['Feature'].unique() if 'fault_x' in f]
    
    for feat in fault_features:
        feat_data = raw_df[raw_df['Feature'] == feat]
        
        plt.figure(figsize=(8, 4))
        plt.errorbar(feat_data['Mean'], feat_data['Block'], 
                     xerr=[feat_data['Mean'] - feat_data['HDI_lower'], feat_data['HDI_upper'] - feat_data['Mean']], 
                     fmt='o', color='royalblue', ecolor='lightgray', elinewidth=3, capsize=0)
        plt.axvline(0, color='black', linestyle='--', linewidth=1)
        plt.title(f'Spatial Stability of {feat.replace("daly_domain_", "")}')
        plt.xlabel('Posterior Beta (Log-Odds Effect)')
        plt.tight_layout()
        plt.savefig(output_dir / f'forest_plot_{feat}.png', dpi=200)
        plt.close()
        
    print(f"  -> Saved Forest Plots to {output_dir}")

if __name__ == "__main__":
    extract_v9_posteriors()