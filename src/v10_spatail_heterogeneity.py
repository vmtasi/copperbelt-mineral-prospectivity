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

def run_v10_inference():
    print("--- 1. Loading Datasets ---")
    data_path = ROOT / 'data' / 'copperbelt_training_v5_with_tectonic_domain.csv'
    df = pd.read_csv(data_path)

    lithology_col = 'litho_contact_litho_class'
    dist_to_lith_col = 'distance_to_lithology_contact'
    dist_to_fault_col = 'distance_to_fault'
    gravity_col = 'bouguer'  

    continuous_features = [dist_to_lith_col, dist_to_fault_col, gravity_col]
    df = df.dropna(subset=['centroid_x', 'centroid_y', lithology_col, 'domain'] + continuous_features).copy()
    
    print("--- 2. Engineering Daly's Tectonic Domains ---")
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

    print("--- 3. Constructing Correctly Scaled Polynomial Features ---")
    scaler_fault = StandardScaler().fit(df[[dist_to_fault_col]])
    scaler_lith = StandardScaler().fit(df[[dist_to_lith_col]])
    scaler_grav = StandardScaler().fit(df[[gravity_col]])

    df['fault_z'] = scaler_fault.transform(df[[dist_to_fault_col]])
    df['lith_z'] = scaler_lith.transform(df[[dist_to_lith_col]])
    df['grav_z'] = scaler_grav.transform(df[[gravity_col]])

    df['fault_z_sq'] = df['fault_z'] ** 2
    df['lith_z_sq'] = df['lith_z'] ** 2

    fault_lin_interactions = []
    fault_sq_interactions = []
    lith_lin_interactions = []
    lith_sq_interactions = []

    for d_col in domain_cols:
        f_lin = f'fault_lin_x_{d_col}'
        f_sq = f'fault_sq_x_{d_col}'
        l_lin = f'lith_lin_x_{d_col}'
        l_sq = f'lith_sq_x_{d_col}'
        
        df[f_lin] = df['fault_z'] * df[d_col]
        df[f_sq] = df['fault_z_sq'] * df[d_col]
        df[l_lin] = df['lith_z'] * df[d_col]
        df[l_sq] = df['lith_z_sq'] * df[d_col]
        
        fault_lin_interactions.append(f_lin)
        fault_sq_interactions.append(f_sq)
        lith_lin_interactions.append(l_lin)
        lith_sq_interactions.append(l_sq)

    df = pd.get_dummies(df, columns=[lithology_col], drop_first=True, dtype=float)
    rock_features = [col for col in df.columns if col.startswith(f'{lithology_col}_')]
    valid_rocks = [c for c in rock_features if df.loc[df[c] == 1, 'deposit_present'].sum() > 0]

    valid_features = ['grav_z'] + domain_cols + fault_lin_interactions + fault_sq_interactions + lith_lin_interactions + lith_sq_interactions + valid_rocks
    
    X_train = df[valid_features].values.astype(float)
    y_train = df['deposit_present'].values.astype(np.int32)
    logit_base_rate = float(np.log(y_train.sum() / (len(y_train) - y_train.sum())))

    print("\n--- 4. Executing V10 Inference Model (100% Data, Numba Backend) ---")
    with pm.Model() as prospectivity_model:
        alpha = pm.Normal('alpha', mu=logit_base_rate, sigma=1, initval=logit_base_rate)
        
        mu_dom = pm.Normal('mu_dom', mu=0.0, sigma=1.0)
        sigma_dom = pm.HalfNormal('sigma_dom', sigma=1.0)
        
        mu_f_lin = pm.Normal('mu_f_lin', mu=0.0, sigma=1.0)
        sigma_f_lin = pm.HalfNormal('sigma_f_lin', sigma=1.0)
        mu_f_sq = pm.Normal('mu_f_sq', mu=0.0, sigma=1.0)
        sigma_f_sq = pm.HalfNormal('sigma_f_sq', sigma=1.0)

        mu_l_lin = pm.Normal('mu_l_lin', mu=0.0, sigma=1.0)
        sigma_l_lin = pm.HalfNormal('sigma_l_lin', sigma=1.0)
        mu_l_sq = pm.Normal('mu_l_sq', mu=0.0, sigma=1.0)
        sigma_l_sq = pm.HalfNormal('sigma_l_sq', sigma=1.0)
        
        beta_coefficients = []
        for feat_name in valid_features:
            if feat_name in domain_cols:
                offset = pm.Normal(f'offset_{feat_name}', mu=0.0, sigma=1.0)
                b = pm.Deterministic(f'beta_{feat_name}', mu_dom + offset * sigma_dom)
            elif feat_name in fault_lin_interactions:
                offset = pm.Normal(f'offset_{feat_name}', mu=0.0, sigma=1.0)
                b = pm.Deterministic(f'beta_{feat_name}', mu_f_lin + offset * sigma_f_lin)
            elif feat_name in fault_sq_interactions:
                offset = pm.Normal(f'offset_{feat_name}', mu=0.0, sigma=1.0)
                b = pm.Deterministic(f'beta_{feat_name}', mu_f_sq + offset * sigma_f_sq)
            elif feat_name in lith_lin_interactions:
                offset = pm.Normal(f'offset_{feat_name}', mu=0.0, sigma=1.0)
                b = pm.Deterministic(f'beta_{feat_name}', mu_l_lin + offset * sigma_l_lin)
            elif feat_name in lith_sq_interactions:
                offset = pm.Normal(f'offset_{feat_name}', mu=0.0, sigma=1.0)
                b = pm.Deterministic(f'beta_{feat_name}', mu_l_sq + offset * sigma_l_sq)
            else:
                b = pm.Normal(f'beta_{feat_name}', mu=0.0, sigma=1.0)
                
            beta_coefficients.append(b)
        
        beta_vector = pm.math.stack(beta_coefficients)
        mu = alpha + pm.math.dot(X_train, beta_vector)
        y_obs = pm.Bernoulli('y_obs', logit_p=mu, observed=y_train)
        
        trace = pm.sample(draws=1500, tune=2000, chains=2, cores=1, target_accept=0.99, progressbar=True, random_seed=42)

    print("\n--- 5. Generating Geological Response Curves (0 - 50 km) ---")
    output_dir = ROOT / 'figures'
    os.makedirs(output_dir, exist_ok=True)
    
    m_array = np.linspace(0, 50000, 200)
    km_axis = m_array / 1000.0  
    
    mu_fault = scaler_fault.mean_[0]
    std_fault = scaler_fault.scale_[0]
    
    Z_fault = (m_array - mu_fault) / std_fault
    Z_fault_sq = Z_fault ** 2

    alpha_samples = trace.posterior['alpha'].values.flatten()
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    colors = plt.cm.tab10(np.linspace(0, 1, len(domain_cols)))

    for idx, d_col in enumerate(domain_cols):
        ax = axes[idx]
        dom_name = d_col.replace('daly_domain_', '')
        
        beta_dom = trace.posterior[f'beta_{d_col}'].values.flatten()
        beta_f_lin = trace.posterior[f'beta_fault_lin_x_{d_col}'].values.flatten()
        beta_f_sq = trace.posterior[f'beta_fault_sq_x_{d_col}'].values.flatten()
        
        logit_matrix = alpha_samples[:, None] + beta_dom[:, None] + np.outer(beta_f_lin, Z_fault) + np.outer(beta_f_sq, Z_fault_sq)
        prob_matrix = 1 / (1 + np.exp(-logit_matrix)) 
        
        mean_curve = np.mean(prob_matrix, axis=0)
        lower_band = np.percentile(prob_matrix, 2.5, axis=0)
        upper_band = np.percentile(prob_matrix, 97.5, axis=0)
        
        ax.plot(km_axis, mean_curve, label='Posterior Mean', color=colors[idx], lw=3)
        ax.fill_between(km_axis, lower_band, upper_band, color=colors[idx], alpha=0.2, label='95% HDI')
        
        ax.set_title(f'Tectonic Domain: {dom_name}', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        if idx >= 3: ax.set_xlabel('Distance to Fault (km)', fontsize=11)
        if idx % 3 == 0: ax.set_ylabel('Conditional Probability', fontsize=11)
        
        ax.set_xlim(0, 50)
        ax.legend(loc='upper right')

    plt.suptitle('Posterior Fault Proximity Response by Daly (2025) Tectonic Domain\n(Predicted probability at mean values of other predictors)', fontsize=16, y=1.02)
    plt.tight_layout()
    
    plot_path = output_dir / 'v10_fault_response_curves_fixed.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nCorrected response curves saved to: {plot_path}")

    print("\n--- Effective Turning Points (D*) from Quadratic Curvature ---")
    for d_col in domain_cols:
        dom_name = d_col.replace('daly_domain_', '')
        beta_f_lin_mean = np.mean(trace.posterior[f'beta_fault_lin_x_{d_col}'].values)
        beta_f_sq_mean = np.mean(trace.posterior[f'beta_fault_sq_x_{d_col}'].values)
        
        if beta_f_sq_mean > 0.05:  
            z_star = -beta_f_lin_mean / (2 * beta_f_sq_mean)
            d_star_km = (z_star * std_fault + mu_fault) / 1000.0
            print(f"{dom_name.ljust(10)}: Peak marginal shift occurs at ~{d_star_km:.1f} km")
        else:
            print(f"{dom_name.ljust(10)}: Predominantly monotonic / weak curvature")

if __name__ == "__main__":
    run_v10_inference()