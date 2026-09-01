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

def run_v10_heterogeneity():
    print("--- 1. Loading Datasets ---")
    data_path = ROOT / 'data' / 'copperbelt_training_v5_with_tectonic_domain.csv'
    df = pd.read_csv(data_path)

    lithology_col = 'litho_contact_litho_class'
    dist_to_lith_col = 'distance_to_lithology_contact'
    dist_to_fault_col = 'distance_to_fault'
    gravity_col = 'bouguer'  

    # Create raw squared features (we scale them afterwards to keep the geometry stable)
    dist_to_lith_sq_col = 'distance_to_lithology_contact_sq'
    dist_to_fault_sq_col = 'distance_to_fault_sq'
    df[dist_to_lith_sq_col] = df[dist_to_lith_col] ** 2
    df[dist_to_fault_sq_col] = df[dist_to_fault_col] ** 2

    continuous_features = [dist_to_lith_col, dist_to_fault_col, dist_to_lith_sq_col, dist_to_fault_sq_col, gravity_col]
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

    print("--- 3. Constructing Spatially Heterogeneous Features ---")
    # Fit independent scalers so we can inverse-transform for the 0-50km response curves later
    scaler_fault = StandardScaler().fit(df[[dist_to_fault_col]])
    scaler_fault_sq = StandardScaler().fit(df[[dist_to_fault_sq_col]])
    scaler_lith = StandardScaler().fit(df[[dist_to_lith_col]])
    scaler_lith_sq = StandardScaler().fit(df[[dist_to_lith_sq_col]])
    scaler_grav = StandardScaler().fit(df[[gravity_col]])

    df[dist_to_fault_col] = scaler_fault.transform(df[[dist_to_fault_col]])
    df[dist_to_fault_sq_col] = scaler_fault_sq.transform(df[[dist_to_fault_sq_col]])
    df[dist_to_lith_col] = scaler_lith.transform(df[[dist_to_lith_col]])
    df[dist_to_lith_sq_col] = scaler_lith_sq.transform(df[[dist_to_lith_sq_col]])
    df[gravity_col] = scaler_grav.transform(df[[gravity_col]])

    # Multiply scaled variables by the domain masks
    fault_lin_interactions = []
    fault_sq_interactions = []
    lith_lin_interactions = []
    lith_sq_interactions = []

    for d_col in domain_cols:
        f_lin = f'fault_lin_x_{d_col}'
        f_sq = f'fault_sq_x_{d_col}'
        l_lin = f'lith_lin_x_{d_col}'
        l_sq = f'lith_sq_x_{d_col}'
        
        df[f_lin] = df[dist_to_fault_col] * df[d_col]
        df[f_sq] = df[dist_to_fault_sq_col] * df[d_col]
        df[l_lin] = df[dist_to_lith_col] * df[d_col]
        df[l_sq] = df[dist_to_lith_sq_col] * df[d_col]
        
        fault_lin_interactions.append(f_lin)
        fault_sq_interactions.append(f_sq)
        lith_lin_interactions.append(l_lin)
        lith_sq_interactions.append(l_sq)

    # Rock classes
    df = pd.get_dummies(df, columns=[lithology_col], drop_first=True, dtype=float)
    rock_features = [col for col in df.columns if col.startswith(f'{lithology_col}_')]
    valid_rocks = [c for c in rock_features if df.loc[df[c] == 1, 'deposit_present'].sum() > 0]

    # Model Matrix
    valid_features = [gravity_col] + domain_cols + fault_lin_interactions + fault_sq_interactions + lith_lin_interactions + lith_sq_interactions + valid_rocks
    
    X_train = df[valid_features].values.astype(float)
    y_train = df['deposit_present'].values.astype(np.int32)
    logit_base_rate = float(np.log(y_train.sum() / (len(y_train) - y_train.sum())))

    print("\n--- 4. Executing V10 Master Inference Model (Numba Backend) ---")
    print("Fitting on 100% of data to extract definitive posterior response curves...")

    with pm.Model() as prospectivity_model:
        alpha = pm.Normal('alpha', mu=logit_base_rate, sigma=1, initval=logit_base_rate)
        
        # Hierarchical hyper-priors
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

    print("\n--- 5. Extracting Posterior Matrices ---")
    summary_df = az.summary(trace, var_names=[f'beta_{f}' for f in valid_features], hdi_prob=0.95)
    
    # Calculate Directional Probability P(beta < 0)
    prob_negative = []
    for f in valid_features:
        samples = trace.posterior[f'beta_{f}'].values.flatten()
        prob_negative.append(np.mean(samples < 0))
    summary_df['P(beta < 0)'] = prob_negative
    
    output_dir = ROOT / 'figures'
    os.makedirs(output_dir, exist_ok=True)
    summary_df.to_csv(output_dir / 'v10_posterior_summary.csv')
    print(summary_df[['mean', 'hdi_2.5%', 'hdi_97.5%', 'P(beta < 0)']].head(15))

    print("\n--- 6. Generating Domain-Specific Response Curves (0 - 50 km) ---")
    # Synthetic array of 0 to 50 km
    km_array = np.linspace(0, 50, 100)
    
    # Scale synthetic array using the fitted standardizers
    fault_scaled = scaler_fault.transform(km_array.reshape(-1, 1)).flatten()
    fault_sq_scaled = scaler_fault_sq.transform((km_array**2).reshape(-1, 1)).flatten()

    alpha_samples = trace.posterior['alpha'].values.flatten()
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(domain_cols)))

    for idx, d_col in enumerate(domain_cols):
        dom_name = d_col.replace('daly_domain_', '')
        
        # Extract the posterior samples specifically for this domain
        beta_dom = trace.posterior[f'beta_{d_col}'].values.flatten()
        beta_f_lin = trace.posterior[f'beta_fault_lin_x_{d_col}'].values.flatten()
        beta_f_sq = trace.posterior[f'beta_fault_sq_x_{d_col}'].values.flatten()
        
        domain_curves = []
        for i in range(len(km_array)):
            # Calculate logit: alpha + Domain_Intercept + (Fault_Lin * scaled_km) + (Fault_Sq * scaled_km^2)
            # Assuming other variables (gravity, lithology) are held at their mean (0.0 in scaled space)
            logit_p = alpha_samples + beta_dom + (beta_f_lin * fault_scaled[i]) + (beta_f_sq * fault_sq_scaled[i])
            prob = 1 / (1 + np.exp(-logit_p))
            domain_curves.append(prob)
            
        domain_curves = np.array(domain_curves) # shape: (100 km_points, N_samples)
        
        mean_curve = np.mean(domain_curves, axis=1)
        lower_band = np.percentile(domain_curves, 2.5, axis=1)
        upper_band = np.percentile(domain_curves, 97.5, axis=1)
        
        plt.plot(km_array, mean_curve, label=dom_name, color=colors[idx], lw=2)
        plt.fill_between(km_array, lower_band, upper_band, color=colors[idx], alpha=0.1)

    plt.title('Posterior Fault Proximity Response by Tectonic Domain', fontsize=14)
    plt.xlabel('Distance to Fault (km)', fontsize=12)
    plt.ylabel('Marginal Probability of Mineralization', fontsize=12)
    plt.xlim(0, 50)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Daly (2025) Domain')
    plt.tight_layout()
    
    plot_path = output_dir / 'v10_fault_response_curves.png'
    plt.savefig(plot_path, dpi=300)
    print(f"\nResponse curves saved to: {plot_path}")

if __name__ == "__main__":
    run_v10_heterogeneity()