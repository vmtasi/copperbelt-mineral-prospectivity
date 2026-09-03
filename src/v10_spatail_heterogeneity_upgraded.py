import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Allow PyTensor to use the native C++ compiler now that Smart App Control is off
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

def run_v10_final():
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
    domain_idx = df['domain_idx'].values

    print("--- 3. Constructing Correctly Scaled Polynomial Features ---")
    scaler_fault = StandardScaler().fit(df[[dist_to_fault_col]])
    scaler_lith = StandardScaler().fit(df[[dist_to_lith_col]])
    scaler_grav = StandardScaler().fit(df[[gravity_col]])

    df['fault_z'] = scaler_fault.transform(df[[dist_to_fault_col]])
    df['lith_z'] = scaler_lith.transform(df[[dist_to_lith_col]])
    df['grav_z'] = scaler_grav.transform(df[[gravity_col]])

    df['fault_z_sq'] = df['fault_z'] ** 2
    df['lith_z_sq'] = df['lith_z'] ** 2

    df = pd.get_dummies(df, columns=[lithology_col], drop_first=True, dtype=float)
    rock_features = [col for col in df.columns if col.startswith(f'{lithology_col}_')]
    valid_rocks = [c for c in rock_features if df.loc[df[c] == 1, 'deposit_present'].sum() > 0]
    
    X_rocks = df[valid_rocks].values.astype(float)
    X_grav = df['grav_z'].values.astype(float)
    X_fault_lin = df['fault_z'].values.astype(float)
    X_fault_sq = df['fault_z_sq'].values.astype(float)
    X_lith_lin = df['lith_z'].values.astype(float)
    X_lith_sq = df['lith_z_sq'].values.astype(float)
    
    y_train = df['deposit_present'].values.astype(np.int32)
    logit_base_rate = float(np.log(y_train.sum() / (len(y_train) - y_train.sum())))

    print(f"\n--- 4. Executing V10 Inference (Native C++ Backend) ---")
    with pm.Model() as prospectivity_model:
        # Hierarchical Intercept (Non-Centered)
        alpha_mu = pm.Normal('alpha_mu', mu=logit_base_rate, sigma=1.0)
        alpha_sigma = pm.HalfNormal('alpha_sigma', sigma=1.0)
        alpha_offset = pm.Normal('alpha_offset', mu=0.0, sigma=1.0, shape=n_domains)
        alpha_dom = pm.Deterministic('alpha_dom', alpha_mu + alpha_offset * alpha_sigma)

        # Hierarchical Slopes for Faults (Non-Centered)
        mu_f_lin = pm.Normal('mu_f_lin', mu=0.0, sigma=1.0)
        sigma_f_lin = pm.HalfNormal('sigma_f_lin', sigma=1.0)
        offset_f_lin = pm.Normal('offset_f_lin', mu=0.0, sigma=1.0, shape=n_domains)
        beta_f_lin = pm.Deterministic('beta_f_lin', mu_f_lin + offset_f_lin * sigma_f_lin)

        mu_f_sq = pm.Normal('mu_f_sq', mu=0.0, sigma=1.0)
        sigma_f_sq = pm.HalfNormal('sigma_f_sq', sigma=1.0)
        offset_f_sq = pm.Normal('offset_f_sq', mu=0.0, sigma=1.0, shape=n_domains)
        beta_f_sq = pm.Deterministic('beta_f_sq', mu_f_sq + offset_f_sq * sigma_f_sq)

        # Hierarchical Slopes for Lithology (Non-Centered)
        mu_l_lin = pm.Normal('mu_l_lin', mu=0.0, sigma=1.0)
        sigma_l_lin = pm.HalfNormal('sigma_l_lin', sigma=1.0)
        offset_l_lin = pm.Normal('offset_l_lin', mu=0.0, sigma=1.0, shape=n_domains)
        beta_l_lin = pm.Deterministic('beta_l_lin', mu_l_lin + offset_l_lin * sigma_l_lin)

        mu_l_sq = pm.Normal('mu_l_sq', mu=0.0, sigma=1.0)
        sigma_l_sq = pm.HalfNormal('sigma_l_sq', sigma=1.0)
        offset_l_sq = pm.Normal('offset_l_sq', mu=0.0, sigma=1.0, shape=n_domains)
        beta_l_sq = pm.Deterministic('beta_l_sq', mu_l_sq + offset_l_sq * sigma_l_sq)

        # Global Features
        beta_grav = pm.Normal('beta_grav', mu=0.0, sigma=1.0)
        beta_rocks = pm.Normal('beta_rocks', mu=0.0, sigma=1.0, shape=len(valid_rocks))

        # Model Assembly
        mu = (
            alpha_dom[domain_idx] + 
            beta_f_lin[domain_idx] * X_fault_lin + 
            beta_f_sq[domain_idx] * X_fault_sq +
            beta_l_lin[domain_idx] * X_lith_lin + 
            beta_l_sq[domain_idx] * X_lith_sq +
            beta_grav * X_grav + 
            pm.math.dot(X_rocks, beta_rocks)
        )
        
        y_obs = pm.Bernoulli('y_obs', logit_p=mu, observed=y_train)
        trace = pm.sample(draws=1500, tune=2500, chains=2, cores=1, target_accept=0.99, progressbar=True, random_seed=42)

    output_dir = ROOT / 'figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # PERMANENT TRACE SAVE (Never lose the sampler output again)
    # ---------------------------------------------------------
    trace_path = output_dir / 'v10_trace.nc'
    trace.to_netcdf(trace_path)
    print(f"\n[+] Raw posterior trace saved permanently to: {trace_path}")

    # ---------------------------------------------------------
    # SUPERVISOR'S CSV EXTRACTION
    # ---------------------------------------------------------
    print("--- 5. Extracting Hierarchical Posterior CSV ---")
    var_names = ['alpha_dom', 'beta_f_lin', 'beta_f_sq', 'beta_l_lin', 'beta_l_sq', 'beta_grav']
    summary_df = az.summary(trace, var_names=var_names, hdi_prob=0.95)
    
    prob_negative = []
    for idx_name in summary_df.index:
        if '[' in idx_name:
            var_name, idx_str = idx_name.split('[')
            idx = int(idx_str.replace(']', ''))
            samples = trace.posterior[var_name].values[:, :, idx].flatten()
        else:
            samples = trace.posterior[idx_name].values.flatten()
        prob_negative.append(np.mean(samples < 0))
        
    summary_df['P(beta < 0)'] = prob_negative
    
    # Map raw PyMC indices to geological domains for readability
    index_mapping = {}
    for var in ['alpha_dom', 'beta_f_lin', 'beta_f_sq', 'beta_l_lin', 'beta_l_sq']:
        for i, dom in enumerate(unique_domains):
            index_mapping[f"{var}[{i}]"] = f"{var}_{dom}"
            
    summary_df = summary_df.rename(index=index_mapping)
    csv_path = output_dir / 'v10_posterior_summary_hierarchical.csv'
    summary_df.to_csv(csv_path)
    print(f"[+] Matrix saved to: {csv_path}")

    # ---------------------------------------------------------
    # GEOLOGICAL RESPONSE CURVES (0 - 50 km)
    # ---------------------------------------------------------
    print("--- 6. Generating Geological Response Curves ---")
    m_array = np.linspace(0, 50000, 200)
    km_axis = m_array / 1000.0  
    
    mu_fault = scaler_fault.mean_[0]
    std_fault = scaler_fault.scale_[0]
    
    Z_fault = (m_array - mu_fault) / std_fault
    Z_fault_sq = Z_fault ** 2

    alpha_samples = trace.posterior['alpha_dom'].values.reshape(-1, n_domains)
    beta_f_lin_samples = trace.posterior['beta_f_lin'].values.reshape(-1, n_domains)
    beta_f_sq_samples = trace.posterior['beta_f_sq'].values.reshape(-1, n_domains)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    colors = plt.cm.tab10(np.linspace(0, 1, n_domains))

    for idx, dom_name in enumerate(unique_domains):
        ax = axes[idx]
        
        logit_matrix = alpha_samples[:, idx][:, None] + np.outer(beta_f_lin_samples[:, idx], Z_fault) + np.outer(beta_f_sq_samples[:, idx], Z_fault_sq)
        prob_matrix = 1 / (1 + np.exp(-logit_matrix)) 
        
        mean_curve = np.mean(prob_matrix, axis=0)
        lower_band = np.percentile(prob_matrix, 2.5, axis=0)
        upper_band = np.percentile(prob_matrix, 97.5, axis=0)
        
        ax.plot(km_axis, mean_curve, label='Posterior Mean', color=colors[idx], lw=3)
        ax.fill_between(km_axis, lower_band, upper_band, color=colors[idx], alpha=0.2, label='95% HDI')
        
        ax.set_title(f'{dom_name}', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        if idx >= 3: ax.set_xlabel('Distance to Fault (km)', fontsize=11)
        if idx % 3 == 0: ax.set_ylabel('Conditional Probability', fontsize=11)
        
        ax.set_xlim(0, 50)
        ax.legend(loc='upper right')

    plt.suptitle('Posterior Fault Proximity Response by Daly (2025) Tectonic Domain\n(Conditional fault-distance response with other covariates held fixed)', fontsize=14, y=1.02)
    plt.tight_layout()
    plot_path = output_dir / 'v10_fault_response_curves_hierarchical.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # ---------------------------------------------------------
    # D* OVERLAY ANALYSIS
    # ---------------------------------------------------------
    print("\n--- 7. Bayesian D* (Turning Point) Analysis vs. Real Data ---")
    for idx, dom_name in enumerate(unique_domains):
        dom_mask = df['daly_domain'] == dom_name
        max_dist_m = df.loc[dom_mask, dist_to_fault_col].max()
        max_dist_km = max_dist_m / 1000.0
        
        b1_trace = beta_f_lin_samples[:, idx]
        b2_trace = beta_f_sq_samples[:, idx]
        
        curvature_mask = b2_trace > 0.05
        prob_curvature = np.mean(curvature_mask)
        
        if prob_curvature > 0.5:
            z_star_trace = -b1_trace[curvature_mask] / (2 * b2_trace[curvature_mask])
            d_star_km_trace = ((z_star_trace * std_fault) + mu_fault) / 1000.0
            
            d_star_med = np.median(d_star_km_trace)
            d_star_lower = np.percentile(d_star_km_trace, 2.5)
            d_star_upper = np.percentile(d_star_km_trace, 97.5)
            p_in_bounds = np.mean((d_star_km_trace >= 0) & (d_star_km_trace <= 50))
            
            print(f"\n[{dom_name}] (Max Observed Data: {max_dist_km:.1f} km)")
            print(f"  Curvature Probability (P(b2 > 0)): {prob_curvature:.2f}")
            print(f"  Posterior D* Minimum : {d_star_med:.1f} km")
            print(f"  95% HDI of D*        : [{d_star_lower:.1f}, {d_star_upper:.1f}] km")
            print(f"  P(D* inside 0-50km)  : {p_in_bounds:.2f}")
            
            if d_star_med > max_dist_km:
                print(f"  ⚠️ WARNING: Median D* ({d_star_med:.1f} km) is extrapolating beyond observed data ({max_dist_km:.1f} km).")
        else:
            print(f"\n[{dom_name}] (Max Observed Data: {max_dist_km:.1f} km)")
            print(f"  Curvature Probability (P(b2 > 0)): {prob_curvature:.2f}")
            print("  Result: Predominantly monotonic / weak curvature. No D* calculated.")

if __name__ == "__main__":
    run_v10_final()