import os
import sys
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Fast C++ execution config
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def run_phase4_heterogeneity():
    print("--- 1. Loading Full Dataset for Inference ---")
    data_path = ROOT / 'data' / 'copperbelt_training_v5_with_tectonic_domain.csv'
    df = pd.read_csv(data_path)

    features = ['distance_to_fault', 'distance_to_lithology_contact', 'bouguer']
    df = df.dropna(subset=['domain', 'deposit_present'] + features).copy()

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
    
    # ---------------------------------------------------------
    # Domain Support Audit
    # ---------------------------------------------------------
    domain_support = df.groupby("daly_domain").agg(
        Cells=("deposit_present", "size"),
        Deposits=("deposit_present", "sum"),
    )
    domain_support["Non_Deposits"] = domain_support["Cells"] - domain_support["Deposits"]
    
    print("\n--- Domain Support ---")
    print(domain_support)

    # Full-data standardization for descriptive/inferential analysis.
    # This model is NOT being used for out-of-sample validation.
    scaler = StandardScaler()
    df[[f'{f}_z' for f in features]] = scaler.fit_transform(df[features])
    
    X_fault = df['distance_to_fault_z'].values
    X_lith = df['distance_to_lithology_contact_z'].values
    X_grav = df['bouguer_z'].values
    domain_idx = df['domain_idx'].values
    y = df['deposit_present'].values.astype(np.int32)
    
    logit_base_rate = float(np.log(y.sum() / (len(y) - y.sum())))

    print("\n--- 2. Compiling Hierarchical Descriptive Model ---")
    with pm.Model() as heterogeneity_model:
        alpha_mu = pm.Normal('alpha_mu', mu=logit_base_rate, sigma=1.0)
        alpha_sigma = pm.HalfNormal('alpha_sigma', sigma=1.0)
        alpha_offset = pm.Normal('alpha_offset', mu=0.0, sigma=1.0, shape=n_domains)
        alpha_dom = pm.Deterministic('alpha_dom', alpha_mu + alpha_offset * alpha_sigma)

        mu_f = pm.Normal('mu_f', mu=0.0, sigma=1.0)
        sigma_f = pm.HalfNormal('sigma_f', sigma=1.0)
        offset_f = pm.Normal('offset_f', mu=0.0, sigma=1.0, shape=n_domains)
        beta_f = pm.Deterministic('beta_f', mu_f + offset_f * sigma_f)

        mu_l = pm.Normal('mu_l', mu=0.0, sigma=1.0)
        sigma_l = pm.HalfNormal('sigma_l', sigma=1.0)
        offset_l = pm.Normal('offset_l', mu=0.0, sigma=1.0, shape=n_domains)
        beta_l = pm.Deterministic('beta_l', mu_l + offset_l * sigma_l)

        mu_g = pm.Normal('mu_g', mu=0.0, sigma=1.0)
        sigma_g = pm.HalfNormal('sigma_g', sigma=1.0)
        offset_g = pm.Normal('offset_g', mu=0.0, sigma=1.0, shape=n_domains)
        beta_g = pm.Deterministic('beta_g', mu_g + offset_g * sigma_g)

        mu = (
            alpha_dom[domain_idx] +
            beta_f[domain_idx] * X_fault +
            beta_l[domain_idx] * X_lith +
            beta_g[domain_idx] * X_grav
        )
        
        y_obs = pm.Bernoulli('y_obs', logit_p=mu, observed=y)
        
        print("--- 3. Sampling Posterior ---")
        trace = pm.sample(draws=1500, tune=2000, chains=2, cores=1, target_accept=0.95, random_seed=42)

    output_dir = ROOT / 'figures' / 'audit'
    os.makedirs(output_dir, exist_ok=True)

    print("--- 4. Exporting Traces and Hyperparameters ---")
    trace_path = output_dir / "phase4_domain_heterogeneity_trace.nc"
    az.to_netcdf(trace, trace_path)
    
    hyper_summary = az.summary(
        trace,
        var_names=["mu_f", "sigma_f", "mu_l", "sigma_l", "mu_g", "sigma_g"],
        hdi_prob=0.95
    )
    hyper_summary.to_csv(output_dir / "phase4_heterogeneity_hyperparameters.csv")

    print("--- 5. Extracting Domain Coefficients & Probabilities ---")
    summary = az.summary(trace, var_names=['beta_f', 'beta_l', 'beta_g'], hdi_prob=0.95)
    
    records = []
    for i, dom in enumerate(unique_domains):
        fault_samples = trace.posterior["beta_f"].values[..., i].flatten()
        lith_samples = trace.posterior["beta_l"].values[..., i].flatten()
        grav_samples = trace.posterior["beta_g"].values[..., i].flatten()

        records.append({
            'Domain': dom,
            
            # Fault
            'Beta_Fault': summary.loc[f'beta_f[{i}]', 'mean'],
            'Fault_HDI_3%': summary.loc[f'beta_f[{i}]', 'hdi_2.5%'],
            'Fault_HDI_97%': summary.loc[f'beta_f[{i}]', 'hdi_97.5%'],
            'P_Fault_neg': np.mean(fault_samples < 0),
            'P_Fault_pos': np.mean(fault_samples > 0),
            
            # Lithology
            'Beta_Lith': summary.loc[f'beta_l[{i}]', 'mean'],
            'Lith_HDI_3%': summary.loc[f'beta_l[{i}]', 'hdi_2.5%'],
            'Lith_HDI_97%': summary.loc[f'beta_l[{i}]', 'hdi_97.5%'],
            'P_Lith_neg': np.mean(lith_samples < 0),
            'P_Lith_pos': np.mean(lith_samples > 0),
            
            # Gravity
            'Beta_Gravity': summary.loc[f'beta_g[{i}]', 'mean'],
            'Grav_HDI_3%': summary.loc[f'beta_g[{i}]', 'hdi_2.5%'],
            'Grav_HDI_97%': summary.loc[f'beta_g[{i}]', 'hdi_97.5%'],
            'P_Grav_neg': np.mean(grav_samples < 0),
            'P_Grav_pos': np.mean(grav_samples > 0)
        })
        
    res_df = pd.DataFrame(records)
    res_df.to_csv(output_dir / 'phase4_domain_coefficients.csv', index=False)
    
    print("\n[+] Phase 4 Complete. Domain-Specific Feature Slopes:")
    print(res_df.round(3).to_string(index=False))
    
    print("\n[+] Hyperparameter Summary (Between-Domain Variation):")
    print(hyper_summary[['mean', 'hdi_2.5%', 'hdi_97.5%']].round(3).to_string())

if __name__ == "__main__":
    run_phase4_heterogeneity()