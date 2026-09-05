import os
import sys
import numpy as np
import pandas as pd
import arviz as az
from pathlib import Path
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def run_phase5_robust():
    print("--- 1. Enforcing Exact V10 Domain Mapping ---")
    # HARDCODED to match V10 PyMC index ordering exactly
    unique_domains = ['CRZ', 'MMSB', 'NKB', 'NRB_3a', 'NRB_3b', 'SRB']
    domain_map = {dom: i for i, dom in enumerate(unique_domains)}

    data_path = ROOT / 'data' / 'copperbelt_training_v5_with_tectonic_domain.csv'
    raw_df = pd.read_csv(data_path)

    def map_daly_domain(x):
        x_str = str(x).lower()
        if '3a' in x_str: return 'NRB_3a'
        elif '3b' in x_str: return 'NRB_3b'
        elif 'crz' in x_str: return 'CRZ'
        elif 'srb' in x_str: return 'SRB'
        elif 'nkb' in x_str: return 'NKB'
        elif 'mmsb' in x_str: return 'MMSB'
        else: return 'Unknown'

    raw_df['daly_domain'] = raw_df['domain'].apply(map_daly_domain)
    # Strictly filter out 'Unknown' or any other variants
    df = raw_df[raw_df['daly_domain'].isin(unique_domains)].copy()
    df = df.dropna(subset=['distance_to_fault', 'distance_to_lithology_contact', 'bouguer', 'deposit_present']).copy()

    print("--- 2. Extracting Scaler Parameters & Observed Limits ---")
    fault_scaler = StandardScaler().fit(df[['distance_to_fault']])
    lith_scaler = StandardScaler().fit(df[['distance_to_lithology_contact']])
    
    # Save the original mu and sigma to convert Z-score turning points back to real kilometers
    mu_f_scale, sigma_f_scale = fault_scaler.mean_[0], fault_scaler.scale_[0]
    mu_l_scale, sigma_l_scale = lith_scaler.mean_[0], lith_scaler.scale_[0]

    observed_limits = {}
    for dom in unique_domains:
        dom_data = df[df['daly_domain'] == dom]
        if len(dom_data) > 0:
            observed_limits[dom] = {
                'max_fault_km': dom_data['distance_to_fault'].max() / 1000.0,
                'max_lith_km': dom_data['distance_to_lithology_contact'].max() / 1000.0
            }
        else:
            observed_limits[dom] = {'max_fault_km': np.nan, 'max_lith_km': np.nan}

    print("--- 3. Loading V10 Posterior Trace ---")
    trace_path = ROOT / 'figures' / 'v10_trace.nc'
    if not trace_path.exists():
        print(f"[!] Trace not found: {trace_path}")
        return

    trace = az.from_netcdf(trace_path)
    alpha = trace.posterior['alpha_dom'].values.reshape(-1, len(domain_map))
    mu_f_lin = trace.posterior['beta_f_lin'].values.reshape(-1, len(domain_map))
    mu_f_sq = trace.posterior['beta_f_sq'].values.reshape(-1, len(domain_map))
    mu_l_lin = trace.posterior['beta_l_lin'].values.reshape(-1, len(domain_map))
    mu_l_sq = trace.posterior['beta_l_sq'].values.reshape(-1, len(domain_map))

    print("--- 4. Calculating Posterior Turning Points (D*) ---")
    tp_records = []
    # Only analyze the domains with deposit support
    for dom in ['NRB_3a', 'NRB_3b']:
        d_idx = domain_map[dom]
        max_f = observed_limits[dom]['max_fault_km']
        max_l = observed_limits[dom]['max_lith_km']

        # Fault D*: Z* = -beta_1 / (2 * beta_2)
        valid_f = np.abs(mu_f_sq[:, d_idx]) > 1e-5
        z_star_f = -mu_f_lin[valid_f, d_idx] / (2.0 * mu_f_sq[valid_f, d_idx])
        d_star_f_km = (mu_f_scale + sigma_f_scale * z_star_f) / 1000.0

        # Lithology D*
        valid_l = np.abs(mu_l_sq[:, d_idx]) > 1e-5
        z_star_l = -mu_l_lin[valid_l, d_idx] / (2.0 * mu_l_sq[valid_l, d_idx])
        d_star_l_km = (mu_l_scale + sigma_l_scale * z_star_l) / 1000.0

        tp_records.append({
            'Domain': dom,
            'Obs_Max_Fault_km': max_f,
            'D*_Fault_Median': np.median(d_star_f_km),
            'D*_Fault_2.5%': np.percentile(d_star_f_km, 2.5),
            'D*_Fault_97.5%': np.percentile(d_star_f_km, 97.5),
            'P(D*_Fault in Obs Range)': np.mean((d_star_f_km >= 0) & (d_star_f_km <= max_f)),
            
            'Obs_Max_Lith_km': max_l,
            'D*_Lith_Median': np.median(d_star_l_km),
            'D*_Lith_2.5%': np.percentile(d_star_l_km, 2.5),
            'D*_Lith_97.5%': np.percentile(d_star_l_km, 97.5),
            'P(D*_Lith in Obs Range)': np.mean((d_star_l_km >= 0) & (d_star_l_km <= max_l)),
        })

    print("--- 5. Evaluating Conditional Response Curves ---")
    distances_km = np.linspace(0, 50, 51)
    distances_m = distances_km * 1000.0
    fault_z = fault_scaler.transform(distances_m.reshape(-1, 1)).flatten()
    lith_z = lith_scaler.transform(distances_m.reshape(-1, 1)).flatten()

    response_records = []
    for dom in ['NRB_3a', 'NRB_3b']:
        d_idx = domain_map[dom]
        max_f = observed_limits[dom]['max_fault_km']
        max_l = observed_limits[dom]['max_lith_km']

        for d_km, f_z, l_z in zip(distances_km, fault_z, lith_z):
            logit_fault = alpha[:, d_idx] + (mu_f_lin[:, d_idx] * f_z) + (mu_f_sq[:, d_idx] * (f_z**2))
            logit_lith = alpha[:, d_idx] + (mu_l_lin[:, d_idx] * l_z) + (mu_l_sq[:, d_idx] * (l_z**2))

            response_records.append({
                'Domain': dom,
                'Distance_km': d_km,
                'Obs_Max_Fault_km': max_f,
                'P_Fault_Conditional_Mean': np.mean(expit(logit_fault)),
                'P_Fault_2_5%': np.percentile(expit(logit_fault), 2.5),
                'P_Fault_97_5%': np.percentile(expit(logit_fault), 97.5),
                'Obs_Max_Lith_km': max_l,
                'P_Lith_Conditional_Mean': np.mean(expit(logit_lith)),
                'P_Lith_2_5%': np.percentile(expit(logit_lith), 2.5),
                'P_Lith_97_5%': np.percentile(expit(logit_lith), 97.5)
            })

    output_dir = ROOT / 'figures' / 'audit'
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(tp_records).to_csv(output_dir / 'phase5_turning_points.csv', index=False)
    pd.DataFrame(response_records).to_csv(output_dir / 'phase5_distance_response.csv', index=False)

    print("\n[+] Turning Points Summary:")
    print(pd.DataFrame(tp_records).round(2).to_string(index=False))

if __name__ == "__main__":
    run_phase5_robust()