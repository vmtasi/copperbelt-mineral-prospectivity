import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation_strategies import get_along_belt_folds

# 5km x 5km grid resolution = 25 km^2 per cell
CELL_AREA_KM2 = 25.0 

def run_spatial_support_audit(n_folds=4):
    print("--- 1. Loading Datasets ---")
    data_path = ROOT / 'data' / 'copperbelt_training_v5_with_tectonic_domain.csv'
    df = pd.read_csv(data_path)

    lithology_col = 'litho_contact_litho_class'
    features = ['distance_to_fault', 'distance_to_lithology_contact', 'bouguer']
    df = df.dropna(subset=['centroid_x', 'centroid_y', 'domain', lithology_col] + features).copy()
    
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
    df['spatial_block'] = get_along_belt_folds(df, n_folds=n_folds)
    
    df_with_dummies = pd.get_dummies(df, columns=[lithology_col], drop_first=True, dtype=float)
    all_rock_features = [col for col in df_with_dummies.columns if col.startswith(f'{lithology_col}_')]
    
    output_dir = ROOT / 'figures' / 'audit'
    os.makedirs(output_dir, exist_ok=True)
    
    print("--- 2a. Generating Region & Domain Summaries ---")
    
    def generate_summary(group_col):
        summary = []
        for name, group in df.groupby(group_col):
            cells = len(group)
            deposits = group['deposit_present'].sum()
            negatives = cells - deposits
            area_km2 = cells * CELL_AREA_KM2
            density = (deposits / area_km2) * 1000 if area_km2 > 0 else 0
            
            valid_rock_count = np.nan
            if group_col == 'spatial_block':
                train_group = df_with_dummies[df_with_dummies['spatial_block'] != name]
                y_tr = train_group['deposit_present'].values
                valid_rocks = [c for c in all_rock_features if ((train_group[c] == 1) & (y_tr == 1)).sum() > 0 and ((train_group[c] == 1) & (y_tr == 0)).sum() > 0]
                valid_rock_count = len(valid_rocks)

            summary.append({
                group_col.capitalize(): name,
                'Area_km2': area_km2,
                'Cells': cells,
                'Deposits': deposits,
                'Non_Deposits': negatives,
                'Positive_Rate': round(deposits / cells, 4),
                'Deposit_Density_per_1000km2': round(density, 2),
                'Valid_Train_Rock_Features': valid_rock_count,
                'X_min': group['centroid_x'].min(),
                'X_max': group['centroid_x'].max(),
                'Y_min': group['centroid_y'].min(),
                'Y_max': group['centroid_y'].max()
            })
        return pd.DataFrame(summary)

    fold_summary = generate_summary('spatial_block')
    dominant_domains = df.groupby('spatial_block')['daly_domain'].apply(lambda x: x.mode()[0]).reset_index(name='Dominant_Domain')
    fold_summary = fold_summary.merge(dominant_domains, left_on='Spatial_block', right_on='spatial_block').drop('spatial_block', axis=1)
    
    domain_summary = generate_summary('daly_domain')
    
    fold_summary.to_csv(output_dir / 'fold_summary.csv', index=False)
    domain_summary.to_csv(output_dir / 'domain_summary.csv', index=False)

    print("--- 2b. Deposit vs Non-Deposit Feature Statistics ---")
    target_feature_summary = []
    
    for group_col in ['spatial_block', 'daly_domain']:
        for name, group in df.groupby(group_col):
            deposits_group = group[group['deposit_present'] == 1]
            non_deposits_group = group[group['deposit_present'] == 0]

            row = {
                'Grouping_Type': group_col.capitalize(),
                'Name': name,
                'Deposits': len(deposits_group),
                'Non_Deposits': len(non_deposits_group)
            }

            for feat in features:
                scale = 1000.0 if 'distance' in feat else 1.0
                dep = deposits_group[feat] / scale
                nondep = non_deposits_group[feat] / scale

                row[f'{feat}_deposit_mean'] = dep.mean()
                row[f'{feat}_deposit_median'] = dep.median()
                row[f'{feat}_deposit_sd'] = dep.std()
                row[f'{feat}_deposit_q25'] = dep.quantile(0.25)
                row[f'{feat}_deposit_q75'] = dep.quantile(0.75)

                row[f'{feat}_nondeposit_mean'] = nondep.mean()
                row[f'{feat}_nondeposit_median'] = nondep.median()
                row[f'{feat}_nondeposit_sd'] = nondep.std()
                row[f'{feat}_nondeposit_q25'] = nondep.quantile(0.25)
                row[f'{feat}_nondeposit_q75'] = nondep.quantile(0.75)

            target_feature_summary.append(row)
            
    pd.DataFrame(target_feature_summary).to_csv(output_dir / 'feature_by_target_summary.csv', index=False)
    
    print("--- 2c. Domain Training Support per Held-Out Fold ---")
    training_support = []
    
    for fold in range(n_folds):
        train = df[df['spatial_block'] != fold]
        test = df[df['spatial_block'] == fold]

        for domain in sorted(df['daly_domain'].unique()):
            train_domain = train[train['daly_domain'] == domain]
            test_domain = test[test['daly_domain'] == domain]

            training_support.append({
                'Held_Out_Fold': fold,
                'Domain': domain,
                'Train_Cells': len(train_domain),
                'Train_Deposits': int(train_domain['deposit_present'].sum()),
                'Train_Non_Deposits': int((train_domain['deposit_present'] == 0).sum()),
                'Test_Cells': len(test_domain),
                'Test_Deposits': int(test_domain['deposit_present'].sum()),
                'Test_Non_Deposits': int((test_domain['deposit_present'] == 0).sum())
            })
            
    pd.DataFrame(training_support).to_csv(output_dir / 'domain_training_support_by_fold.csv', index=False)

    print("--- 3. Generating Fold x Domain Overlap Matrix ---")
    overlap_df = df.groupby(['spatial_block', 'daly_domain']).agg(
        Cells=('deposit_present', 'count'),
        Deposits=('deposit_present', 'sum')
    ).reset_index()
    
    overlap_df['Non_Deposits'] = overlap_df['Cells'] - overlap_df['Deposits']
    overlap_df['Area_km2'] = overlap_df['Cells'] * CELL_AREA_KM2
    overlap_df['Deposit_Rate'] = overlap_df['Deposits'] / overlap_df['Cells']
    overlap_df['Deposit_Density_per_1000km2'] = (overlap_df['Deposits'] / overlap_df['Area_km2']) * 1000
    
    domain_proportions = (
        df.groupby(['spatial_block', 'daly_domain'])
          .size()
          .groupby(level=0)
          .transform(lambda x: (x / x.sum()) * 100)
    )
    overlap_df['Percent_of_Fold'] = domain_proportions.values

    overlap_df['Cell_Text'] = overlap_df.apply(lambda x: f"Area: {x['Area_km2']} km2 | D: {x['Deposits']} | {x['Percent_of_Fold']:.1f}%", axis=1)
    heatmap_table = overlap_df.pivot(index='spatial_block', columns='daly_domain', values='Cell_Text').fillna("None")
    
    overlap_df.to_csv(output_dir / 'fold_domain_overlap.csv', index=False)
    heatmap_table.to_csv(output_dir / 'fold_domain_heatmap_visual.csv')
    
    print("--- 4. Calculating Feature Extrapolation Metrics ---")
    support_metrics = []
    
    for fold in range(n_folds):
        train = df[df['spatial_block'] != fold]
        test = df[df['spatial_block'] == fold]
        
        for feat in features:
            train_vals = train[feat].values
            test_vals = test[feat].values
            
            w_dist_raw = wasserstein_distance(train_vals, test_vals)
            train_sd = train_vals.std()
            w_dist_std = w_dist_raw / train_sd if train_sd > 0 else np.nan
            
            train_min, train_max = train_vals.min(), train_vals.max()
            q01, q99 = np.percentile(train_vals, 1), np.percentile(train_vals, 99)
            
            p_extrap_minmax = np.sum((test_vals < train_min) | (test_vals > train_max)) / len(test_vals) * 100
            p_extrap_robust = np.sum((test_vals < q01) | (test_vals > q99)) / len(test_vals) * 100
            
            support_metrics.append({
                'Test_Fold': fold,
                'Feature': feat,
                'Wasserstein_Raw': w_dist_raw,
                'Wasserstein_Std': w_dist_std,
                'Percent_Outside_MinMax': round(p_extrap_minmax, 2),
                'Percent_Outside_1_99': round(p_extrap_robust, 2)
            })
            
    pd.DataFrame(support_metrics).to_csv(output_dir / 'feature_support.csv', index=False)
    print(f"\n[+] Audit Complete. 6 CSVs saved to: {output_dir}")

if __name__ == "__main__":
    run_spatial_support_audit()