"""
Distance Representation Sensitivity Analysis — Master Runner
=============================================================
Supplementary sensitivity analysis for frozen V11 mineral prospectivity model.

Executes the 4-model × 4-fold sensitivity experiment plus the 8-fit tail
diagnostic. Generates all required CSV outputs and a summary report.

This script is SUPPLEMENTARY to frozen V11. It does NOT modify any existing
V11 files, traces, audit outputs, or manuscript files.

Usage:
    python src/run_distance_sensitivity.py
"""

import os
import sys
import time
import hashlib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.distance_sensitivity_models import (
    load_modeling_data, build_scalers, transform_distances,
    build_sensitivity_model, predict_oof, compute_response_curve,
    compute_dstar_km, compute_support_quantiles, calc_pr_auc,
    DOMAINS, N_DOMAINS, N_FOLDS, GRID_POINTS, EPS_BETA_SQ, LEGACY_FILTER,
    MCMC_DRAWS, MCMC_TUNE, MCMC_CHAINS, MCMC_CORES, MCMC_TARGET_ACCEPT, MCMC_SEED,
    OUTPUT_DIR, TRACE_OUTPUT_DIR, FROZEN_TRACE_DIR,
)
from sklearn.metrics import brier_score_loss

# ── Model specifications ────────────────────────────────────────────────

MODEL_SPECS = {
    'A': {'representation': 'raw', 'quadratic': True,  'label': 'Raw-Quadratic (V11 frozen)'},
    'B': {'representation': 'log', 'quadratic': True,  'label': 'Log-Quadratic'},
    'C': {'representation': 'raw', 'quadratic': False, 'label': 'Raw-Linear'},
    'D': {'representation': 'log', 'quadratic': False, 'label': 'Log-Linear'},
}

# ── File integrity checks ───────────────────────────────────────────────

PROTECTED_FILES = [
    ROOT / 'src' / 'v11_spatial_stability_final.py',
    ROOT / 'src' / 'validation_strategies.py',
    ROOT / 'figures' / 'v11_fold_1_trace.nc',
    ROOT / 'figures' / 'v11_fold_2_trace.nc',
    ROOT / 'figures' / 'v11_fold_3_trace.nc',
    ROOT / 'figures' / 'v11_fold_4_trace.nc',
    ROOT / 'AI_PROJECT_CONTEXT.md',
]


def compute_file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def snapshot_protected_files():
    """Record hashes of all protected files."""
    return {str(p): compute_file_hash(p) for p in PROTECTED_FILES if p.exists()}


def verify_protected_files(snapshot):
    """Verify no protected file has changed."""
    for path_str, original_hash in snapshot.items():
        current_hash = compute_file_hash(Path(path_str))
        if current_hash != original_hash:
            raise RuntimeError(
                f"PROTECTED FILE CHANGED: {path_str}\n"
                f"  Original hash: {original_hash}\n"
                f"  Current hash:  {current_hash}\n"
                "STOPPING IMMEDIATELY. Do not attempt automatic repair."
            )
    print("[SAFETY] All protected files verified unchanged.")


# ── Main execution ───────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 70)
    print("DISTANCE REPRESENTATION SENSITIVITY ANALYSIS")
    print("Supplementary to frozen V11 — no V11 files will be modified")
    print("=" * 70)

    # Snapshot protected files BEFORE any work
    snapshot = snapshot_protected_files()
    print(f"[SAFETY] Snapshotted {len(snapshot)} protected files.")

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n--- Loading modeling data ---")
    df, all_rock_features = load_modeling_data()
    print(f"  Total observations: {len(df)}")
    print(f"  Rock features: {len(all_rock_features)}")

    # ── Phase 0: Support quantiles ───────────────────────────────────────
    print("\n--- Computing support quantiles ---")
    support_rows = []
    for fold in range(N_FOLDS):
        train_df = df[df['spatial_block'] != fold].copy()
        support_rows.extend(compute_support_quantiles(train_df, fold))
    pd.DataFrame(support_rows).to_csv(OUTPUT_DIR / 'support_quantiles.csv', index=False)
    print(f"  Saved support_quantiles.csv")

    # ── Phase 1: Main model matrix ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 1: MAIN MODEL MATRIX (12 new fits + Model A from frozen traces)")
    print("=" * 70)

    summary_rows = []
    response_fault_rows = []
    response_lith_rows = []
    traces_cache = {}  # (model, fold) -> trace

    for model_name, spec in MODEL_SPECS.items():
        rep = spec['representation']
        quad = spec['quadratic']
        label = spec['label']

        for fold in range(N_FOLDS):
            print(f"\n--- Model {model_name} ({label}), Fold {fold+1} ---")

            train_df = df[df['spatial_block'] != fold].copy()
            test_df = df[df['spatial_block'] == fold].copy()
            y_train = train_df['deposit_present'].values.astype(np.int32)
            y_test = test_df['deposit_present'].values.astype(np.int32)

            # Build scalers on training data only
            scalers = build_scalers(train_df, rep)

            # Transform both train and test
            transform_distances(train_df, scalers, rep, quad)
            transform_distances(test_df, scalers, rep, quad)

            # Valid rock features (train-only filter)
            valid_rocks = [c for c in all_rock_features
                           if ((train_df[c] == 1) & (y_train == 1)).sum() > 0
                           and ((train_df[c] == 1) & (y_train == 0)).sum() > 0]

            train_domain_idx = train_df['domain_idx'].values
            test_domain_idx = test_df['domain_idx'].values

            if model_name == 'A':
                # Reconstruct from frozen traces — no refitting
                trace_path = FROZEN_TRACE_DIR / f'v11_fold_{fold+1}_trace.nc'
                print(f"  Loading frozen trace: {trace_path.name}")
                trace = az.from_netcdf(trace_path)
            else:
                # Fit new model
                model = build_sensitivity_model(
                    train_df, y_train, train_domain_idx, valid_rocks,
                    include_quadratic=quad, n_domains=N_DOMAINS
                )
                with model:
                    trace = pm.sample(
                        draws=MCMC_DRAWS, tune=MCMC_TUNE,
                        chains=MCMC_CHAINS, cores=MCMC_CORES,
                        target_accept=MCMC_TARGET_ACCEPT,
                        progressbar=True, random_seed=MCMC_SEED
                    )
                # Save trace
                trace_path = TRACE_OUTPUT_DIR / f'model_{model_name}_fold_{fold+1}_trace.nc'
                trace.to_netcdf(trace_path)
                print(f"  Saved trace: {trace_path.name}")

            traces_cache[(model_name, fold)] = trace

            # OOF predictions
            mean_probs = predict_oof(trace, test_df, test_domain_idx,
                                     valid_rocks, quad, N_DOMAINS)
            fold_auc = roc_auc_score(y_test, mean_probs)
            fold_pr_auc = calc_pr_auc(y_test, mean_probs)
            fold_brier = brier_score_loss(y_test, mean_probs)
            print(f"  ROC-AUC={fold_auc:.3f}, PR-AUC={fold_pr_auc:.3f}, Brier={fold_brier:.3f}")

            # Per-domain summaries
            for dom_idx, dom in enumerate(DOMAINS):
                # Scaler params for this predictor
                for pred_key in ['fault', 'lith']:
                    _, sc_mu, sc_sigma = scalers[pred_key]

                    # D* (quadratic models only)
                    if quad:
                        dstar_info = compute_dstar_km(
                            trace, dom_idx, pred_key, rep,
                            sc_mu, sc_sigma, N_DOMAINS
                        )
                        dstar_draws = dstar_info.pop('_d_star_draws', np.array([]))

                        # Support check
                        dom_train = train_df[train_df['daly_domain'] == dom]
                        col = 'distance_to_fault' if pred_key == 'fault' else 'distance_to_lithology_contact'
                        if len(dom_train) > 0:
                            sup_min = dom_train[col].min() / 1000.0
                            sup_max = dom_train[col].max() / 1000.0
                            sup_p95 = np.percentile(dom_train[col].values / 1000.0, 95)
                            if len(dstar_draws) > 0:
                                p_in_support = float(np.mean(
                                    (dstar_draws >= sup_min) & (dstar_draws <= sup_max)
                                ))
                            else:
                                p_in_support = np.nan
                        else:
                            sup_min = sup_max = sup_p95 = p_in_support = np.nan

                        # Beta medians
                        lin_name = f'beta_{pred_key[0]}_lin'
                        sq_name = f'beta_{pred_key[0]}_sq'
                        b1_all = trace.posterior[lin_name].values.reshape(-1, N_DOMAINS)[:, dom_idx]
                        b2_all = trace.posterior[sq_name].values.reshape(-1, N_DOMAINS)[:, dom_idx]
                        beta_lin_med = float(np.median(b1_all))
                        beta_sq_med = float(np.median(b2_all))
                    else:
                        # Linear model — no D*
                        dstar_info = {
                            'd_star_median_km': np.nan,
                            'd_star_ci_lower_km': np.nan,
                            'd_star_ci_upper_km': np.nan,
                            'p_beta_sq_positive': np.nan,
                        }
                        dom_train = train_df[train_df['daly_domain'] == dom]
                        col = 'distance_to_fault' if pred_key == 'fault' else 'distance_to_lithology_contact'
                        if len(dom_train) > 0:
                            sup_min = dom_train[col].min() / 1000.0
                            sup_max = dom_train[col].max() / 1000.0
                            sup_p95 = np.percentile(dom_train[col].values / 1000.0, 95)
                        else:
                            sup_min = sup_max = sup_p95 = np.nan
                        p_in_support = np.nan

                        lin_name = f'beta_{pred_key[0]}_lin'
                        b1_all = trace.posterior[lin_name].values.reshape(-1, N_DOMAINS)[:, dom_idx]
                        beta_lin_med = float(np.median(b1_all))
                        beta_sq_med = np.nan

                    summary_rows.append({
                        'model': model_name,
                        'fold': fold + 1,
                        'domain': dom,
                        'predictor': pred_key,
                        'transformation': rep,
                        'functional_form': 'quadratic' if quad else 'linear',
                        'auc': fold_auc,
                        'pr_auc': fold_pr_auc,
                        'd_star_median_km': dstar_info['d_star_median_km'],
                        'd_star_ci_lower_km': dstar_info['d_star_ci_lower_km'],
                        'd_star_ci_upper_km': dstar_info['d_star_ci_upper_km'],
                        'p_beta_sq_positive': dstar_info['p_beta_sq_positive'],
                        'p_d_star_in_support': p_in_support,
                        'support_min_km': sup_min,
                        'support_max_km': sup_max,
                        'support_p95_km': sup_p95,
                        'beta_lin_median': beta_lin_med,
                        'beta_sq_median': beta_sq_med,
                    })

                    # Response curves
                    grid_max_km = sup_max if np.isfinite(sup_max) else 120.0
                    grid_km = np.linspace(0, min(grid_max_km * 1.1, 120.0), GRID_POINTS)
                    rc = compute_response_curve(
                        trace, dom_idx, pred_key, rep,
                        grid_km, sc_mu, sc_sigma, quad, N_DOMAINS
                    )
                    target_list = response_fault_rows if pred_key == 'fault' else response_lith_rows
                    for i, km in enumerate(grid_km):
                        target_list.append({
                            'model': model_name,
                            'fold': fold + 1,
                            'domain': dom,
                            'grid_km': float(km),
                            'response_mean': float(rc['response_mean'][i]),
                            'response_ci_lower': float(rc['response_ci_lower'][i]),
                            'response_ci_upper': float(rc['response_ci_upper'][i]),
                        })

            # Verify protected files after each model×fold
            verify_protected_files(snapshot)

    # Save Phase 1 outputs
    print("\n--- Saving Phase 1 outputs ---")
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / 'sensitivity_summary.csv', index=False)
    pd.DataFrame(response_fault_rows).to_csv(OUTPUT_DIR / 'response_curves_fault.csv', index=False)
    pd.DataFrame(response_lith_rows).to_csv(OUTPUT_DIR / 'response_curves_lithology.csv', index=False)
    print("  Saved sensitivity_summary.csv, response_curves_fault.csv, response_curves_lithology.csv")

    # ── Phase 2: Tail sensitivity ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2: TAIL-SENSITIVITY DIAGNOSTIC (8 new fits)")
    print("=" * 70)

    tail_rows = []

    for model_name in ['A', 'B']:
        spec = MODEL_SPECS[model_name]
        rep = spec['representation']
        label = spec['label']

        for fold in range(N_FOLDS):
            print(f"\n--- Tail-suppressed Model {model_name} ({label}), Fold {fold+1} ---")

            train_df_full = df[df['spatial_block'] != fold].copy()
            test_df = df[df['spatial_block'] == fold].copy()
            y_test = test_df['deposit_present'].values.astype(np.int32)

            # Calculate 95th percentiles from TRAINING DATA ONLY
            p95_fault = train_df_full['distance_to_fault'].quantile(0.95)
            p95_lith = train_df_full['distance_to_lithology_contact'].quantile(0.95)
            print(f"  P95 fault: {p95_fault/1000:.1f} km, P95 lith: {p95_lith/1000:.1f} km")

            # Trim training data (NEVER touch test data)
            train_df = train_df_full[
                (train_df_full['distance_to_fault'] <= p95_fault) &
                (train_df_full['distance_to_lithology_contact'] <= p95_lith)
            ].copy()
            y_train = train_df['deposit_present'].values.astype(np.int32)
            print(f"  Training: {len(train_df_full)} -> {len(train_df)} ({len(train_df_full)-len(train_df)} removed)")
            print(f"  Positives: {train_df_full['deposit_present'].sum()} -> {y_train.sum()}")

            # Build scalers on TRIMMED training data
            scalers = build_scalers(train_df, rep)
            transform_distances(train_df, scalers, rep, True)
            transform_distances(test_df, scalers, rep, True)

            valid_rocks = [c for c in all_rock_features
                           if ((train_df[c] == 1) & (y_train == 1)).sum() > 0
                           and ((train_df[c] == 1) & (y_train == 0)).sum() > 0]

            train_domain_idx = train_df['domain_idx'].values
            test_domain_idx = test_df['domain_idx'].values

            # Fit new model on trimmed training data
            model = build_sensitivity_model(
                train_df, y_train, train_domain_idx, valid_rocks,
                include_quadratic=True, n_domains=N_DOMAINS
            )
            with model:
                trace = pm.sample(
                    draws=MCMC_DRAWS, tune=MCMC_TUNE,
                    chains=MCMC_CHAINS, cores=MCMC_CORES,
                    target_accept=MCMC_TARGET_ACCEPT,
                    progressbar=True, random_seed=MCMC_SEED
                )

            # Save trace with clear naming
            trace_path = TRACE_OUTPUT_DIR / f'model_{model_name}_tail_fold_{fold+1}_trace.nc'
            trace.to_netcdf(trace_path)
            print(f"  Saved trace: {trace_path.name}")

            # OOF predictions on UNMODIFIED test fold
            mean_probs = predict_oof(trace, test_df, test_domain_idx,
                                     valid_rocks, True, N_DOMAINS)
            fold_auc = roc_auc_score(y_test, mean_probs)
            fold_pr_auc = calc_pr_auc(y_test, mean_probs)
            fold_brier = brier_score_loss(y_test, mean_probs)
            print(f"  ROC-AUC={fold_auc:.3f}, PR-AUC={fold_pr_auc:.3f}, Brier={fold_brier:.3f}")

            # Per-domain summaries
            for dom_idx, dom in enumerate(DOMAINS):
                for pred_key in ['fault', 'lith']:
                    _, sc_mu, sc_sigma = scalers[pred_key]

                    dstar_info = compute_dstar_km(
                        trace, dom_idx, pred_key, rep,
                        sc_mu, sc_sigma, N_DOMAINS
                    )
                    dstar_draws = dstar_info.pop('_d_star_draws', np.array([]))

                    dom_train = train_df[train_df['daly_domain'] == dom]
                    col = 'distance_to_fault' if pred_key == 'fault' else 'distance_to_lithology_contact'
                    if len(dom_train) > 0:
                        sup_min = dom_train[col].min() / 1000.0
                        sup_max = dom_train[col].max() / 1000.0
                        sup_p95 = np.percentile(dom_train[col].values / 1000.0, 95)
                        if len(dstar_draws) > 0:
                            p_in_support = float(np.mean(
                                (dstar_draws >= sup_min) & (dstar_draws <= sup_max)
                            ))
                        else:
                            p_in_support = np.nan
                    else:
                        sup_min = sup_max = sup_p95 = p_in_support = np.nan

                    lin_name = f'beta_{pred_key[0]}_lin'
                    sq_name = f'beta_{pred_key[0]}_sq'
                    b1_all = trace.posterior[lin_name].values.reshape(-1, N_DOMAINS)[:, dom_idx]
                    b2_all = trace.posterior[sq_name].values.reshape(-1, N_DOMAINS)[:, dom_idx]

                    tail_rows.append({
                        'model': model_name,
                        'fold': fold + 1,
                        'domain': dom,
                        'predictor': pred_key,
                        'transformation': rep,
                        'functional_form': 'quadratic',
                        'tail_removed': 'upper_5pct_removed',
                        'auc': fold_auc,
                        'pr_auc': fold_pr_auc,
                        'd_star_median_km': dstar_info['d_star_median_km'],
                        'd_star_ci_lower_km': dstar_info['d_star_ci_lower_km'],
                        'd_star_ci_upper_km': dstar_info['d_star_ci_upper_km'],
                        'p_beta_sq_positive': dstar_info['p_beta_sq_positive'],
                        'p_d_star_in_support': p_in_support,
                        'support_min_km': sup_min,
                        'support_max_km': sup_max,
                        'support_p95_km': sup_p95,
                        'beta_lin_median': float(np.median(b1_all)),
                        'beta_sq_median': float(np.median(b2_all)),
                    })

            verify_protected_files(snapshot)

    # Add full-data rows to tail summary for comparison
    full_summary = pd.read_csv(OUTPUT_DIR / 'sensitivity_summary.csv')
    full_ab = full_summary[full_summary['model'].isin(['A', 'B'])].copy()
    full_ab['tail_removed'] = 'full_data'
    tail_df = pd.concat([full_ab, pd.DataFrame(tail_rows)], ignore_index=True)
    tail_df.to_csv(OUTPUT_DIR / 'tail_sensitivity_summary.csv', index=False)
    print("\n  Saved tail_sensitivity_summary.csv")

    # ── Phase 3: Robustness assessment ───────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 3: ROBUSTNESS ASSESSMENT (Section 17.1 criteria)")
    print("=" * 70)

    robustness_rows = []
    summary_df = pd.read_csv(OUTPUT_DIR / 'sensitivity_summary.csv')

    for fold in range(1, N_FOLDS + 1):
        for dom in DOMAINS:
            for pred in ['fault', 'lith']:
                row_a = summary_df[
                    (summary_df['model'] == 'A') &
                    (summary_df['fold'] == fold) &
                    (summary_df['domain'] == dom) &
                    (summary_df['predictor'] == pred)
                ]
                row_b = summary_df[
                    (summary_df['model'] == 'B') &
                    (summary_df['fold'] == fold) &
                    (summary_df['domain'] == dom) &
                    (summary_df['predictor'] == pred)
                ]

                if len(row_a) == 0 or len(row_b) == 0:
                    continue

                ra = row_a.iloc[0]
                rb = row_b.iloc[0]

                # Criterion 1: curvature probability agreement
                p_a = ra['p_beta_sq_positive']
                p_b = rb['p_beta_sq_positive']
                if np.isfinite(p_a) and np.isfinite(p_b):
                    crit1_diff = abs(p_a - p_b)
                    crit1_pass = crit1_diff < 0.15
                else:
                    crit1_diff = np.nan
                    crit1_pass = None

                # Criterion 2: D* median agreement (within 25%)
                da = ra['d_star_median_km']
                db = rb['d_star_median_km']
                if np.isfinite(da) and np.isfinite(db):
                    denom = max(abs(da), abs(db))
                    if denom > 0:
                        crit2_diff = abs(da - db) / denom
                        crit2_pass = crit2_diff <= 0.25
                    else:
                        crit2_diff = 0.0
                        crit2_pass = True
                else:
                    crit2_diff = np.nan
                    crit2_pass = None

                # Criterion 3: both D* in empirical support
                pa_sup = ra['p_d_star_in_support']
                pb_sup = rb['p_d_star_in_support']
                if np.isfinite(da) and np.isfinite(db):
                    sup_min_a = ra['support_min_km']
                    sup_max_a = ra['support_max_km']
                    a_in_support = (np.isfinite(sup_min_a) and np.isfinite(sup_max_a)
                                    and sup_min_a <= da <= sup_max_a)
                    sup_min_b = rb['support_min_km']
                    sup_max_b = rb['support_max_km']
                    b_in_support = (np.isfinite(sup_min_b) and np.isfinite(sup_max_b)
                                    and sup_min_b <= db <= sup_max_b)
                    crit3_pass = a_in_support and b_in_support
                else:
                    crit3_pass = None

                # Overall
                if crit1_pass is not None and crit2_pass is not None and crit3_pass is not None:
                    robust = crit1_pass and crit2_pass and crit3_pass
                else:
                    robust = None

                robustness_rows.append({
                    'fold': fold,
                    'domain': dom,
                    'predictor': pred,
                    'p_beta_sq_pos_A': p_a,
                    'p_beta_sq_pos_B': p_b,
                    'crit1_diff': crit1_diff,
                    'crit1_pass': crit1_pass,
                    'd_star_median_A_km': da,
                    'd_star_median_B_km': db,
                    'crit2_relative_diff': crit2_diff,
                    'crit2_pass': crit2_pass,
                    'crit3_both_in_support': crit3_pass,
                    'robust_all_criteria': robust,
                })

    robustness_df = pd.DataFrame(robustness_rows)
    robustness_df.to_csv(OUTPUT_DIR / 'robustness_assessment.csv', index=False)
    print(f"  Saved robustness_assessment.csv")

    # Print summary
    if len(robustness_df) > 0:
        evaluable = robustness_df[robustness_df['robust_all_criteria'].notna()]
        if len(evaluable) > 0:
            n_robust = evaluable['robust_all_criteria'].sum()
            n_total = len(evaluable)
            print(f"  Evaluable cases: {n_total}")
            print(f"  Robust (all 3 criteria): {n_robust} ({100*n_robust/n_total:.0f}%)")
            print(f"  Representation-dependent: {n_total - n_robust} ({100*(n_total-n_robust)/n_total:.0f}%)")

    # ── Final safety check ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL SAFETY CHECK")
    print("=" * 70)
    verify_protected_files(snapshot)

    elapsed = time.time() - t0
    print(f"\nTotal execution time: {elapsed/60:.1f} minutes")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("Experiment complete.")


if __name__ == '__main__':
    main()

