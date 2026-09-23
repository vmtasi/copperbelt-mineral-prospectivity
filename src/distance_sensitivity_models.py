"""
Distance Representation Sensitivity Analysis — Core Module
===========================================================
Supplementary sensitivity analysis for frozen V11 mineral prospectivity model.

This module is supplementary to and does NOT modify the frozen V11 implementation
(src/v11_spatial_stability_final.py), its posterior traces, or any existing audit
artifacts.

Unit convention:
    - Source data distances are in METRES.
    - For log-distance models (B, D), distances are converted to KILOMETRES
      before computing log1p: x_log = log(1 + D_km), where D_km = D_m / 1000.
    - All D* values are reported in KILOMETRES.
    - This is consistent with V11's own D* reporting convention (line 218 of
      v11_spatial_stability_final.py divides by 1000 at the final step).
"""

import os
import sys
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation_strategies import get_along_belt_folds

# ── Constants ────────────────────────────────────────────────────────────

DATA_PATH = ROOT / 'data' / 'copperbelt_training_v5_with_tectonic_domain.csv'
FROZEN_TRACE_DIR = ROOT / 'figures'
OUTPUT_DIR = ROOT / 'figures' / 'audit' / 'distance_representation_sensitivity'
TRACE_OUTPUT_DIR = OUTPUT_DIR / 'traces'

DOMAINS = ['CRZ', 'MMSB', 'NKB', 'NRB_3a', 'NRB_3b', 'SRB']
N_DOMAINS = 6
N_FOLDS = 4
GRID_POINTS = 201
EPS_BETA_SQ = 1e-5
LEGACY_FILTER = 0.05

# MCMC settings matching V11
MCMC_DRAWS = 1500
MCMC_TUNE = 2500
MCMC_CHAINS = 2
MCMC_CORES = 1
MCMC_TARGET_ACCEPT = 0.99
MCMC_SEED = 42


# ── Data loading ─────────────────────────────────────────────────────────

def map_daly_domain(x):
    s = str(x).lower()
    if '3a' in s: return 'NRB_3a'
    elif '3b' in s: return 'NRB_3b'
    elif 'crz' in s: return 'CRZ'
    elif 'srb' in s: return 'SRB'
    elif 'nkb' in s: return 'NKB'
    elif 'mmsb' in s: return 'MMSB'
    return 'Unknown'


def load_modeling_data():
    """Load and prepare the modeling dataset exactly as V11 does."""
    df = pd.read_csv(DATA_PATH)
    lithology_col = 'litho_contact_litho_class'
    continuous = ['distance_to_fault', 'distance_to_lithology_contact', 'bouguer']
    df = df.dropna(subset=['centroid_x', 'centroid_y', 'domain', lithology_col] + continuous).copy()
    df['daly_domain'] = df['domain'].apply(map_daly_domain)
    df = df[df['daly_domain'] != 'Unknown'].copy()
    df['domain_idx'] = df['daly_domain'].map({d: i for i, d in enumerate(DOMAINS)})
    df['spatial_block'] = get_along_belt_folds(df, n_folds=N_FOLDS)
    df = pd.get_dummies(df, columns=[lithology_col], drop_first=True, dtype=float)
    all_rock_features = [c for c in df.columns if c.startswith(f'{lithology_col}_')]
    return df, all_rock_features


def calc_pr_auc(y_true, y_prob):
    if sum(y_true) == 0 or sum(y_true) == len(y_true):
        return np.nan
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return auc(recall, precision)


# ── Scaler construction ─────────────────────────────────────────────────

def build_scalers(train_df, representation):
    """
    Build training-fold-only scalers for the given representation.

    Parameters
    ----------
    train_df : DataFrame with raw distance columns
    representation : 'raw' or 'log'

    Returns
    -------
    dict with keys 'fault', 'lith', 'grav', each containing
    (scaler_object, mu, sigma) where mu/sigma are in the working space.
    """
    scalers = {}
    for key, col in [('fault', 'distance_to_fault'), ('lith', 'distance_to_lithology_contact')]:
        if representation == 'raw':
            sc = StandardScaler().fit(train_df[[col]])
        elif representation == 'log':
            # Convert metres to km, then log1p
            vals = np.log1p(train_df[col].values / 1000.0).reshape(-1, 1)
            sc = StandardScaler().fit(vals)
        else:
            raise ValueError(f"Unknown representation: {representation}")
        scalers[key] = (sc, float(sc.mean_[0]), float(sc.scale_[0]))

    sc_grav = StandardScaler().fit(train_df[['bouguer']])
    scalers['grav'] = (sc_grav, float(sc_grav.mean_[0]), float(sc_grav.scale_[0]))
    return scalers


def transform_distances(df, scalers, representation, include_quadratic):
    """
    Transform distance columns in-place using the given scalers.

    Parameters
    ----------
    df : DataFrame (modified in-place)
    scalers : dict from build_scalers
    representation : 'raw' or 'log'
    include_quadratic : bool
    """
    for key, col in [('fault', 'distance_to_fault'), ('lith', 'distance_to_lithology_contact')]:
        sc, _, _ = scalers[key]
        if representation == 'raw':
            z = sc.transform(df[[col]])
        elif representation == 'log':
            vals = np.log1p(df[col].values / 1000.0).reshape(-1, 1)
            z = sc.transform(vals)
        df[f'{key}_z'] = z
        if include_quadratic:
            df[f'{key}_z_sq'] = z ** 2

    sc_grav, _, _ = scalers['grav']
    df['grav_z'] = sc_grav.transform(df[['bouguer']])


# ── PyMC model builder ───────────────────────────────────────────────────

def build_sensitivity_model(train_df, y_train, train_domain_idx, valid_rocks,
                            include_quadratic, n_domains=N_DOMAINS):
    """
    Build a PyMC model matching V11 hierarchical structure.

    For quadratic models (A, B): includes domain-varying linear and quadratic
    distance effects.
    For linear models (C, D): includes domain-varying linear distance effects
    only. Quadratic parameters are removed entirely.

    Global Bouguer gravity and retained lithological-class coefficients are
    always included.
    """
    X_grav = train_df['grav_z'].values
    X_f_lin = train_df['fault_z'].values
    X_l_lin = train_df['lith_z'].values
    X_rocks = train_df[valid_rocks].values

    logit_base_rate = float(np.log(y_train.sum() / (len(y_train) - y_train.sum())))

    with pm.Model() as model:
        # Domain-varying intercept (non-centred)
        alpha_mu = pm.Normal('alpha_mu', mu=logit_base_rate, sigma=1.0)
        alpha_sigma = pm.HalfNormal('alpha_sigma', sigma=1.0)
        alpha_offset = pm.Normal('alpha_offset', mu=0.0, sigma=1.0, shape=n_domains)
        alpha_dom = pm.Deterministic('alpha_dom', alpha_mu + alpha_offset * alpha_sigma)

        # Domain-varying fault linear
        mu_f_lin = pm.Normal('mu_f_lin', mu=0.0, sigma=1.0)
        sigma_f_lin = pm.HalfNormal('sigma_f_lin', sigma=1.0)
        offset_f_lin = pm.Normal('offset_f_lin', mu=0.0, sigma=1.0, shape=n_domains)
        beta_f_lin = pm.Deterministic('beta_f_lin', mu_f_lin + offset_f_lin * sigma_f_lin)

        # Domain-varying lithology linear
        mu_l_lin = pm.Normal('mu_l_lin', mu=0.0, sigma=1.0)
        sigma_l_lin = pm.HalfNormal('sigma_l_lin', sigma=1.0)
        offset_l_lin = pm.Normal('offset_l_lin', mu=0.0, sigma=1.0, shape=n_domains)
        beta_l_lin = pm.Deterministic('beta_l_lin', mu_l_lin + offset_l_lin * sigma_l_lin)

        # Linear predictor starts with intercept + linear distance + gravity + rocks
        mu = (
            alpha_dom[train_domain_idx] +
            beta_f_lin[train_domain_idx] * X_f_lin +
            beta_l_lin[train_domain_idx] * X_l_lin
        )

        if include_quadratic:
            X_f_sq = train_df['fault_z_sq'].values
            X_l_sq = train_df['lith_z_sq'].values

            # Domain-varying fault quadratic
            mu_f_sq = pm.Normal('mu_f_sq', mu=0.0, sigma=1.0)
            sigma_f_sq = pm.HalfNormal('sigma_f_sq', sigma=1.0)
            offset_f_sq = pm.Normal('offset_f_sq', mu=0.0, sigma=1.0, shape=n_domains)
            beta_f_sq = pm.Deterministic('beta_f_sq', mu_f_sq + offset_f_sq * sigma_f_sq)

            # Domain-varying lithology quadratic
            mu_l_sq = pm.Normal('mu_l_sq', mu=0.0, sigma=1.0)
            sigma_l_sq = pm.HalfNormal('sigma_l_sq', sigma=1.0)
            offset_l_sq = pm.Normal('offset_l_sq', mu=0.0, sigma=1.0, shape=n_domains)
            beta_l_sq = pm.Deterministic('beta_l_sq', mu_l_sq + offset_l_sq * sigma_l_sq)

            mu = mu + beta_f_sq[train_domain_idx] * X_f_sq + beta_l_sq[train_domain_idx] * X_l_sq

        # Global Bouguer gravity
        beta_grav = pm.Normal('beta_grav', mu=0.0, sigma=1.0)
        mu = mu + beta_grav * X_grav

        # Global retained lithological-class coefficients
        beta_rocks = pm.Normal('beta_rocks', mu=0.0, sigma=1.0, shape=len(valid_rocks))
        mu = mu + pm.math.dot(X_rocks, beta_rocks)

        y_obs = pm.Bernoulli('y_obs', logit_p=mu, observed=y_train)

    return model


# ── OOF prediction ───────────────────────────────────────────────────────

def predict_oof(trace, test_df, test_domain_idx, valid_rocks, include_quadratic,
                n_domains=N_DOMAINS):
    """
    Compute out-of-fold posterior mean probabilities from a fitted trace.
    """
    alpha = trace.posterior['alpha_dom'].values.reshape(-1, n_domains)
    bf_lin = trace.posterior['beta_f_lin'].values.reshape(-1, n_domains)
    bl_lin = trace.posterior['beta_l_lin'].values.reshape(-1, n_domains)
    bg = trace.posterior['beta_grav'].values.flatten()
    br = trace.posterior['beta_rocks'].values.reshape(-1, len(valid_rocks))

    X_grav = test_df['grav_z'].values
    X_f_lin = test_df['fault_z'].values
    X_l_lin = test_df['lith_z'].values
    X_rocks = test_df[valid_rocks].values

    logit = (
        alpha[:, test_domain_idx] +
        bf_lin[:, test_domain_idx] * X_f_lin +
        bl_lin[:, test_domain_idx] * X_l_lin +
        bg[:, None] * X_grav +
        np.dot(br, X_rocks.T)
    )

    if include_quadratic:
        bf_sq = trace.posterior['beta_f_sq'].values.reshape(-1, n_domains)
        bl_sq = trace.posterior['beta_l_sq'].values.reshape(-1, n_domains)
        X_f_sq = test_df['fault_z_sq'].values
        X_l_sq = test_df['lith_z_sq'].values
        logit = logit + bf_sq[:, test_domain_idx] * X_f_sq + bl_sq[:, test_domain_idx] * X_l_sq

    prob = 1.0 / (1.0 + np.exp(-logit))
    return np.mean(prob, axis=0)


# ── Response curve computation ───────────────────────────────────────────

def compute_response_curve(trace, domain_idx, predictor, representation,
                           grid_km, scaler_mu, scaler_sigma,
                           include_quadratic, n_domains=N_DOMAINS):
    """
    Compute posterior response curve for a single distance predictor over
    a grid of raw-km values.

    Convention: Holds all other continuous predictors at zero (training-fold
    mean in standardized space) and categorical predictors at zero (reference
    rock class). Reports the conditional distance contribution on the
    probability scale.

    Parameters
    ----------
    trace : arviz InferenceData
    domain_idx : int (index into DOMAINS)
    predictor : 'fault' or 'lith'
    representation : 'raw' or 'log'
    grid_km : 1-d array of raw distances in km
    scaler_mu : float (training mean in working space, metres or log-km)
    scaler_sigma : float (training std in working space)
    include_quadratic : bool
    n_domains : int

    Returns
    -------
    dict with keys: response_mean, response_ci_lower, response_ci_upper
    """
    # Forward transform grid_km to standardized z
    if representation == 'raw':
        z = (grid_km * 1000.0 - scaler_mu) / scaler_sigma
    elif representation == 'log':
        z = (np.log1p(grid_km) - scaler_mu) / scaler_sigma
    else:
        raise ValueError(f"Unknown representation: {representation}")

    lin_name = f'beta_{predictor[0]}_lin'
    alpha = trace.posterior['alpha_dom'].values.reshape(-1, n_domains)[:, domain_idx]
    b1 = trace.posterior[lin_name].values.reshape(-1, n_domains)[:, domain_idx]

    # eta = alpha + b1*z [+ b2*z^2]
    eta = alpha[:, None] + b1[:, None] * z[None, :]

    if include_quadratic:
        sq_name = f'beta_{predictor[0]}_sq'
        b2 = trace.posterior[sq_name].values.reshape(-1, n_domains)[:, domain_idx]
        eta = eta + b2[:, None] * (z[None, :] ** 2)

    prob = expit(eta)
    return {
        'response_mean': np.mean(prob, axis=0),
        'response_ci_lower': np.percentile(prob, 2.5, axis=0),
        'response_ci_upper': np.percentile(prob, 97.5, axis=0),
    }


# ── D* computation ───────────────────────────────────────────────────────

def compute_dstar_km(trace, domain_idx, predictor, representation,
                     scaler_mu, scaler_sigma, n_domains=N_DOMAINS):
    """
    Compute posterior D* in raw kilometres.

    For raw representation:
        z* = -b1 / (2*b2)
        D*_km = (mu_m + sigma_m * z*) / 1000

    For log representation:
        z* = -b1 / (2*b2)
        x_log* = mu_log + sigma_log * z*
        D*_km = exp(x_log*) - 1

    Returns dict with unconditional D* statistics and P(b2 > 0).
    """
    lin_name = f'beta_{predictor[0]}_lin'
    sq_name = f'beta_{predictor[0]}_sq'

    b1 = trace.posterior[lin_name].values.reshape(-1, n_domains)[:, domain_idx]
    b2 = trace.posterior[sq_name].values.reshape(-1, n_domains)[:, domain_idx]

    valid = np.abs(b2) > EPS_BETA_SQ
    p_b2_pos = float(np.mean(b2 > 0))

    if not np.any(valid):
        return {
            'd_star_median_km': np.nan,
            'd_star_ci_lower_km': np.nan,
            'd_star_ci_upper_km': np.nan,
            'p_beta_sq_positive': p_b2_pos,
        }

    z_star = -b1[valid] / (2.0 * b2[valid])

    if representation == 'raw':
        d_star_km = (scaler_mu + scaler_sigma * z_star) / 1000.0
    elif representation == 'log':
        x_log_star = scaler_mu + scaler_sigma * z_star
        d_star_km = np.exp(x_log_star) - 1.0
    else:
        raise ValueError(f"Unknown representation: {representation}")

    return {
        'd_star_median_km': float(np.median(d_star_km)),
        'd_star_ci_lower_km': float(np.percentile(d_star_km, 2.5)),
        'd_star_ci_upper_km': float(np.percentile(d_star_km, 97.5)),
        'p_beta_sq_positive': p_b2_pos,
        '_d_star_draws': d_star_km,  # kept for support check
    }


def compute_support_quantiles(train_df, fold):
    """Compute predictor quantile statistics from training data."""
    rows = []
    for key, col in [('fault', 'distance_to_fault'), ('lith', 'distance_to_lithology_contact')]:
        vals = train_df[col].values / 1000.0  # km
        rows.append({
            'fold': fold + 1,
            'predictor': key,
            'min': float(np.min(vals)),
            'p05': float(np.percentile(vals, 5)),
            'p25': float(np.percentile(vals, 25)),
            'p50': float(np.percentile(vals, 50)),
            'p75': float(np.percentile(vals, 75)),
            'p90': float(np.percentile(vals, 90)),
            'p95': float(np.percentile(vals, 95)),
            'p99': float(np.percentile(vals, 99)),
            'max': float(np.max(vals)),
            'n_train': len(vals),
        })
    return rows

