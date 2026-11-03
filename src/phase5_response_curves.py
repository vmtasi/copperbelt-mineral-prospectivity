"""Per-fold V14 distance-response diagnostics.

This downstream diagnostic reconstructs the fitted linear distance components
from each V14 posterior trace.  Scalers are fitted on that fold's training
cells only, exactly as in the frozen V11 specification.  The primary inference
and OOF evaluation remain in the V14 model run and Phase 7 scripts.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.runtime_shim  # noqa: F401,E402

import numpy as np
import pandas as pd
from scipy.special import expit

from src.numerics import StandardScaler
from src.trace_io import load_trace, resolve_trace_path, trace_exists
from src.validation_strategies import get_along_belt_folds

DATA_PATH = ROOT / "data" / "copperbelt_training_v14_centroid_contact.csv"
TRACE_DIR = ROOT / "figures" / "v14_centroid_contact"
OUTPUT_DIR = ROOT / "figures" / "audit"
DOMAINS = ["CRZ", "MMSB", "NKB", "NRB_3a", "NRB_3b", "SRB"]
MODELED_DOMAINS = ["NRB_3a", "NRB_3b"]
FEATURES = ["distance_to_fault", "distance_to_lithology_contact", "bouguer"]


def map_daly_domain(value):
    text = str(value).lower()
    for token, code in (
        ("3a", "NRB_3a"), ("3b", "NRB_3b"), ("crz", "CRZ"),
        ("srb", "SRB"), ("nkb", "NKB"), ("mmsb", "MMSB"),
    ):
        if token in text:
            return code
    return "Unknown"


def run_phase5_response_curves():
    raw = pd.read_csv(DATA_PATH)
    frame = raw.dropna(subset=["centroid_x", "centroid_y", "domain"] + FEATURES).copy()
    frame["daly_domain"] = frame["domain"].map(map_daly_domain)
    frame = frame[frame["daly_domain"].isin(DOMAINS)].copy()
    frame["spatial_block"] = get_along_belt_folds(frame, n_folds=4)

    turning_points = []
    response_records = []
    distance_km = np.linspace(0.0, 50.0, 51)
    distance_m = distance_km * 1000.0

    for fold in range(4):
        train = frame[frame["spatial_block"] != fold].copy()
        requested = TRACE_DIR / f"v14_centroid_fold_{fold + 1}_trace.nc"
        if not trace_exists(requested):
            raise FileNotFoundError(f"Missing V14 trace for fold {fold + 1}: {requested}")
        trace = load_trace(resolve_trace_path(requested))

        scaler_fault = StandardScaler().fit(train[["distance_to_fault"]])
        scaler_lith = StandardScaler().fit(train[["distance_to_lithology_contact"]])
        mu_fault, sigma_fault = float(scaler_fault.mean_[0]), float(scaler_fault.scale_[0])
        mu_lith, sigma_lith = float(scaler_lith.mean_[0]), float(scaler_lith.scale_[0])

        alpha = trace.posterior["alpha_dom"].values.reshape(-1, len(DOMAINS))
        fault_lin = trace.posterior["beta_f_lin"].values.reshape(-1, len(DOMAINS))
        fault_sq = trace.posterior["beta_f_sq"].values.reshape(-1, len(DOMAINS))
        lith_lin = trace.posterior["beta_l_lin"].values.reshape(-1, len(DOMAINS))
        lith_sq = trace.posterior["beta_l_sq"].values.reshape(-1, len(DOMAINS))
        fault_z = scaler_fault.transform(distance_m.reshape(-1, 1)).reshape(-1)
        lith_z = scaler_lith.transform(distance_m.reshape(-1, 1)).reshape(-1)

        for domain in MODELED_DOMAINS:
            domain_index = DOMAINS.index(domain)
            domain_train = train[train["daly_domain"] == domain]
            max_fault_km = float(domain_train["distance_to_fault"].max() / 1000.0)
            max_lith_km = float(domain_train["distance_to_lithology_contact"].max() / 1000.0)

            valid_fault = np.abs(fault_sq[:, domain_index]) > 1e-5
            fault_star_z = (
                -fault_lin[valid_fault, domain_index]
                / (2.0 * fault_sq[valid_fault, domain_index])
            )
            fault_star_km = (mu_fault + sigma_fault * fault_star_z) / 1000.0

            valid_lith = np.abs(lith_sq[:, domain_index]) > 1e-5
            lith_star_z = (
                -lith_lin[valid_lith, domain_index]
                / (2.0 * lith_sq[valid_lith, domain_index])
            )
            lith_star_km = (mu_lith + sigma_lith * lith_star_z) / 1000.0

            turning_points.append(
                {
                    "Fold": fold + 1,
                    "Domain": domain,
                    "Obs_Max_Fault_km": max_fault_km,
                    "D*_Fault_Median_km": float(np.median(fault_star_km)),
                    "D*_Fault_2.5%_km": float(np.percentile(fault_star_km, 2.5)),
                    "D*_Fault_97.5%_km": float(np.percentile(fault_star_km, 97.5)),
                    "P(D*_Fault_in_Obs_Range)": float(
                        np.mean((fault_star_km >= 0.0) & (fault_star_km <= max_fault_km))
                    ),
                    "Obs_Max_Lith_km": max_lith_km,
                    "D*_Lith_Median_km": float(np.median(lith_star_km)),
                    "D*_Lith_2.5%_km": float(np.percentile(lith_star_km, 2.5)),
                    "D*_Lith_97.5%_km": float(np.percentile(lith_star_km, 97.5)),
                    "P(D*_Lith_in_Obs_Range)": float(
                        np.mean((lith_star_km >= 0.0) & (lith_star_km <= max_lith_km))
                    ),
                }
            )

            fault_logit = (
                alpha[:, domain_index, None]
                + fault_lin[:, domain_index, None] * fault_z[None, :]
                + fault_sq[:, domain_index, None] * (fault_z[None, :] ** 2)
            )
            lith_logit = (
                alpha[:, domain_index, None]
                + lith_lin[:, domain_index, None] * lith_z[None, :]
                + lith_sq[:, domain_index, None] * (lith_z[None, :] ** 2)
            )
            fault_probability = expit(fault_logit)
            lith_probability = expit(lith_logit)

            for index, km in enumerate(distance_km):
                response_records.append(
                    {
                        "Fold": fold + 1,
                        "Domain": domain,
                        "Distance_km": float(km),
                        "Obs_Max_Fault_km": max_fault_km,
                        "P_Fault_Conditional_Mean": float(fault_probability[:, index].mean()),
                        "P_Fault_2_5%": float(np.percentile(fault_probability[:, index], 2.5)),
                        "P_Fault_97_5%": float(np.percentile(fault_probability[:, index], 97.5)),
                        "Obs_Max_Lith_km": max_lith_km,
                        "P_Lith_Conditional_Mean": float(lith_probability[:, index].mean()),
                        "P_Lith_2_5%": float(np.percentile(lith_probability[:, index], 2.5)),
                        "P_Lith_97_5%": float(np.percentile(lith_probability[:, index], 97.5)),
                    }
                )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    turning_path = OUTPUT_DIR / "phase5_turning_points_v14_centroid.csv"
    response_path = OUTPUT_DIR / "phase5_distance_response_v14_centroid.csv"
    pd.DataFrame(turning_points).to_csv(turning_path, index=False)
    pd.DataFrame(response_records).to_csv(response_path, index=False)
    print(pd.DataFrame(turning_points).round(3).to_string(index=False))
    print(f"[+] {turning_path}")
    print(f"[+] {response_path}")


if __name__ == "__main__":
    run_phase5_response_curves()
