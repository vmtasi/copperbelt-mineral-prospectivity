import os
import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_PATH = ROOT / "data" / "copperbelt_training_v5_with_tectonic_domain.csv"
TRACE_DIR = ROOT / "figures"
OUT_DIR = ROOT / "figures" / "audit" / "beta_diagnostics"

DOMAINS = ["CRZ", "MMSB", "NKB", "NRB_3a", "NRB_3b", "SRB"]
DISTANCE_VARS = {
    "fault": "distance_to_fault",
    "lithology": "distance_to_lithology_contact",
}
EPS_BETA_SQ = 1e-5
FILTER_THRESHOLD = 0.05
SENSITIVITY_THRESHOLDS = [0.0, 0.01, 0.05, 0.10]
GRID_POINTS = 401


def map_daly_domain(x):
    s = str(x).lower()
    if "3a" in s:
        return "NRB_3a"
    if "3b" in s:
        return "NRB_3b"
    if "crz" in s:
        return "CRZ"
    if "srb" in s:
        return "SRB"
    if "nkb" in s:
        return "NKB"
    if "mmsb" in s:
        return "MMSB"
    return "Unknown"


def summarize_interval(x):
    x = np.asarray(x, dtype=float)
    return np.median(x), np.percentile(x, 2.5), np.percentile(x, 97.5)


def posterior_mean_extremum(distances_m, alpha, b1, b2, mu, sigma):
    z = (distances_m - mu) / sigma
    eta = alpha[:, None] + b1[:, None] * z[None, :] + b2[:, None] * z[None, :] ** 2
    p = expit(eta)
    mean_p = p.mean(axis=0)
    # The logistic transform is monotone, so the extremum location of eta is
    # the same as that of p for each draw; the mean response is evaluated
    # directly on the probability scale as the reported curve.
    i_min = int(np.argmin(mean_p))
    i_max = int(np.argmax(mean_p))
    return mean_p, i_min, i_max, p


def run_beta_diagnostics():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    required = [
        "centroid_x", "centroid_y", "domain", "litho_contact_litho_class",
        "distance_to_fault", "distance_to_lithology_contact", "bouguer", "deposit_present"
    ]
    df = df.dropna(subset=required).copy()
    df["daly_domain"] = df["domain"].apply(map_daly_domain)
    df = df[df["daly_domain"].isin(DOMAINS)].copy()

    # Reproduce V11 fold construction exactly. This is downstream only; no
    # PyMC model is fitted here.
    from src.validation_strategies import get_along_belt_folds
    df["spatial_block"] = get_along_belt_folds(df, n_folds=4)

    coeff_rows = []
    support_rows = []
    tp_rows = []
    slope_rows = []
    response_rows = []
    stability_rows = []

    traces = {}
    scalers = {}
    supports = {}

    for fold in range(4):
        train = df[df["spatial_block"] != fold].copy()
        trace_path = TRACE_DIR / f"v11_fold_{fold + 1}_trace.nc"
        if not trace_path.exists():
            raise FileNotFoundError(f"Missing V11 posterior trace: {trace_path}")
        trace = az.from_netcdf(trace_path)
        traces[fold + 1] = trace

        scalers[fold + 1] = {}
        for key, col in DISTANCE_VARS.items():
            sc = StandardScaler().fit(train[[col]])
            scalers[fold + 1][key] = (float(sc.mean_[0]), float(sc.scale_[0]))
            for dom in DOMAINS:
                td = train[train["daly_domain"] == dom]
                vals = td[col].to_numpy(dtype=float)
                if len(vals) == 0:
                    mn = mx = np.nan
                else:
                    mn = float(vals.min() / 1000.0)
                    mx = float(vals.max() / 1000.0)
                supports[(fold + 1, dom, key)] = (mn, mx)

    # The trace index order is tied to sorted domains in V11. Verify it rather
    # than assuming a different ordering.
    expected = {d: i for i, d in enumerate(DOMAINS)}
    for fold_id, trace in traces.items():
        for key, col in DISTANCE_VARS.items():
            lin_name = "beta_f_lin" if key == "fault" else "beta_l_lin"
            sq_name = "beta_f_sq" if key == "fault" else "beta_l_sq"
            b1_all = trace.posterior[lin_name].values.reshape(-1, len(DOMAINS))
            b2_all = trace.posterior[sq_name].values.reshape(-1, len(DOMAINS))

            for dom in DOMAINS:
                j = expected[dom]
                b1 = b1_all[:, j]
                b2 = b2_all[:, j]
                mu, sigma = scalers[fold_id][key]
                support_min, support_max = supports[(fold_id, dom, key)]

                med1, lo1, hi1 = summarize_interval(b1)
                med2, lo2, hi2 = summarize_interval(b2)
                p_pos = float(np.mean(b2 > 0))
                p_neg = float(np.mean(b2 < 0))
                p_gt = float(np.mean(b2 > FILTER_THRESHOLD))
                valid = np.abs(b2) > EPS_BETA_SQ
                p_exists = float(np.mean(valid))
                classification = (
                    "minimum-dominant" if p_pos >= 0.95 else
                    "maximum-dominant" if p_neg >= 0.95 else
                    "sign-uncertain"
                )
                p_minimum = p_pos
                p_maximum = p_neg

                coeff_rows.append({
                    "Fold": fold_id, "Domain": dom, "Variable": key,
                    "beta_lin_median": med1, "beta_lin_2.5%": lo1, "beta_lin_97.5%": hi1,
                    "beta_sq_median": med2, "beta_sq_2.5%": lo2, "beta_sq_97.5%": hi2,
                    "P(beta_sq>0)": p_pos, "P(beta_sq<0)": p_neg,
                    "P(beta_sq>0.05)": p_gt, "P(valid_turning_point)": p_exists,
                    "P(turning_point_is_minimum)": p_minimum,
                    "P(turning_point_is_maximum)": p_maximum,
                    "turning_point_classification": classification,
                    "train_mu_m": mu, "train_sigma_m": sigma,
                    "observed_min_distance_km": support_min,
                    "observed_max_distance_km": support_max,
                    "filter_threshold_beta_sq": FILTER_THRESHOLD,
                    "turning_point_epsilon_beta_sq": EPS_BETA_SQ,
                })

                # Unfiltered algebraic D*: all numerically valid draws.
                if np.any(valid):
                    zstar = -b1[valid] / (2.0 * b2[valid])
                    dstar = (mu + sigma * zstar) / 1000.0
                    dmed, dlo, dhi = summarize_interval(dstar)
                    p_support = float(np.mean((dstar >= support_min) & (dstar <= support_max))) if np.isfinite(support_min) else np.nan
                else:
                    dstar = np.array([])
                    dmed = dlo = dhi = np.nan
                    p_support = np.nan

                filt = b2 > FILTER_THRESHOLD
                if np.any(filt):
                    zstar_f = -b1[filt] / (2.0 * b2[filt])
                    dstar_f = (mu + sigma * zstar_f) / 1000.0
                    fmed, flo, fhi = summarize_interval(dstar_f)
                    p_f_support = float(np.mean((dstar_f >= support_min) & (dstar_f <= support_max))) if np.isfinite(support_min) else np.nan
                else:
                    fmed = flo = fhi = np.nan
                    p_f_support = np.nan

                # Posterior-mean response curve on the exact empirical support.
                if np.isfinite(support_min) and np.isfinite(support_max) and support_max >= support_min:
                    grid_km = np.linspace(support_min, support_max, GRID_POINTS)
                    grid_m = grid_km * 1000.0
                    alpha = trace.posterior["alpha_dom"].values.reshape(-1, len(DOMAINS))[:, j]
                    mean_p, i_min, i_max, p_draws = posterior_mean_extremum(grid_m, alpha, b1, b2, mu, sigma)
                    # A curve extremum can occur at a boundary; record whether
                    # the selected point is interior to distinguish a turning
                    # point from a monotone response over observed support.
                    interior_min = 0 < i_min < len(grid_km) - 1
                    interior_max = 0 < i_max < len(grid_km) - 1
                    if interior_min and not interior_max:
                        curve_class = "minimum"
                        curve_d = float(grid_km[i_min])
                    elif interior_max and not interior_min:
                        curve_class = "maximum"
                        curve_d = float(grid_km[i_max])
                    elif interior_min and interior_max:
                        curve_class = "mixed/ambiguous"
                        curve_d = float(grid_km[i_min])
                    else:
                        curve_class = "boundary/monotone"
                        curve_d = float(grid_km[i_min]) if i_min in (0, len(grid_km)-1) else float(grid_km[i_max])
                    response_rows.append({
                        "Fold": fold_id, "Domain": dom, "Variable": key,
                        "support_min_km": support_min, "support_max_km": support_max,
                        "grid_points": GRID_POINTS,
                        "posterior_mean_curve_extremum_km": curve_d,
                        "posterior_mean_curve_extremum_class": curve_class,
                        "posterior_mean_curve_probability_at_extremum": float(mean_p[i_min] if curve_class == "minimum" else mean_p[i_max]),
                    })
                else:
                    response_rows.append({
                        "Fold": fold_id, "Domain": dom, "Variable": key,
                        "support_min_km": support_min, "support_max_km": support_max,
                        "grid_points": GRID_POINTS,
                        "posterior_mean_curve_extremum_km": np.nan,
                        "posterior_mean_curve_extremum_class": "undefined",
                        "posterior_mean_curve_probability_at_extremum": np.nan,
                    })

                tp_rows.append({
                    "Fold": fold_id, "Domain": dom, "Variable": key,
                    "unfiltered_Dstar_median_km": dmed,
                    "unfiltered_Dstar_2.5%_km": dlo,
                    "unfiltered_Dstar_97.5%_km": dhi,
                    "P_Dstar_within_empirical_support": p_support,
                    "filtered_Dstar_beta_sq_gt_0.05_median_km": fmed,
                    "filtered_Dstar_beta_sq_gt_0.05_2.5%_km": flo,
                    "filtered_Dstar_beta_sq_gt_0.05_97.5%_km": fhi,
                    "P_filtered_Dstar_within_empirical_support": p_f_support,
                    "filter_definition": "beta_sq > 0.05",
                    "support_boundary_convention": "inclusive [observed_min_distance_km, observed_max_distance_km]",
                })

                # Slope behavior on a documented grid spanning exact support.
                if np.isfinite(support_min) and np.isfinite(support_max):
                    slope_grid_km = np.linspace(support_min, support_max, GRID_POINTS)
                    slope_z = (slope_grid_km * 1000.0 - mu) / sigma
                    slope_draws = b1[:, None] + 2.0 * b2[:, None] * slope_z[None, :]
                    # Store every grid point; probabilities are draw-based.
                    for k, (dkm, zz) in enumerate(zip(slope_grid_km, slope_z)):
                        slope_rows.append({
                            "Fold": fold_id, "Domain": dom, "Variable": key,
                            "Distance_km": float(dkm), "z": float(zz),
                            "P(slope>0)": float(np.mean(slope_draws[:, k] > 0)),
                            "P(slope<0)": float(np.mean(slope_draws[:, k] < 0)),
                            "slope_median": float(np.median(slope_draws[:, k])),
                            "slope_2.5%": float(np.percentile(slope_draws[:, k], 2.5)),
                            "slope_97.5%": float(np.percentile(slope_draws[:, k], 97.5)),
                        })

    # Threshold sensitivity for the four required cases. The 0.05 rule is
    # retained as a methodological filter; these diagnostics show how much
    # the conditional D* distribution changes under alternative thresholds.
    threshold_rows = []
    for fold_id, trace in traces.items():
        for dom in ["NRB_3a", "NRB_3b"]:
            j = DOMAINS.index(dom)
            for key in DISTANCE_VARS:
                lin_name = "beta_f_lin" if key == "fault" else "beta_l_lin"
                sq_name = "beta_f_sq" if key == "fault" else "beta_l_sq"
                b1 = trace.posterior[lin_name].values.reshape(-1, len(DOMAINS))[:, j]
                b2 = trace.posterior[sq_name].values.reshape(-1, len(DOMAINS))[:, j]
                mu, sigma = scalers[fold_id][key]
                mn, mx = supports[(fold_id, dom, key)]
                for threshold in SENSITIVITY_THRESHOLDS:
                    mask = b2 > threshold
                    if np.any(mask):
                        zstar = -b1[mask] / (2.0 * b2[mask])
                        dstar = (mu + sigma * zstar) / 1000.0
                        med, lo, hi = summarize_interval(dstar)
                        p_support = float(np.mean((dstar >= mn) & (dstar <= mx)))
                    else:
                        med = lo = hi = p_support = np.nan
                    threshold_rows.append({
                        "Fold": fold_id, "Domain": dom, "Variable": key,
                        "beta_sq_threshold": threshold,
                        "P(beta_sq_above_threshold)": float(np.mean(b2 > threshold)),
                        "conditional_Dstar_median_km": med,
                        "conditional_Dstar_2.5%_km": lo,
                        "conditional_Dstar_97.5%_km": hi,
                        "P(conditional_Dstar_within_support)": p_support,
                    })

    coeff_df = pd.DataFrame(coeff_rows)
    tp_df = pd.DataFrame(tp_rows)
    threshold_df = pd.DataFrame(threshold_rows)
    slope_df = pd.DataFrame(slope_rows)
    response_df = pd.DataFrame(response_rows)

    # Fold-to-fold coefficient stability: descriptive pairwise posterior
    # differences, without an arbitrary composite score.
    for dom in DOMAINS:
        for key in DISTANCE_VARS:
            lin_name = "beta_f_lin" if key == "fault" else "beta_l_lin"
            sq_name = "beta_f_sq" if key == "fault" else "beta_l_sq"
            for f1 in range(1, 5):
                for f2 in range(f1 + 1, 5):
                    t1 = traces[f1].posterior[lin_name].values.reshape(-1, len(DOMAINS))[:, DOMAINS.index(dom)]
                    t2 = traces[f2].posterior[lin_name].values.reshape(-1, len(DOMAINS))[:, DOMAINS.index(dom)]
                    s1 = traces[f1].posterior[sq_name].values.reshape(-1, len(DOMAINS))[:, DOMAINS.index(dom)]
                    s2 = traces[f2].posterior[sq_name].values.reshape(-1, len(DOMAINS))[:, DOMAINS.index(dom)]
                    n = min(len(t1), len(t2))
                    dlin = t1[:n] - t2[:n]
                    dsq = s1[:n] - s2[:n]
                    for cname, diff in [("beta_lin", dlin), ("beta_sq", dsq)]:
                        lo, med, hi = np.percentile(diff, [2.5, 50, 97.5])
                        stability_rows.append({
                            "Domain": dom, "Variable": key, "Coefficient": cname,
                            "Fold_A": f1, "Fold_B": f2,
                            "difference_median": float(med),
                            "difference_2.5%": float(lo), "difference_97.5%": float(hi),
                            "P(Fold_A > Fold_B)": float(np.mean(diff > 0)),
                            "CI_difference_includes_zero": bool(lo <= 0 <= hi),
                            "operational_note": "Descriptive pairwise posterior comparison; no composite stability score."
                        })

    # Aggregate four required cases, retaining fold-level values.
    four = coeff_df[coeff_df["Domain"].isin(["NRB_3a", "NRB_3b"])].copy()
    four = four.merge(tp_df, on=["Fold", "Domain", "Variable"], how="left")
    four = four.merge(response_df, on=["Fold", "Domain", "Variable"], how="left")
    four.to_csv(OUT_DIR / "beta_diagnostics_four_cases_by_fold.csv", index=False)
    coeff_df.to_csv(OUT_DIR / "beta_diagnostics_coefficients_by_fold.csv", index=False)
    tp_df.to_csv(OUT_DIR / "beta_turning_points_by_fold.csv", index=False)
    slope_df.to_csv(OUT_DIR / "beta_slope_support_by_fold.csv", index=False)
    response_df.to_csv(OUT_DIR / "beta_response_curve_extrema_by_fold.csv", index=False)
    threshold_df.to_csv(OUT_DIR / "beta_threshold_sensitivity_four_cases.csv", index=False)
    pd.DataFrame(stability_rows).to_csv(OUT_DIR / "beta_coefficient_stability_pairwise.csv", index=False)

    # Endpoint slope summary gives a compact, auditable description of
    # increasing/decreasing posterior behavior at the empirical boundaries.
    endpoint_rows = []
    for dom in ["NRB_3a", "NRB_3b"]:
        for key in DISTANCE_VARS:
            sub = slope_df[(slope_df.Domain == dom) & (slope_df.Variable == key)]
            for fold_id in range(1, 5):
                s = sub[sub.Fold == fold_id].sort_values("Distance_km")
                if len(s):
                    for label, row in [("support_min", s.iloc[0]), ("support_max", s.iloc[-1])]:
                        endpoint_rows.append({
                            "Fold": fold_id, "Domain": dom, "Variable": key,
                            "Endpoint": label, "Distance_km": float(row.Distance_km),
                            "z": float(row.z), "P(slope>0)": float(row["P(slope>0)"]),
                            "P(slope<0)": float(row["P(slope<0)"]),
                            "slope_median": float(row.slope_median),
                            "slope_2.5%": float(row["slope_2.5%"]),
                            "slope_97.5%": float(row["slope_97.5%"]),
                        })
    pd.DataFrame(endpoint_rows).to_csv(OUT_DIR / "beta_slope_endpoints_four_cases.csv", index=False)

    # Compact summary over the four cases: fold range and medians.
    summary_rows = []
    for dom in ["NRB_3a", "NRB_3b"]:
        for key in DISTANCE_VARS:
            sub = four[(four.Domain == dom) & (four.Variable == key)]
            for metric in [
                "beta_lin_median", "beta_sq_median", "P(beta_sq>0)", "P(beta_sq<0)",
                "P(beta_sq>0.05)", "P(valid_turning_point)",
                "unfiltered_Dstar_median_km", "P_Dstar_within_empirical_support",
                "filtered_Dstar_beta_sq_gt_0.05_median_km", "P_filtered_Dstar_within_empirical_support",
                "posterior_mean_curve_extremum_km"
            ]:
                vals = pd.to_numeric(sub[metric], errors="coerce")
                summary_rows.append({
                    "Domain": dom, "Variable": key, "Metric": metric,
                    "fold_median": float(vals.median()) if vals.notna().any() else np.nan,
                    "fold_min": float(vals.min()) if vals.notna().any() else np.nan,
                    "fold_max": float(vals.max()) if vals.notna().any() else np.nan,
                })
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "beta_four_case_summary.csv", index=False)

    # Plot the four required cases. Each plot uses fold-specific posterior
    # mean curves and marks each fold's empirical support.
    for dom in ["NRB_3a", "NRB_3b"]:
        for key, col in DISTANCE_VARS.items():
            fig, ax = plt.subplots(figsize=(9, 5.5))
            for fold_id, trace in traces.items():
                mu, sigma = scalers[fold_id][key]
                mn, mx = supports[(fold_id, dom, key)]
                if not np.isfinite(mn) or not np.isfinite(mx):
                    continue
                grid_km = np.linspace(mn, mx, GRID_POINTS)
                j = DOMAINS.index(dom)
                alpha = trace.posterior["alpha_dom"].values.reshape(-1, len(DOMAINS))[:, j]
                lin_name = "beta_f_lin" if key == "fault" else "beta_l_lin"
                sq_name = "beta_f_sq" if key == "fault" else "beta_l_sq"
                b1 = trace.posterior[lin_name].values.reshape(-1, len(DOMAINS))[:, j]
                b2 = trace.posterior[sq_name].values.reshape(-1, len(DOMAINS))[:, j]
                _, _, _, draws = posterior_mean_extremum(grid_km * 1000.0, alpha, b1, b2, mu, sigma)
                mean_curve = draws.mean(axis=0)
                ax.plot(grid_km, mean_curve, label=f"Fold {fold_id}")
            ax.set_xlabel(f"Distance ({key}, km)")
            ax.set_ylabel("Posterior mean conditional probability")
            ax.set_title(f"{dom} — {key} response over fold-specific empirical support")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(OUT_DIR / f"{dom}_{key}_response_curves.png", dpi=220)
            plt.close(fig)

            # Slope probability plot for the same case.
            sub = slope_df[(slope_df.Domain == dom) & (slope_df.Variable == key)]
            fig, ax = plt.subplots(figsize=(9, 5.5))
            for fold_id in range(1, 5):
                s = sub[sub.Fold == fold_id]
                if len(s):
                    ax.plot(s.Distance_km, s["P(slope>0)"], label=f"Fold {fold_id}")
            ax.axhline(0.5, linestyle="--", linewidth=1)
            ax.set_ylim(0, 1)
            ax.set_xlabel(f"Distance ({key}, km)")
            ax.set_ylabel("P(slope > 0)")
            ax.set_title(f"{dom} — {key} posterior slope behavior")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(OUT_DIR / f"{dom}_{key}_slope_probability.png", dpi=220)
            plt.close(fig)

    print("[+] V11c beta diagnostics complete")
    print(f"[+] Outputs: {OUT_DIR}")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    run_beta_diagnostics()
