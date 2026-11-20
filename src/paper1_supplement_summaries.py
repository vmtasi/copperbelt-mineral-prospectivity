"""Trace- and OOF-derived summaries for the Paper 1 supplement (no refitting).

Reads only frozen V14 benchmark artefacts (see paper_1/benchmark/SHA256SUMS)
plus the externally archived distance-representation traces, and writes new
versioned outputs under figures/audit/paper1_supplement_v14/. Nothing frozen
is modified and no MCMC is run.

Outputs
-------
coefficients_nrb.csv            Domain distance coefficients with 95% CIs (M-check: L426)
direction_by_representation.csv P(beta_lin < 0) for Models A-D (abstract "recur" claim)
dstar_informative_subset.json   D* robustness on the 16 NRB_3a/NRB_3b cases (M24)
scale_posteriors.csv            Hierarchical scale and population posteriors (M27)
fold_auc_intervals.csv          Fold-specific ROC-AUC with stratified bootstrap CIs (M28)
probability_accuracy.json       Brier skill and calibration diagnostics (M19)
reliability_bins.csv            Reliability table for the pooled OOF probabilities
uncertainty_by_domain.csv       95% credible-interval width by domain/support
lithology_retained_by_fold.csv  Retained lithology dummies and reference class per fold
manifest.json                   Input hashes, software versions, seeds

Run:  python -m src.paper1_supplement_summaries
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.numerics import roc_auc_score  # noqa: E402
from src.validation_strategies import get_along_belt_folds  # noqa: E402

DATA = ROOT / "data" / "copperbelt_training_v14_centroid_contact.csv"
V14_DIR = ROOT / "figures" / "v14_centroid_contact"
OOF = V14_DIR / "v14_centroid_oof_predictions.csv"
UNCERTAINTY = ROOT / "figures" / "v14_centroid_oof_posterior_uncertainty.csv"
SENS_DIR = ROOT / "figures" / "audit" / "distance_representation_sensitivity_v14_centroid"
ROBUSTNESS = SENS_DIR / "robustness_assessment.csv"
OUT = ROOT / "figures" / "audit" / "paper1_supplement_v14"

DOMAINS = ["CRZ", "MMSB", "NKB", "NRB_3a", "NRB_3b", "SRB"]  # sorted, as in the frozen runner
INFORMATIVE = ["NRB_3a", "NRB_3b"]
LITHO_COL = "litho_contact_litho_class"
LITHO_NAMES = {
    4: "Metasedimentary", 6: "Silicic", 7: "Metaigneous",
    14: "Alluvium-Fluvial", 16: "Alluvium-Saline", 20: "Water",
}
N_BOOT = 4000
SEED = 42


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def map_daly_domain(value) -> str:
    text = str(value).lower()
    for key, name in (("3a", "NRB_3a"), ("3b", "NRB_3b"), ("crz", "CRZ"),
                      ("srb", "SRB"), ("nkb", "NKB"), ("mmsb", "MMSB")):
        if key in text:
            return name
    return "Unknown"


def load_frame() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    cols = ["centroid_x", "centroid_y", LITHO_COL, "domain",
            "distance_to_fault", "distance_to_lithology_contact", "bouguer"]
    df = df.dropna(subset=cols).copy()
    df["daly_domain"] = df["domain"].apply(map_daly_domain)
    df = df[df["daly_domain"] != "Unknown"].copy()
    df["spatial_block"] = get_along_belt_folds(df, n_folds=4)
    return df


def flat(trace, name: str) -> np.ndarray:
    values = trace.posterior[name].values
    return values.reshape(values.shape[0] * values.shape[1], *values.shape[2:])


def q(x: np.ndarray) -> tuple[float, float, float]:
    lo, med, hi = np.quantile(x, [0.025, 0.5, 0.975])
    return float(med), float(lo), float(hi)


def load_trace(path: Path):
    import arviz as az

    return az.from_netcdf(path)


# --------------------------------------------------------------------------- #
def coefficients_nrb() -> pd.DataFrame:
    rows = []
    for fold in range(1, 5):
        trace = load_trace(V14_DIR / f"v14_centroid_fold_{fold}_trace.nc")
        for predictor, prefix in (("fault", "f"), ("contact", "l")):
            lin = flat(trace, f"beta_{prefix}_lin")
            sq = flat(trace, f"beta_{prefix}_sq")
            for domain in INFORMATIVE:
                k = DOMAINS.index(domain)
                lm, ll, lh = q(lin[:, k])
                sm, sl, sh = q(sq[:, k])
                rows.append({
                    "fold": fold, "domain": domain, "predictor": predictor,
                    "beta_lin_median": lm, "beta_lin_q025": ll, "beta_lin_q975": lh,
                    "p_beta_lin_neg": float((lin[:, k] < 0).mean()),
                    "beta_sq_median": sm, "beta_sq_q025": sl, "beta_sq_q975": sh,
                    "p_beta_sq_pos": float((sq[:, k] > 0).mean()),
                })
    return pd.DataFrame(rows)


def direction_by_representation() -> pd.DataFrame:
    sources = {"A": lambda f: V14_DIR / f"v14_centroid_fold_{f}_trace.nc"}
    for model in ("B", "C", "D"):
        sources[model] = lambda f, m=model: SENS_DIR / "traces" / f"model_{m}_fold_{f}_trace.nc"
    rows = []
    for model, path_for in sources.items():
        for fold in range(1, 5):
            path = path_for(fold)
            if not path.is_file():
                raise FileNotFoundError(
                    f"{path} is missing; restore the external sensitivity traces "
                    "(paper_1/benchmark/SHA256SUMS_external)."
                )
            trace = load_trace(path)
            for predictor, prefix in (("fault", "f"), ("contact", "l")):
                lin = flat(trace, f"beta_{prefix}_lin")
                for domain in INFORMATIVE:
                    k = DOMAINS.index(domain)
                    med, lo, hi = q(lin[:, k])
                    rows.append({
                        "model": model, "fold": fold, "domain": domain,
                        "predictor": predictor,
                        "transformation": "raw" if model in "AC" else "log1p_km",
                        "basis": "quadratic" if model in "AB" else "linear",
                        "beta_lin_median": med, "beta_lin_q025": lo, "beta_lin_q975": hi,
                        "p_beta_lin_neg": float((lin[:, k] < 0).mean()),
                    })
    return pd.DataFrame(rows)


def dstar_informative_subset() -> dict:
    r = pd.read_csv(ROBUSTNESS)
    out = {}
    for label, subset in (("all_48", r), ("informative_16", r[r["domain"].isin(INFORMATIVE)])):
        out[label] = {
            "cases": int(len(subset)),
            "criterion_1_curvature": int(subset["crit1_pass"].sum()),
            "criterion_2_dstar": int(subset["crit2_pass"].sum()),
            "criterion_3_support": int(subset["crit3_both_in_support"].sum()),
            "all_three": int(subset["robust_all_criteria"].sum()),
        }
    informative = r[r["domain"].isin(INFORMATIVE)]
    out["informative_cases"] = informative[[
        "fold", "domain", "predictor", "p_beta_sq_pos_A", "p_beta_sq_pos_B",
        "crit1_pass", "d_star_median_A_km", "d_star_median_B_km", "crit2_pass",
        "crit3_both_in_support", "robust_all_criteria",
    ]].to_dict(orient="records")
    return out


def scale_posteriors() -> pd.DataFrame:
    import arviz as az

    names = ["alpha_mu", "alpha_sigma", "mu_f_lin", "sigma_f_lin", "mu_f_sq", "sigma_f_sq",
             "mu_l_lin", "sigma_l_lin", "mu_l_sq", "sigma_l_sq", "beta_grav"]
    rows = []
    for fold in range(1, 5):
        trace = load_trace(V14_DIR / f"v14_centroid_fold_{fold}_trace.nc")
        summary = az.summary(trace, var_names=names, kind="diagnostics")
        for name in names:
            med, lo, hi = q(flat(trace, name))
            rows.append({
                "fold": fold, "parameter": name, "median": med, "q025": lo, "q975": hi,
                "ess_bulk": float(summary.loc[name, "ess_bulk"]),
                "r_hat": float(summary.loc[name, "r_hat"]),
            })
    return pd.DataFrame(rows)


def stratified_bootstrap_auc(y: np.ndarray, p: np.ndarray, rng) -> tuple[float, float]:
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    stats = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        stats[b] = roc_auc_score(y[idx], p[idx])
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return float(lo), float(hi)


def fold_auc_intervals(oof: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for block, part in oof.groupby("fold"):
        y, p = part["deposit_present"].to_numpy(), part["prob_v14"].to_numpy()
        lo, hi = stratified_bootstrap_auc(y, p, rng)
        rows.append({"fold": int(block) + 1, "n": len(part), "positives": int(y.sum()),
                     "roc_auc": float(roc_auc_score(y, p)), "ci95_lower": lo, "ci95_upper": hi,
                     "method": f"class-stratified bootstrap, B={N_BOOT}, seed={SEED}"})
    y, p = oof["deposit_present"].to_numpy(), oof["prob_v14"].to_numpy()
    lo, hi = stratified_bootstrap_auc(y, p, rng)
    rows.append({"fold": "pooled", "n": len(oof), "positives": int(y.sum()),
                 "roc_auc": float(roc_auc_score(y, p)), "ci95_lower": lo, "ci95_upper": hi,
                 "method": f"class-stratified bootstrap, B={N_BOOT}, seed={SEED}"})
    return pd.DataFrame(rows)


def logistic_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Unpenalised logistic regression y ~ a + b*x by Newton-Raphson."""
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(100):
        mu = 1 / (1 + np.exp(-X @ beta))
        w = mu * (1 - mu)
        step = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (y - mu))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def probability_accuracy(oof: pd.DataFrame, frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    y = oof["deposit_present"].to_numpy().astype(float)
    p = oof["prob_v14"].to_numpy()
    eps = 1e-12
    logit_p = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1))

    # Out-of-fold baseline: each held-out cell receives its training-partition prevalence.
    train_prev = {
        block: frame.loc[frame["spatial_block"] != block, "deposit_present"].mean()
        for block in range(4)
    }
    base_oof = oof["fold"].map(train_prev).to_numpy()
    pooled_prev = y.mean()

    brier = float(np.mean((p - y) ** 2))
    brier_oof_base = float(np.mean((base_oof - y) ** 2))
    brier_pooled_base = float(np.mean((pooled_prev - y) ** 2))
    intercept, slope = logistic_fit(logit_p, y)
    # calibration-in-the-large: intercept with logit(p) as offset
    a = 0.0
    for _ in range(100):
        mu = 1 / (1 + np.exp(-(a + logit_p)))
        step = np.sum(y - mu) / np.sum(mu * (1 - mu))
        a += step
        if abs(step) < 1e-12:
            break

    per_fold = {}
    for block, part in oof.groupby("fold"):
        yy, pp = part["deposit_present"].to_numpy(float), part["prob_v14"].to_numpy()
        bb = train_prev[block]
        b_model, b_base = float(np.mean((pp - yy) ** 2)), float(np.mean((bb - yy) ** 2))
        per_fold[str(int(block) + 1)] = {
            "brier": b_model, "baseline_training_prevalence": float(bb),
            "brier_baseline": b_base, "brier_skill": 1 - b_model / b_base,
            "mean_predicted": float(pp.mean()), "observed_prevalence": float(yy.mean()),
        }

    bins = pd.qcut(p, 10, duplicates="drop")
    rel = (pd.DataFrame({"p": p, "y": y, "bin": bins})
           .groupby("bin", observed=True)
           .agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"),
                positives=("y", "sum"))
           .reset_index())
    rel["bin"] = rel["bin"].astype(str)

    result = {
        "n": int(len(y)), "positives": int(y.sum()), "pooled_prevalence": float(pooled_prev),
        "brier_dvhm": brier,
        "brier_baseline_oof_training_prevalence": brier_oof_base,
        "brier_baseline_pooled_prevalence": brier_pooled_base,
        "brier_skill_vs_oof_baseline": 1 - brier / brier_oof_base,
        "brier_skill_vs_pooled_prevalence": 1 - brier / brier_pooled_base,
        "calibration_intercept_in_the_large": float(a),
        "calibration_slope": slope, "calibration_slope_intercept": intercept,
        "mean_predicted": float(p.mean()),
        "per_fold": per_fold,
        "notes": ("Baseline 'oof_training_prevalence' assigns each held-out cell the "
                  "positive rate of its own training partition and is the fair "
                  "out-of-fold comparator; 'pooled_prevalence' uses the overall rate "
                  "and therefore leaks the held-out label frequency."),
    }
    return result, rel


def uncertainty_by_domain(frame: pd.DataFrame) -> pd.DataFrame:
    u = pd.read_csv(UNCERTAINTY)
    train_domains = {
        block: set(frame.loc[frame["spatial_block"] != block, "daly_domain"]) for block in range(4)
    }
    positive_domains = set(frame.loc[frame["deposit_present"] == 1, "daly_domain"])
    u["domain_in_training"] = [d in train_domains[b] for d, b in zip(u["daly_domain"], u["spatial_block"])]
    u["domain_has_positives"] = u["daly_domain"].isin(positive_domains)

    def summarise(group: pd.DataFrame) -> pd.Series:
        w = group["prob_interval_width95"]
        return pd.Series({"cells": len(group), "median_width": w.median(),
                          "q90_width": w.quantile(0.9), "max_width": w.max(),
                          "share_width_gt_0_2": (w > 0.2).mean(),
                          "share_width_gt_0_6": (w > 0.6).mean()})

    by_domain = u.groupby(["daly_domain", "domain_has_positives"]).apply(summarise, include_groups=False).reset_index()
    by_domain.insert(0, "grouping", "domain")
    by_status = u.groupby(["domain_in_training", "domain_has_positives"]).apply(summarise, include_groups=False).reset_index()
    by_status.insert(0, "grouping", "training_support")
    wide = u[u["prob_interval_width95"] > 0.6]
    tail = pd.DataFrame([{
        "grouping": "cells_width_gt_0_6", "cells": len(wide),
        "daly_domain": "; ".join(f"{k}:{v}" for k, v in wide["daly_domain"].value_counts().items()),
        "domain_in_training": int(wide["domain_in_training"].sum()),
        "domain_has_positives": int(wide["domain_has_positives"].sum()),
    }])
    return pd.concat([by_domain, by_status, tail], ignore_index=True)


def lithology_retained(frame: pd.DataFrame) -> pd.DataFrame:
    codes = sorted(frame[LITHO_COL].astype(int).unique())
    reference = codes[0]  # pd.get_dummies(drop_first=True) drops the first sorted level
    rows = []
    for block in range(4):
        train = frame[frame["spatial_block"] != block]
        retained, excluded = [], []
        for code in codes[1:]:
            sel = train[LITHO_COL].astype(int) == code
            has_pos = bool((sel & (train["deposit_present"] == 1)).any())
            has_neg = bool((sel & (train["deposit_present"] == 0)).any())
            (retained if has_pos and has_neg else excluded).append(code)
        rows.append({
            "fold": block + 1,
            "reference_class": f"{reference} {LITHO_NAMES[reference]}",
            "retained_classes": "; ".join(f"{c} {LITHO_NAMES[c]}" for c in retained),
            "n_retained": len(retained),
            "excluded_classes_pooled_with_reference": "; ".join(f"{c} {LITHO_NAMES[c]}" for c in excluded),
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    oof = pd.read_csv(OOF)
    assert len(oof) == len(frame) == 1872 and int(oof["deposit_present"].sum()) == 138

    coefficients_nrb().to_csv(OUT / "coefficients_nrb.csv", index=False)
    direction_by_representation().to_csv(OUT / "direction_by_representation.csv", index=False)
    (OUT / "dstar_informative_subset.json").write_text(
        json.dumps(dstar_informative_subset(), indent=2, default=bool), encoding="utf-8")
    scale_posteriors().to_csv(OUT / "scale_posteriors.csv", index=False)
    fold_auc_intervals(oof).to_csv(OUT / "fold_auc_intervals.csv", index=False)
    accuracy, reliability = probability_accuracy(oof, frame)
    (OUT / "probability_accuracy.json").write_text(json.dumps(accuracy, indent=2), encoding="utf-8")
    reliability.to_csv(OUT / "reliability_bins.csv", index=False)
    uncertainty_by_domain(frame).to_csv(OUT / "uncertainty_by_domain.csv", index=False)
    lithology_retained(frame).to_csv(OUT / "lithology_retained_by_fold.csv", index=False)

    inputs = [DATA, OOF, UNCERTAINTY, ROBUSTNESS,
              *(V14_DIR / f"v14_centroid_fold_{k}_trace.nc" for k in range(1, 5)),
              *sorted((SENS_DIR / "traces").glob("model_[BCD]_fold_*_trace.nc"))]
    packages = ["numpy", "pandas", "scipy", "arviz", "pymc", "pytensor", "scikit-learn",
                "geopandas", "shapely", "rasterio", "matplotlib"]
    manifest = {
        "script": "src/paper1_supplement_summaries.py",
        "refit": False, "mcmc_run": False, "bootstrap_draws": N_BOOT, "seed": SEED,
        "python": platform.python_version(),
        "packages": {p: _version(p) for p in packages},
        "inputs": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in inputs},
        "outputs": sorted(p.name for p in OUT.glob("*") if p.name != "manifest.json"),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {len(manifest['outputs'])} outputs to {OUT.relative_to(ROOT)}")


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    main()
