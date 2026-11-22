"""PyMC model construction for the V15 ablation ladder."""

from __future__ import annotations

from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from src.v15 import config


def _base_intercept(arrays: dict[str, Any], link: str) -> float:
    y = arrays["y_train"]
    if link == "cloglog":
        return float(np.log((y.sum() + 0.5) / (arrays["area_train_sum"] + 0.5)))
    return float(np.log((y.sum() + 0.5) / (len(y) - y.sum() + 0.5)))


def build_model(
    model_name: str,
    arrays: dict[str, Any],
    spec: dict[str, Any],
    prior_scale: float = 1.0,
):
    family = spec["family"]
    link = spec["link"]
    n_domains = arrays["n_domains"]
    n_rocks = arrays["n_rocks"]
    train = arrays["train"]
    base = _base_intercept(arrays, link)
    scale = float(prior_scale)

    with pm.Model() as model:
        if family == "global":
            alpha = pm.Normal("alpha", mu=base, sigma=config.PRIORS["intercept_mu_sigma"] * scale)
        else:
            alpha_mu = pm.Normal("alpha_mu", mu=base, sigma=config.PRIORS["intercept_mu_sigma"] * scale)
            alpha_sigma = pm.HalfNormal("alpha_sigma", sigma=config.PRIORS["intercept_sigma_scale"] * scale)
            alpha_offset = pm.Normal("alpha_offset", mu=0.0, sigma=1.0, shape=n_domains)
            alpha_dom = pm.Deterministic("alpha_dom", alpha_mu + alpha_offset * alpha_sigma)

        def shared(name: str):
            return pm.Normal(name, mu=0.0, sigma=config.PRIORS["slope_mu_sigma"] * scale)

        def varying(name: str):
            mu = pm.Normal(f"mu_{name}", mu=0.0, sigma=config.PRIORS["slope_mu_sigma"] * scale)
            sigma = pm.HalfNormal(f"sigma_{name}", sigma=config.PRIORS["slope_sigma_scale"] * scale)
            offset = pm.Normal(f"offset_{name}", mu=0.0, sigma=1.0, shape=n_domains)
            return pm.Deterministic(name, mu + offset * sigma)

        if family in {"global", "domain_intercept"}:
            beta_f_lin = shared("beta_f_lin")
            beta_f_sq = shared("beta_f_sq")
            beta_l_lin = shared("beta_l_lin")
            beta_l_sq = shared("beta_l_sq")
        else:
            beta_f_lin = varying("beta_f_lin")
            beta_f_sq = varying("beta_f_sq")
            beta_l_lin = varying("beta_l_lin")
            beta_l_sq = varying("beta_l_sq")

        if family == "domain_distance_gravity":
            beta_grav = varying("beta_grav")
        else:
            beta_grav = shared("beta_grav")

        if n_rocks:
            beta_rocks = pm.Normal("beta_rocks", mu=0.0, sigma=config.PRIORS["rock_sigma"] * scale, shape=n_rocks)
        else:
            beta_rocks = None
        if spec["area"] == "covariate":
            beta_area = pm.Normal("beta_area", mu=0.0, sigma=config.PRIORS["area_sigma"] * scale)
        else:
            beta_area = None

        if family == "global":
            eta = alpha
        else:
            eta = alpha_dom[train["domain_idx"]]

        if family in {"global", "domain_intercept"}:
            eta = eta + beta_f_lin * train["fault_z"] + beta_f_sq * train["fault_z2"]
            eta = eta + beta_l_lin * train["lith_z"] + beta_l_sq * train["lith_z2"]
            eta = eta + beta_grav * train["grav_z"]
        else:
            eta = eta + beta_f_lin[train["domain_idx"]] * train["fault_z"]
            eta = eta + beta_f_sq[train["domain_idx"]] * train["fault_z2"]
            eta = eta + beta_l_lin[train["domain_idx"]] * train["lith_z"]
            eta = eta + beta_l_sq[train["domain_idx"]] * train["lith_z2"]
            if family == "domain_distance_gravity":
                eta = eta + beta_grav[train["domain_idx"]] * train["grav_z"]
            else:
                eta = eta + beta_grav * train["grav_z"]

        if beta_rocks is not None:
            eta = eta + pt.dot(train["rocks"], beta_rocks)
        if spec["area"] == "offset":
            eta = eta + train["log_area"]
        elif spec["area"] == "covariate":
            eta = eta + beta_area * train["log_area_z"]

        if link == "cloglog":
            probability = -pt.expm1(-pt.exp(eta))
            pm.Bernoulli(
                "y_obs",
                p=pt.clip(probability, config.PCELL, 1.0 - config.PCELL),
                observed=arrays["y_train"],
            )
        else:
            pm.Bernoulli("y_obs", logit_p=eta, observed=arrays["y_train"])
    return model
