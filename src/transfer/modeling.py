"""Bayesian and MAP estimators for the transfer models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

from src.transfer import config

PCELL = 1e-12


def cloglog_probability(eta: np.ndarray) -> np.ndarray:
    return np.clip(-np.expm1(-np.exp(np.clip(eta, -30.0, 20.0))), PCELL, 1.0 - PCELL)


def log_probability(y: np.ndarray, eta: np.ndarray) -> np.ndarray:
    lam = np.exp(np.clip(eta, -30.0, 20.0))
    log_p = np.log(np.clip(-np.expm1(-lam), PCELL, 1.0))
    return np.where(y == 1, log_p, -lam)


def _score_eta(y: np.ndarray, eta: np.ndarray) -> np.ndarray:
    lam = np.exp(np.clip(eta, -30.0, 20.0))
    derivative = np.where(
        np.abs(lam) < 1e-6,
        1.0 - lam / 2.0,
        lam / np.expm1(lam),
    )
    return y * derivative - (1.0 - y) * lam


def build_pymc_model(data: dict[str, Any]):
    import pymc as pm
    import pytensor.tensor as pt

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=data["intercept_mean"], sigma=config.MODEL_PRIOR_SD)
        eta = alpha + data["offset_train"]
        for index, name in enumerate(data["features"]):
            coefficient = pm.Normal(
                f"beta_{name}", mu=0.0, sigma=config.MODEL_PRIOR_SD
            )
            eta = eta + coefficient * data["X_train"][:, index]
        probability = -pt.expm1(-pt.exp(eta))
        pm.Bernoulli(
            "y_obs",
            p=pt.clip(probability, PCELL, 1.0 - PCELL),
            observed=data["y_train"],
        )
    return model


def sample_pymc(data: dict[str, Any], draws: int, tune: int, chains: int, cores: int, target_accept: float, seed: int):
    import arviz as az
    import pymc as pm

    model = build_pymc_model(data)
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=seed,
            progressbar=True,
            idata_kwargs={"log_likelihood": False},
        )
    variables = [name for name in trace.posterior.data_vars if name != "y_obs"]
    summary = az.summary(
        trace,
        var_names=variables,
        hdi_prob=0.95,
        round_to=8,
    )
    summary["median"] = [
        float(np.median(np.asarray(trace.posterior[name].values)))
        for name in summary.index
    ]
    diverging = int(np.asarray(trace.sample_stats["diverging"]).sum())
    info = {
        "diverging": diverging,
        "max_rhat": float(np.nanmax(summary["r_hat"])),
        "min_ess_bulk": float(np.nanmin(summary["ess_bulk"])),
        "min_ess_tail": float(np.nanmin(summary["ess_tail"])),
    }
    info["converged"] = bool(
        diverging <= config.MCMC_TOLERANCE["max_divergences"]
        and info["max_rhat"] < config.MCMC_TOLERANCE["max_rhat"]
        and info["min_ess_bulk"] >= config.MCMC_TOLERANCE["min_ess_bulk"]
        and info["min_ess_tail"] >= config.MCMC_TOLERANCE["min_ess_tail"]
    )
    return trace, summary.reset_index(names="parameter"), info


def posterior_probability(trace, data: dict[str, Any]) -> np.ndarray:
    def samples(name: str) -> np.ndarray:
        values = np.asarray(trace.posterior[name].values)
        return values.reshape(-1, *values.shape[2:])

    eta = samples("alpha")[:, None] + data["offset_test"][None, :]
    for index, name in enumerate(data["features"]):
        eta = eta + samples(f"beta_{name}")[:, None] * data["X_test"][:, index][None, :]
    return cloglog_probability(eta)


@dataclass
class MAPFit:
    beta: np.ndarray
    covariance: np.ndarray
    standard_error: np.ndarray
    gradient: np.ndarray
    hessian: np.ndarray
    optimization_success: bool
    optimization_message: str
    intercept_mean: float
    feature_names: list[str]


def _prior_vector(data: dict[str, Any]) -> np.ndarray:
    return np.r_[data["intercept_mean"], np.zeros(len(data["features"]))]


def _map_objective(
    beta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    prior_mean: np.ndarray,
    return_gradient: bool,
):
    eta = beta[0] + offset + X @ beta[1:]
    value = -float(np.sum(log_probability(y, eta)))
    residual = beta - prior_mean
    value += 0.5 * float(residual @ residual) / config.MODEL_PRIOR_SD**2
    if not return_gradient:
        return value
    gradient = np.empty_like(beta)
    score = _score_eta(y, eta)
    gradient[0] = -score.sum() + residual[0] / config.MODEL_PRIOR_SD**2
    gradient[1:] = -(X.T @ score) + residual[1:] / config.MODEL_PRIOR_SD**2
    return value, gradient


def fit_map(data: dict[str, Any]) -> MAPFit:
    prior_mean = _prior_vector(data)
    initial = prior_mean.copy()

    def objective(beta):
        return _map_objective(
            beta,
            data["X_train"],
            data["y_train"],
            data["offset_train"],
            prior_mean,
            True,
        )

    result = minimize(
        objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": config.MAP["maxiter"],
            "ftol": config.MAP["ftol"],
            "gtol": config.MAP["gtol"],
        },
    )
    beta = np.asarray(result.x, dtype=float)
    p = beta.size
    hessian = np.empty((p, p), dtype=float)
    for index in range(p):
        step = 1e-4 * (1.0 + abs(beta[index]))
        upper = beta.copy()
        lower = beta.copy()
        upper[index] += step
        lower[index] -= step
        grad_upper = objective(upper)[1]
        grad_lower = objective(lower)[1]
        hessian[:, index] = (grad_upper - grad_lower) / (2.0 * step)
    hessian = (hessian + hessian.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    covariance = eigenvectors @ np.diag(1.0 / eigenvalues) @ eigenvectors.T
    return MAPFit(
        beta=beta,
        covariance=covariance,
        standard_error=np.sqrt(np.diag(covariance)),
        gradient=np.asarray(result.jac, dtype=float),
        hessian=hessian,
        optimization_success=bool(result.success),
        optimization_message=str(result.message),
        intercept_mean=float(data["intercept_mean"]),
        feature_names=list(data["features"]),
    )


def laplace_probability(
    fit: MAPFit,
    data: dict[str, Any],
    draws: int | None = None,
    seed: int = config.SAMPLER["random_seed"],
) -> np.ndarray:
    draws = draws or config.MAP["laplace_draws"]
    rng = np.random.default_rng(seed)
    sampled = rng.multivariate_normal(fit.beta, fit.covariance, size=draws)
    eta = sampled[:, 0, None] + data["offset_test"][None, :]
    eta = eta + sampled[:, 1:] @ data["X_test"].T
    return cloglog_probability(eta)


def map_prediction_summary(
    fit: MAPFit,
    data: dict[str, Any],
) -> dict[str, np.ndarray]:
    probability = laplace_probability(fit, data)
    return {
        "prob": probability.mean(axis=0),
        "prob_lower": np.percentile(probability, 2.5, axis=0),
        "prob_upper": np.percentile(probability, 97.5, axis=0),
    }


def coefficient_frame(fit: MAPFit, method: str) -> list[dict[str, Any]]:
    names = ["intercept", *[f"beta_{name}" for name in fit.feature_names]]
    return [
        {
            "estimation": method,
            "term": name,
            "estimate": float(fit.beta[index]),
            "standard_error": float(fit.standard_error[index]),
            "ci_lower": float(fit.beta[index] - 1.96 * fit.standard_error[index]),
            "ci_upper": float(fit.beta[index] + 1.96 * fit.standard_error[index]),
        }
        for index, name in enumerate(names)
    ]


def fit_intercept_oracle(
    y: np.ndarray,
    offset: np.ndarray,
    linear_score: np.ndarray,
    intercept_mean: float,
) -> float:
    def objective(alpha):
        eta = alpha[0] + offset + linear_score
        value = -float(np.sum(log_probability(y, eta)))
        value += 0.5 * (alpha[0] - intercept_mean) ** 2
        gradient = -float(_score_eta(y, eta).sum()) + (alpha[0] - intercept_mean)
        return value, np.array([gradient])

    result = minimize(
        objective,
        np.array([intercept_mean]),
        jac=True,
        method="L-BFGS-B",
    )
    return float(result.x[0])


def fit_scaled_oracle(
    y: np.ndarray,
    offset: np.ndarray,
    linear_score: np.ndarray,
    intercept_mean: float,
) -> tuple[float, float]:
    def objective(parameters):
        alpha, multiplier = parameters
        eta = alpha + offset + multiplier * linear_score
        value = -float(np.sum(log_probability(y, eta)))
        value += 0.5 * (alpha - intercept_mean) ** 2 + 0.5 * (multiplier - 1.0) ** 2
        derivative = _score_eta(y, eta)
        gradient = np.array(
            [
                -float(derivative.sum()) + (alpha - intercept_mean),
                -float(derivative @ linear_score) + (multiplier - 1.0),
            ]
        )
        return value, gradient

    result = minimize(
        objective,
        np.array([intercept_mean, 1.0]),
        jac=True,
        method="L-BFGS-B",
    )
    return float(result.x[0]), float(result.x[1])

