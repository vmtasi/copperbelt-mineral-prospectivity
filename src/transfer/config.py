"""Frozen paths and analysis constants for the Paper 1 transfer study."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V15 = ROOT / "paper_1" / "v15"
PHASE2B = V15 / "phase2b_two_domain"
OUTPUT = ROOT / "paper_1" / "transfer"
MODEL_FRAME = V15 / "data" / "v15_model_frame.csv"
PHASE2B_FRAME = PHASE2B / "data" / "primary_frame.csv"

Y_COL = "deposit_present"
FOLD_COL = "fold"
AREA_COL = "log_area_km2"
OUTSIDE_NRB = "outside_nrb"

RAW_FEATURES = {
    "fault": "fault_stored_km",
    "contact": "contact_km",
    "gravity": "gravity_mgal",
}
PRIMARY_MODEL = "T0"
SENSITIVITY_MODELS = {
    "T0_rebuilt": {"fault_column": "fault_rebuilt_km"},
    "T0_linear": {"quadratic": False},
    "T0_nrb": {"outside_nrb": True},
}

MODEL_PRIOR_SD = 1.0
SAMPLER = {
    "chains": 4,
    "draws": 1200,
    "tune": 1800,
    "cores": 4,
    "target_accept": 0.97,
    "random_seed": 42,
}
MAP = {
    "maxiter": 3000,
    "ftol": 1e-12,
    "gtol": 1e-8,
    "laplace_draws": 1000,
}
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 42
RANDOM_CV_REPEATS = 20
RANDOM_CV_FOLDS = 4
INNER_BLOCK_SIZE_M = 25000.0
AOA_CV_FOLDS = 5

MCMC_TOLERANCE = {
    "max_rhat": 1.05,
    "min_ess_bulk": 200,
    "min_ess_tail": 200,
    "max_divergences": 0,
}
MAP_MCMC_MAX_LOG_SCORE_DIFFERENCE = 0.002

PREDICTOR_VERDICT_RULE = {
    "requirement": (
        "transfers only when every relevant coefficient retains the same sign "
        "in at least 3 of 4 leave-one-sector-out fits and the oriented "
        "within-sector AUC 95% interval lower bound exceeds 0.5 in at least "
        "3 of 4 sectors; partly transfers when exactly one condition holds; "
        "otherwise does not transfer"
    ),
    "marginal_predictors": ["fault", "contact", "gravity"],
}

PARTITION_RULE = {
    "axis": (
        "first principal component of all complete-case centroid coordinates, "
        "oriented so the minimum projection is 0 km at the south-eastern end"
    ),
    "P_GAP": (
        "K=4 contiguous sectors cut at midpoints of the three largest eligible "
        "gaps between consecutive positive-cell projections; every sector must "
        "be at least 100 km long; ties are resolved by feasibility"
    ),
    "P_POS": (
        "K=4 contiguous sectors cut between positive-cell projections so each "
        "sector contains 34 or 35 positives, with earlier full groups receiving "
        "35 when the remainder is two"
    ),
}

