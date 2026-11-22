"""Frozen configuration for the Paper 1 V15 experiment.

Only this module and the generated specification JSON define the experiment.
V14 paths are read-only controls; every new artefact is written below
``paper_1/v15``.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V15 = ROOT / "paper_1" / "v15"
V14_TABLE = ROOT / "data" / "copperbelt_training_v14_centroid_contact.csv"
V14_OOF = ROOT / "figures" / "v14_centroid_contact" / "v14_centroid_oof_predictions.csv"
V14_RUN_MANIFEST = ROOT / "figures" / "v14_centroid_contact" / "run_manifest.json"
V14_FOLD_METRICS = ROOT / "figures" / "v14_centroid_contact" / "v14_centroid_fold_metrics.csv"
GRID_GPKG = Path(r"C:\Project\data for projects\grid_internal_contact_distance_1899.gpkg")
LITHO_RASTER = Path(r"C:\Project\data for projects\lithography_square_93m.tif")
CONTACT_GPKG = Path(r"C:\Project\data for projects\internal_contacts_only.gpkg")
GRID_CONTACT_GPKG = Path(r"C:\Project\data for projects\grid_internal_contact_distance_1899.gpkg")
TRACTS_GPKG = (
    ROOT
    / "data"
    / "raw"
    / "rdc_rocks"
    / "derived"
    / "spatial_atlas_v1"
    / "layers"
    / "usgs_permissive_tracts.gpkg"
)
BENCHMARK_SUMS = ROOT / "paper_1" / "benchmark" / "SHA256SUMS"

DOMAINS = ["CRZ", "MMSB", "NKB", "NRB_3a", "NRB_3b", "SRB"]
CONTINUOUS = {
    "fault": "distance_to_fault",
    "lith": "distance_to_lithology_contact",
    "gravity": "bouguer",
}
LITHO_COL = "litho_contact_litho_class"
Y_COL = "deposit_present"
FOLD_SEED = 42
BOOTSTRAP_SEED = 42
PERTURB_SEED = 42

SAMPLER = {
    "draws": 1000,
    "tune": 1500,
    "chains": 2,
    "cores": 2,
    "target_accept": 0.95,
    "random_seed": 42,
}

MODELS = {
    "M0_global": {
        "family": "global",
        "link": "cloglog",
        "area": "offset",
        "include_litho": True,
    },
    "M1_domain_intercept": {
        "family": "domain_intercept",
        "link": "cloglog",
        "area": "offset",
        "include_litho": True,
    },
    "M2_domain_distance": {
        "family": "domain_distance",
        "link": "cloglog",
        "area": "offset",
        "include_litho": True,
    },
    "M3_domain_distance_gravity": {
        "family": "domain_distance_gravity",
        "link": "cloglog",
        "area": "offset",
        "include_litho": True,
    },
    "V14_control": {
        "family": "domain_distance",
        "link": "logit",
        "area": "none",
        "include_litho": True,
    },
    "M2_logit_no_area": {
        "family": "domain_distance",
        "link": "logit",
        "area": "none",
        "include_litho": True,
    },
    "M2_logit_area_covariate": {
        "family": "domain_distance",
        "link": "logit",
        "area": "covariate",
        "include_litho": True,
    },
    "M2_cloglog_no_litho": {
        "family": "domain_distance",
        "link": "cloglog",
        "area": "offset",
        "include_litho": False,
    },
    "M2_logit_raster_centroid_litho": {
        "family": "domain_distance",
        "link": "logit",
        "area": "none",
        "include_litho": True,
        "litho_source": "raster_centroid",
    },
}

PRIMARY_MODELS = [
    "M0_global",
    "M1_domain_intercept",
    "M2_domain_distance",
    "M3_domain_distance_gravity",
]
SENSITIVITY_MODELS = [
    "V14_control",
    "M2_logit_no_area",
    "M2_logit_area_covariate",
    "M2_cloglog_no_litho",
]

PRIORS = {
    "intercept_mu_sigma": 1.0,
    "intercept_sigma_scale": 1.0,
    "slope_mu_sigma": 1.0,
    "slope_sigma_scale": 1.0,
    "rock_sigma": 1.0,
    "area_sigma": 1.0,
    "prior_scale": 1.0,
}

PCELL = 1e-12
MODEL_FRAME_COLUMNS = [
    "id",
    "deposit_present",
    "fold",
    "daly_domain",
    "centroid_x",
    "centroid_y",
    "distance_to_fault",
    "distance_to_lithology_contact",
    "bouguer",
    "litho_contact_litho_class",
    "litho_class_raster_centroid",
    "area_km2",
    "log_area_km2",
    "small_cell",
]
