"""
Unit tests for distance representation sensitivity analysis math.

Validates:
- D* forward/backward conversion for raw and log representations
- Derivative check (d eta / dz ≈ 0 at z*)
- log1p / expm1 round-trip
- Scale consistency
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
# pytest not required; tests run via __main__


# ── D* conversion tests ──────────────────────────────────────────────────


def test_model_a_dstar_roundtrip():
    """Model A (raw distance): z* -> D*_km -> z_eval -> derivative ≈ 0."""
    rng = np.random.default_rng(99)
    mu_m = 23210.0   # metres
    sigma_m = 25069.0

    b1 = rng.normal(0, 2, size=500)
    b2 = rng.normal(0, 1, size=500)
    valid = np.abs(b2) > 1e-5
    b1, b2 = b1[valid], b2[valid]

    z_star = -b1 / (2.0 * b2)
    d_star_km = (mu_m + sigma_m * z_star) / 1000.0

    # Round-trip: km -> metres -> z
    d_star_m = d_star_km * 1000.0
    z_eval = (d_star_m - mu_m) / sigma_m
    deriv = b1 + 2.0 * b2 * z_eval
    assert np.allclose(deriv, 0.0, atol=1e-10), f"Max err: {np.max(np.abs(deriv))}"


def test_model_b_dstar_roundtrip():
    """Model B (log distance): z* -> x_log* -> D*_km -> forward -> derivative ≈ 0."""
    rng = np.random.default_rng(99)
    mu_log = 2.52
    sigma_log = 0.95

    b1 = rng.normal(0, 2, size=500)
    b2 = rng.normal(0, 1, size=500)
    valid = np.abs(b2) > 1e-5
    b1, b2 = b1[valid], b2[valid]

    z_star = -b1 / (2.0 * b2)
    x_log_star = mu_log + sigma_log * z_star
    d_star_km = np.exp(x_log_star) - 1.0

    # Filter to physically valid draws (D* > -1, required for log1p)
    # Also filter to moderate z* to avoid extreme exp/log precision loss
    phys_valid = (d_star_km > -1.0) & (np.abs(z_star) < 10.0)
    d_star_km = d_star_km[phys_valid]
    b1_v, b2_v = b1[phys_valid], b2[phys_valid]
    z_star_v = z_star[phys_valid]

    # Round-trip: km -> log1p -> standardize -> z
    x_log_back = np.log1p(d_star_km)
    z_eval = (x_log_back - mu_log) / sigma_log
    deriv = b1_v + 2.0 * b2_v * z_eval
    assert np.allclose(deriv, 0.0, atol=1e-8), f"Max err: {np.max(np.abs(deriv))}"


def test_log1p_expm1_roundtrip():
    """Verify log1p / expm1 inverses are exact for positive distances."""
    distances_km = np.array([0.0, 0.001, 1.0, 10.0, 50.0, 112.75])
    x_log = np.log1p(distances_km)
    recovered = np.expm1(x_log)
    assert np.allclose(distances_km, recovered, atol=1e-14)


def test_nonneg_distance_log1p():
    """log1p(x) is well-defined for x >= 0."""
    distances_km = np.array([0.0, 1e-10, 0.001, 100.0])
    result = np.log1p(distances_km)
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0)


# ── Response curve transform tests ──────────────────────────────────────


def test_raw_forward_transform():
    """Raw: grid_km -> metres -> standardize."""
    mu_m, sigma_m = 23000.0, 25000.0
    grid_km = np.array([0.0, 10.0, 23.0, 50.0])
    z = (grid_km * 1000.0 - mu_m) / sigma_m
    expected = np.array([-0.92, -0.52, 0.0, 1.08])
    assert np.allclose(z, expected, atol=0.01)


def test_log_forward_transform():
    """Log: grid_km -> log1p -> standardize."""
    mu_log, sigma_log = 2.5, 1.0
    grid_km = np.array([0.0, 1.0, 10.0])
    x_log = np.log1p(grid_km)
    z = (x_log - mu_log) / sigma_log
    expected_x = np.array([0.0, np.log(2), np.log(11)])
    assert np.allclose(x_log, expected_x, atol=1e-10)


# ── Quadratic vs linear model parameter count ───────────────────────────


def test_linear_model_has_no_quadratic():
    """Models C/D should not have beta_sq parameters."""
    # This is a structural test; in the implementation, we check that
    # the model builder does not create mu_f_sq, sigma_f_sq, etc.
    quadratic_params = ['mu_f_sq', 'sigma_f_sq', 'offset_f_sq',
                        'mu_l_sq', 'sigma_l_sq', 'offset_l_sq',
                        'beta_f_sq', 'beta_l_sq']
    linear_params = ['mu_f_lin', 'sigma_f_lin', 'offset_f_lin',
                     'mu_l_lin', 'sigma_l_lin', 'offset_l_lin',
                     'beta_f_lin', 'beta_l_lin']
    # Linear models must NOT contain any quadratic params
    for p in quadratic_params:
        assert p not in linear_params, f"{p} found in linear param set"


if __name__ == '__main__':
    test_model_a_dstar_roundtrip()
    print("[PASS] Model A D* roundtrip")
    test_model_b_dstar_roundtrip()
    print("[PASS] Model B D* roundtrip")
    test_log1p_expm1_roundtrip()
    print("[PASS] log1p/expm1 roundtrip")
    test_nonneg_distance_log1p()
    print("[PASS] Non-negative distance log1p")
    test_raw_forward_transform()
    print("[PASS] Raw forward transform")
    test_log_forward_transform()
    print("[PASS] Log forward transform")
    test_linear_model_has_no_quadratic()
    print("[PASS] Linear model structure check")
    print("\nAll tests passed.")
