"""Contract tests for the numerical fallbacks used by the QGIS runtime."""

import numpy as np

from src.numerics import average_precision_score, precision_recall_curve
from src.oof_provenance import load_v11_oof


def test_average_precision_matches_frozen_v11_oof_values():
    oof = load_v11_oof()
    expected_fold = [0.06122244292275711, 0.11540956072226605,
                     0.14066293400023203, 0.1622054630019324]
    for fold, expected in enumerate(expected_fold):
        frame = oof[oof["fold"] == fold]
        actual = average_precision_score(
            frame["deposit_present"].to_numpy(), frame["prob_v11"].to_numpy()
        )
        assert np.isclose(actual, expected, atol=1e-12)

    pooled = average_precision_score(
        oof["deposit_present"].to_numpy(), oof["prob_v11"].to_numpy()
    )
    assert np.isclose(pooled, 0.1341428127365783, atol=1e-12)


def test_average_precision_is_not_trapezoidal_threshold_area():
    oof = load_v11_oof()
    frame = oof[oof["fold"] == 0]
    y = frame["deposit_present"].to_numpy()
    p = frame["prob_v11"].to_numpy()
    precision, recall, _ = precision_recall_curve(y, p)
    trapezoid = abs(float(np.trapz(precision, recall)))
    average_precision = average_precision_score(y, p)
    assert np.isclose(average_precision, 0.06122244292275711, atol=1e-12)
    assert np.isclose(trapezoid, 0.05404742882227891, atol=1e-12)
    assert not np.isclose(average_precision, trapezoid, atol=1e-6)
