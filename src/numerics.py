"""Numerical helpers with scikit-learn-compatible pure NumPy/SciPy fallbacks.

The active interpreter in this environment (QGIS CPython 3.12) provides NumPy,
SciPy and pandas but not scikit-learn; the project virtualenv's scikit-learn
build targets CPython 3.10 and cannot be imported by CPython 3.12.  When
scikit-learn is importable these names are simply re-exported, so behaviour is
unchanged on a normal environment.  The fallbacks implement the same
estimators/objectives (standardisation with ddof=0; L2-penalised logistic
regression with an unpenalised intercept and C=1.0) and the same metric
definitions (rank-based ROC AUC, trapezoidal PR AUC, mean squared error Brier
score).  Their agreement with the frozen scikit-learn outputs is asserted in
``tests/test_numerics_compat.py``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata

try:  # pragma: no cover - exercised only where scikit-learn exists
    from sklearn.linear_model import LogisticRegression as _SkLogisticRegression
    from sklearn.metrics import (
        auc,
        average_precision_score,
        brier_score_loss,
        precision_recall_curve,
        roc_auc_score,
    )
    from sklearn.decomposition import PCA as _SkPCA
    from sklearn.preprocessing import StandardScaler as _SkStandardScaler
    from sklearn.utils import resample

    SKLEARN_AVAILABLE = True
    LogisticRegression = _SkLogisticRegression
    StandardScaler = _SkStandardScaler
    PCA = _SkPCA
except Exception:  # ImportError or binary incompatibility
    SKLEARN_AVAILABLE = False

    class StandardScaler:
        """Column-wise standardisation matching ``sklearn.preprocessing``."""

        def __init__(self, copy: bool = True):
            self.copy = copy

        def fit(self, X, y=None):
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            self.mean_ = X.mean(axis=0)
            scale = X.std(axis=0, ddof=0)
            scale = np.where(scale == 0.0, 1.0, scale)
            self.scale_ = scale
            self.var_ = scale ** 2
            self.n_features_in_ = X.shape[1]
            return self

        def transform(self, X):
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return (X - self.mean_) / self.scale_

        def fit_transform(self, X, y=None):
            return self.fit(X, y).transform(X)

    class PCA:
        """Principal component analysis matching ``sklearn.decomposition.PCA``.

        Only the leading component(s) are required by the fold construction.
        The SVD sign convention follows ``sklearn.utils.extmath.svd_flip`` so
        that ``get_along_belt_folds`` reproduces the frozen fold assignment.
        """

        def __init__(self, n_components: int = 1):
            self.n_components = n_components

        def fit(self, X, y=None):
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            self.mean_ = X.mean(axis=0)
            centred = X - self.mean_
            _, singular_values, vt = np.linalg.svd(centred, full_matrices=False)
            max_abs_cols = np.argmax(np.abs(vt), axis=1)
            signs = np.sign(vt[np.arange(vt.shape[0]), max_abs_cols])
            signs[signs == 0] = 1.0
            vt = vt * signs[:, None]
            self.components_ = vt[: self.n_components]
            self.explained_variance_ = (
                singular_values[: self.n_components] ** 2 / max(len(X) - 1, 1)
            )
            self.n_features_in_ = X.shape[1]
            return self

        def transform(self, X):
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return (X - self.mean_) @ self.components_.T

        def fit_transform(self, X, y=None):
            return self.fit(X, y).transform(X)

    class LogisticRegression:
        """L2-penalised logistic regression (lbfgs), sklearn-compatible API."""

        def __init__(
            self,
            max_iter: int = 1000,
            C: float = 1.0,
            solver: str = "lbfgs",
            random_state=None,
            tol: float = 1e-10,
        ):
            if solver != "lbfgs":
                raise NotImplementedError("Fallback supports solver='lbfgs' only.")
            self.max_iter = max_iter
            self.C = C
            self.solver = solver
            self.random_state = random_state
            self.tol = tol

        def fit(self, X, y):
            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=float).ravel()
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            n_samples, n_features = X.shape
            design = np.column_stack([np.ones(n_samples), X])

            def objective(theta):
                coef = theta[1:]
                z = design @ theta
                loss = np.sum(np.logaddexp(0.0, z) - y * z)
                value = 0.5 * float(coef @ coef) + self.C * loss
                residual = 1.0 / (1.0 + np.exp(-z)) - y
                grad = np.empty_like(theta)
                grad[0] = self.C * residual.sum()
                grad[1:] = coef + self.C * (X.T @ residual)
                return value, grad

            result = minimize(
                objective,
                np.zeros(n_features + 1),
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": self.max_iter, "ftol": 1e-15, "gtol": 1e-8},
            )
            self.coef_ = result.x[1:].reshape(1, -1)
            self.intercept_ = np.array([result.x[0]])
            self.classes_ = np.array([0.0, 1.0])
            self.n_iter_ = np.array([result.nit])
            self.converged_ = bool(result.success)
            self.optimizer_message_ = str(result.message)
            self.n_features_in_ = n_features
            return self

        def decision_function(self, X):
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return X @ self.coef_[0] + self.intercept_[0]

        def predict_proba(self, X):
            z = self.decision_function(X)
            positive = 1.0 / (1.0 + np.exp(-z))
            return np.column_stack([1.0 - positive, positive])

        def predict(self, X):
            return (self.decision_function(X) > 0.0).astype(float)

    def roc_auc_score(y_true, y_score):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score, dtype=float)
        positive = y_true == 1
        n_pos = int(positive.sum())
        n_neg = int(y_true.size - n_pos)
        if n_pos == 0 or n_neg == 0:
            raise ValueError("Only one class present in y_true.")
        ranks = rankdata(y_score, method="average")
        return float(
            (ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        )

    def average_precision_score(y_true, y_score):
        # Non-interpolated average precision, matching scikit-learn.
        y_true = np.asarray(y_true)
        if int((y_true == 1).sum()) == 0:
            return float(nan)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return float(-np.sum(np.diff(recall) * precision[:-1]))

    def brier_score_loss(y_true, y_prob):
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)
        return float(np.mean((y_prob - y_true) ** 2))

    def _binary_clf_curve(y_true, y_score, pos_label=1):
        """Tie-aware cumulative TP/FP counts, mirroring scikit-learn."""
        y_true = (np.asarray(y_true) == pos_label).astype(float)
        y_score = np.asarray(y_score, dtype=float)
        desc = np.argsort(y_score, kind="mergesort")[::-1]
        y_score = y_score[desc]
        y_true = y_true[desc]
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
        tps = np.cumsum(y_true)[threshold_idxs]
        fps = 1 + threshold_idxs - tps
        return fps, tps, y_score[threshold_idxs]

    def precision_recall_curve(y_true, probas_pred, pos_label=1):
        fps, tps, thresholds = _binary_clf_curve(y_true, probas_pred, pos_label)
        ps = tps + fps
        precision = np.zeros_like(tps)
        np.divide(tps, ps, out=precision, where=(ps != 0))
        if tps[-1] == 0:
            recall = np.ones_like(tps)
        else:
            recall = tps / tps[-1]
        sl = slice(None, None, -1)
        return (
            np.hstack((precision[sl], 1.0)),
            np.hstack((recall[sl], 0.0)),
            thresholds[sl],
        )

    def auc(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.shape[0] < 2:
            raise ValueError("At least 2 points are needed to compute area.")
        direction = 1.0
        dx = np.diff(x)
        if np.any(dx < 0):
            if np.all(dx <= 0):
                direction = -1.0
            else:
                raise ValueError("x is neither increasing nor decreasing.")
        return direction * float(np.trapz(y, x))

    def resample(*arrays, replace=True, n_samples=None, random_state=None):
        arrays = [np.asarray(array) for array in arrays]
        if not arrays:
            raise ValueError("At least one array to resample.")
        if n_samples is None:
            n_samples = arrays[0].shape[0]
        if random_state is not None:
            np.random.seed(random_state)
        indices = np.random.randint(0, arrays[0].shape[0], size=n_samples)
        # sklearn returns a list, even for a single input array
        return [array[indices] for array in arrays]

