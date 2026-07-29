"""C++ accelerated AGraph expression implementation.

This sub-package provides a C++17/Eigen implementation of the acyclic-graph
expression engine. It exposes the same public API as the pure-Python sibling
package :mod:`~bingo.expressions.agraph.pyagraph` with a thin Python shim to
align public semantics (loss/score kinds, keyword-only fit tolerance, implicit
regression helpers).

If the compiled extension ``_cppagraph`` is not available (e.g. the C++
build was skipped), importing this package will raise :exc:`ImportError`.
The parent package's ``__init__.py`` handles the fallback to pyagraph.
"""

from ._cppagraph import (  # type: ignore[import-not-found]
    # DataContainer class
    DataContainer,
    # Operator ID constants
    VARIABLE,
    CONSTANT,
    INTEGER,
    ADDITION,
    SUBTRACTION,
    MULTIPLICATION,
    DIVISION,
    POWER,
    SAFE_POWER,
    SQUARE,
    CUBE,
    SQRT,
    ABS,
    EXPONENTIAL,
    LOGARITHM,
    SIN,
    COS,
    TAN,
    ARCSIN,
    ARCCOS,
    ARCTAN,
    SINH,
    COSH,
    TANH,
    # Operator property sets
    TERMINAL_IDS,
    ARITY_2_IDS,
    # NumPy lookup arrays
    IS_TERMINAL_ARRAY,
    IS_ARITY_2_ARRAY,
    # Name mapping
    OPERATOR_NAMES,
    # Evaluation engine
    evaluate,
    evaluate_with_derivative,
    CachedEvaluator,
    # Simplification / stack reduction
    get_utilized_commands,
    reduce,
    # CAS simplification pipeline
    cas_simplify,
    # Expression class (bound below as _CppAGraphExpression)
    AGraphExpression as _CppAGraphExpression,
)

from typing import Tuple

import numpy as _np
import scipy.optimize as _sopt


# ---------------------------- helpers (public semantics) -------------------- #

_POS_INF = float("inf")
_NEG_INF = float("-inf")


def _mean_absolute_error(residuals: _np.ndarray) -> float:
    return float(_np.mean(_np.abs(residuals)))


def _mean_squared_error(residuals: _np.ndarray) -> float:
    return float(_np.mean(residuals ** 2))


def _root_mean_squared_error(residuals: _np.ndarray) -> float:
    return float(_np.sqrt(_np.mean(residuals ** 2)))


def _relative_mse(residuals: _np.ndarray, y: _np.ndarray) -> float:
    y = _np.asarray(y, dtype=float).ravel()
    if _np.any(y == 0):
        raise ValueError("relative_mse rejects zero-valued targets")
    return float(_np.mean((residuals / y) ** 2))


def _r2_score(predictions: _np.ndarray, y: _np.ndarray) -> float:
    predictions = _np.asarray(predictions, dtype=float).ravel()
    y = _np.asarray(y, dtype=float).ravel()
    ss_res = _np.sum((y - predictions) ** 2)
    ss_tot = _np.sum((y - _np.mean(y)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def _laplace_nmll(residuals: _np.ndarray, n_constants: int) -> float:
    n = len(residuals)
    k = n_constants + 1
    b = 1 / _np.sqrt(n)
    mse = _np.mean(residuals ** 2)
    if mse <= 0:
        mse = _np.finfo(float).tiny
    log_like = -n / 2 * _np.log(mse) - n / 2 - n / 2 * _np.log(2 * _np.pi)
    return float((1 - b) * log_like + _np.log(b) / 2 * k)


_SIMPLE_LOSS_METRICS = {
    "mse": _mean_squared_error,
    "mae": _mean_absolute_error,
    "rmse": _root_mean_squared_error,
}
_VALID_LOSS_KINDS = frozenset(set(_SIMPLE_LOSS_METRICS) | {"relative_mse", "correlation", "laplace_nmll"})
_VALID_SCORE_KINDS = frozenset({"r2", "laplace_nmll"})


# ---------------------------- public shim class ----------------------------- #


class AGraphExpression(_CppAGraphExpression):
    """Python-level shim over the C++ AGraph to align public semantics.

    Adds:
    - keyword-only ``tolerance`` for fit / fit_implicit
    - explicit/implicit loss/score kinds matching the Python contract
    - ``gradient`` convenience returning (f(x), df/dx)
    - non-finite normalization in public paths
    """

    # sklearn-like API --------------------------------------------------- #
    def predict(self, X):  # type: ignore[override]
        X = _np.atleast_2d(_np.asarray(X, dtype=float))
        return self._evaluate(X).ravel()

    def gradient(self, X) -> Tuple[_np.ndarray, _np.ndarray]:
        X = _np.atleast_2d(_np.asarray(X, dtype=float))
        f, df_dx = self._evaluate_with_x_gradient(X)
        return f.ravel(), df_dx

    def fit(self, X, y, *, tolerance: float = 1e-5):  # type: ignore[override]
        """Fit constants to explicit-regression data.

        Publicly enforces keyword-only tolerance. Delegates to the C++
        implementation when available; if its signature differs, falls back
        gracefully without the tolerance.
        """
        # Ensure derived layer is ready before fitting (mirrors Python impl)
        try:
            return super().fit(X, y, tolerance=tolerance)  # type: ignore[misc]
        except TypeError:  # old signature without tolerance
            return super().fit(X, y)

    def fit_implicit(self, X, dx_dt, *, tolerance: float = 1e-5):
        """Fit constants to implicit-regression data.

        Delegates to C++ if available, otherwise performs a local least-squares
        solve using the public implicit residual vector.
        """
        # Try delegating first (newer C++ builds)
        try:
            return super().fit_implicit(X, dx_dt, tolerance=tolerance)  # type: ignore[misc]
        except AttributeError:
            # Fallback: scipy least_squares on implicit residuals
            X = _np.atleast_2d(_np.asarray(X, dtype=float))
            dx_dt = _np.atleast_2d(_np.asarray(dx_dt, dtype=float))
            if len(self.constants) == 0:
                return self

            x0 = _np.array(self.constants, dtype=float)

            def residuals(params):
                self.constants = params
                return self._implicit_residual_vector(X, dx_dt)

            try:
                res = _sopt.least_squares(residuals, x0, ftol=tolerance, xtol=tolerance)
                self.constants = res.x
            except Exception:
                pass
            return self

    # explicit metrics --------------------------------------------------- #
    def _explicit_predictions(self, X, y):
        X = _np.atleast_2d(_np.asarray(X, dtype=float))
        y = _np.asarray(y, dtype=float).ravel()
        predictions = self.predict(X)
        if not _np.all(_np.isfinite(predictions)):
            return None, y
        return predictions, y

    def loss(self, X, y, *, kind: str = "mse") -> float:
        if kind not in _VALID_LOSS_KINDS:
            raise ValueError(f"kind must be one of {sorted(_VALID_LOSS_KINDS)!r}, got {kind!r}")
        predictions, y = self._explicit_predictions(X, y)
        if predictions is None:
            return _POS_INF
        residuals = predictions - y
        if kind in _SIMPLE_LOSS_METRICS:
            value = _SIMPLE_LOSS_METRICS[kind](residuals)
        elif kind == "laplace_nmll":
            value = -_laplace_nmll(residuals, len(self.constants))
        elif kind == "relative_mse":
            value = _relative_mse(residuals, y)
        else:  # "correlation"
            # 1 - r**2; perfectly associated yields 0
            preds = _np.asarray(predictions, dtype=float).ravel()
            y1 = _np.asarray(y, dtype=float).ravel()
            if _np.std(preds) == 0 or _np.std(y1) == 0:
                value = 1.0
            else:
                r = _np.corrcoef(preds, y1)[0, 1]
                value = 1.0 - r ** 2
        return float(value) if _np.isfinite(value) else _POS_INF

    def score(self, X, y, *, kind: str = "r2", metric: str | None = None) -> float:  # type: ignore[override]
        """Higher-is-better score.

        ``kind`` vocabulary matches the Python backend ("r2", "laplace_nmll").
        For backward compatibility, a legacy ``metric`` argument is accepted and
        interpreted as a loss-kind; the corresponding loss is returned.
        """
        if metric is not None:
            # Back-compat path expected by historical tests
            return self.loss(X, y, kind=metric)
        if kind not in _VALID_SCORE_KINDS:
            raise ValueError(f"kind must be one of {sorted(_VALID_SCORE_KINDS)!r}, got {kind!r}")
        predictions, y = self._explicit_predictions(X, y)
        if predictions is None:
            return _NEG_INF
        if kind == "laplace_nmll":
            value = _laplace_nmll(predictions - y, len(self.constants))
        else:  # "r2"
            value = _r2_score(predictions, y)
        return float(value) if _np.isfinite(value) else _NEG_INF

    # implicit metrics --------------------------------------------------- #
    def _implicit_residual_vector(self, X, dx_dt, *, required_params=None):
        f, df_dx = self._evaluate_with_x_gradient(X)
        dot_product = df_dx * dx_dt
        if required_params is not None:
            n_params_used = (_np.abs(dot_product) > 1e-16).sum(axis=1)
            if not _np.any(n_params_used >= required_params):
                return _np.full((X.shape[0],), _POS_INF)
        denominator = _np.sum(_np.abs(dot_product), axis=1)
        with _np.errstate(divide="ignore", invalid="ignore"):
            residual = _np.sum(dot_product, axis=1) / denominator
        residual[~_np.isfinite(denominator)] = _POS_INF
        return residual

    def implicit_loss(self, X, dx_dt, *, required_params=None) -> float:
        X = _np.atleast_2d(_np.asarray(X, dtype=float))
        dx_dt = _np.atleast_2d(_np.asarray(dx_dt, dtype=float))
        residual = self._implicit_residual_vector(X, dx_dt, required_params=required_params)
        if not _np.all(_np.isfinite(residual)):
            return _POS_INF
        value = _mean_absolute_error(residual)
        return float(value) if _np.isfinite(value) else _POS_INF

    def implicit_score(self, X, dx_dt, *, required_params=None) -> float:
        loss = self.implicit_loss(X, dx_dt, required_params=required_params)
        return _NEG_INF if loss == _POS_INF else -loss


__all__ = [
    "DataContainer",
    # Operator ID constants
    "VARIABLE",
    "CONSTANT",
    "INTEGER",
    "ADDITION",
    "SUBTRACTION",
    "MULTIPLICATION",
    "DIVISION",
    "POWER",
    "SAFE_POWER",
    "SQUARE",
    "CUBE",
    "SQRT",
    "ABS",
    "EXPONENTIAL",
    "LOGARITHM",
    "SIN",
    "COS",
    "TAN",
    "ARCSIN",
    "ARCCOS",
    "ARCTAN",
    "SINH",
    "COSH",
    "TANH",
    # Operator property sets
    "TERMINAL_IDS",
    "ARITY_2_IDS",
    # NumPy lookup arrays
    "IS_TERMINAL_ARRAY",
    "IS_ARITY_2_ARRAY",
    # Name mapping
    "OPERATOR_NAMES",
    # Evaluation engine
    "evaluate",
    "evaluate_with_derivative",
    "CachedEvaluator",
    # Simplification / stack reduction
    "get_utilized_commands",
    "reduce",
    # CAS simplification pipeline
    "cas_simplify",
    # Expression class (shimmed)
    "AGraphExpression",
]
