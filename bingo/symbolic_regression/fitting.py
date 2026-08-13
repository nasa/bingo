"""Pure fitting measures and configurable SciPy-based expression fitting."""

# Built-in analytic expression derivatives are intentionally internal hooks.
# They are shared by the Python and C++ Expression backends.
# pylint: disable=protected-access

from dataclasses import dataclass

import numpy as np
from scipy import optimize


@dataclass(frozen=True)
class FitResult:
    """Constants selected by a fitter without exposing solver-specific results.

    Parameters
    ----------
    constants : array-like
        Candidate simplified Expression constants.
    success : bool
        Whether the fitting algorithm converged numerically.
    message : str, optional
        Human-readable solver status.
    """

    constants: object
    success: bool
    message: str | None = None


class ResidualMeasure:
    """A vector-valued fitting measure with optional residual derivatives.

    ``value(expression, data, constants)`` must return one residual per sample.
    Optional ``jacobian`` and ``residual_hessian`` callables use the same
    arguments and return derivatives with respect to ``constants``.

    Parameters
    ----------
    value : callable
        Returns a one-dimensional residual vector.
    jacobian : callable, optional
        Returns the residual Jacobian with respect to constants.
    residual_hessian : callable, optional
        Returns one constant Hessian per residual.
    """

    def __init__(self, value, *, jacobian=None, residual_hessian=None):
        self.value = value
        self.jacobian = jacobian
        self.residual_hessian = residual_hessian

    def __call__(self, expression, data, constants):
        return self.value(expression, data, constants)


class ScalarMeasure:
    """A scalar-valued fitting measure with optional constant derivatives.

    Parameters
    ----------
    value : callable
        Returns a scalar fitting value.
    gradient : callable, optional
        Returns its gradient with respect to constants.
    hessian : callable, optional
        Returns its Hessian with respect to constants.
    """

    def __init__(self, value, *, gradient=None, hessian=None):
        self.value = value
        self.gradient = gradient
        self.hessian = hessian

    def __call__(self, expression, data, constants):
        return self.value(expression, data, constants)


def explicit_residuals():
    """Return ordinary explicit-regression residuals and their Jacobian.

    Returns
    -------
    ResidualMeasure
        A pure measure over two-array ``ObjectiveData(X, y)``.
    """

    def value(expression, data, constants):
        X, y = data.arrays
        return expression.predict(X, constants=constants) - y

    def jacobian(expression, data, constants):
        X, _ = data.arrays
        trial = expression.copy()
        trial.constants = constants
        _, derivative = trial._evaluate_with_const_gradient(np.atleast_2d(X))
        return derivative

    def residual_hessian(expression, data, constants):
        X, _ = data.arrays
        trial = expression.copy()
        trial.constants = constants
        _, _, hessian = trial._evaluate_with_const_hessian(np.atleast_2d(X))
        return hessian

    return ResidualMeasure(
        value, jacobian=jacobian, residual_hessian=residual_hessian
    )


def implicit_residuals(*, required_params=None):
    """Return normalized implicit-regression residuals.

    Parameters
    ----------
    required_params : int, optional
        Implicit-regression anti-triviality guard.

    Returns
    -------
    ResidualMeasure
        A pure measure over two-array ``ObjectiveData(X, dx_dt)``.
    """

    def value(expression, data, constants):
        X, dx_dt = data.arrays
        trial = expression.copy()
        trial.constants = constants
        _, df_dx = trial.gradient(X)
        dot_product = df_dx * dx_dt
        if required_params is not None:
            n_params_used = (np.abs(dot_product) > 1e-16).sum(axis=1)
            if not np.any(n_params_used >= required_params):
                return np.full(X.shape[0], np.inf)
        denominator = np.sum(np.abs(dot_product), axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            residual = np.sum(dot_product, axis=1) / denominator
        residual[~np.isfinite(denominator)] = np.inf
        return residual

    return ResidualMeasure(value)


def expression_loss(kind="mse"):
    """Adapt a named explicit Expression loss into a scalar fitting measure.

    Parameters
    ----------
    kind : str, optional
        A loss kind accepted by :meth:`Expression.loss`.

    Returns
    -------
    ScalarMeasure
        A pure measure over two-array ``ObjectiveData(X, y)``.
    """

    def value(expression, data, constants):
        X, y = data.arrays
        trial = expression.copy()
        trial.constants = constants
        return trial.loss(X, y, kind=kind)

    return ScalarMeasure(value)


def implicit_loss(*, required_params=None):
    """Adapt Expression implicit loss into a scalar fitting measure.

    Parameters
    ----------
    required_params : int, optional
        Implicit-regression anti-triviality guard.

    Returns
    -------
    ScalarMeasure
        A pure measure over two-array ``ObjectiveData(X, dx_dt)``.
    """

    def value(expression, data, constants):
        X, dx_dt = data.arrays
        trial = expression.copy()
        trial.constants = constants
        return trial.implicit_loss(X, dx_dt, required_params=required_params)

    return ScalarMeasure(value)


class ScipyFitter:
    """Fit an Expression with a selected SciPy root or minimize method.

    Root methods consume :class:`ResidualMeasure` instances. Minimize methods
    consume :class:`ScalarMeasure` instances. Plain callables are classified by
    their value at the fitter-owned initial constants.

    Parameters
    ----------
    method : str, optional
        A :func:`scipy.optimize.root` or :func:`scipy.optimize.minimize`
        method. Root methods require residual measures; all other methods are
        passed to ``minimize`` and require scalar measures.
    tolerance : float, optional
        Solver convergence tolerance.
    """

    _ROOT_METHODS = frozenset(
        {
            "hybr",
            "lm",
            "df-sane",
            "broyden1",
            "broyden2",
            "anderson",
            "linearmixing",
            "diagbroyden",
            "excitingmixing",
            "krylov",
        }
    )

    def __init__(self, method="lm", *, tolerance=1e-5):
        self.method = method
        self.tolerance = tolerance

    def __call__(self, expression, data, measure):
        initial = np.asarray(expression.constants, dtype=float)
        if self.method == "least_squares":
            return self._least_squares(expression, data, measure, initial)
        if self.method in self._ROOT_METHODS:
            return self._root(expression, data, measure, initial)
        return self._minimize(expression, data, measure, initial)

    def _least_squares(self, expression, data, measure, initial):
        if isinstance(measure, ScalarMeasure):
            raise TypeError("SciPy least_squares requires a residual fitting measure")

        def value(constants):
            residuals = np.asarray(measure(expression, data, constants), dtype=float)
            if residuals.ndim != 1:
                raise ValueError("residual fitting measures must return a one-dimensional vector")
            return residuals

        jacobian = getattr(measure, "jacobian", None)
        if not np.all(np.isfinite(value(initial))):
            return FitResult(initial, False, "non-finite residuals at initial constants")
        result = optimize.least_squares(
            value,
            initial,
            jac=(
                lambda constants: np.asarray(
                    jacobian(expression, data, constants), dtype=float
                )
            )
            if jacobian
            else "2-point",
            ftol=self.tolerance,
            xtol=self.tolerance,
        )
        return FitResult(result.x, bool(result.success), str(result.message))

    def _root(self, expression, data, measure, initial):
        if isinstance(measure, ScalarMeasure):
            raise TypeError("SciPy root methods require a residual fitting measure")

        def value(constants):
            residuals = np.asarray(measure(expression, data, constants), dtype=float)
            if residuals.ndim != 1:
                raise ValueError("residual fitting measures must return a one-dimensional vector")
            return residuals

        jacobian = getattr(measure, "jacobian", None)
        if not np.all(np.isfinite(value(initial))):
            return FitResult(initial, False, "non-finite residuals at initial constants")
        result = optimize.root(
            value,
            initial,
            jac=(
                lambda constants: np.asarray(
                    jacobian(expression, data, constants), dtype=float
                )
            )
            if jacobian
            else None,
            method=self.method,
            tol=self.tolerance,
        )
        return FitResult(result.x, bool(result.success), str(result.message))

    def _minimize(self, expression, data, measure, initial):
        if isinstance(measure, ResidualMeasure):
            raise TypeError("SciPy minimize methods require a scalar fitting measure")

        def value(constants):
            result = np.asarray(measure(expression, data, constants), dtype=float)
            if result.ndim != 0:
                raise ValueError("scalar fitting measures must return a scalar")
            return float(result)

        gradient = getattr(measure, "gradient", None)
        hessian = getattr(measure, "hessian", None)
        result = optimize.minimize(
            value,
            initial,
            method=self.method,
            jac=(
                lambda constants: np.asarray(
                    gradient(expression, data, constants), dtype=float
                )
            )
            if gradient
            else None,
            hess=(
                lambda constants: np.asarray(
                    hessian(expression, data, constants), dtype=float
                )
            )
            if hessian
            else None,
            tol=self.tolerance,
        )
        return FitResult(result.x, bool(result.success), str(result.message))
