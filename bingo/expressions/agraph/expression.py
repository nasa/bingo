"""Acyclic graph expression — standalone pure-Python implementation.

``AGraphExpression`` represents a mathematical expression as an acyclic
graph encoded in a uint8 command stack.  It provides evaluation,
automatic differentiation, scikit-learn-style ``fit``/``predict``/``score``,
and multiple output formats (sympy, LaTeX, ONNX).
"""

import copy
import warnings

import numpy as np
import scipy.optimize
from sympy import sympify

from .operators import CONSTANT
from . import evaluation as evaluation_backend
from . import simplification as simplification_backend
from .formatting import get_formatted_string
from .parsing import eq_string_to_command_array_and_constants


# ---------- scoring metrics ----------


def _mean_absolute_error(fitness_vector):
    return np.mean(np.abs(fitness_vector))


def _mean_squared_error(fitness_vector):
    return np.mean(fitness_vector**2)


def _root_mean_squared_error(fitness_vector):
    return np.sqrt(np.mean(fitness_vector**2))


def _bic(fitness_vector, n_constants):
    """Bayesian Information Criterion.

    BIC = k * ln(n) - 2 * ln(L̂)

    where *k* = ``n_constants + 1`` (the extra 1 accounts for the noise
    variance σ as a free parameter), *n* is the number of data points,
    and *L̂* is the maximised Gaussian log-likelihood evaluated at the
    MLE noise variance.
    """
    n = len(fitness_vector)
    k = n_constants + 1
    mse = np.mean(fitness_vector**2)
    if mse <= 0:
        mse = np.finfo(float).tiny
    log_likelihood = -n / 2 * np.log(mse) - n / 2 - n / 2 * np.log(2 * np.pi)
    return k * np.log(n) - 2 * log_likelihood


def _laplace_nmll(fitness_vector, n_constants):
    """Negative normalised marginal log-likelihood (Laplace approximation).

    NMLL = - (1 - b) * ln(L̂) - ln(b) / 2 * k

    where *b* = 1 / sqrt(n) is a normalisation factor, *k* =
    ``n_constants + 1``, *n* is the number of data points, and *L̂* is
    the maximised Gaussian log-likelihood.

    Parameters
    ----------
    fitness_vector : array, shape (n,)
        Residual vector.
    n_constants : int
        Number of optimisable constants in the expression.

    Returns
    -------
    float
    """
    n = len(fitness_vector)
    k = n_constants + 1
    b = 1 / np.sqrt(n)
    mse = np.mean(fitness_vector**2)
    if mse <= 0:
        mse = np.finfo(float).tiny
    log_like = -n / 2 * np.log(mse) - n / 2 - n / 2 * np.log(2 * np.pi)
    nmll_laplace = (1 - b) * log_like + np.log(b) / 2 * k
    return nmll_laplace


_METRIC_MAP = {
    "mae": _mean_absolute_error,
    "mse": _mean_squared_error,
    "rmse": _root_mean_squared_error,
}


class AGraphExpression:
    """Acyclic graph expression.

    Parameters
    ----------
    equation : str or sympy.Expr, optional
        An equation to initialise the expression from.
    """

    def __init__(self, *, equation=None):
        self._hash = None

        if equation is not None:
            cmd, consts, ints = eq_string_to_command_array_and_constants(str(equation))
            self._command_array = cmd
            self._simplified_command_array = np.empty([0, 3], dtype=np.uint8)
            self._simplified_constants = consts
            self._integers = ints
            self._is_fitted = len(consts) == 0
            self._modified = True
        else:
            self._command_array = np.empty([0, 3], dtype=np.uint8)
            self._simplified_command_array = np.empty([0, 3], dtype=np.uint8)
            self._simplified_constants = ()
            self._integers = ()
            self._is_fitted = True
            self._modified = False

    # ------------------------------------------------------------------ #
    #  Properties                                                         #
    # ------------------------------------------------------------------ #

    @property
    def command_array(self):
        """Nx3 uint8 array: the command stack (read-only view)."""
        self._command_array.flags.writeable = False
        return self._command_array

    @command_array.setter
    def command_array(self, value):
        self._command_array = np.asarray(value, dtype=np.uint8)
        self._notify_modification()

    @property
    def mutable_command_array(self):
        """Nx3 uint8 array: writable command stack.

        Accessing this property marks the expression as modified.
        """
        self._command_array.flags.writeable = True
        self._notify_modification()
        return self._command_array

    @property
    def constants(self):
        """Numeric constants used in the (simplified) equation."""
        if self._modified:
            self._update()
        return self._simplified_constants

    @constants.setter
    def constants(self, value):
        """Set the constants for the simplified equation."""
        if self._modified:
            self._update()
        self._simplified_constants = tuple(float(v) for v in value)

    @property
    def integers(self):
        """Integer values used in the equation."""
        return self._integers

    @property
    def complexity(self):
        """Number of utilized commands in the simplified stack."""
        if self._modified:
            self._update()
        return self._simplified_command_array.shape[0]

    # ------------------------------------------------------------------ #
    #  Format properties                                                  #
    # ------------------------------------------------------------------ #

    @property
    def console(self):
        """Human-readable console string."""
        return self._format("console")

    @property
    def sympy(self):
        """Sympy expression representation."""
        return sympify(self._format("sympy"))

    @property
    def latex(self):
        """LaTeX string representation."""
        return self._format("latex")

    @property
    def onnx(self):
        """ONNX model representation."""
        from .onnx_interface import make_onnx_model

        if self._modified:
            self._update()
        return make_onnx_model(
            self._simplified_command_array,
            self._simplified_constants,
            self._integers,
        )

    def __str__(self):
        """Human-readable console string."""
        return self._format("console")

    def _format(self, fmt):
        if self._modified:
            self._update()
        return get_formatted_string(
            fmt,
            self._simplified_command_array,
            self._simplified_constants,
            self._integers,
        )

    # ------------------------------------------------------------------ #
    #  Evaluation                                                         #
    # ------------------------------------------------------------------ #

    def _evaluate(self, x):
        """Evaluate the expression at *x*.

        Parameters
        ----------
        x : MxD array of numeric
            Input data.

        Returns
        -------
        Mx1 array of numeric
            f(x)
        """
        if self._modified:
            self._update()
        try:
            return evaluation_backend.evaluate(
                self._simplified_command_array,
                x,
                self._simplified_constants,
                self._integers,
            )
        except (ArithmeticError, OverflowError, ValueError, FloatingPointError) as err:
            warnings.warn(f"{err} in expression evaluation")
            return np.full((x.shape[0], 1), np.nan)

    def _evaluate_with_x_gradient(self, x):
        """Evaluate the expression and its gradient w.r.t. *x*.

        Returns
        -------
        tuple of (Mx1 array, MxD array)
            (f(x), df/dx)
        """
        if self._modified:
            self._update()
        try:
            return evaluation_backend.evaluate_with_derivative(
                self._simplified_command_array,
                x,
                self._simplified_constants,
                self._integers,
                True,
            )
        except (ArithmeticError, OverflowError, ValueError, FloatingPointError) as err:
            warnings.warn(f"{err} in expression gradient evaluation")
            nan = np.full(x.shape, np.nan)
            return nan, nan.copy()

    def _evaluate_with_const_gradient(self, x):
        """Evaluate the expression and its gradient w.r.t. constants.

        Returns
        -------
        tuple of (Mx1 array, MxL array)
            (f(x), df/dc)  where L = len(constants)
        """
        if self._modified:
            self._update()
        try:
            return evaluation_backend.evaluate_with_derivative(
                self._simplified_command_array,
                x,
                self._simplified_constants,
                self._integers,
                False,
            )
        except (ArithmeticError, OverflowError, ValueError, FloatingPointError) as err:
            warnings.warn(f"{err} in const gradient evaluation")
            nc = len(self._simplified_constants)
            nan = np.full((x.shape[0], nc), np.nan)
            return nan, nan.copy()

    # ------------------------------------------------------------------ #
    #  sklearn-like interface                                             #
    # ------------------------------------------------------------------ #

    def predict(self, X):
        """Predict target values for *X*.

        Parameters
        ----------
        X : array-like, shape (M, D)
            Input data.

        Returns
        -------
        numpy array, shape (M,)
            Predictions.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return self._evaluate(X).ravel()

    def fit(self, X, y, optimizer="lm", metric="mse", **scipykwargs):
        """Optimize constants to fit the data.

        Uses ``scipy.optimize.least_squares`` (Levenberg-Marquardt by
        default) to minimize residuals.

        Parameters
        ----------
        X : array-like, shape (M, D)
        y : array-like, shape (M,)
        optimizer : str, optional
            Method passed to ``scipy.optimize.least_squares``.
            Default ``"lm"`` (Levenberg-Marquardt).
        metric : str, optional
            Unused for now (least-squares always minimises SSE).
            Retained for API symmetry.
        **scipykwargs : keyword arguments
            Additional keyword arguments passed to ``scipy.optimize.least_squares``.

        Returns
        -------
        self
        """
        if self._modified:
            self._update()

        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        self._is_fitted = True

        if len(self.constants) == 0:
            return self

        x0 = np.array(self.constants, dtype=float)

        def residuals(params):
            self.constants = params
            return self._evaluate(X).ravel() - y

        def jacobian(params):
            self.constants = params
            _, jac = self._evaluate_with_const_gradient(X)
            return jac

        try:
            result = scipy.optimize.least_squares(
                residuals, x0, jac=jacobian, method=optimizer, **scipykwargs
            )
            self.constants = result.x
        except Exception:  # noqa: broad-except — don't crash on bad fits
            pass

        return self

    def score(self, X, y, metric="mse"):
        """Score the expression on data.

        Parameters
        ----------
        X : array-like, shape (M, D)
        y : array-like, shape (M,)
        metric : str, optional
            One of ``"mae"``, ``"mse"``, ``"rmse"``, ``"bic"``,
            ``"laplace_nmll"``. Default ``"mse"``.

        Returns
        -------
        float
            The score (lower is better for all supported metrics).
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        residuals = self.predict(X) - y

        if metric == "bic":
            return float(_bic(residuals, len(self.constants)))

        if metric == "laplace_nmll":
            return float(_laplace_nmll(residuals, len(self.constants)))

        metric_fn = _METRIC_MAP.get(metric, _mean_squared_error)
        return float(metric_fn(residuals))

    def __sklearn_is_fitted__(self):
        """Whether the expression has been fitted (constants optimized).

        Used by scikit-learn's ``check_is_fitted``.
        """
        if self._modified:
            self._update()
        return self._is_fitted

    # ------------------------------------------------------------------ #
    #  Simplification / utility                                           #
    # ------------------------------------------------------------------ #

    def get_utilized_commands(self):
        """Find which raw commands are utilized by the output.

        Returns
        -------
        bytearray
        """
        return simplification_backend.get_utilized_commands(self._command_array)

    def distance(self, other):
        """Element-wise distance between two command arrays.

        Parameters
        ----------
        other : AGraphExpression

        Returns
        -------
        int
        """
        return int(np.sum(self.command_array != other.command_array))

    def copy(self):
        """Deep copy of the expression."""
        return copy.deepcopy(self)

    # ------------------------------------------------------------------ #
    #  Internal update logic                                              #
    # ------------------------------------------------------------------ #

    def _notify_modification(self):
        self._modified = True
        self._hash = None

    def _update(self):
        """Reduce the stack and renumber constants/integers."""
        self._simplified_command_array = simplification_backend.reduce_stack(
            self._command_array
        )

        # Renumber constants sequentially
        const_mask = self._simplified_command_array[:, 0] == CONSTANT
        num_const = int(np.count_nonzero(const_mask))
        self._simplified_command_array[const_mask, 1] = np.arange(num_const)
        self._simplified_command_array[const_mask, 2] = np.arange(num_const)

        # Integer indices are not renumbered — they point directly into
        # self._integers which is fixed from the unsimplified source.

        # Manage constant values
        old = self._simplified_constants
        if num_const <= len(old):
            self._simplified_constants = old[:num_const]
        else:
            self._simplified_constants = old + (1.0,) * (num_const - len(old))

        self._is_fitted = num_const == 0
        self._modified = False

    # ------------------------------------------------------------------ #
    #  Hash / equality                                                    #
    # ------------------------------------------------------------------ #

    def __hash__(self):
        if self._modified or self._hash is None:
            self._update()
            self._hash = hash(tuple(map(tuple, self._simplified_command_array)))
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, AGraphExpression):
            return NotImplemented
        return hash(self) == hash(other)

    # ------------------------------------------------------------------ #
    #  Serialization                                                      #
    # ------------------------------------------------------------------ #

    def __getstate__(self):
        state = self.__dict__.copy()
        # command_array is already uint8 — store directly
        state.pop("_hash", None)
        # Don't store derived simplified arrays
        del state["_simplified_command_array"]
        return state

    def __setstate__(self, state):
        state["_simplified_command_array"] = np.empty([0, 3], dtype=np.uint8)
        state["_modified"] = True
        state["_hash"] = None
        self.__dict__.update(state)

    def __deepcopy__(self, memodict=None):
        new = AGraphExpression.__new__(AGraphExpression)
        new._command_array = np.copy(self._command_array)
        new._simplified_command_array = np.copy(self._simplified_command_array)
        new._simplified_constants = tuple(self._simplified_constants)
        new._integers = tuple(self._integers)
        new._is_fitted = self._is_fitted
        new._modified = self._modified
        new._hash = None
        return new
