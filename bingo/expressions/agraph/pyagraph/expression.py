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

from .evaluation import (
    evaluate,
    evaluate_with_const_hessian,
    evaluate_with_derivative,
)
from .evaluation.cached_evaluation import CachedEvaluator
from .operators import (
    VARIABLE,
    IS_TERMINAL_ARRAY,
    IS_ARITY_2_ARRAY,
)
from .simplification import get_utilized_commands, reduce, simplify as cas_simplify
from .formatting import get_formatted_string
from .parsing import eq_string_to_command_array_and_constants


# ---------- score / loss vocabulary ----------
#
# Score is a higher-is-better measure of Expression quality; loss is a
# lower-is-better objective used by fitness evaluation.  Non-finite public
# evaluation normalizes to negative-infinite score and infinite loss.

_POS_INF = float("inf")
_NEG_INF = float("-inf")


def _mean_absolute_error(residuals):
    return np.mean(np.abs(residuals))


def _mean_squared_error(residuals):
    return np.mean(residuals**2)


def _root_mean_squared_error(residuals):
    return np.sqrt(np.mean(residuals**2))


def _relative_mse(residuals, y):
    """Squared residuals normalized pointwise by the target value.

    Rejects zero-valued targets, which would make the normalization
    undefined.
    """
    y = np.asarray(y, dtype=float).ravel()
    if np.any(y == 0):
        raise ValueError("relative_mse rejects zero-valued targets")
    return np.mean((residuals / y) ** 2)


def _correlation_loss(predictions, y):
    """Loss derived from the Pearson correlation between predictions and y.

    Returns ``1 - r**2`` so that perfectly associated data (``|r| = 1``)
    yields ``0`` and uncorrelated data yields ``1``.  It measures
    association without applying any output transformation to the
    Expression.
    """
    predictions = np.asarray(predictions, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if np.std(predictions) == 0 or np.std(y) == 0:
        return 1.0
    r = np.corrcoef(predictions, y)[0, 1]
    return 1.0 - r**2


def _r2_score(predictions, y):
    """Coefficient of determination R^2 (higher is better)."""
    predictions = np.asarray(predictions, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def _laplace_nmll(residuals, n_constants):
    """Normalised marginal log-likelihood (Laplace approximation).

    NMLL = (1 - b) * ln(L̂) + ln(b) / 2 * k

    where *b* = 1 / sqrt(n) is a normalisation factor, *k* =
    ``n_constants + 1``, *n* is the number of data points, and *L̂* is
    the maximised Gaussian log-likelihood.  Higher values indicate a
    better fit; this is a score (its negation is the corresponding loss).

    Parameters
    ----------
    residuals : array, shape (n,)
        Residual vector.
    n_constants : int
        Number of optimisable constants in the expression.

    Returns
    -------
    float
    """
    n = len(residuals)
    k = n_constants + 1
    b = 1 / np.sqrt(n)
    mse = np.mean(residuals**2)
    if mse <= 0:
        mse = np.finfo(float).tiny
    log_like = -n / 2 * np.log(mse) - n / 2 - n / 2 * np.log(2 * np.pi)
    return (1 - b) * log_like + np.log(b) / 2 * k


# Loss kinds that depend only on the residual vector share one dispatch map;
# ``relative_mse``, ``correlation``, and ``laplace_nmll`` need extra arguments
# and are handled separately.
_SIMPLE_LOSS_METRICS = {
    "mse": _mean_squared_error,
    "mae": _mean_absolute_error,
    "rmse": _root_mean_squared_error,
}

# Loss kinds are lower-is-better; score kinds are higher-is-better.
_VALID_LOSS_KINDS = frozenset(
    set(_SIMPLE_LOSS_METRICS) | {"relative_mse", "correlation", "laplace_nmll"}
)
_VALID_SCORE_KINDS = frozenset({"r2", "laplace_nmll"})


class AGraphExpression:
    """Acyclic graph expression.

    Parameters
    ----------
    equation : str or sympy.Expr, optional
        An equation to initialise the expression from.
    simplification : {"reduce", "cas"}, optional
        Which simplification strategy to use when deriving the
        evaluation-facing stack from the raw stack.  ``"reduce"``
        performs cheap dead-code elimination / constant
        folding on the stack.  ``"cas"`` (default) runs the full computer
        algebra simplification pipeline.
    propagate_constants : bool, optional
        Whether setting simplified constants also updates raw constants.
        Default ``False``.

    Notes
    -----
    **Fitted lifecycle.**  An Expression is *fitted* when it has no optimizable
    constants, or when an applicable fitting method has been attempted for its
    current raw structure (see :attr:`is_fitted`).  The lifecycle is
    structure-only:

    - A raw structural change (setting the raw command array, raw constants, or
      raw integers, or mutating the raw command array in place) unsets it.
    - A fitting attempt (:meth:`fit` or :meth:`fit_implicit`) establishes it,
      even when the solver does not numerically converge.
    - Direct simplified-constant assignment preserves the current state but
      cannot establish it.
    - Copying and serialization preserve it exactly.
    """

    _VALID_SIMPLIFICATIONS = frozenset({"reduce", "cas"})

    def __init__(
        self, equation=None, *, simplification="cas", propagate_constants=False
    ):
        if simplification not in self._VALID_SIMPLIFICATIONS:
            raise ValueError(
                f"simplification must be one of "
                f"{self._VALID_SIMPLIFICATIONS!r}, got {simplification!r}"
            )
        self._simplification = simplification
        self._propagate_constants = propagate_constants
        self._constant_mapping = ()
        self._hash = None

        # ``_fit_attempted`` records whether an applicable fitting method has
        # been run for the *current raw structure*.  Combined with the
        # optimizable-constant count it defines the structure-only ``is_fitted``
        # lifecycle: a raw structural change clears it, a fitting attempt
        # establishes it, and direct constant assignment leaves it untouched.
        self._fit_attempted = False

        if equation is not None:
            cmd, consts, ints = eq_string_to_command_array_and_constants(str(equation))
            self._raw_command_array = cmd
            self._raw_constants = consts
            self._raw_integers = ints
            self._command_array = np.empty([0, 3], dtype=np.uint8)
            self._constants = ()
            self._integers = ()
            self._modified = True
        else:
            self._raw_command_array = np.empty([0, 3], dtype=np.uint8)
            self._raw_constants = ()
            self._raw_integers = ()
            self._command_array = np.empty([0, 3], dtype=np.uint8)
            self._constants = ()
            self._integers = ()
            self._modified = False

    # ------------------------------------------------------------------ #
    #  Properties                                                         #
    # ------------------------------------------------------------------ #

    # -- Raw (GA-facing) layer ------------------------------------------- #

    @property
    def raw_command_array(self):
        """Nx3 uint8 array: GA-facing command stack (read-only view)."""
        self._raw_command_array.flags.writeable = False
        return self._raw_command_array

    @raw_command_array.setter
    def raw_command_array(self, value):
        self._raw_command_array = np.asarray(value, dtype=np.uint8)
        self._notify_modification()

    @property
    def mutable_raw_command_array(self):
        """Nx3 uint8 array: writable GA-facing command stack.

        Accessing this property marks the expression as modified.
        """
        self._raw_command_array.flags.writeable = True
        self._notify_modification()
        return self._raw_command_array

    @property
    def raw_constants(self):
        """Constant values in the GA-facing (raw) equation."""
        return self._raw_constants

    @raw_constants.setter
    def raw_constants(self, value):
        self._raw_constants = tuple(float(v) for v in value)
        self._notify_modification()

    @property
    def raw_integers(self):
        """Integer values in the GA-facing (raw) equation."""
        return self._raw_integers

    @raw_integers.setter
    def raw_integers(self, value):
        self._raw_integers = tuple(int(v) for v in value)
        self._notify_modification()

    # -- Simplified (evaluation-facing) layer ----------------------------- #

    @property
    def command_array(self):
        """Nx3 uint8 array: simplified command stack (read-only, derived)."""
        if self._modified:
            self._update()
        self._command_array.flags.writeable = False
        return self._command_array

    @property
    def constants(self):
        """Numeric constants used in the simplified equation."""
        if self._modified:
            self._update()
        return self._constants

    @constants.setter
    def constants(self, value):
        """Set constants in the simplified equation only.

        Used by :meth:`fit` — does **not** touch :attr:`raw_constants`
        unless :attr:`propagate_constants` is enabled.
        The provided values are used directly for evaluation without
        triggering re-simplification.
        """
        if self._modified:
            self._update()
        self._constants = tuple(float(v) for v in value)
        if self._propagate_constants and self._constant_mapping:
            raw = list(self._raw_constants)
            for simp_idx, raw_idx in enumerate(self._constant_mapping):
                if simp_idx < len(self._constants) and raw_idx < len(raw):
                    raw[raw_idx] = self._constants[simp_idx]
            self._raw_constants = tuple(raw)

    @property
    def integers(self):
        """Integer values used in the simplified equation."""
        if self._modified:
            self._update()
        return self._integers

    @property
    def constant_mapping(self):
        """Index mapping from simplified constants to raw constants.

        Returns a tuple where ``constant_mapping[simplified_idx]`` is
        the corresponding index into :attr:`raw_constants`.
        """
        if self._modified:
            self._update()
        return self._constant_mapping

    @property
    def propagate_constants(self):
        """Whether setting simplified constants also updates raw constants."""
        return self._propagate_constants

    @propagate_constants.setter
    def propagate_constants(self, value):
        self._propagate_constants = bool(value)

    @property
    def complexity(self):
        """Number of utilized commands in the simplified stack."""
        if self._modified:
            self._update()
        return self._command_array.shape[0]

    @property
    def tree_complexity(self):
        """Tree-based node count (counts shared sub-expressions multiple times)."""
        if self._modified:
            self._update()
        command_array = self._command_array
        if command_array.shape[0] == 0:
            return 0
        is_arity_2 = IS_ARITY_2_ARRAY
        is_terminal = IS_TERMINAL_ARRAY
        count = 0
        stack = [command_array.shape[0] - 1]
        while stack:
            row = command_array[stack.pop()]
            count += 1
            if not is_terminal[row[0]]:
                stack.append(int(row[1]))
                if is_arity_2[row[0]]:
                    stack.append(int(row[2]))
        return count

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
            self._command_array,
            self._constants,
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
            self._command_array,
            self._constants,
            self._integers,
        )

    # ------------------------------------------------------------------ #
    #  Evaluation                                                         #
    # ------------------------------------------------------------------ #

    def _evaluate(self, x, constants=None, output_dimension=1):
        """Evaluate the expression at *x*.

        Parameters
        ----------
        x : MxD array of numeric
            Input data.
        constants : tuple of numeric or numpy arrays, optional
            Numeric constants used for this evaluation. If omitted, the
            expression's stored simplified constants are used.
        output_dimension : int, optional
            Number of output columns to return if evaluation fails. Default 1.

        Returns
        -------
        Mx1 or MxB array of numeric
            f(x)
        """
        if self._modified:
            self._update()
        if constants is None:
            constants = self._constants
        try:
            return evaluate(
                self._command_array,
                x,
                constants,
                self._integers,
            )
        except (ArithmeticError, OverflowError, ValueError, FloatingPointError) as err:
            warnings.warn(f"{err} in expression evaluation")
            return np.full((x.shape[0], output_dimension), np.nan)

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
            return evaluate_with_derivative(
                self._command_array,
                x,
                self._constants,
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
            return evaluate_with_derivative(
                self._command_array,
                x,
                self._constants,
                self._integers,
                False,
            )
        except (ArithmeticError, OverflowError, ValueError, FloatingPointError) as err:
            warnings.warn(f"{err} in const gradient evaluation")
            nc = len(self._constants)
            nan = np.full((x.shape[0], nc), np.nan)
            return nan, nan.copy()

    def _evaluate_with_const_hessian(self, x):
        """Evaluate the expression with constant gradient and Hessian.

        Returns
        -------
        tuple of (Mx1 array, MxL array, MxLxL array)
            ``(f(x), df/dc, d2f/dc2)`` where ``L = len(constants)``.
        """
        if self._modified:
            self._update()
        try:
            return evaluate_with_const_hessian(
                self._command_array,
                x,
                self._constants,
                self._integers,
            )
        except (ArithmeticError, OverflowError, ValueError, FloatingPointError) as err:
            warnings.warn(f"{err} in const hessian evaluation")
            num_samples = x.shape[0]
            num_constants = len(self._constants)
            nan_value = np.full((num_samples, 1), np.nan)
            nan_gradient = np.full((num_samples, num_constants), np.nan)
            nan_hessian = np.full(
                (num_samples, num_constants, num_constants), np.nan
            )
            return nan_value, nan_gradient, nan_hessian

    # ------------------------------------------------------------------ #
    #  sklearn-like interface                                             #
    # ------------------------------------------------------------------ #

    def predict(self, X, *, constants=None):
        """Predict target values for *X*.

        Parameters
        ----------
        X : array-like, shape (M, D)
            Input data.
        constants : array-like, shape (L,) or (L, B), optional
            Temporary simplified constant values, where ``L`` is the number of
            constants in the expression. A one-dimensional array produces the
            usual one-dimensional prediction. A constant-major two-dimensional
            array evaluates ``B`` constant sets in one vectorized prediction and
            produces one prediction column per set. This does not modify the expression's
            stored constants. C-contiguous ``float64`` arrays provide the
            zero-copy input path in the C++ backend.

        Returns
        -------
        numpy array, shape (M,) or (M, B)
            Predictions. The output is two-dimensional only when ``constants``
            is two-dimensional.

        Raises
        ------
        ValueError
            If ``constants`` is not one- or two-dimensional, or does not have
            one row per simplified expression constant.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if constants is None:
            return self._evaluate(X).ravel()

        if self._modified:
            self._update()
        constants = np.asarray(constants, dtype=float)
        n_constants = len(self._constants)
        if constants.ndim not in (1, 2):
            raise ValueError("constants must be a one- or two-dimensional array")
        if constants.shape[0] != n_constants:
            raise ValueError(
                "constants must have one entry per simplified expression constant"
            )
        if constants.ndim == 1:
            return self._evaluate(X, tuple(constants)).ravel()

        predictions = self._evaluate(
            X, tuple(constants), output_dimension=constants.shape[1]
        )
        if n_constants == 0:
            return np.repeat(predictions, constants.shape[1], axis=1)
        return predictions

    def gradient(self, X):
        """Predictions and the gradient of the output with respect to inputs.

        This is the derivative surface used by implicit regression.

        Parameters
        ----------
        X : array-like, shape (M, D)
            Input data.

        Returns
        -------
        tuple of (numpy array shape (M,), numpy array shape (M, D))
            ``(f(x), df/dx)``.  A non-finite evaluation propagates as
            ``nan`` entries rather than raising.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        f_of_x, df_dx = self._evaluate_with_x_gradient(X)
        return f_of_x.ravel(), df_dx

    def fit(self, X, y, *, tolerance=1e-5):
        """Fit constants to explicit-regression data.

        Runs Levenberg-Marquardt (``scipy.optimize.root(method='lm')``)
        with analytic Jacobians from the :class:`CachedEvaluator`, always
        minimizing the ordinary residual vector ``f(x) - y`` regardless of
        the loss later used to rank the Expression.

        Attempting the fit establishes the fitted state for the current raw
        structure even when the solver does not numerically converge.

        Parameters
        ----------
        X : array-like, shape (M, D)
        y : array-like, shape (M,)
        tolerance : float, keyword-only, optional
            Solver convergence tolerance.  Default ``1e-5``.

        Returns
        -------
        self
        """
        if self._modified:
            self._update()

        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.size:
            raise ValueError("X and y must have the same number of samples")
        if len(self.constants) == 0:
            return self

        x0 = np.array(self.constants, dtype=float)
        cached = CachedEvaluator(self._command_array, X, self._integers)

        def residuals(params):
            return cached.forward_eval(tuple(params)).ravel() - y

        def jacobian(params):
            _, jac = cached.forward_eval_with_const_derivative(tuple(params))
            return jac

        try:
            result = scipy.optimize.root(
                residuals,
                x0,
                jac=jacobian,
                method="lm",
                tol=tolerance,
            )
            self.commit_fit(result.x)
        except Exception:  # noqa: broad-except — don't crash on bad fits
            pass

        return self

    def fit_implicit(self, X, dx_dt, *, tolerance=1e-5):
        """Fit constants to implicit-regression data.

        Runs SciPy least squares to minimize the ordinary implicit residual
        vector (the per-sample normalized alignment between the Expression's
        input gradient and the observed trajectory derivatives ``dx_dt``).

        Attempting the fit establishes the fitted state for the current raw
        structure even when the solver does not numerically converge.

        Parameters
        ----------
        X : array-like, shape (M, D)
        dx_dt : array-like, shape (M, D)
            Observed trajectory derivatives aligned with *X*.
        tolerance : float, keyword-only, optional
            Solver convergence tolerance.  Default ``1e-5``.

        Returns
        -------
        self
        """
        if self._modified:
            self._update()

        X = np.atleast_2d(np.asarray(X, dtype=float))
        dx_dt = np.atleast_2d(np.asarray(dx_dt, dtype=float))
        if X.shape != dx_dt.shape:
            raise ValueError("X and dx_dt must have the same shape")
        if len(self.constants) == 0:
            return self

        x0 = np.array(self.constants, dtype=float)
        original_constants = self.constants

        def residuals(params):
            self.constants = params
            return self._implicit_residual_vector(X, dx_dt)

        try:
            result = scipy.optimize.least_squares(
                residuals,
                x0,
                ftol=tolerance,
                xtol=tolerance,
            )
            self.constants = original_constants
            self.commit_fit(result.x)
        except Exception:  # noqa: broad-except — don't crash on bad fits
            self.constants = original_constants

        return self

    def _explicit_predictions(self, X, y):
        """Coerce inputs and predict; return ``(predictions, y)``.

        ``predictions`` is ``None`` when evaluation is non-finite, signalling
        the caller to normalize to an infinite loss / score.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.size:
            raise ValueError("X and y must have the same number of samples")
        predictions = self.predict(X)
        if not np.all(np.isfinite(predictions)):
            return None, y
        return predictions, y

    def loss(self, X, y, *, kind="mse"):
        """Lower-is-better explicit-regression loss.

        Parameters
        ----------
        X : array-like, shape (M, D)
        y : array-like, shape (M,)
        kind : str, keyword-only, optional
            One of ``"mse"`` (default), ``"mae"``, ``"rmse"``,
            ``"relative_mse"``, ``"correlation"``, ``"laplace_nmll"``.

        Returns
        -------
        float
            The loss.  Non-finite expression evaluation returns
            positive infinity.
        """
        if kind not in _VALID_LOSS_KINDS:
            raise ValueError(
                f"kind must be one of {sorted(_VALID_LOSS_KINDS)!r}, got {kind!r}"
            )
        predictions, y = self._explicit_predictions(X, y)
        if predictions is None:
            return _POS_INF

        residuals = predictions - y
        if kind in _SIMPLE_LOSS_METRICS:
            value = _SIMPLE_LOSS_METRICS[kind](residuals)
        elif kind == "laplace_nmll":
            # Loss is the negation of the corresponding (higher-is-better) score.
            value = -_laplace_nmll(residuals, len(self.constants))
        elif kind == "relative_mse":
            value = _relative_mse(residuals, y)
        else:  # "correlation"
            value = _correlation_loss(predictions, y)

        return float(value) if np.isfinite(value) else _POS_INF

    def score(self, X, y, *, kind="r2"):
        """Higher-is-better explicit-regression score.

        Parameters
        ----------
        X : array-like, shape (M, D)
        y : array-like, shape (M,)
        kind : str, keyword-only, optional
            One of ``"r2"`` (default) or ``"laplace_nmll"``.

        Returns
        -------
        float
            The score.  Non-finite expression evaluation returns
            negative infinity.
        """
        if kind not in _VALID_SCORE_KINDS:
            raise ValueError(
                f"kind must be one of {sorted(_VALID_SCORE_KINDS)!r}, got {kind!r}"
            )
        predictions, y = self._explicit_predictions(X, y)
        if predictions is None:
            return _NEG_INF

        if kind == "laplace_nmll":
            value = _laplace_nmll(predictions - y, len(self.constants))
        else:  # "r2"
            value = _r2_score(predictions, y)

        return float(value) if np.isfinite(value) else _NEG_INF

    def implicit_loss(self, X, dx_dt, *, required_params=None):
        """Lower-is-better implicit-regression loss.

        The loss aggregates the per-sample alignment between the
        Expression's input gradient and the observed trajectory
        derivatives ``dx_dt``.

        Parameters
        ----------
        X : array-like, shape (M, D)
        dx_dt : array-like, shape (M, D)
            Observed trajectory derivatives aligned with *X*.
        required_params : int, keyword-only, optional
            Anti-triviality guard.  When provided, at least one sample must
            have at least this many derivative components with magnitude
            greater than ``1e-16``; otherwise the loss is positive infinity.

        Returns
        -------
        float
            The loss.  Non-finite expression evaluation and guard failure
            both return positive infinity.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        dx_dt = np.atleast_2d(np.asarray(dx_dt, dtype=float))
        if X.shape != dx_dt.shape:
            raise ValueError("X and dx_dt must have the same shape")
        residual = self._implicit_residual_vector(
            X, dx_dt, required_params=required_params
        )
        if not np.all(np.isfinite(residual)):
            return _POS_INF
        value = _mean_absolute_error(residual)
        return float(value) if np.isfinite(value) else _POS_INF

    def implicit_score(self, X, dx_dt, *, required_params=None):
        """Higher-is-better implicit-regression score.

        The score is the negation of :meth:`implicit_loss`.

        Parameters
        ----------
        X : array-like, shape (M, D)
        dx_dt : array-like, shape (M, D)
        required_params : int, keyword-only, optional
            See :meth:`implicit_loss`.

        Returns
        -------
        float
            The score.  Non-finite expression evaluation and guard failure
            both return negative infinity.
        """
        loss = self.implicit_loss(X, dx_dt, required_params=required_params)
        return _NEG_INF if loss == _POS_INF else -loss

    def _implicit_residual_vector(self, X, dx_dt, required_params=None):
        """Per-sample implicit residual (normalized gradient alignment).

        Returns an ``(M,)`` vector whose entries are the signed, normalized
        alignment of the Expression's input gradient with ``dx_dt`` at each
        sample.  Non-finite normalizations become ``inf``.  When
        *required_params* is set and the anti-triviality guard fails, every
        entry is ``inf``.
        """
        _, df_dx = self._evaluate_with_x_gradient(X)
        dot_product = df_dx * dx_dt

        if required_params is not None:
            n_params_used = (np.abs(dot_product) > 1e-16).sum(axis=1)
            if not np.any(n_params_used >= required_params):
                return np.full((X.shape[0],), _POS_INF)

        denominator = np.sum(np.abs(dot_product), axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            residual = np.sum(dot_product, axis=1) / denominator
        residual[~np.isfinite(denominator)] = _POS_INF
        return residual

    @property
    def is_fitted(self):
        """Whether the Expression is fitted (structure-only lifecycle).

        ``True`` when the Expression has no optimizable constants, or when a
        fitting method has been attempted for its current raw structure.
        A raw structural change unsets it; direct simplified-constant
        assignment preserves but cannot establish it.
        """
        if self._modified:
            self._update()
        return self._fit_attempted or len(self._constants) == 0

    def __sklearn_is_fitted__(self):
        """Whether the expression has been fitted.

        Used by scikit-learn's ``check_is_fitted``; delegates to
        :attr:`is_fitted`.
        """
        return self.is_fitted

    def _set_fit_attempted(self, value):
        """Restore the fitted lifecycle after constructing from raw state."""
        self._fit_attempted = bool(value)

    def commit_fit(self, constants):
        """Atomically install validated fitted constants for this structure."""
        if self._modified:
            self._update()
        constants = tuple(float(value) for value in constants)
        if len(constants) != len(self._constants):
            raise ValueError(
                "constants must have one entry per simplified expression constant"
            )
        if not np.all(np.isfinite(constants)):
            raise ValueError("fitted constants must be finite")
        self.constants = constants
        self._fit_attempted = True
        return self

    def clear_fit(self):
        """Clear fittedness without changing constants or raw structure."""
        if self._modified:
            self._update()
        if self._constants:
            self._fit_attempted = False
        return self

    # ------------------------------------------------------------------ #
    #  Simplification / utility                                           #
    # ------------------------------------------------------------------ #

    def get_utilized_commands(self):
        """Find which raw commands are utilized by the output.

        Returns
        -------
        bytearray
        """
        return get_utilized_commands(self._raw_command_array)

    def promote_simplification(self):
        """Replace the raw stack with its simplified form.

        After simplification the raw values equal the simplified values —
        unused rows are removed, references are remapped, and
        constants/integers are renumbered to match.

        Returns
        -------
        self
            For method chaining.
        """
        if self._modified:
            self._update()

        # Promote simplified values back to raw storage
        self._raw_command_array = self._command_array.copy()
        self._raw_constants = self._constants
        self._raw_integers = self._integers
        self._constant_mapping = tuple(range(len(self._constants)))
        self._modified = False
        return self

    def get_operator_counts(self, tree=True, terminals="exclude"):
        """Count the occurrences of each operator in the expression.

        Parameters
        ----------
        tree : bool, optional
            If True, perform depth-first tree traversal (counts repeated
            sub-graphs multiple times).  If False, count unique nodes in
            the DAG.  Default is True.
        terminals : {"include", "exclude", "combine"}
            How to handle terminal nodes:

            - ``"include"``: Count each terminal type separately.
            - ``"exclude"``: Don't count terminal nodes.
            - ``"combine"``: Combine all terminals into a single
              ``VARIABLE`` (0) category.

            Default is ``"exclude"``.

        Returns
        -------
        dict
            Mapping from operator ID (``int``) to count (``int``).
        """
        if self._modified:
            self._update()
        command_array = self._command_array

        if not tree:
            return self._dag_operator_counts(command_array, terminals)
        return self._tree_operator_counts(command_array, terminals)

    @staticmethod
    def _dag_operator_counts(command_array, terminals):
        """Vectorised operator counting over the DAG (no repeated nodes)."""
        nodes = command_array[:, 0]
        terminal_mask = IS_TERMINAL_ARRAY[nodes]
        counts = {}

        # Non-terminal operators — always counted.
        nt_nodes = nodes[~terminal_mask]
        if nt_nodes.size:
            ids, cnts = np.unique(nt_nodes, return_counts=True)
            for op_id, cnt in zip(ids, cnts):
                counts[int(op_id)] = int(cnt)

        # Terminal handling.
        if terminals == "exclude":
            return counts

        t_nodes = nodes[terminal_mask]
        if t_nodes.size == 0:
            return counts

        if terminals == "combine":
            counts[VARIABLE] = counts.get(VARIABLE, 0) + int(t_nodes.size)
        else:  # "include"
            ids, cnts = np.unique(t_nodes, return_counts=True)
            for op_id, cnt in zip(ids, cnts):
                counts[int(op_id)] = int(cnt)

        return counts

    @staticmethod
    def _tree_operator_counts(command_array, terminals):
        """Depth-first tree traversal operator counting."""
        if command_array.shape[0] == 0:
            return {}

        # Local references for speed inside the hot loop.
        is_terminal = IS_TERMINAL_ARRAY
        is_arity_2 = IS_ARITY_2_ARRAY
        exclude = terminals == "exclude"
        combine = terminals == "combine"

        counts = {}
        stack = [command_array.shape[0] - 1]
        while stack:
            idx = stack.pop()
            row = command_array[idx]
            node = row[0]

            if is_terminal[node]:
                if exclude:
                    continue
                key = VARIABLE if combine else int(node)
                counts[key] = counts.get(key, 0) + 1
                continue

            int_node = int(node)
            counts[int_node] = counts.get(int_node, 0) + 1
            stack.append(int(row[1]))
            if is_arity_2[node]:
                stack.append(int(row[2]))

        return counts

    def distance(self, other):
        """Element-wise distance between two raw command arrays.

        Parameters
        ----------
        other : AGraphExpression

        Returns
        -------
        int
        """
        return int(np.sum(self.raw_command_array != other.raw_command_array))

    def copy(self):
        """Deep copy of the expression."""
        return copy.deepcopy(self)

    # ------------------------------------------------------------------ #
    #  Internal update logic                                              #
    # ------------------------------------------------------------------ #

    def _notify_modification(self):
        # A raw structural change unsets the fitted state (structure-only
        # lifecycle); the property re-derives ``True`` for constant-free stacks.
        self._modified = True
        self._fit_attempted = False
        self._hash = None

    def _update(self):
        """Run the simplification backend to derive command_array, constants,
        and integers from the raw inputs."""
        if self._simplification == "cas":
            (
                self._command_array,
                self._constants,
                self._integers,
                self._constant_mapping,
            ) = cas_simplify(
                self._raw_command_array,
                self._raw_constants,
                self._raw_integers,
            )
        else:
            (
                self._command_array,
                self._constants,
                self._integers,
                self._constant_mapping,
            ) = reduce(
                self._raw_command_array,
                self._raw_constants,
                self._raw_integers,
            )
        self._modified = False

    # ------------------------------------------------------------------ #
    #  Hash / equality                                                    #
    # ------------------------------------------------------------------ #

    def __hash__(self):
        if self._modified:
            self._update()
        if self._hash is None:
            self._hash = hash(tuple(map(tuple, self._command_array)))
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, AGraphExpression):
            return NotImplemented
        return hash(self) == hash(other)

    # ------------------------------------------------------------------ #
    #  Serialization                                                      #
    # ------------------------------------------------------------------ #

    def __getstate__(self):
        state = {}
        # Essential: raw layer (source of truth)
        # Store command array as (n_rows, bytes) for compact pickling;
        # bytes objects have ~15 B overhead vs ~129 B for ndarray.
        arr = self._raw_command_array
        state["_raw_command_array"] = (arr.shape[0], arr.tobytes())
        state["_raw_constants"] = self._raw_constants
        state["_raw_integers"] = self._raw_integers
        # Preserve the structure-only fitted lifecycle exactly.
        if self._fit_attempted:
            state["_fit_attempted"] = True
            # Fitted constants differ from raw when fitting moved them;
            # only include them when they actually differ.
            if self._constants != self._raw_constants:
                state["_constants"] = self._constants
        # Only include non-default settings to keep pickle compact.
        if self._simplification != "cas":
            state["_simplification"] = self._simplification
        if self._propagate_constants:
            state["_propagate_constants"] = True
        return state

    def __setstate__(self, state):
        raw = state["_raw_command_array"]
        if isinstance(raw, tuple):
            n_rows, data = raw
            self._raw_command_array = np.frombuffer(data, dtype=np.uint8).reshape(n_rows, 3).copy()
        else:
            # Backward compat: old pickles stored an ndarray directly.
            self._raw_command_array = np.asarray(raw, dtype=np.uint8)
        self._raw_constants = state["_raw_constants"]
        self._raw_integers = state.get("_raw_integers", ())
        self._simplification = state.get("_simplification", "cas")
        self._propagate_constants = state.get("_propagate_constants", False)
        # Backward compat: older pickles encoded fittedness by the mere
        # presence of "_constants".
        self._fit_attempted = state.get("_fit_attempted", "_constants" in state)
        self._hash = None

        if "_constants" in state:
            # Derive the simplified layer from raw, then restore the fitted
            # constant values so evaluation matches the pre-serialization
            # Expression exactly.
            self._command_array = np.empty([0, 3], dtype=np.uint8)
            self._constants = ()
            self._integers = ()
            self._constant_mapping = ()
            self._modified = True
            self._update()
            self._constants = tuple(state["_constants"])
        else:
            # Derived layer recomputed lazily on first access.
            self._constants = self._raw_constants
            self._command_array = np.empty([0, 3], dtype=np.uint8)
            self._integers = ()
            self._constant_mapping = ()
            self._modified = True

    def __deepcopy__(self, memodict=None):
        new = AGraphExpression.__new__(AGraphExpression)
        new._simplification = self._simplification
        new._propagate_constants = self._propagate_constants
        new._raw_command_array = np.copy(self._raw_command_array)
        new._raw_constants = tuple(self._raw_constants)
        new._raw_integers = tuple(self._raw_integers)
        new._command_array = np.copy(self._command_array)
        new._constants = tuple(self._constants)
        new._integers = tuple(self._integers)
        new._constant_mapping = tuple(self._constant_mapping)
        new._fit_attempted = self._fit_attempted
        new._modified = self._modified
        new._hash = None
        return new
