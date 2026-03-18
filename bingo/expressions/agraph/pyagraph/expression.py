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

from .evaluation import evaluate, evaluate_with_derivative
from .evaluation.cached_evaluation import CachedEvaluator
from .operators import (
    VARIABLE,
    IS_TERMINAL_ARRAY,
    IS_ARITY_2_ARRAY,
)
from .simplification import get_utilized_commands, reduce, simplify as cas_simplify
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
    """Normalised marginal log-likelihood (Laplace approximation).

    NMLL = (1 - b) * ln(L̂) + ln(b) / 2 * k

    where *b* = 1 / sqrt(n) is a normalisation factor, *k* =
    ``n_constants + 1``, *n* is the number of data points, and *L̂* is
    the maximised Gaussian log-likelihood.  Higher values indicate a
    better fit.

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
    simplification : {"reduce", "cas"}, optional
        Which simplification strategy to use when deriving the
        evaluation-facing stack from the raw stack.  ``"reduce"``
        performs cheap dead-code elimination / constant
        folding on the stack.  ``"cas"`` (default) runs the full computer
        algebra simplification pipeline.
    propagate_constants : bool, optional
        Whether setting simplified constants also updates raw constants.
        Default ``False``.
    """

    _VALID_SIMPLIFICATIONS = frozenset({"reduce", "cas"})

    def __init__(
        self, *, equation=None, simplification="cas", propagate_constants=False
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

        if equation is not None:
            cmd, consts, ints = eq_string_to_command_array_and_constants(str(equation))
            self._raw_command_array = cmd
            self._raw_constants = consts
            self._raw_integers = ints
            self._command_array = np.empty([0, 3], dtype=np.uint8)
            self._constants = ()
            self._integers = ()
            self._is_fitted = False
            self._modified = True
        else:
            self._raw_command_array = np.empty([0, 3], dtype=np.uint8)
            self._raw_constants = ()
            self._raw_integers = ()
            self._command_array = np.empty([0, 3], dtype=np.uint8)
            self._constants = ()
            self._integers = ()
            self._is_fitted = True
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
            return evaluate(
                self._command_array,
                x,
                self._constants,
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

    def fit(self, X, y, metric="mse", **scipykwargs):
        """Optimize constants to fit the data.

        Uses ``scipy.optimize.root(method='lm')`` (Levenberg-Marquardt)
        with analytic Jacobians from the :class:`CachedEvaluator`.

        Parameters
        ----------
        X : array-like, shape (M, D)
        y : array-like, shape (M,)
        metric : str, optional
            Unused for now (root always minimises residuals).
            Retained for API symmetry.
        **scipykwargs : keyword arguments
            Additional keyword arguments forwarded to
            ``scipy.optimize.root``.  Common options include ``tol``
            and ``options={'maxiter': N}``.

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
        cached = CachedEvaluator(self._command_array, X, self._integers)

        def residuals(params):
            self.constants = params
            return cached.forward_eval(self._constants).ravel() - y

        def jacobian(params):
            self.constants = params
            _, jac = cached.forward_eval_with_const_derivative(self._constants)
            return jac

        try:
            result = scipy.optimize.root(
                residuals,
                x0,
                jac=jacobian,
                method="lm",
                **scipykwargs,
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
            The score (lower is better for all metrics except
            ``"laplace_nmll"`` where higher is better).
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
        self._modified = True
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
        self._is_fitted = len(self._constants) == 0
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
        # Fitted constants differ from raw when fit() has been called;
        # only include them when they actually differ.
        if self._is_fitted and self._constants != self._raw_constants:
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
        self._constants = state.get("_constants", self._raw_constants)
        self._is_fitted = "_constants" in state
        # Derived — will be recomputed on first access.
        self._command_array = np.empty([0, 3], dtype=np.uint8)
        self._integers = ()
        self._constant_mapping = ()
        self._modified = True
        self._hash = None

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
        new._is_fitted = self._is_fitted
        new._modified = self._modified
        new._hash = None
        return new
