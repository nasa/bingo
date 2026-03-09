"""Cached evaluation for use during constant optimisation (fitting).

During fitting, the command stack and input data *X* are fixed — only
the constants change between iterations.  This module provides a
:class:`CachedEvaluator` that exploits that invariance in three ways:

1. **Fused residual / jacobian**: Cache the forward buffer from the
   residual evaluation and reuse it for the immediately following
   jacobian call (which receives the same constants), eliminating one
   redundant forward pass per Levenberg-Marquardt iteration.

2. **Partial forward re-evaluation**: Pre-compute a dependency mask
   indicating which stack rows transitively depend on ``CONSTANT``
   nodes.  On subsequent evaluations only the constant-dependent rows
   are recomputed; the constant-independent rows are read from a
   persistent cache.

3. **Partial reverse re-evaluation**: During the reverse (gradient)
   pass, cache contributions from rows whose reverse operation is
   invariant across constant changes.  ``ADDITION`` and
   ``SUBTRACTION`` reverse functions do not read forward values, so
   they are invariant whenever their adjoint is invariant.  All other
   non-terminal operators are treated conservatively as variant when
   any of their forward inputs depend on constants.
"""

import numpy as np

from .evaluation import _reshape_reverse_eval, _reshape_output
from .operator_eval import forward_eval_function, reverse_eval_function
from ..operators import (
    CONSTANT,
    ADDITION,
    SUBTRACTION,
    IS_TERMINAL_MAP,
    IS_ARITY_2_MAP,
)


# ------------------------------------------------------------------ #
#  Dependency mask                                                     #
# ------------------------------------------------------------------ #


def _build_dependency_mask(stack):
    """Return a boolean array marking constant-dependent rows.

    A row *i* depends on constants if:

    * ``stack[i, 0] == CONSTANT``, or
    * any of its operands (``stack[i, 1]``, ``stack[i, 2]``) points to
      a row that depends on constants.

    Parameters
    ----------
    stack : Nx3 numpy array
        The command stack.

    Returns
    -------
    numpy array of bool, shape (N,)
        ``True`` for rows that (transitively) depend on a ``CONSTANT``.
    """
    n = stack.shape[0]
    depends = np.zeros(n, dtype=bool)
    for i in range(n):
        node = int(stack[i, 0])
        if node == CONSTANT:
            depends[i] = True
        elif not IS_TERMINAL_MAP[node]:
            p1 = int(stack[i, 1])
            if depends[p1]:
                depends[i] = True
            elif IS_ARITY_2_MAP[node]:
                p2 = int(stack[i, 2])
                if depends[p2]:
                    depends[i] = True
    return depends


# ------------------------------------------------------------------ #
#  Reverse variant mask                                                #
# ------------------------------------------------------------------ #


def _build_reverse_variant_mask(stack, forward_depends):
    """Return masks identifying variant reverse-pass rows.

    A reverse operation at row *i* produces **variant** contributions
    (contributions that change when constants change) when the adjoint
    ``reverse[i]`` is variant **or** the forward values it reads are
    variant.

    ``ADDITION`` and ``SUBTRACTION`` reverse functions do not read
    forward values, so they contribute variant output only when
    ``reverse[i]`` itself is variant.  All other non-terminal operators
    are treated conservatively: they are variant when
    ``reverse_variant[i]`` **or** ``forward_depends[i]`` is ``True``.

    Parameters
    ----------
    stack : Nx3 numpy array
        The command stack.
    forward_depends : numpy array of bool, shape (N,)
        ``True`` for rows whose forward value depends on constants.

    Returns
    -------
    reverse_variant : numpy array of bool, shape (N,)
        ``True`` for rows whose adjoint ``reverse[i]`` can change.
    row_variant : numpy array of bool, shape (N,)
        ``True`` for non-terminal rows whose reverse operation must be
        re-executed on each call.
    """
    n = stack.shape[0]
    reverse_variant = np.zeros(n, dtype=bool)
    row_variant = np.zeros(n, dtype=bool)

    for i in range(n - 1, -1, -1):
        node = int(stack[i, 0])
        if IS_TERMINAL_MAP[node]:
            continue

        # ADD/SUB: reverse function only reads reverse[i], not forward.
        if node in (ADDITION, SUBTRACTION):
            is_variant = bool(reverse_variant[i])
        else:
            is_variant = bool(reverse_variant[i]) or bool(forward_depends[i])

        row_variant[i] = is_variant

        if is_variant:
            p1 = int(stack[i, 1])
            reverse_variant[p1] = True
            if IS_ARITY_2_MAP[node]:
                p2 = int(stack[i, 2])
                reverse_variant[p2] = True

    return reverse_variant, row_variant


# ------------------------------------------------------------------ #
#  CachedEvaluator                                                     #
# ------------------------------------------------------------------ #


class CachedEvaluator:
    """Evaluation context that caches invariant computation.

    Designed to be created at the start of a ``fit()`` call and
    discarded when fitting is complete.

    Parameters
    ----------
    stack : Nx3 numpy array
        The (simplified) command stack.
    x : MxD numpy array
        The training input data.
    integers : tuple of int
        Integer lookup for the stack.
    """

    def __init__(self, stack, x, integers):
        self._stack = stack
        self._x = x
        self._integers = integers
        self._n = stack.shape[0]

        # Dependency mask (computed once)
        self._depends_on_constant = _build_dependency_mask(stack)

        # Reverse variant masks (computed once)
        self._reverse_variant, self._row_variant = \
            _build_reverse_variant_mask(stack, self._depends_on_constant)

        # Cached forward buffer for constant-independent rows
        self._static_forward = None  # populated on first eval

        # Cached reverse buffer for invariant reverse contributions
        self._static_reverse = None  # populated on first derivative eval
        self._static_derivative = None

        # Fused residual/jacobian state
        self._last_params = None
        self._last_forward = None

    # ---- public API ---- #

    def forward_eval(self, constants):
        """Evaluate the expression with the given *constants*.

        Parameters
        ----------
        constants : tuple or array of float
            Constant values to use.

        Returns
        -------
        Mx1 numpy array
            Expression output f(x).
        """
        forward = self._compute_forward(constants)

        # Cache for potential jacobian reuse
        self._last_params = constants
        self._last_forward = forward

        return _reshape_output(forward[-1], constants, self._x)

    def forward_eval_with_const_derivative(self, constants):
        """Evaluate and compute derivative w.r.t. constants.

        If the cached forward buffer was computed with the same
        *constants* (e.g. from a preceding ``forward_eval`` call), the
        forward pass is skipped entirely.

        Parameters
        ----------
        constants : tuple or array of float
            Constant values to use.

        Returns
        -------
        tuple of (Mx1 array, MxL array)
            ``(f(x), df/dc)`` where L = ``len(constants)``.
        """
        if (
            self._last_forward is not None
            and _params_equal(self._last_params, constants)
        ):
            forward = self._last_forward
        else:
            forward = self._compute_forward(constants)
            self._last_params = constants
            self._last_forward = forward

        deriv_shape = (self._x.shape[0], len(constants))
        derivative = self._reverse_eval_constants(forward, deriv_shape)

        return _reshape_output(forward[-1], constants, self._x), derivative

    # ---- internals ---- #

    def _compute_forward(self, constants):
        """Forward evaluation reusing cached constant-independent rows."""
        stack = self._stack
        x = self._x
        integers = self._integers
        depends = self._depends_on_constant

        if self._static_forward is None:
            # First call — evaluate everything and cache static rows.
            forward = [None] * self._n
            static = [None] * self._n
            for i, (node, param1, param2) in enumerate(stack):
                forward[i] = forward_eval_function(
                    node, param1, param2, x, constants, integers, forward
                )
                if not depends[i]:
                    static[i] = forward[i]
            self._static_forward = static
            return forward

        # Subsequent calls — copy static rows, recompute dependent ones.
        forward = list(self._static_forward)  # shallow copy
        for i, (node, param1, param2) in enumerate(stack):
            if depends[i]:
                forward[i] = forward_eval_function(
                    node, param1, param2, x, constants, integers, forward
                )
        return forward

    def _reverse_eval_constants(self, forward, deriv_shape):
        """Reverse pass computing derivatives w.r.t. constants."""
        if self._static_reverse is None:
            return self._reverse_eval_first(forward, deriv_shape)
        return self._reverse_eval_cached(forward, deriv_shape)

    def _reverse_eval_first(self, forward, deriv_shape):
        """First reverse call --- run full pass and populate static cache.

        Two reverse passes are performed:

        1. The standard full pass to produce the correct derivative.
        2. A *static-only* pass that processes only invariant rows,
           building ``_static_reverse`` and ``_static_derivative``.
           These capture the portion of the computation that does not
           depend on constants and can be reused on subsequent calls.
        """
        stack = self._stack
        row_variant = self._row_variant
        reverse_variant = self._reverse_variant
        n = self._n
        m = deriv_shape[0]

        # --- Full reverse pass (correct result) ---
        derivative = np.zeros(deriv_shape)
        reverse = [0] * n
        reverse[-1] = 1.0
        for i in range(n - 1, -1, -1):
            node, param1, param2 = stack[i]
            if node == CONSTANT:
                derivative[:, param1] += _reshape_reverse_eval(
                    reverse[i], m
                )
            elif not IS_TERMINAL_MAP[node]:
                reverse_eval_function(
                    node, i, param1, param2, forward, reverse
                )

        # --- Static reverse pass (invariant rows only) ---
        static_reverse = [0] * n
        static_reverse[-1] = 1.0
        static_derivative = np.zeros(deriv_shape)
        for i in range(n - 1, -1, -1):
            node, param1, param2 = stack[i]
            if row_variant[i]:
                continue
            if node == CONSTANT:
                if not reverse_variant[i]:
                    static_derivative[:, param1] += _reshape_reverse_eval(
                        static_reverse[i], m
                    )
            elif not IS_TERMINAL_MAP[node]:
                reverse_eval_function(
                    node, i, param1, param2, forward, static_reverse
                )

        self._static_reverse = static_reverse
        self._static_derivative = static_derivative

        return derivative

    def _reverse_eval_cached(self, forward, deriv_shape):
        """Subsequent reverse calls --- reuse cached invariant parts.

        Starts from a copy of the static reverse buffer and static
        derivative, then re-runs only the variant rows to accumulate
        their dynamic contributions.
        """
        stack = self._stack
        row_variant = self._row_variant
        reverse_variant = self._reverse_variant
        n = self._n
        m = deriv_shape[0]

        # Deep-copy arrays to avoid corrupting the cache.
        reverse = [
            v.copy() if isinstance(v, np.ndarray) else v
            for v in self._static_reverse
        ]
        derivative = self._static_derivative.copy()

        for i in range(n - 1, -1, -1):
            node, param1, param2 = stack[i]
            if node == CONSTANT:
                if reverse_variant[i]:
                    derivative[:, param1] += _reshape_reverse_eval(
                        reverse[i], m
                    )
            elif row_variant[i]:
                reverse_eval_function(
                    node, i, param1, param2, forward, reverse
                )

        return derivative


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #


def _params_equal(a, b):
    """Fast equality check for constant tuples/arrays."""
    if a is b:
        return True
    try:
        if len(a) != len(b):
            return False
        return all(x == y for x, y in zip(a, b))
    except TypeError:
        return False
