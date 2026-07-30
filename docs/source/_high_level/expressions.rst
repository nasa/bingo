Expressions
===========

An ``Expression`` is a standalone mathematical model that represents a
candidate symbolic relationship. It owns fitting, scoring, prediction, and
derivative operations, and is independent of the evolutionary framework — you
can build one, fit it, and evaluate it without running an evolution.

The default implementation is the acyclic-graph expression,
``AGraphExpression``. A pure-Python backend is always available; an accelerated
C++ backend is used automatically when it is installed. Both backends expose the
same public contract.

.. code-block:: python

    import numpy as np
    from bingo.expressions import AGraphExpression

    expr = AGraphExpression(equation="X_0 * 1.0 + 1.0")

    X = np.linspace(0.1, 5.0, 20).reshape(-1, 1)
    y = 3.0 * X.ravel() + 7.0

Prediction and derivatives
--------------------------

``predict`` evaluates the expression on input data and returns a 1-D array of
predictions:

.. code-block:: python

    predictions = expr.predict(X)

``gradient`` additionally returns the derivative of the output with respect to
the inputs — the surface implicit regression is built on:

.. code-block:: python

    f_of_x, df_dx = expr.gradient(X)   # df_dx has shape (M, D)

Fitting
-------

An expression can carry *optimizable constants*. Fitting adjusts those constants
to match data. Explicit fitting uses Levenberg-Marquardt and always minimizes
the expression's ordinary residual vector, independent of the loss used to rank
expressions during evolution. Implicit fitting minimizes its normalized
gradient-alignment residual with a backend-specific numerical least-squares
solver. The convergence tolerance is a keyword-only argument:

.. code-block:: python

    expr.fit(X, y, tolerance=1e-5)   # explicit regression: f(x) ≈ y

For implicit regression — where the expression's input gradient should be
consistent with observed trajectory derivatives — use ``fit_implicit``:

.. code-block:: python

    expr.fit_implicit(X, dx_dt, tolerance=1e-5)

Score and loss
--------------

Bingo distinguishes two polarities of quality measure:

- **Score** is *higher-is-better*. Explicit regression uses :math:`R^2` by
  default and may use the Laplace NMLL. Implicit regression uses the negative of
  its aggregate loss.
- **Loss** is *lower-is-better* and is what Bingo fitness evaluation minimizes.

.. code-block:: python

    expr.score(X, y)                     # R^2 (default), higher is better
    expr.score(X, y, kind="laplace_nmll")

    expr.loss(X, y)                      # mean squared error (default)
    expr.loss(X, y, kind="mae")
    expr.loss(X, y, kind="rmse")
    expr.loss(X, y, kind="relative_mse") # rejects zero-valued targets
    expr.loss(X, y, kind="correlation")
    expr.loss(X, y, kind="laplace_nmll") # negation of the NMLL score

Implicit regression has matching ``implicit_score`` and ``implicit_loss``
methods. Both accept an optional anti-triviality guard, ``required_params``:
at least one sample must have the requested number of derivative components with
magnitude greater than :math:`10^{-16}`, otherwise the loss is infinite (and the
score negative-infinite):

.. code-block:: python

    expr.implicit_loss(X, dx_dt, required_params=2)
    expr.implicit_score(X, dx_dt)

Non-finite evaluation
---------------------

When an expression cannot be evaluated to a finite value on the given data,
Bingo normalizes the outcome rather than propagating ``nan``: loss becomes
positive infinity and score becomes negative infinity. This holds for both the
explicit and implicit paths.

The fitted lifecycle
--------------------

An expression tracks whether it has been *fitted*. It is fitted when it has no
optimizable constants, or when an applicable fitting method has been attempted
for its current structure. The lifecycle is structure-only:

- A structural change (editing the raw command array or raw constants) unsets
  the fitted state.
- A fitting attempt establishes it — even if the solver does not numerically
  converge.
- Directly assigning simplified constants preserves the current state but cannot
  establish it.
- Copying and serialization (``copy.deepcopy``, ``pickle``) preserve the fitted
  state exactly.

.. code-block:: python

    expr = AGraphExpression(equation="X_0 + 1.0")
    expr.is_fitted            # False — has an unfitted constant
    expr.fit(X, y)
    expr.is_fitted            # True

    expr.raw_constants = (0.0,)
    expr.is_fitted            # False — structure changed

``is_fitted`` is also honored by scikit-learn's ``check_is_fitted`` through the
``__sklearn_is_fitted__`` hook.
