Expressions Migration
=====================

Bingo's symbolic-regression API is Expression-based. The only public names in
``bingo.symbolic_regression`` are ``SymbolicRegressor``,
``ExplicitRegression``, and ``ImplicitRegression``. Import expression
generation and variation from ``bingo.expressions``:

.. code-block:: python

    from bingo.expressions import (
        AGraphExpression,
        AGraphGenerator,
        AGraphCrossover,
        AGraphMutation,
        ComponentGenerator,
        EvolvableExpression,
    )
    from bingo.symbolic_regression import SymbolicRegressor

Removed APIs
------------

``AGraph``, ``Equation``, explicit and implicit training-data containers,
``PairwiseAtomicPotential``, ``PairwiseAtomicTrainingData``, and
``ImplicitRegressionSchmidt`` have been removed. Objectives now receive numeric
arrays directly: ``ExplicitRegression(X, y)`` and
``ImplicitRegression(X, dx_dt)``. Expression fitting accepts those same arrays.

``SymbolicRegressor`` replaces ``stack_size`` with ``min_stack_size`` and
``max_stack_size``, ``use_simplification`` with ``simplification``, and
``metric`` with ``loss``. The removed ``clo_alg`` and ``clo_threshold`` settings
are replaced by expression-owned fitting configured with ``fit_tolerance``.

Loss and serialization
----------------------

Evolution minimizes **loss**. Expression ``score`` is higher-is-better, while
``loss`` is lower-is-better. Existing AGraph/Equation instances and checkpoints
cannot be loaded by this version; there is no checkpoint migration path.

Backends
--------

``AGraphExpression`` uses the C++ backend when available and otherwise uses the
pure-Python backend. Select a backend explicitly with
``bingo.expressions.agraph.set_backend("python")`` or ``set_backend("cpp")``.
