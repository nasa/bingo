Expressions Migration
=====================

Bingo's symbolic-regression API is Expression-based. Public objectives and
fitting tools include ``SymbolicRegressor``, ``ExplicitRegression``,
``ImplicitRegression``, ``CustomRegression``, ``ObjectiveData``,
``ScipyFitter``, and ``FitResult``. Evidence estimation is provided by
``SmcEvidenceEstimator`` and ``EvidenceResult``. Import expression generation
and variation from ``bingo.expressions``:

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

Generic local optimization
--------------------------

``bingo.local_optimizers`` has been removed with no compatibility module.
``LocalOptimizer``, ``ScipyOptimizer``, ``SmcpyOptimizer``,
``LocalOptFitnessFunction``, and ``NormalizedMarginalLikelihood`` have no
generic replacement. Generic evolutionary optimization continues to evaluate
Chromosomes directly through ``Evaluation``.

The Chromosome local-optimization protocol
(``needs_local_optimization``, ``get_number_local_optimization_params``,
``get_local_optimization_params``, and ``set_local_optimization_params``) and
the ``needs_opt_list`` arguments to ``MultipleFloatChromosome`` and
``MultipleFloatChromosomeGenerator`` have been removed with no replacement.

Expression users should use ``ExplicitRegression``, ``ImplicitRegression``, or
``CustomRegression`` to select an Expression fitting policy. Use
``ScipyFitter`` for configurable SciPy fitting. For Evidence estimation, use
``SmcEvidenceEstimator`` and install the optional ``evidence`` dependency.

``VectorBasedFunction``, ``GradientMixin``, and ``VectorGradientMixin`` have
been removed. Implement custom objectives by subclassing ``FitnessFunction``
and returning a scalar lower-is-better fitness value. Expression
``loss(..., "laplace_nmll")`` remains available for Laplace normalized marginal
log-likelihood.

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
