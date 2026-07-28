## Problem

Bingo currently carries two incompatible symbolic-regression stacks: legacy
AGraph/Equation APIs under `bingo.symbolic_regression` and the newer
Expression interface under `bingo.expressions`. The legacy stack combines
expression representation, evolutionary state, local optimization, fitness
aggregation, and estimator concerns. This duplication leaves Python and C++
behavior divergent and prevents the Expression interface from becoming the
single supported model surface.

## Why Now

The newer AGraphExpression, its evolution adapter, and backend selection
already exist. Keeping legacy APIs alongside them increases maintenance cost,
retains conflicting score/loss semantics, and lets obsolete generic local
optimization control retained symbolic-regression workflows. A clean major
version can remove that ambiguity without compatibility shims or checkpoint
migration obligations.

## Proposed Direction

Make Expression the sole symbolic model abstraction. `AGraphExpression` owns
evaluation, derivatives, fitting, score, and loss semantics; its Python and
C++ implementations are semantically equivalent. `EvolvableExpression` owns
only Chromosome state. Explicit and implicit regression objectives own
fit-before-score policy and use private Objective data for subset evaluation.
`SymbolicRegressor` is the only scikit-learn-facing integration point.

## Scope

- Remove legacy symbolic-regression AGraph/Equation implementations and their
  public exports, without compatibility aliases.
- Retain `SymbolicRegressor`, `ExplicitRegression`, and `ImplicitRegression`.
- Make `bingo.expressions` the public home for expressions, AGraph generation,
  variation, and `EvolvableExpression`.
- Establish matching Python/C++ Expression APIs for explicit and implicit
  fitting, prediction, derivatives, loss, score, `is_fitted`, copying, and
  serialization.
- Replace retained symbolic-regression local-optimizer wiring with
  expression-owned LM fitting.
- Remove unsupported specialized legacy regression modes:
  `PairwiseAtomicPotential`, `PairwiseAtomicTrainingData`, and
  `ImplicitRegressionSchmidt`.
- Publish migration documentation covering imports, renamed settings, removed
  APIs, score/loss polarity, and serialization incompatibility.
- Exclude legacy API compatibility, legacy dill/pickle migration, and
  seed-for-seed evolutionary trajectory compatibility.

## Key Questions

- Preserve the selected `is_fitted` lifecycle across raw mutation, direct
  constant assignment, copying, and serialization in both backends.
- Ensure C++ and Python normalize numerical failures identically while
  retaining non-finite residual vectors internally for LM rejection.
- Keep generic `VectorBasedFunction` and local-optimizer infrastructure only
  for remaining generic users; remove it from retained symbolic-regression
  objectives.

## Modules And Surfaces

- `bingo.expressions.agraph`: AGraphExpression, backend selection, component
  generation, crossover, mutation, and C++ bindings.
- `EvolvableExpression`: remove the external local-optimization adapter while
  retaining Chromosome behavior.
- `ExplicitRegression` and `ImplicitRegression`: direct `FitnessFunction`
  objectives, private subsettable Objective data, and fit-before-score.
- `SymbolicRegressor`: scikit-learn validation, active backend use, generation
  configuration, estimator loss selection, and `fit_tolerance` forwarding.
- Legacy symbolic-regression exports, obsolete specialized modes, tests, and
  C++ declarations/bindings.
- Sphinx migration documentation, README links, and the domain glossary.

## Validation Plan

- Parametrize Expression, objective, and estimator tests over all locally
  available backends; Python is always required and unavailable C++ cases skip
  locally.
- Build C++ in CI and require its parametrized cases.
- Compare fixed-expression numerical behavior within tolerance: predictions,
  input gradients, fitted constants, explicit and implicit losses/scores,
  `is_fitted`, serialization, and normalized non-finite outcomes.
- Cover Objective data slicing through `RandomSubsetEvaluation` and the
  retained implicit `required_params` guard.
- Validate `SymbolicRegressor` scikit-learn behavior, including feature-count
  tracking and estimator validation.
- Remove tests for deliberately removed public surfaces and run the full test
  suite with the C++ extension enabled.

## Risks And Unknowns

- C++ solver and error-normalization behavior may require small backend-specific
  implementation adjustments to meet the semantic parity gate.
- Removing exports may reveal examples, docs, or optional modules still
  importing legacy surfaces; the import audit must be complete before release.
- Random-subset evaluation must reuse fitted constants for an unchanged raw
  Expression rather than silently retriggering fitting on every subset.

## Follow-on Issues

- Core Python Expression score/loss, fitting, and `is_fitted` lifecycle.
- C++ Expression parity for the same public contract.
- Explicit/implicit Objective rewrite and EvolvableExpression simplification.
- SymbolicRegressor migration and public-export cleanup.
- Removal sweep, migration documentation, and backend-parity integration tests.

These are implementation slices within one complete major-version PR, not
independent compatibility releases.