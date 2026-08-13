## Problem

Bingo has two incompatible ways to select continuous parameters. Expressions
own constants, temporary-constant evaluation, derivatives, and fittedness, but
`bingo.local_optimizers` mutates arbitrary Chromosomes through a separate
three-method protocol. The generic protocol has no demonstrated production use
outside the zero-minimization examples, while its SciPy and SMCPy wrappers
duplicate behavior now owned by Expressions.

The current Expression objectives are intentionally simple but cannot combine a
user-selected fitting algorithm, fitting measure, and ranking Loss. The SMCPy
implementation is also modeled as a local optimizer even though its primary
purpose is Evidence estimation. Its better Laplace-proposal implementation
lives on the `smc_laplace` branch and still depends on the obsolete Chromosome
protocol.

## Why Now

The Expression migration has established matching Python and C++ model
surfaces, Expression-owned LM fitting, private subsettable Objective data, and
structure-only fittedness. Keeping the generic local-optimizer stack would
retain a second parameter model immediately after that consolidation. This is
also the right point to port the `smc_laplace` behavior before further work
builds on the older develop-branch estimator.

## Proposed Direction

Per [ADR 0002](../../adr/0002-expression-specific-fitting.md), retain
general-purpose evolutionary optimization while making continuous fitting an
Expression-specific capability.

Retain `ExplicitRegression` and `ImplicitRegression` as stable, opinionated
public Objectives with their current observable behavior. Internally, allow
them to share the same fitting lifecycle as a new `CustomRegression` Objective.
`CustomRegression` requires three independent choices: a fitter, a fitting
measure, and a lower-is-better scalar ranking Loss.

Introduce a generic `ObjectiveData(*arrays)` container. It validates equal
first-axis lengths, exposes the aligned arrays as a tuple, and applies indexing
to every array. Explicit and implicit data containers may build on it while
retaining semantic accessors such as `X`, `y`, and `dx_dt`.

A fitting measure is a pure callable over an Expression, Objective data, and
candidate constants. Residual measures return vectors and may expose Jacobians
or per-residual Hessians; scalar measures return scalars and may expose
gradients or Hessians. Fitters declare the capabilities they accept, own their
initialization policy, and return `FitResult(constants, success, message)`.
They do not mutate Expressions during trial evaluations.

The Objective validates the returned constant count and finiteness, then asks
the Expression to commit the constants and establish fittedness. Numerical
non-convergence may commit the best finite constants with `success=False`.
Contract errors, invalid constants, and user-code exceptions propagate without
establishing fittedness. Expressions with no optimizable constants skip fitting.
Fittedness remains structure-only and independent of Objective, data subset,
or fitting policy. Users deliberately refit through `Expression.clear_fit()`.

Ship the existing LM residual behavior, one configurable `ScipyFitter` for
compatible SciPy root and minimize methods, adapters for named Expression
losses, explicit and implicit residual measures, and user callable measures and
fitters. Preserve Python/C++ parity for temporary constants, derivatives, fit
commit, fittedness, copying, and serialization.

Model SMCPy as Evidence estimation rather than fitting. `SmcEvidenceEstimator`
returns `EvidenceResult` containing higher-is-better `smc_nmll`, posterior MAP
constants, status, and diagnostics. It installs MAP constants only after
successful sampling and Evidence calculation; failure preserves the pre-SMC
constants and maps to infinite ranking Loss. Posterior samples default to
omitted and are returned only when `return_posterior=True`; results are never
attached automatically to Expressions.

Port the Laplace proposal algorithm and validated numerical behavior from the
`smc_laplace` branch into this interface rather than cherry-picking its legacy
files. Keep `laplace_nmll` for the existing closed-form approximation and use
`smc_nmll` for the sequential-Monte-Carlo estimate. Both names mean normalized
marginal log-likelihood and are higher-is-better; ranking adapters negate them
for Bingo's lower-is-better fitness contract.

SMC accepts a root seed and derives a stable child generator from that seed and
the Expression's raw structure, excluding constants and fitness. This makes the
same structure reproducible across serial and multiprocessing evaluation.
`seed=None` permits non-reproducible sampling. Move SMCPy from required
dependencies to an optional `evidence` dependency.

## Scope

- Add shared Objective data, fitting, custom regression, and Evidence modules
  under `bingo.symbolic_regression`.
- Publicly export `CustomRegression`, `ObjectiveData`, `ScipyFitter`,
  `FitResult`, `SmcEvidenceEstimator`, and `EvidenceResult`.
- Keep capability protocols and ranking adapters in their defining modules.
- Preserve current `ExplicitRegression`, `ImplicitRegression`, and
  `SymbolicRegressor` behavior.
- Support fitting on residual vectors, named Expression losses, and user-defined
  scalar or residual measures.
- Support arbitrary aligned arrays through `ObjectiveData` and preserve common
  subset fitting and ranking under `RandomSubsetEvaluation`.
- Remove `bingo.local_optimizers` without compatibility modules or aliases.
- Remove local-optimization methods from `Chromosome`,
  `MultipleValueChromosome`, and `MultipleFloatChromosome`, including
  `needs_opt_list` generator configuration.
- Remove generic `VectorBasedFunction` metrics `"negative nmll laplace"` and
  `"bic"`, which depend on the deleted parameter-count protocol.
- Retain generic evolutionary-optimization examples while removing their local
  fitting behavior.
- Publish migration documentation mapping every removed API to an Expression
  replacement or explicitly stating that no replacement exists.
- Require custom callables to be pickleable only when used with multiprocessing
  or checkpoint workflows; serial evaluation accepts any compatible callable.
- Exclude compatibility stubs, old checkpoint migration, retained generic
  continuous fitting, automatic posterior retention, and automatic refitting
  when Objective configuration changes.
- Keep Bingo's existing general-purpose evolutionary-optimization product
  description without adding local-fitting qualification.

## Key Questions

- Which minimal runtime capability checks give clear errors for incompatible
  fitter and measure combinations without creating a large class hierarchy?
- Which stable raw-structure serialization should seed SMC identically across
  Python and C++ Expression backends?
- Which SMCPy posterior object can be retained when requested while remaining
  pickleable and avoiding unnecessary copies?
- Which numerical regularization from `smc_laplace` is required when the
  Laplace covariance is singular or not positive semidefinite?

## Modules And Surfaces

- Expression APIs: validated fit commit, explicit fit reset, temporary-constant
  evaluation, derivative capabilities, copying, and serialization.
- Regression objectives: shared lifecycle, aligned Objective data,
  `ExplicitRegression`, `ImplicitRegression`, and `CustomRegression`.
- Fitting: `FitResult`, LM behavior, `ScipyFitter`, measure adapters, capability
  validation, and custom callable contracts.
- Evidence: `EvidenceResult`, `SmcEvidenceEstimator`, NMLL ranking adapter,
  Laplace proposal construction, MAP commit, posterior capture, and RNG policy.
- Evaluation: common subset semantics and multiprocessing serialization.
- Generic chromosomes and fitness metrics: complete local-optimization protocol
  removal.
- Packaging, public exports, examples, migration documentation, and tests.

## Validation Plan

- Test aligned-array validation, tuple exposure, indexing, deep copying, and
  `RandomSubsetEvaluation` behavior for generic, explicit, and implicit data.
- Test pure fitting measures against candidate constants and verify that trial
  evaluations never mutate Expression constants or fittedness.
- Test residual/scalar capability matching, optional derivatives, invalid
  shapes, non-finite results, user exceptions, and SciPy method dispatch.
- Test successful, non-converged, and invalid `FitResult` handling, including
  exact constant-count validation and atomic fit commit.
- Preserve existing explicit LM, implicit least-squares, ranking Loss,
  anti-triviality guard, and `SymbolicRegressor` tests.
- Parametrize Expression lifecycle tests across Python and available C++
  backends, including `clear_fit()`, fit commit, copy, pickle, and raw mutation.
- Port branch tests for SMCPy and add deterministic fixed-seed tests for
  proposal covariance, NMLL sign, MAP installation, rollback, optional posterior
  capture, and serial/multiprocessing stream parity.
- Test missing optional SMCPy dependencies with a focused installation/import
  smoke test and test the `evidence` extra in distribution validation.
- Audit the repository for every deleted import, protocol method, metric name,
  `needs_opt_list`, documentation reference, and tutorial cell.
- Run the full Python suite and C++ parity suite after each vertical slice; run
  documentation, examples, and distribution validation before the hard switch
  merges.

## Risks And Unknowns

- Structure-only fittedness permits reuse of constants fitted by a different
  Objective or data subset. This is deliberate and requires explicit
  `clear_fit()` when users change fitting intent.
- Deriving identical stable structure digests across Python and C++ may require
  a backend-neutral byte representation rather than backend object hashing.
- The `smc_laplace` covariance calculation assumes per-sample constant Hessians
  and may need regularization for singular or indefinite curvature.
- SMCPy may use global or library-owned random state internally, limiting full
  serial/multiprocessing reproducibility despite deterministic proposal inputs.
- Optional posterior objects may be large or non-pickleable; capture must not
  affect ordinary evolutionary ranking.
- Immediate deletion can break external users of generic local optimization,
  so migration documentation must be complete even though no compatibility
  period is provided.

## Follow-on Issues

1. Add shared Objective data and the cross-backend Expression fit lifecycle,
   including validated fit commit and `clear_fit()`.
2. Add fitting measures, `FitResult`, `ScipyFitter`, and `CustomRegression`;
   preserve explicit and implicit presets over the shared machinery.
3. Add the Evidence interface and port `smc_laplace` behavior, deterministic
   seeding, atomic MAP commit, and opt-in posterior capture.
4. Remove generic local optimization, coupled metrics, and required SMCPy;
   update exports, examples, migration documentation, and distribution checks.

These are test-passing vertical slices on one migration branch. The completed
initiative merges as one hard public API switch.