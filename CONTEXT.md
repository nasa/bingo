# Bingo

Bingo evolves and simplifies symbolic mathematical expressions for regression and general-purpose evolutionary optimization.

## Project Support

**Installation-compatible Python**:
A Python version allowed by Bingo's package metadata. Versions 3.11 and 3.12 are installation-compatible but unverified.
_Avoid_: supported Python

**CI-verified Python**:
A Python version covered by Bingo's required continuous-integration validation. Bingo's support guarantee begins with Python 3.13.
_Avoid_: installation-compatible Python

**Required PR validation**:
The always-running standard pull-request validation comprising linting and test jobs on Python 3.13 and 3.14. It is not currently enforced by GitHub branch protection.
_Avoid_: selectively run verification, merge-blocking check

**Standard validation event scope**:
Standard validation runs for every pull request and for direct updates to the `main` and `develop` branches, but not for other branch pushes.
_Avoid_: all-branch push validation

**Coverage reporting**:
The Python 3.13 full-suite validation job generates Bingo's only coverage report. It publishes coverage artifacts and a job summary through GitHub Actions, without an external coverage service. Python 3.14 runs the full suite without coverage instrumentation.
_Avoid_: duplicate matrix coverage reports, external coverage service

**Full-suite validation**:
Execution of Bingo's complete test suite, including MPI-dependent integration coverage. Required PR validation runs the full suite on every CI-verified Python version.
_Avoid_: unit-only validation

**Selective user-facing verification**:
Advisory documentation and example validation that runs after every update to the `main` and `develop` branches. On pull requests, documentation runs for changes to `docs/**`, `bingo/**`, build or dependency metadata, or `.github/**`; examples run for changes to `examples/**`, `bingo/**`, build or dependency metadata, or `.github/**`. It is not a branch-protected check.
_Avoid_: required PR validation

**Performance benchmark validation**:
Non-required performance benchmarking run after updates to the `main` and `develop` branches. It is not coupled to GitHub branch-protection status.
_Avoid_: required PR validation

**Distribution validation**:
Building Bingo's source and binary distributions, checking their metadata with `twine check`, and smoke-testing a clean wheel installation after updates to the `main` and `develop` branches. It does not publish artifacts.
_Avoid_: release publication

**Release publication**:
The tag-triggered publication, signing, and GitHub release process for distributions that have passed the full suite from a clean installed artifact. It applies to production release tags and release-candidate tags; pushing an eligible tag is the publication authorization and does not require manual environment approval.
_Avoid_: distribution validation

**Production release tag**:
An exact numeric `X.Y.Z` Git tag, such as `0.5.8`, that is eligible to trigger release publication. Other tag shapes are not production releases.
_Avoid_: arbitrary tag, version-prefixed tag

**Release-candidate tag**:
An exact numeric `X.Y.ZrcN` Git tag, such as `0.5.8rc1`, that is eligible to publish a public pre-release to PyPI and create a signed GitHub prerelease. It may tag a commit on `develop`.
_Avoid_: TestPyPI build, arbitrary tag

**Tag-artifact version equality**:
The release workflow verifies that every built distribution's normalized version exactly equals the triggering production or release-candidate tag before publication. This workflow check enforces the release convention without narrowing `setuptools-scm`'s historic tag parsing.
_Avoid_: inferred release version

**Published wheel coverage**:
Bingo publishes Linux x86_64 wheels for Python 3.11 through 3.14 and an sdist. Every wheel receives a clean-install smoke test; the Python 3.13 and 3.14 wheels additionally run the full installed-artifact suite. The sdist receives a clean-install smoke test on Python 3.13.
_Avoid_: untested published wheel

**Manual CI dispatch**:
An operator-triggered execution for diagnostics or release validation. It may run benchmarks, documentation, examples, and distribution validation, but cannot publish a release.
_Avoid_: release publication

**Shared CI setup**:
The reusable environment preparation for CI jobs, including repository checkout, Python, MPI, dependencies, submodules, and the safe-forking environment setting. It is owned by a local composite action.
_Avoid_: workflow policy

**Workflow policy**:
The event triggers, job matrix, check requirements, and permissions that govern a CI workflow. It is owned by the top-level workflow.
_Avoid_: shared CI setup

**Superseded CI run**:
A pull-request or branch workflow run made obsolete by a newer commit on the same pull request or branch. Superseded CI runs are cancelled; tag-triggered release publication is not cancelled.
_Avoid_: release publication

**Least-privilege CI permissions**:
Validation workflows have read-only repository-content access. Release publication receives OIDC token-writing access, and GitHub-release creation receives repository-content write access; no other write permissions are granted.
_Avoid_: workflow-wide write permissions

**Documentation deployment**:
The repository-writing publication of a successfully built documentation site to the `gh-pages` branch after an update to `main`. Documentation validation is a separate, read-only operation.
_Avoid_: documentation validation

## Expression Simplification

**Bounded constant folding**:
A deterministic constant-folding policy that preserves exhaustive folding for expressions with at most seven distinct constants and uses a distinct-first local approximation from eight constants onward.
_Avoid_: unbounded folding, complete folding

**Exhaustive constant folding**:
The existing policy that searches every non-empty subset of an expression's distinct constants in pursuit of the fewest `CONSTANT` nodes.
_Avoid_: greedy folding

Example dialogue:

> Developer: "This expression has twelve distinct constants, so it uses bounded constant folding."
>
> Domain expert: "Correct. Seven or fewer uses exhaustive constant folding; eight or more uses a distinct-first local policy for predictable simplification cost."

## Symbolic Regression

**Expression**:
A standalone mathematical model that represents a candidate symbolic
relationship and owns fitting, scoring, prediction, and derivative operations.
_Avoid_: equation regressor

**Batched constant prediction**:
Prediction over multiple temporary constant sets for one Expression and one
input dataset. Constants use constant-major shape `(L, B)`, with one constant
set per column, and produce predictions shaped `(M, B)`. It is distinct from
batching input samples, does not include derivative or scoring operations, and
does not alter the Expression's constants or fitted state. `L` counts constants
in the simplified Expression and follows their simplified order.
_Avoid_: batched constant evaluation, input minibatching

**Explicit regression**:
Symbolic regression that searches for an Expression whose predicted output
matches a target value for each input.

**Implicit regression**:
Symbolic regression that searches for an Expression whose input gradient is
consistent with observed trajectory derivatives.

**Required parameters**:
An optional implicit-regression anti-triviality guard. An Expression passes
when at least one sample has at least the requested number of derivative
components with magnitude greater than $10^{-16}$; otherwise its loss is
infinite.

**Objective data**:
Private, indexable aligned arrays retained by a regression objective so Bingo
can evaluate a common subset of samples across a population.
_Avoid_: public training-data API

**Score**:
A higher-is-better measure of an Expression's quality. Explicit regression
uses $R^2$ by default and may use Laplace NMLL; implicit regression uses the
negative of its aggregate loss. Non-finite expression evaluation returns a
negative infinite score.
_Avoid_: lower-is-better score

**Loss**:
A lower-is-better objective used by Bingo fitness evaluation. Laplace NMLL
loss is the negation of the corresponding score. Non-finite expression
evaluation returns an infinite loss.

**LM fitting**:
Constant fitting that always minimizes an Expression's ordinary residual vector.
It is independent of the loss used to rank evolved Expressions.

**Fitted expression**:
An Expression with no optimizable constants, or one whose applicable fitting
method has been attempted for its current raw structure. Numerical
non-convergence does not unset this state; a raw structural change does.
Fittedness is independent of the data subset used for the fitting attempt.
Direct constant assignment does not establish fittedness.
Serialization preserves fittedness exactly.

**Relative MSE**:
An explicit-regression loss that averages squared residuals normalized by the
corresponding target value. It is available as ``relative_mse`` only and
rejects zero-valued targets.

**Correlation loss**:
An explicit-regression loss derived from the correlation coefficient between
an Expression's predictions and the observed targets. It assesses association
without applying a hidden output transformation to the Expression.
