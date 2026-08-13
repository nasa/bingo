---
title: Deliver customizable regression fitting
category: enhancement
state: ready-for-agent
blocked_by:
  - 01-expression-fitting-foundations
created: 2026-08-13
---

## Parent

[Expression fitting and Evidence work brief](../WORK-BRIEF.md)

## What to build

Deliver an advanced regression Objective whose fitting algorithm, fitting
measure, and ranking Loss are independently configurable. `CustomRegression`
requires all three choices explicitly and works with arbitrary aligned
Objective data while preserving common-subset fitting and ranking.

Fitting measures evaluate candidate constants without mutating the Expression.
They may be residual-vector or scalar measures and may optionally provide the
derivatives required by a fitter. Fitters own initialization and return a
structured result; the Objective validates and commits the result through the
Expression lifecycle.

Ship a configurable SciPy fitter for compatible root and minimize methods,
built-in explicit and implicit residual measures, adapters for named Expression
losses, and support for user callable fitters and measures. Preserve
`ExplicitRegression` and `ImplicitRegression` as simple public presets with
their current observable behavior.

## Acceptance criteria

- [ ] `CustomRegression` requires Objective data, a fitter, a fitting measure,
  and a lower-is-better scalar ranking Loss.
- [ ] Fitting measures receive an Expression, Objective data, and candidate
  constants; trial evaluations do not change Expression constants or
  fittedness.
- [ ] Scalar and residual measures expose distinct value contracts with
  optional gradient, Jacobian, Hessian, or per-residual Hessian capabilities.
- [ ] Fitters return `FitResult` with constants, success, and an optional
  message, and incompatible fitter/measure combinations fail with clear errors.
- [ ] Numerical non-convergence may commit the best finite constants while
  invalid shape, non-finite constants, contract violations, and user exceptions
  leave the Expression unfitted and propagate appropriately.
- [ ] Expressions without optimizable constants skip fitting and proceed
  directly to ranking.
- [ ] `ScipyFitter` supports the selected SciPy root and minimize methods,
  fitter-owned initialization, and available derivative capabilities without
  exposing SciPy result objects as the public contract.
- [ ] Existing LM explicit fitting, implicit least-squares fitting,
  `required_params`, score/Loss direction, and `SymbolicRegressor` behavior are
  unchanged after the preset Objectives adopt the shared machinery.
- [ ] Custom callable portability is documented: serial workflows accept any
  compatible callable, while multiprocessing and checkpoints require
  pickleable callables.
- [ ] `CustomRegression`, `ObjectiveData`, `ScipyFitter`, and `FitResult` are
  exported from the symbolic-regression package.

## Blocked by

- [01 - Build Expression fitting foundations](01-expression-fitting-foundations.md)