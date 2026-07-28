---
title: Evaluate Expressions through regression Objectives
category: enhancement
state: ready-for-agent
blocked_by:
  - 01-python-expression-contract
created: 2026-07-28
---

## Parent

[Expressions major migration work brief](../WORK-BRIEF.md)

## What to build

Make explicit and implicit regression Objectives evaluate
`EvolvableExpression` through its Expression rather than through the legacy
external local-optimizer protocol. Both Objectives accept arrays directly,
retain private Objective data for population-subset evaluation, establish
fitting only when the Expression is not fitted, and return lower-is-better
Loss.

This slice removes the external local-optimization adapter from
`EvolvableExpression`. It preserves the implicit `required_params` guard as an
Objective-level anti-triviality policy.

## Acceptance criteria

- [ ] Explicit and implicit Objectives directly subclass `FitnessFunction`,
  accept their selected public inputs, and use private subsettable Objective
  data rather than public training-data APIs.
- [ ] Objectives use `Expression.is_fitted` and expression-owned fitting with
  `fit_tolerance=1e-5`; retained symbolic-regression workflows no longer use
  `LocalOptFitnessFunction` or `ScipyOptimizer`.
- [ ] `EvolvableExpression` retains Chromosome behavior but no longer exposes
  `needs_local_optimization`, local-optimization parameter count, or parameter
  setter methods.
- [ ] Implicit regression preserves the `required_params` guard: at least one
  sample must have the requested number of derivative components above
  $10^{-16}$, otherwise evaluation returns infinite Loss.
- [ ] Unit tests cover Objective fitting, Loss direction, Objective data
  slicing through `RandomSubsetEvaluation`, and the implicit guard.

## Blocked by

- [Define Python Expression lifecycle and scoring](01-python-expression-contract.md)