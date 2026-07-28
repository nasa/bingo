---
title: Retire legacy symbolic-regression APIs
category: enhancement
state: ready-for-agent
blocked_by:
  - 04-symbolic-regressor-expressions
created: 2026-07-28
---

## Parent

[Expressions major migration work brief](../WORK-BRIEF.md)

## What to build

Complete the major-version cutover once the Expression-backed workflow is
demonstrably complete. Remove legacy AGraph/Equation imports and public
exports, public training-data APIs, specialized legacy regression modes, and
their obsolete tests and C++ declarations. Do not provide compatibility aliases
or old-checkpoint migration.

Publish migration guidance that identifies the new public surfaces, renamed
settings, removed APIs, score/loss semantics, backend behavior, and the
serialization break. Audit examples, documentation, and optional modules for
imports of retired surfaces.

## Acceptance criteria

- [ ] Public symbolic-regression exports contain only the retained
  `SymbolicRegressor`, `ExplicitRegression`, and `ImplicitRegression` surface;
  Expression generation and variation live under `bingo.expressions`.
- [ ] Legacy AGraph/Equation implementations, public training-data APIs,
  `PairwiseAtomicPotential`, `PairwiseAtomicTrainingData`, and
  `ImplicitRegressionSchmidt` are removed with their obsolete tests and unused
  C++ declarations.
- [ ] No repository examples, docs, tests, or optional modules import retired
  public surfaces.
- [ ] Migration documentation and README guidance explain replacement imports,
  renamed settings, Loss/Score polarity, and checkpoint incompatibility.
- [ ] The full Python suite passes; CI builds C++ and runs required
  backend-parity and estimator integration coverage.

## Blocked by

- [Run SymbolicRegressor on Expressions](04-symbolic-regressor-expressions.md)