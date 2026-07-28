---
title: Run SymbolicRegressor on Expressions
category: enhancement
state: ready-for-agent
blocked_by:
  - 02-cpp-expression-parity
  - 03-expression-regression-objectives
created: 2026-07-28
---

## Parent

[Expressions major migration work brief](../WORK-BRIEF.md)

## What to build

Make `SymbolicRegressor` run end-to-end on the active Expression backend and
the new generation and variation surfaces. It remains the sole scikit-learn
integration point, validates estimator input, tracks feature count, and
configures retained explicit regression with the selected loss and fitting
tolerance.

The new estimator surface uses `min_stack_size`, `max_stack_size`,
`simplification`, `loss`, and `fit_tolerance=1e-5`; it removes legacy
generation and external-local-optimization settings.

## Acceptance criteria

- [ ] `SymbolicRegressor` creates and evolves `EvolvableExpression` using the
  active Expression backend and Expression-native generation and variation.
- [ ] Estimator validation and feature-count behavior remain scikit-learn
  compatible, while Expressions remain framework-neutral.
- [ ] The public estimator exposes the selected generation, simplification,
  loss, and fitting-tolerance settings; `stack_size`, `use_simplification`,
  `metric`, `clo_alg`, and `clo_threshold` are absent.
- [ ] An end-to-end explicit-regression run succeeds for Python and, when
  available, C++ backends.
- [ ] Estimator tests demonstrate the selected score/loss contract and do not
  route fitting through generic external local optimization.

## Blocked by

- [Match C++ Expression semantics](02-cpp-expression-parity.md)
- [Evaluate Expressions through regression Objectives](03-expression-regression-objectives.md)
