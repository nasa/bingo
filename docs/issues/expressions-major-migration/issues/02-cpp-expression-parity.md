---
title: Match C++ Expression semantics
category: enhancement
state: ready-for-agent
blocked_by:
  - 01-python-expression-contract
created: 2026-07-28
---

## Parent

[Expressions major migration work brief](../WORK-BRIEF.md)

## What to build

Bring the C++ AGraphExpression implementation and Python bindings to semantic
parity with the approved Python Expression contract. Users selecting either
available backend receive matching explicit and implicit evaluation, fitting,
derivative, loss, score, lifecycle, serialization, and numerical-failure
behavior.

Add backend-parametrized fixed-fixture tests. C++ cases may skip in local
environments without the extension, while CI requires them after building the
extension.

## Acceptance criteria

- [ ] The C++ backend exposes the same public Expression operations and
  keyword-only fitting tolerance as the Python backend.
- [ ] Fixed expressions and data produce numerically equivalent predictions,
  gradients, fitted constants, explicit/implicit losses and scores within the
  documented tolerance.
- [ ] Both backends agree on `is_fitted`, copying, serialization, and
  normalized non-finite public outcomes.
- [ ] Backend-parametrized tests always execute Python cases and skip only
  unavailable local C++ cases.
- [ ] The C++ extension builds and its parity tests pass in CI.

## Blocked by

- [Define Python Expression lifecycle and scoring](01-python-expression-contract.md)
