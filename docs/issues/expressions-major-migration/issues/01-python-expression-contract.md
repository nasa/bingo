---
title: Define Python Expression lifecycle and scoring
category: enhancement
state: ready-for-agent
blocked_by: []
created: 2026-07-28
---

## Parent

[Expressions major migration work brief](../WORK-BRIEF.md)

## What to build

Deliver a complete, directly usable pure-Python `AGraphExpression` contract.
The Expression remains framework-neutral while owning explicit and implicit
evaluation, fitting, derivatives, loss, and score behavior. It exposes the
selected `is_fitted` lifecycle, preserves that lifecycle through copying and
serialization, and normalizes non-finite public evaluation results.

The slice includes user-facing documentation for this Expression surface and
focused tests that demonstrate the behavior without requiring evolution or the
C++ backend.

## Acceptance criteria

- [ ] `AGraphExpression` supports explicit and implicit fitting, prediction,
  derivatives, higher-is-better score, and lower-is-better loss with the
  selected score/loss vocabulary.
- [ ] `is_fitted` follows the approved structure-only lifecycle: raw changes
  invalidate it; fitting attempts establish it; direct simplified-constant
  assignment preserves but cannot establish it; copying and serialization
  preserve it exactly.
- [ ] Fitting uses keyword-only `tolerance`; legacy fit metrics and arbitrary
  public solver options are absent.
- [ ] Non-finite public evaluation normalizes to infinite loss and negative
  infinite score, with focused unit coverage for explicit and implicit paths.
- [ ] Expression tests pass with the Python backend, including score/loss,
  fitting, lifecycle, and serialization cases.

## Blocked by

None - can start immediately.