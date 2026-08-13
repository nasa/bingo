---
title: Build Expression fitting foundations
category: enhancement
state: ready-for-agent
blocked_by: []
created: 2026-08-13
---

## Parent

[Expression fitting and Evidence work brief](../WORK-BRIEF.md)

## What to build

Establish the shared data and lifecycle foundations needed by every retained
regression Objective. Users can represent arbitrary aligned Objective arrays,
subset them consistently across a population, and rely on matching Python and
C++ Expression behavior when fitted constants are committed or fitting is
explicitly reset.

Objective data validates equal first-axis lengths, exposes its arrays as a
tuple, and applies an index to every aligned array. Existing explicit and
implicit Objectives adopt this shared behavior without changing their public
inputs or subset semantics.

Expressions gain an atomic operation that validates and commits fitted
constants and an explicit operation that clears fittedness. Fittedness remains
structure-only: direct constant assignment cannot establish it, raw structural
changes clear it, and copying and serialization preserve it exactly.

## Acceptance criteria

- [ ] `ObjectiveData` accepts arbitrary aligned arrays, rejects mismatched
  first-axis lengths, exposes the arrays as a tuple, and returns aligned
  `ObjectiveData` when indexed.
- [ ] Explicit and implicit Objective data use the shared container behavior
  while retaining semantic access to their selected inputs and their current
  `RandomSubsetEvaluation` behavior.
- [ ] Python and C++ Expressions validate exact constant count and finite
  numeric values before atomically committing fitted constants.
- [ ] Invalid fit commits leave constants and fittedness unchanged; valid fit
  commits establish fittedness for the current raw structure.
- [ ] `clear_fit()` unsets fittedness for Expressions with optimizable
  constants and remains a no-op for Expressions without them.
- [ ] Python and C++ tests demonstrate matching fit commit, reset, raw mutation,
  direct assignment, copying, and serialization behavior.
- [ ] Existing explicit, implicit, and Expression tests remain green.

## Blocked by

None - can start immediately.