---
title: Retire generic local optimization
category: enhancement
state: ready-for-agent
blocked_by:
  - 02-custom-regression-fitting
  - 03-smc-evidence-estimation
created: 2026-08-13
---

## Parent

[Expression fitting and Evidence work brief](../WORK-BRIEF.md)

## What to build

Complete the hard API switch to Expression-specific fitting after custom
regression and Evidence estimation provide the retained use cases. Remove the
generic local-optimizer package, its Chromosome parameter protocol, and every
repository-owned consumer without compatibility modules or aliases.

Generic evolutionary optimization remains supported. Its zero-minimization
examples and tutorials continue to demonstrate evolution but no longer perform
continuous local fitting. Expression users migrate to `ExplicitRegression`,
`ImplicitRegression`, or `CustomRegression`; generic Chromosome local fitting
has no replacement.

Move SMCPy out of core dependencies into an optional `evidence` extra, remove
generic metrics coupled to local-parameter counts, update public exports and
migration documentation, and validate the resulting source and installed
distributions.

## Acceptance criteria

- [ ] `bingo.local_optimizers` and its tests are deleted, and importing it fails
  normally without compatibility stubs.
- [ ] Local-optimization methods are removed from the Chromosome hierarchy,
  including parameter getters/setters, optimization predicates, and
  `needs_opt_list` generator configuration.
- [ ] Generic `VectorBasedFunction` metrics `"negative nmll laplace"` and
  `"bic"` are removed with their parameter-count dependencies; Expression
  `laplace_nmll` remains supported.
- [ ] Generic zero-minimization examples and tutorials run without local
  fitting and continue to demonstrate general-purpose evolutionary
  optimization.
- [ ] SMCPy is absent from required dependencies and available through the
  optional `evidence` extra; core Bingo imports and symbolic regression work
  without SMCPy installed.
- [ ] Migration documentation maps each deleted public surface to its
  Expression replacement or states explicitly that no replacement exists.
- [ ] Public exports include the new fitting and Evidence APIs and contain no
  deleted local-optimizer symbols.
- [ ] A repository-wide audit finds no stale imports, protocol calls, metric
  names, `needs_opt_list` usage, or documentation instructions outside the
  migration record.
- [ ] The full Python suite and C++ parity suite pass, documentation and examples
  validate, and distribution checks cover both core installation and the
  `evidence` extra.

## Blocked by

- [02 - Deliver customizable regression fitting](02-custom-regression-fitting.md)
- [03 - Deliver SMC Evidence estimation](03-smc-evidence-estimation.md)