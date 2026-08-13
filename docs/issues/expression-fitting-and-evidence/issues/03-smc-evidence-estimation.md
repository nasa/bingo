---
title: Deliver SMC Evidence estimation
category: enhancement
state: ready-for-agent
blocked_by:
  - 02-custom-regression-fitting
created: 2026-08-13
---

## Parent

[Expression fitting and Evidence work brief](../WORK-BRIEF.md)

## What to build

Deliver sequential-Monte-Carlo Evidence estimation as a symbolic-regression
capability rather than a generic local optimizer. Port the Laplace proposal
algorithm and validated numerical behavior from the `smc_laplace` branch into
`SmcEvidenceEstimator`, adapting it to pure fitting measures, temporary
Expression constants, and the shared fit lifecycle instead of carrying over
the legacy Chromosome protocol.

The estimator returns a structured `EvidenceResult` containing
higher-is-better `smc_nmll`, posterior MAP constants, status, and diagnostics.
Successful estimation atomically installs the final posterior MAP constants
after Evidence calculation. Proposal or sampling failure preserves the
pre-estimation fitted constants and becomes infinite Loss when used for
ranking.

SMC is reproducible by default across serial and multiprocessing evaluation.
A root seed derives a stable child generator from a backend-neutral digest of
the raw Expression structure, excluding constants and fitness; `seed=None`
permits non-reproducible sampling. Posterior samples are omitted by default and
returned only when explicitly requested, never attached automatically to an
Expression.

## Acceptance criteria

- [ ] `SmcEvidenceEstimator` and `EvidenceResult` implement the public Evidence
  contract without depending on generic local-optimization methods.
- [ ] The estimator ports the `smc_laplace` Hessian-based proposal behavior and
  handles singular or indefinite covariance estimates with tested numerical
  regularization.
- [ ] `smc_nmll` is higher-is-better normalized marginal log-likelihood and is
  visibly distinct from the existing `laplace_nmll` approximation; the ranking
  adapter negates it for Bingo's lower-is-better fitness contract.
- [ ] Successful estimation installs only posterior MAP Expression constants,
  excluding nuisance parameters such as noise standard deviation, after
  Evidence calculation completes.
- [ ] Proposal and sampling failures return an unsuccessful result, preserve
  pre-SMC constants, and rank as infinite Loss.
- [ ] A fixed root seed and raw structure produce reproducible streams across
  Python and C++ Expressions and across serial and multiprocessing evaluation;
  current constants and fitness do not affect derivation.
- [ ] `seed=None` enables intentionally non-reproducible estimation.
- [ ] `return_posterior=False` omits posterior samples by default;
  `return_posterior=True` returns them directly without attaching results to
  Expressions or ordinary evolutionary populations.
- [ ] Evidence results and configured estimators support the documented
  checkpoint and multiprocessing behavior when their retained diagnostics are
  pickleable.
- [ ] `SmcEvidenceEstimator` and `EvidenceResult` are exported from the
  symbolic-regression package.

## Blocked by

- [02 - Deliver customizable regression fitting](02-custom-regression-fitting.md)