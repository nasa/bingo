# Expression-Specific Fitting

Bingo retains general-purpose evolutionary optimization but removes its generic
Chromosome local-optimization protocol and `bingo.local_optimizers` package.
Continuous fitting belongs to symbolic-regression Expressions because fitting
depends on Expression constants, temporary-constant evaluation, derivatives,
and fittedness; preserving a second generic protocol would duplicate those
semantics without a demonstrated non-expression use case.

`ExplicitRegression` and `ImplicitRegression` remain opinionated public
objectives. Advanced workflows use `CustomRegression`, which independently
selects a fitting policy, fitting measure, and ranking loss over aligned
Objective data. Evidence estimation is separate from fitting, although a
successful estimator may atomically install posterior MAP constants.

## Consequences

This is a hard API switch without compatibility aliases. Generic Chromosome
local fitting has no replacement, while Expression workflows gain custom SciPy
fitting and SMC NMLL evidence estimation. Laplace NMLL and SMC NMLL denote the
same higher-is-better normalized marginal log-likelihood concept estimated by
different methods.