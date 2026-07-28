# Bingo

Bingo evolves and simplifies symbolic mathematical expressions for regression and general-purpose evolutionary optimization.

## Expression Simplification

**Bounded constant folding**:
A deterministic constant-folding policy that preserves exhaustive folding for expressions with at most seven distinct constants and uses a distinct-first local approximation from eight constants onward.
_Avoid_: unbounded folding, complete folding

**Exhaustive constant folding**:
The existing policy that searches every non-empty subset of an expression's distinct constants in pursuit of the fewest `CONSTANT` nodes.
_Avoid_: greedy folding

Example dialogue:

> Developer: "This expression has twelve distinct constants, so it uses bounded constant folding."
>
> Domain expert: "Correct. Seven or fewer uses exhaustive constant folding; eight or more uses a distinct-first local policy for predictable simplification cost."

## Symbolic Regression

**Expression**:
A standalone mathematical model that represents a candidate symbolic
relationship and owns fitting, scoring, prediction, and derivative operations.
_Avoid_: equation regressor

**Explicit regression**:
Symbolic regression that searches for an Expression whose predicted output
matches a target value for each input.

**Implicit regression**:
Symbolic regression that searches for an Expression whose input gradient is
consistent with observed trajectory derivatives.

**Required parameters**:
An optional implicit-regression anti-triviality guard. An Expression passes
when at least one sample has at least the requested number of derivative
components with magnitude greater than $10^{-16}$; otherwise its loss is
infinite.

**Objective data**:
Private, indexable aligned arrays retained by a regression objective so Bingo
can evaluate a common subset of samples across a population.
_Avoid_: public training-data API

**Score**:
A higher-is-better measure of an Expression's quality. Explicit regression
uses $R^2$ by default and may use Laplace NMLL; implicit regression uses the
negative of its aggregate loss. Non-finite expression evaluation returns a
negative infinite score.
_Avoid_: lower-is-better score

**Loss**:
A lower-is-better objective used by Bingo fitness evaluation. Laplace NMLL
loss is the negation of the corresponding score. Non-finite expression
evaluation returns an infinite loss.

**LM fitting**:
Constant fitting that always minimizes an Expression's ordinary residual vector.
It is independent of the loss used to rank evolved Expressions.

**Fitted expression**:
An Expression with no optimizable constants, or one whose applicable fitting
method has been attempted for its current raw structure. Numerical
non-convergence does not unset this state; a raw structural change does.
Fittedness is independent of the data subset used for the fitting attempt.
Direct constant assignment does not establish fittedness.
Serialization preserves fittedness exactly.

**Relative MSE**:
An explicit-regression loss that averages squared residuals normalized by the
corresponding target value. It is available as ``relative_mse`` only and
rejects zero-valued targets.

**Correlation loss**:
An explicit-regression loss derived from the correlation coefficient between
an Expression's predictions and the observed targets. It assesses association
without applying a hidden output transformation to the Expression.
