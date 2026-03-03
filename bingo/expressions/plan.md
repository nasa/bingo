# ideal user usage
- when performing bingo analysis, they would import an expression type, and a its corresponding generator, mutation, and crossover classes. 
- there should also be a datacontainer object that converts numpy arrays into whatever container is ideal for the expression
- each could be configed with there init
- each expression should have a scikit-learn type regressor interface, fit, predict, score
- each expression should also have a function for predicting derivatives, and hessians and/or some general form of derivative function. derivatives can be as a function of input featres(X) or of the numerical constants in the expression
- string representation should be a sympy readable string
- there should be converters to onnx and latex formats as `expr.onnx` and `expr.latex` properties. there should be a `.sympy` property, too

# general implementation details
- the score function should have several scoring functions available, mae, mse, rmse, bic, laplacenml
- constants should be accessible
- there shuold be a hash function for comparing equality


# Step 1: Pure Python AGraph expression  ✅ COMPLETED

## What was implemented

Pure-python reimplementation of the AGraph expression as a standalone class
(`AGraphExpression`) in `bingo/expressions/agraph/`.  No C++ logic, no
inheritance from the evolutionary framework (`Chromosome`/`Equation`).

### Operator renumbering (uint8-safe, no negatives)

| ID | Name | Arity | Group |
|----|------|-------|-------|
| 0 | VARIABLE | terminal | Terminals |
| 1 | CONSTANT | terminal | Terminals |
| 2 | INTEGER | terminal | Terminals |
| 3 | ADDITION | 2 | Arithmetic |
| 4 | SUBTRACTION | 2 | Arithmetic |
| 5 | MULTIPLICATION | 2 | Arithmetic |
| 6 | DIVISION | 2 | Arithmetic |
| 7 | POWER | 2 | Power |
| 8 | SAFE_POWER | 2 | Power |
| 9 | SQUARE | 1 | Power |
| 10 | CUBE | 1 | Power |
| 11 | SQRT | 1 | Power |
| 12 | ABS | 1 | Misc |
| 13 | EXPONENTIAL | 1 | Exp/Log |
| 14 | LOGARITHM | 1 | Exp/Log |
| 15 | SIN | 1 | Trig |
| 16 | COS | 1 | Trig |
| 17 | TAN | 1 | Trig |
| 18 | ARCSIN | 1 | Trig |
| 19 | ARCCOS | 1 | Trig |
| 20 | ARCTAN | 1 | Trig |
| 21 | SINH | 1 | Hyperbolic |
| 22 | COSH | 1 | Hyperbolic |
| 23 | TANH | 1 | Hyperbolic |

### Key design decisions

- **Standalone class** — `AGraphExpression` does not inherit from
  `Chromosome` or `Equation`.  Evolutionary compatibility will be added
  via an adapter in a future step.
- **Separate `_integers` tuple** — integer parameter values are stored
  externally (like constants), not in the command array itself.
  `INTEGER` operator param indexes into the integers tuple.
- **`fit()` optimises constants only** — uses
  `scipy.optimize.least_squares` (Levenberg-Marquardt).  Full evolutionary
  structure search is a separate `Regressor` class (future step).
- **`__str__` returns sympy-readable string** — different from the old
  AGraph which defaulted to "console" format.
- **Command array dtype is `uint8`** — smaller memory footprint.
- **Stack reduction only** — CAS simplification deferred.

### Files created

```
bingo/expressions/
├── __init__.py               # Re-exports AGraphExpression, DataContainer
├── plan.md                   # This file
├── data_container.py         # DataContainer(x, y) with auto-reshape, slicing
└── agraph/
    ├── __init__.py           # Re-exports AGraphExpression
    ├── expression.py         # AGraphExpression class
    ├── operators.py          # 24 operator constants + metadata maps
    ├── operator_eval.py      # Per-operator forward/reverse eval functions
    ├── evaluation.py         # evaluate(), evaluate_with_derivative()
    ├── simplification.py     # get_utilized_commands(), reduce_stack()
    ├── formatting.py         # String generation: console, latex, sympy, stack
    ├── parsing.py            # String/sympy → command_array, constants, integers
    └── onnx_interface.py     # ONNX model generation

tests/unit/expressions/
├── __init__.py
├── test_data_container.py
└── agraph/
    ├── __init__.py
    ├── test_expression.py
    ├── test_operators.py
    ├── test_operator_eval.py
    ├── test_evaluation.py
    ├── test_simplification.py
    ├── test_formatting.py
    ├── test_parsing.py
    └── test_onnx.py
```

### Scoring metrics available

- `mae` — mean absolute error
- `mse` — mean squared error
- `rmse` — root mean squared error
- `bic` and `nmll laplace`

### Test results

136 tests pass.  No regressions in existing test suite.

---

# Step 2: Genetic operators  ✅ COMPLETED

## What was implemented

Genetic operators and an evolutionary adapter for AGraph expressions.
All components work with variable-size command stacks and integrate with
bingo's evolutionary framework via `EvolvableExpression`.

### New files

```
bingo/expressions/
├── evolvable.py                    # EvolvableExpression (Chromosome adapter)
└── agraph/
    ├── component_generator.py      # ComponentGenerator (random commands)
    ├── generator.py                # AGraphGenerator (random individuals)
    ├── crossover.py                # AGraphCrossover (variable-size single-point)
    └── mutation.py                 # AGraphMutation (5 strategies)

tests/unit/expressions/
├── test_evolvable.py               # 16 tests
└── agraph/
    ├── test_component_generator.py # 18 tests
    ├── test_generator.py           # 10 tests
    ├── test_crossover.py           # 13 tests
    └── test_mutation.py            # 16 tests
```

### Key design decisions

- **EvolvableExpression adapter** — wraps `AGraphExpression` and inherits
  from `Chromosome` (fitness, genetic_age, fit_set, copy, __str__, distance).
  Also implements `needs_local_optimization`, `get_number_local_optimization_params`,
  `set_local_optimization_params` for constant tuning.
- **No INTEGER in generation** — `ComponentGenerator` only generates
  VARIABLE and CONSTANT terminals. INTEGERs are preserved through
  crossover/mutation but never randomly introduced.
- **Variable-size stacks** — generator picks random size in
  `[min_size, max_size]`; crossover handles different-length parents;
  fork mutation can grow the stack up to `max_size`.
- **`preserve_constants=True`** in crossover — parent constant values
  are carried to children with proper index remapping.
- **5 mutation types** — command, node, parameter, prune, fork. Fork
  appends rows when below `max_size`; repurposes unutilized rows at capacity.
- **`simplify()` method on AGraphExpression** — replaces raw command
  array with the simplified version, renumbers constants/integers.

### Test results

250 tests pass. No regressions in existing test suite.

# Step 3: C++ subpackage (NOT YET STARTED)

- `bingo/expressions/agraph_cpp/` — explicit C++ backend subpackage
- Clear separation: users import from `agraph` (python) or `agraph_cpp`
- Operator IDs shared between Python and C++ via the same `operators.py`

# Step 4: CAS simplification (NOT YET STARTED)

- Port the CAS simplification backend from
  `bingo/symbolic_regression/agraph/simplification_backend/`
- Add `simplify()` alongside `reduce()` in
  `bingo/expressions/agraph/simplification.py`

# Step 5: Advanced features (NOT YET STARTED)

- Hessian computation (second-order derivatives)
- General derivative function interface