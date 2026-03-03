# Bingo Copilot Instructions

## Repository Overview

Bingo is an open-source Python package developed by NASA for symbolic regression and general-purpose evolutionary optimization using genetic programming. The project implements parallel island evolution strategies with MPI support and includes optional C++ performance optimizations via BingoCpp.

## Technology Stack

### Primary Language & Version
- **Python**: >=3.9 (supports 3.9, 3.10, 3.11, 3.12)
- **C++**: Used for performance-critical components via pybind11

### Core Dependencies
- `numpy`: Numerical computations
- `scipy`: Scientific computing
- `mpi4py>=4.0`: Parallel computing with MPI
- `sympy`: Symbolic mathematics
- `scikit-learn>=1.1`: Machine learning utilities
- `dill>=0.2.9`: Serialization
- `smcpy>=0.1.4`: Statistical methods
- `pybind11`: Python-C++ bindings

### Build System
- **setuptools** with CMake for C++ extensions
- **pytest** for testing
- **Sphinx** for documentation

## Project Structure

```
bingo/
├── bingo/                          # Main Python package
│   ├── chromosomes/                # Genetic representation
│   ├── evaluation/                 # Fitness evaluation
│   ├── evolutionary_algorithms/    # EA implementations
│   ├── evolutionary_optimizers/    # Optimization strategies (Island, Archipelago)
│   ├── local_optimizers/           # Local optimization methods
│   ├── selection/                  # Selection operators (Tournament, etc.)
│   ├── stats/                      # Statistics and tracking
│   ├── symbolic_regression/        # Symbolic regression specific code
│   │   ├── agraph/                # Acyclic graph representation
│   │   └── benchmarking/          # Benchmark problems
│   ├── util/                       # Utilities
│   └── variation/                  # Genetic operators (crossover, mutation)
├── bingocpp/                       # C++ performance library (submodule)
├── tests/                          # Test suite
├── examples/                       # Usage examples
├── docs/                           # Sphinx documentation
└── .github/workflows/              # CI/CD pipelines
```

## Coding Standards

### Python Style
- Follow **PEP 8** conventions
- Use **numpy-style docstrings** (configured in `.pydocstyle`)
- Maintain compatibility with Python 3.9+
- Use type hints where appropriate but not required

### Linting & Code Quality
- **pylint**: Configuration in `.pylintrc`
- **pydocstyle**: Configuration in `.pydocstyle`
  - Convention: numpy
  - Ignored rules: D204, D401, D400, D205, D404, D104, D105, D210, D403, D200, D209

### Documentation
- Use numpy-style docstrings for all public APIs
- Include docstring sections: Parameters, Returns, Raises, Examples where applicable
- Documentation built with Sphinx and hosted at https://nasa.github.io/bingo/

## Design Patterns & Architecture

### Evolutionary Algorithm Components
- **Abstract base classes**: Use ABCMeta for interfaces (e.g., `FitnessFunction`, `Variation`)
- **Genetic operators**: Implement as callable classes with `__call__` method
- **Variation strategies**: 
  - `VarOr`: Mutation OR crossover OR replication
  - `VarAnd`: Mutation AND crossover
- **Fitness evaluation**: Vector-based with aggregation metrics (MAE, RMSE, MSE, etc.)

### Code Organization Principles
- Each module should have a clear, single responsibility
- Use composition over inheritance where practical
- Maintain separation between evolutionary framework and symbolic regression specifics
- Support both serial and parallel execution modes

## Testing Requirements

### Test Framework & Tools
- **Runner**: `pytest` (primary), executing via `pytest tests` or `tests/.run_tests.sh`
- **Mocking**: `pytest-mock` (use `mocker` fixture, avoid `unittest.mock.patch` decorators)
- **Coverage**: `pytest-cov` (target >90% coverage for Python code)
- **Timeouts**: `pytest-timeout` (prevent hanging specific tests, esp. with MPI)

### Directory Structure & Naming
- **Location**: `tests/` directory at project root
- **Organization**: Mirror `bingo/` package structure in `tests/unit/`
  - Example: `bingo/symbolic_regression/agraph/` -> `tests/unit/symbolic_regression/agraph/`
- **File Naming**: `test_*.py` (e.g., `test_agraph.py`)
- **Function Naming**: `test_*` (descriptive usage, e.g., `test_crossover_maintains_size`)
- **Class Naming**: `Test*` (CamelCase, e.g., `class TestAGraph:`)

### Best Practices
- **Fixtures**: Use `pytest` fixtures (`@pytest.fixture`) over `setUp`/`tearDown`
  - Define shared fixtures in `conftest.py` at appropriate levels
  - Use appropriate scopes (`function`, `module`, `session`) to optimize runtime
- **Parametrization**: Use `@pytest.mark.parametrize` for data-driven tests
  - Provide `ids` for readable test output
- **Arrange-Act-Assert**: Structure tests clearly

### Mocking Guidelines
- **Target**: Mock where objects are *used*, not where they are defined
- **MPI**: Mock `mpi4py` interactions for unit tests; use real MPI only in integration tests
- **C++ Extensions**: 
  - Test Python bindings behavior, not internal C++ logic
  - Verify consistency between C++ implementation and Python fallback

### Test Categories
- **Unit**: Fast, isolated, run in-memory (`tests/unit/`)
- **Integration**: Component interaction, real MPI execution (`tests/integration/`)
- **Performance**: Benchmarks and end-to-end runs (`tests/performance_benchmarking/`)

## Build & Development

### Installation from Source
```bash
git clone --recurse-submodules https://github.com/nasa/bingo.git
cd bingo
pip install -r requirements.txt
./.build_bingocpp.sh  # Optional: build C++ extensions
pytest tests
```

### Building C++ Extensions
- BingoCpp is a git submodule providing performance optimizations
- Built via CMake integration in setup.py
- Optional but recommended for performance
- If build fails, Python fallback implementations are used

### Environment Setup
- Set `PYTHONPATH` to bingo root directory for development
- Set `RDMAV_FORK_SAFE=1` for MPI environments
- Use conda or pip for dependency management

## Key Components & Usage

### Symbolic Regression
- Primary use case: symbolic regression via genetic programming
- **SymbolicRegressor**: scikit-learn compatible interface
- **AGraph**: Acyclic graph representation of equations
- Supports explicit and implicit regression
- Includes benchmark suite from literature (Koza, Nguyen problems)

### Evolutionary Optimization
- **Island**: Single population evolutionary algorithm
- **Archipelago**: Multiple islands with migration (serial or parallel)
- **FitnessPredictor**: Coevolution of fitness prediction models

### Parallel Computing
- MPI support via mpi4py for distributed evolution
- Parallel archipelago implementations
- Consider thread safety in parallel contexts

## Constraints & Conventions

### DO
- Use numpy arrays for numerical data
- Implement genetic operators as reusable, composable components
- Support serialization with dill for distributed computing
- Follow the abstract base class patterns in the codebase
- Write comprehensive docstrings for public APIs
- Ensure thread-safety for parallel operations

### DON'T
- Don't break backward compatibility without careful consideration
- Don't add dependencies without strong justification
- Don't modify working code unnecessarily
- Don't remove or skip existing tests
- Don't use Python features requiring >3.9 without updating requirements
- Don't commit sensitive data or credentials

## CI/CD

### GitHub Actions Workflow
- Runs on every push and pull request
- Python 3.13 used for CI
- Steps: checkout (with submodules) → install MPI → install deps → build BingoCpp → run tests → coverage
- Must pass tests and coverage checks

### What CI Checks
- All pytest tests pass
- Code coverage reported to Coveralls
- BingoCpp builds successfully (with fallback if it fails)

## Security & Compliance

### License
- **Apache License 2.0**
- NASA Open Source Agreement applies
- No copyright claimed by US Government under Title 17, U.S. Code
- All contributions must respect this license

### Security Practices
- Don't introduce security vulnerabilities
- Be cautious with user input in symbolic expressions
- Avoid arbitrary code execution in equation evaluation
- Use safe serialization practices

## Common Tasks

### Adding a New Genetic Operator
1. Create a new class in appropriate module (e.g., `bingo/variation/`)
2. Inherit from relevant base class (e.g., `Variation`)
3. Implement required abstract methods
4. Add comprehensive docstrings
5. Write unit tests in `tests/unit/variation/`
6. Add usage examples if applicable

### Adding a New Fitness Function
1. Create class inheriting from `FitnessFunction` or `VectorBasedFunction`
2. Implement `evaluate_fitness_vector(self, individual)` method
3. Add docstring explaining the metric
4. Add tests in `tests/unit/evaluation/`

### Adding a Benchmark Problem
1. Define function in `bingo/symbolic_regression/benchmarking/benchmark_definitions.py`
2. Name with `bench_` prefix
3. Return a `Benchmark` object
4. Include source reference in docstring
5. Add test case

## Additional Resources

- **Documentation**: https://nasa.github.io/bingo/
- **Repository**: https://github.com/nasa/bingo
- **Paper**: Randall et al. "Bingo: a customizable framework for symbolic regression with genetic programming." GECCO 2022.

## Questions & Clarifications

When in doubt:
- Check existing similar implementations in the codebase
- Refer to the online documentation
- Follow established patterns for evolutionary algorithms
- Prioritize code clarity and maintainability
- Ask for clarification on ambiguous requirements
