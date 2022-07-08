# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring

import pytest
import numpy as np

from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer, ROOT_SET, \
    MINIMIZE_SET
from bingo.local_optimizers.local_opt_fitness import \
    LocalOptFitnessFunction
from bingo.symbolic_regression.explicit_regression \
    import ExplicitTrainingData as pyExplicitTrainingData, \
    ExplicitRegression as pyExplicitRegression
from bingo.symbolic_regression.agraph.agraph import AGraph as pyagraph
from bingo.symbolic_regression.agraph.operator_definitions import *
try:
    from bingocpp import ExplicitTrainingData as cppExplicitTrainingData, \
        ExplicitRegression as cppExplicitRegression, \
        Equation as cppEquation, \
        AGraph as cppagraph
    bingocpp = True
except ImportError:
    bingocpp = False

CPP_PARAM = pytest.param('Cpp',
                         marks=pytest.mark.skipif(not bingocpp,
                                                  reason='BingoCpp import '
                                                         'failure'))


@pytest.fixture(params=['Python', CPP_PARAM])
def engine(request):
    return request.param


@pytest.fixture
def explicit_training_data(engine):
    if engine == 'Python':
        return pyExplicitTrainingData
    return cppExplicitTrainingData


@pytest.fixture
def explicit_regression(engine):
    if engine == 'Python':
        return pyExplicitRegression
    return cppExplicitRegression


@pytest.fixture
def agraph_implementation(engine):
    if engine == 'Python':
        return pyagraph
    return cppagraph


@pytest.fixture
def training_data(explicit_training_data):
    x = np.arange(-10, 11).reshape(-1, 1)
    y = 2 * x**2 + 3 * x
    return explicit_training_data(x, y)


@pytest.fixture
def norm_individual(agraph_implementation):  # (1.0)((X_0)(X_0)) + (1.0)(X_0)
    individual = agraph_implementation()
    individual.command_array = np.array([[CONSTANT, -1, -1],
                                         [VARIABLE, 0, 0],
                                         [VARIABLE, 0, 0],
                                         [MULTIPLICATION, 1, 2],
                                         [MULTIPLICATION, 0, 3],
                                         [CONSTANT, -1, -1],
                                         [VARIABLE, 0, 0],
                                         [MULTIPLICATION, 5, 6],
                                         [ADDITION, 4, 7]], dtype=int)
    return individual


@pytest.fixture
def opt_individual(agraph_implementation):  # (2)((X_0)(X_0)) + (3)(X_0)
    individual = agraph_implementation()
    individual.command_array = np.array([[CONSTANT, -1, -1],
                                         [VARIABLE, 0, 0],
                                         [VARIABLE, 0, 0],
                                         [MULTIPLICATION, 1, 2],
                                         [MULTIPLICATION, 0, 3],
                                         [CONSTANT, -1, -1],
                                         [VARIABLE, 0, 0],
                                         [MULTIPLICATION, 5, 6],
                                         [ADDITION, 4, 7]], dtype=int)
    individual.set_local_optimization_params(np.array([2, 3]))
    return individual


@pytest.mark.parametrize('method', MINIMIZE_SET)
@pytest.mark.parametrize('metric', ['mae', 'mse', 'rmse'])
def test_explicit_regression_clo_linear(
        explicit_regression, training_data, metric, method,
        norm_individual, opt_individual):
    np.random.seed(1)
    fitness = explicit_regression(training_data=training_data, metric=metric)

    scipy_opt = ScipyOptimizer(fitness, method=method)
    # special options to make TNC converge at better fitness
    if method == 'TNC':
        scipy_opt.options = {'method': method,
                             'options': {'maxfun': 1000, 'xtol': 1e-16}}

    optimizer = LocalOptFitnessFunction(fitness, scipy_opt)
    optimizer(norm_individual)

    assert fitness(norm_individual) == \
           pytest.approx(fitness(opt_individual), abs=1e-5)


@pytest.mark.parametrize('method', ROOT_SET)
def test_explicit_regression_clo_linear_root(
        explicit_regression, training_data, method,
        norm_individual, opt_individual):
    np.random.seed(1)
    fitness = explicit_regression(training_data=training_data)
    optimizer = LocalOptFitnessFunction(
        fitness, ScipyOptimizer(fitness, method=method))
    optimizer(norm_individual)
    assert fitness(norm_individual) == \
           pytest.approx(fitness(opt_individual), abs=1e-5)
