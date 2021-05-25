import pytest
import numpy as np

from bingo.local_optimizers.continuous_local_opt import MINIMIZE_SET, ContinuousLocalOptimization

from bingo.symbolic_regression.explicit_regression \
    import ExplicitTrainingData as pyExplicitTrainingData, \
    ExplicitRegression as pyExplicitRegression
from bingo.symbolic_regression.equation import Equation as pyEquation
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
    x = np.arange(1, 10).reshape(-1, 1)
    y = 2 * x + 3
    return explicit_training_data(x, y)


@pytest.fixture
def norm_individual(agraph_implementation):
    individual = agraph_implementation()
    individual.command_array = np.array([[CONSTANT, -1, -1],
                                         [VARIABLE, 0, 0],
                                         [MULTIPLICATION, 0, 1],
                                         [CONSTANT, -1, -1],
                                         [ADDITION, 2, 3]], dtype=int)
    return individual


@pytest.fixture
def opt_individual(agraph_implementation):
    individual = agraph_implementation()
    individual.command_array = np.array([[CONSTANT, -1, -1],
                                         [VARIABLE, 0, 0],
                                         [MULTIPLICATION, 0, 1],
                                         [CONSTANT, -1, -1],
                                         [ADDITION, 2, 3]], dtype=int)
    individual.set_local_optimization_params(np.array([2, 3]))
    return individual


@pytest.mark.parametrize('algorithm', MINIMIZE_SET)
def test_explicit_regression_clo_linear_mae(explicit_regression, training_data, algorithm,
                                            norm_individual, opt_individual):
    fitness = explicit_regression(training_data=training_data, metric="mae")
    optimizer = ContinuousLocalOptimization(fitness, algorithm)
    optimizer(norm_individual)
    assert fitness(norm_individual) == pytest.approx(fitness(opt_individual), abs=1e-5)


@pytest.mark.parametrize('algorithm', MINIMIZE_SET)
def test_explicit_regression_clo_linear_mse(explicit_regression, training_data, algorithm,
                                            norm_individual, opt_individual):
    fitness = explicit_regression(training_data=training_data, metric="mse")
    optimizer = ContinuousLocalOptimization(fitness, algorithm)
    optimizer(norm_individual)
    assert fitness(norm_individual) == pytest.approx(fitness(opt_individual))


@pytest.mark.parametrize('algorithm', MINIMIZE_SET)
def test_explicit_regression_clo_linear_rmse(explicit_regression, training_data, algorithm,
                                            norm_individual, opt_individual):
    fitness = explicit_regression(training_data=training_data, metric="rmse")
    optimizer = ContinuousLocalOptimization(fitness, algorithm)
    optimizer(norm_individual)
    assert fitness(norm_individual) == pytest.approx(fitness(opt_individual), abs=1e-5)
