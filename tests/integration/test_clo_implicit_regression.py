import pytest
import numpy as np

from bingo.local_optimizers.continuous_local_opt import MINIMIZE_SET, ContinuousLocalOptimization

from bingo.symbolic_regression.implicit_regression \
    import ImplicitTrainingData as pyImplicitTrainingData, \
    ImplicitRegression as pyImplicitRegression
from bingo.symbolic_regression.agraph.agraph import AGraph as pyagraph
from bingo.symbolic_regression.agraph.operator_definitions import *
try:
    from bingocpp import ImplicitTrainingData as cppImplicitTrainingData, \
        ImplicitRegression as cppImplicitRegression, \
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
def implicit_training_data(engine):
    if engine == 'Python':
        return pyImplicitTrainingData
    return cppImplicitTrainingData


@pytest.fixture
def implicit_regression(engine):
    if engine == 'Python':
        return pyImplicitRegression
    return cppImplicitRegression


@pytest.fixture
def agraph_implementation(engine):
    if engine == 'Python':
        return pyagraph
    return cppagraph


@pytest.fixture
def training_data(implicit_training_data):
    # x = np.arange(1, 10, 2).reshape(-1, 1)
    # dx_dt = np.ones((len(x), 1))
    # x = np.hstack((np.arange(10).reshape((-1, 1)), np.zeros((10, 1))))
    # dx_dt = np.array([[1, 0]] * len(x))
    x = np.hstack((np.arange(10).reshape((-1, 1)), (np.arange(10).reshape((-1, 1)))))
    dx_dt = np.array([[1, 1]] * len(x))
    return implicit_training_data(x, dx_dt)


@pytest.fixture
def norm_individual(agraph_implementation):
    individual = agraph_implementation()
    individual.command_array = np.array([[CONSTANT, -1, -1],
                                         [VARIABLE, 0, 0],
                                         [MULTIPLICATION, 0, 1],
                                         [CONSTANT, -1, -1],
                                         [VARIABLE, 1, 1],
                                         [MULTIPLICATION, 3, 4],
                                         [ADDITION, 2, 5]], dtype=int)
    return individual


@pytest.fixture
def opt_individual(agraph_implementation):
    individual = agraph_implementation()
    individual.command_array = np.array([[CONSTANT, -1, -1],
                                         [VARIABLE, 0, 0],
                                         [MULTIPLICATION, 0, 1],
                                         [CONSTANT, -1, -1],
                                         [VARIABLE, 1, 1],
                                         [MULTIPLICATION, 3, 4],
                                         [ADDITION, 2, 5]], dtype=int)
    individual.set_local_optimization_params(np.array([-1, 1]))
    return individual


@pytest.mark.parametrize('algorithm', ['Nelder-Mead'])
def test_implicit_regression_clo_linear_mae(implicit_regression, training_data, algorithm,
                                            norm_individual, opt_individual):
    fitness = implicit_regression(training_data=training_data)
    print(fitness.evaluate_fitness_vector(opt_individual))
    print(fitness(opt_individual))
    optimizer = ContinuousLocalOptimization(fitness, algorithm)
    optimizer(norm_individual)
    assert fitness(norm_individual) == pytest.approx(fitness(opt_individual), abs=1e-5)

# TODO test required params case
# TODO test divide by 0 case (e.g. df/dt is all 0s?)
