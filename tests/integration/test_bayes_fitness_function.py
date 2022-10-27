# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
from time import time
import pytest
import numpy as np

from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer, ROOT_SET, \
    MINIMIZE_SET
from bingo.local_optimizers.local_opt_fitness import \
    LocalOptFitnessFunction
from bingo.local_optimizers.smcpy_optimizer import SmcpyOptimizer
from bingo.symbolic_regression.bayes_fitness_function \
    import BayesFitnessFunction
from bingo.symbolic_regression.explicit_regression \
    import ExplicitTrainingData as pyExplicitTrainingData, \
    ExplicitRegression as pyExplicitRegression
from bingo.symbolic_regression.agraph.agraph import AGraph as pyagraph, force_use_of_python_backends, evaluation_backend
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
HYBRID_PARAM = pytest.param('Hybrid',
                         marks=pytest.mark.skipif(not bingocpp,
                                                  reason='BingoCpp import '
                                                         'failure'))


@pytest.fixture(params=[CPP_PARAM, HYBRID_PARAM, 'Python'])
def engine(request):
    return request.param


@pytest.fixture
def explicit_training_data(engine):
    if engine in ['Python']:
        return pyExplicitTrainingData
    return cppExplicitTrainingData


@pytest.fixture
def explicit_regression(engine):
    if engine in ['Python', 'Hybrid']:
        return pyExplicitRegression
    return cppExplicitRegression


@pytest.fixture
def agraph_implementation(engine):
    if engine == 'Python':
        force_use_of_python_backends()
        return pyagraph
    if engine == 'Hybrid':
        return pyagraph
    return cppagraph


@pytest.fixture
def training_data(explicit_training_data):
    # np.random.seed(0)
    x = np.linspace(0, np.pi*1.5, 100).reshape(-1, 1)
    y = 2*np.sin(x) + 3 
    noise_std = np.mean(np.abs(y)) * 0.5
    y += np.random.normal(0, noise_std, y.shape)
    return explicit_training_data(x, y)

@pytest.fixture
def true_equation(agraph_implementation):
    true_commands = np.array([[0, 0, 0],
                              [1, 0, 0],
                              [1, 1, 1],
                              [6, 0, 0],
                              [4, 1, 3],
                              [2, 4, 2]])
    true_equ = agraph_implementation(use_simplification=True)
    true_equ.command_array = true_commands
    return true_equ



def test_bayes_fitness_function(explicit_regression, training_data, true_equation, engine):
    # np.random.seed(0)

    fitness = explicit_regression(training_data=training_data, metric="mse")
    scipy_opt = ScipyOptimizer(fitness, method="lm")
    # optimizer = LocalOptFitnessFunction(fitness, scipy_opt)
    # bff = BayesFitnessFunction(optimizer,
    #                            num_particles=500,
    #                            mcmc_steps=10,
    #                            num_multistarts=8)


    smcoptimizer = SmcpyOptimizer(fitness, scipy_opt, 
                                  num_particles=500,
                                  mcmc_steps=10,
                                  num_multistarts=8)

    

    # bff(true_equation)
    # print(f"bff: {true_equation}")

    t0=time()
    smcoptimizer(true_equation)
    t1=time()
    print(f"smc ({t1-t0}s): {true_equation}")
    print(f"Engine: {engine}")
    from bingo.symbolic_regression.agraph.agraph import evaluation_backend
    print(f"backend: {evaluation_backend.ENGINE}")
    

    assert False