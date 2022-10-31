# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
from time import time
import pytest
import numpy as np

from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.local_optimizers.smcpy_optimizer import SmcpyOptimizer

from bingo.symbolic_regression.explicit_regression import (
    ExplicitTrainingData as pyExplicitTrainingData,
    ExplicitRegression as pyExplicitRegression,
)
from bingo.symbolic_regression.agraph.agraph import (
    AGraph as pyagraph,
    force_use_of_python_backends,
)
from bingo.symbolic_regression.agraph.operator_definitions import *
from bingo.symbolic_regression import AGraphGenerator, ComponentGenerator

try:
    from bingocpp import (
        ExplicitTrainingData as cppExplicitTrainingData,
        ExplicitRegression as cppExplicitRegression,
        AGraph as cppagraph,
    )

    bingocpp = True
except ImportError:
    bingocpp = False

CPP_PARAM = pytest.param(
    "Cpp",
    marks=pytest.mark.skipif(
        not bingocpp, reason="BingoCpp import " "failure"
    ),
)
HYBRID_PARAM = pytest.param(
    "Hybrid",
    marks=pytest.mark.skipif(
        not bingocpp, reason="BingoCpp import " "failure"
    ),
)


@pytest.fixture(params=[CPP_PARAM, HYBRID_PARAM, "Python"])
def engine(request):
    return request.param


@pytest.fixture
def explicit_training_data(engine):
    if engine in ["Python", "Hybrid"]:
        return pyExplicitTrainingData
    return cppExplicitTrainingData


@pytest.fixture
def explicit_regression(engine):
    if engine in ["Python", "Hybrid"]:
        return pyExplicitRegression
    return cppExplicitRegression


@pytest.fixture
def agraph_implementation(engine):
    if engine == "Python":
        force_use_of_python_backends()
        return pyagraph
    if engine == "Hybrid":
        return pyagraph
    return cppagraph


@pytest.fixture
def training_data(explicit_training_data):
    np.random.seed(1)
    x = np.linspace(0, np.pi * 1.5, 5).reshape(-1, 1)
    y = 2 * np.sin(x) + 3
    noise_std = 0.2
    y += np.random.normal(0, noise_std, y.shape)
    return explicit_training_data(x, y)


@pytest.fixture
def true_equation(agraph_implementation):
    true_commands = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 1], [6, 0, 0], [4, 1, 3], [2, 4, 2]]
    )
    true_equ = agraph_implementation(use_simplification=True)
    true_equ.command_array = true_commands
    return true_equ


def test_smc_optimizer(
    explicit_regression, training_data, true_equation, engine
):
    np.random.seed(0)

    fitness = explicit_regression(training_data=training_data, metric="mse")
    scipy_opt = ScipyOptimizer(fitness, method="lm")
    smcpy_opt = SmcpyOptimizer(
        fitness,
        scipy_opt,
        num_particles=2000,
        mcmc_steps=25,
        num_multistarts=8,
    )

    smcpy_opt(true_equation)
    fit_constants = true_equation.get_local_optimization_params()
    map_fitness = fitness(true_equation)

    assert fit_constants[0] == pytest.approx(3.0, rel=0.2)
    assert fit_constants[1] == pytest.approx(2.0, rel=0.2)
    assert np.sqrt(map_fitness) == pytest.approx(0.2, rel=0.1)


# @pytest.fixture
# def other_equations(engine):
#     use_python = engine in ["Python", "Hybrid"]
#     component_generator = ComponentGenerator(input_x_dimension=1)
#     for comp in ["*", "+", "/", "-", "sin", "cos", "exp", "log"]:
#         component_generator.add_operator(comp)
#     generator = AGraphGenerator(
#         64, component_generator, use_python=use_python, use_simplification=True
#     )
#     np.random.seed(1)
#     return [generator() for i in range(1000)]


# def test_bayes_fitness_function_other(
#     explicit_regression,
#     training_data,
#     other_equations,
#     engine,
#     agraph_implementation,
# ):
#     np.random.seed(0)

#     fitness = explicit_regression(training_data=training_data, metric="mse")
#     scipy_opt = ScipyOptimizer(fitness, method="lm")
#     # optimizer = LocalOptFitnessFunction(fitness, scipy_opt)
#     # bff = BayesFitnessFunction(optimizer,
#     #                            num_particles=500,
#     #                            mcmc_steps=10,
#     #                            num_multistarts=8)

#     smcoptimizer = SmcpyOptimizer(
#         fitness, scipy_opt, num_particles=200, mcmc_steps=10, num_multistarts=8
#     )

#     # bff(true_equation)
#     # print(f"bff: {true_equation}")

#     t0 = time()
#     fits = [smcoptimizer(equ)[0] for equ in other_equations]
#     # for equ in other_equations[2:3]:
#     #     print(f"Complexity: {equ.get_complexity()}: {equ}")
#     #     for i in range(1):
#     #         print(f"{i}: {smcoptimizer(equ)[0]}")
#     t1 = time()
#     print(f"Time: {t1-t0}s")
#     print(f"Engine: {engine}")
#     from bingo.symbolic_regression.agraph.agraph import evaluation_backend

#     print(f"backend: {evaluation_backend.ENGINE}")

#     print(f"Nans: {np.count_nonzero(np.isnan(fits))}")

#     assert False
