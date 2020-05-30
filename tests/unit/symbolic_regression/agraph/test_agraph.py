# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np
import pytest
import dill

from bingo.symbolic_regression.agraph.operator_definitions import *
from bingo.symbolic_regression.agraph.agraph import AGraph as pyagraph

try:
    from bingocpp.build import symbolic_regression as bingocpp
except ImportError:
    bingocpp = None

CPP_PARAM = pytest.param("Cpp",
                         marks=pytest.mark.skipif(not bingocpp,
                                                  reason='BingoCpp import '
                                                         'failure'))


@pytest.fixture(params=["Python", CPP_PARAM])
def engine(request):
    return request.param


@pytest.fixture
def agraph_implementation(engine):
    if engine == "Python":
        return pyagraph
    return bingocpp.AGraph


@pytest.fixture
def addition_agraph(agraph_implementation):
    sample = agraph_implementation()
    sample.command_array = np.array([[VARIABLE, 0, 0],
                                     [VARIABLE, 1, 1],
                                     [ADDITION, 1, 0]], dtype=int)
    return sample


@pytest.fixture
def addition_agraph_with_constants(agraph_implementation):
    sample = agraph_implementation()
    sample.command_array = np.array([[CONSTANT, -1, -1],
                                     [CONSTANT, -1, -1],
                                     [ADDITION, 1, 0]], dtype=int)
    return sample


@pytest.fixture
def sin_agraph(agraph_implementation):
    sample = agraph_implementation()
    sample.command_array = np.array([[VARIABLE, 0, 0],
                                     [SIN, 0, 0]], dtype=int)
    return sample

# def _sample_agraph_1(test_graph):  # sin(X_0 + 2.0) + 2.0
#     test_graph.command_array = np.array([[VARIABLE, 0, 0],
#                                          [CONSTANT, 0, 0],
#                                          [ADDITION, 0, 1],
#                                          [SIN, 2, 2],
#                                          [ADDITION, 0, 1],
#                                          [ADDITION, 3, 1]], dtype=int)
#     test_graph.genetic_age = 10
#     _ = test_graph.needs_local_optimization()
#     test_graph.set_local_optimization_params([2.0, ])
#     test_graph.fitness = 1
#     return test_graph


def test_agraph_copy_and_distance(addition_agraph):
    agraph_2 = addition_agraph.copy()
    agraph_2.mutable_command_array[0, 0] = CONSTANT
    agraph_2.mutable_command_array[1, 1:] = 0
    agraph_2.mutable_command_array[2, 0] = SUBTRACTION

    assert addition_agraph.distance(agraph_2) == 4
    assert agraph_2.distance(addition_agraph) == 4


def test_agraph_complexity(addition_agraph, sin_agraph):
    assert addition_agraph.get_complexity() == 3
    assert sin_agraph.get_complexity() == 2


def test_local_opt_interface(addition_agraph_with_constants):
    assert addition_agraph_with_constants.needs_local_optimization()
    n_p = addition_agraph_with_constants.get_number_local_optimization_params()
    assert n_p == 2
    params = (2, 3)
    addition_agraph_with_constants.set_local_optimization_params(params)
    assert addition_agraph_with_constants.constants == params
    assert not addition_agraph_with_constants.needs_local_optimization()


def test_engine_identification(engine, addition_agraph):
    assert addition_agraph.engine == engine


@pytest.mark.parametrize("format_", ["latex", "console", "stack"])
@pytest.mark.parametrize("raw", [True, False])
def test_can_get_formatted_strings(format_, raw, addition_agraph):
    string = addition_agraph.get_formatted_string(format_, raw=raw)
    assert isinstance(string, str)


def test_can_pickle(addition_agraph):
    _ = dill.loads(dill.dumps(addition_agraph))