# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
from collections import namedtuple
import pytest
import numpy as np

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
def sample_agraph(agraph_implementation):  # sin(X_0 + 2.0) + 2.0
    test_graph = agraph_implementation()
    test_graph.command_array = np.array([[VARIABLE, 0, 0],
                                         [CONSTANT, 0, 0],
                                         [ADDITION, 0, 1],
                                         [SIN, 2, 2],
                                         [ADDITION, 0, 1],
                                         [ADDITION, 3, 1]], dtype=int)
    test_graph.genetic_age = 10
    _ = test_graph.needs_local_optimization()
    test_graph.set_local_optimization_params([2.0, ])
    test_graph.fitness = 1
    return test_graph


@pytest.fixture
def sample_agraph_values():
    values = namedtuple('Data', ['x', 'f_of_x', 'grad_x', 'grad_c'])
    x = np.vstack((np.linspace(-1.0, 0.0, 11),
                   np.linspace(0.0, 1.0, 11))).transpose()
    f_of_x = (np.sin(x[:, 0] + 2.0) + 2.0).reshape((-1, 1))
    grad_x = np.zeros(x.shape)
    grad_x[:, 0] = np.cos(x[:, 0] + 2.0)
    grad_c = (np.cos(x[:, 0] + 2.0) + 1.0).reshape((-1, 1))
    return values(x, f_of_x, grad_x, grad_c)


def test_evaluate_agraph(sample_agraph, sample_agraph_values):
    np.testing.assert_allclose(
        sample_agraph.evaluate_equation_at(sample_agraph_values.x),
        sample_agraph_values.f_of_x)


def test_evaluate_agraph_x_gradient(sample_agraph, sample_agraph_values):
    f_of_x, df_dx = \
        sample_agraph.evaluate_equation_with_x_gradient_at(
            sample_agraph_values.x)
    np.testing.assert_allclose(f_of_x, sample_agraph_values.f_of_x)
    np.testing.assert_allclose(df_dx, sample_agraph_values.grad_x)


def test_evaluate_agraph_c_gradient(sample_agraph, sample_agraph_values):
    f_of_x, df_dc = \
        sample_agraph.evaluate_equation_with_local_opt_gradient_at(
            sample_agraph_values.x)
    np.testing.assert_allclose(f_of_x, sample_agraph_values.f_of_x)
    np.testing.assert_allclose(df_dc, sample_agraph_values.grad_c)


# def test_evaluate_overflow_exception(mocker,
#                                      sample_agraph_1,
#                                      sample_agraph_1_values):
#     mocker.patch(EVALUATE)
#     bingo.symbolic_regression.agraph.evaluation_backend.backend.evaluate.side_effect = OverflowError
#
#     values = sample_agraph_1.evaluate_equation_at(sample_agraph_1_values.x)
#     assert np.isnan(values).all()
#
#
# def test_evaluate_gradient_overflow_exception(mocker,
#                                               sample_agraph_1,
#                                               sample_agraph_1_values):
#     mocker.patch(EVALUATE_WTIH_DERIV)
#     bingo.symbolic_regression.agraph.evaluation_backend.backend.evaluate_with_derivative.side_effect = OverflowError
#
#     values = sample_agraph_1.evaluate_equation_with_x_gradient_at(
#         sample_agraph_1_values.x)
#     assert np.isnan(values).all()
#
#
# def test_evaluate_local_opt_gradient_overflow_exception(mocker,
#                                                         sample_agraph_1,
#                                                         sample_agraph_1_values):
#     mocker.patch(EVALUATE_WTIH_DERIV)
#     bingo.symbolic_regression.agraph.evaluation_backend.backend.evaluate_with_derivative.side_effect = OverflowError
#
#     values = sample_agraph_1.evaluate_equation_with_local_opt_gradient_at(
#         sample_agraph_1_values.x)
#     assert np.isnan(values).all()