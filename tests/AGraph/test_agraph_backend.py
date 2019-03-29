# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
from collections import namedtuple
import pytest
import numpy as np

from bingo.SymbolicRegression.AGraph import Backend as PythonBackend

try:
    from bingocpp.build import bingocpp as CppBackend
    CPP_LOADED = True
except ImportError:
    CppBackend = None
    CPP_LOADED = False


@pytest.fixture(params=[
    PythonBackend,
    pytest.param(CppBackend,
                 marks=pytest.mark.skipif(not CPP_LOADED,
                                          reason='BingoCpp import failure'))
])
def backend(request):
    return request.param


@pytest.fixture
def sample_agraph_values():
    values = namedtuple('Point', ['x', 'constants'])
    x = np.vstack((np.linspace(-1.0, 0.0, 11),
                   np.linspace(0.0, 1.0, 11))).transpose()
    constants = [10, 3.14]
    return values(x, constants)


@pytest.fixture
def operator_evals_x0(sample_agraph_values):
    x_0 = sample_agraph_values.x[:, 0].reshape((-1, 1))
    c_0 = np.full(x_0.shape, sample_agraph_values.constants[0])
    return [x_0,
            c_0,
            x_0+x_0,
            x_0-x_0,
            x_0*x_0,
            x_0/x_0,
            np.sin(x_0),
            np.cos(x_0),
            np.exp(x_0),
            np.log(np.abs(x_0)),
            np.power(np.abs(x_0), x_0),
            np.abs(x_0),
            np.sqrt(np.abs(x_0))]


@pytest.fixture
def operator_x_derivs(sample_agraph_values):
    def last_nan(array):
        array[-1] = np.nan
        return array
    x_0 = sample_agraph_values.x[:, 0].reshape((-1, 1))
    return [np.ones(x_0.shape),
            np.zeros(x_0.shape),
            np.full(x_0.shape, 2.0),
            np.zeros(x_0.shape),
            2*x_0,
            last_nan(np.zeros(x_0.shape)),
            np.cos(x_0),
            -np.sin(x_0),
            np.exp(x_0),
            1.0 / x_0,
            last_nan(np.power(np.abs(x_0), x_0)*(np.log(np.abs(x_0)) + 1)),
            np.sign(x_0),
            0.5*np.sign(x_0) / np.sqrt(np.abs(x_0))]


@pytest.fixture
def operator_c_derivs(sample_agraph_values):
    c_1 = np.full((sample_agraph_values.x.shape[0], 1),
                  sample_agraph_values.constants[1])
    return [np.zeros(c_1.shape),
            np.ones(c_1.shape),
            np.full(c_1.shape, 2.0),
            np.zeros(c_1.shape),
            2*c_1,
            np.zeros(c_1.shape),
            np.cos(c_1),
            -np.sin(c_1),
            np.exp(c_1),
            1.0 / c_1,
            np.power(np.abs(c_1), c_1)*(np.log(np.abs(c_1)) + 1),
            np.sign(c_1),
            0.5*np.sign(c_1) / np.sqrt(np.abs(c_1))]


@pytest.fixture
def sample_stack():
    test_stack = np.array([[0, 0, 0],  # sin(X_0) + 1.0
                           [1, 0, 0],
                           [2, 0, 1],
                           [6, 0, 2],
                           [2, 3, 1]])
    return test_stack


@pytest.fixture
def all_funcs_stack():
    test_stack = np.array([[0, 0, 0],
                           [1, 0, 0],
                           [2, 1, 0],
                           [3, 2, 0],
                           [4, 3, 0],
                           [5, 4, 0],
                           [6, 5, 0],
                           [7, 6, 0],
                           [8, 7, 0],
                           [9, 8, 0],
                           [10, 9, 0],
                           [11, 10, 0],
                           [12, 11, 0]])
    return test_stack


@pytest.fixture(params=['all_funcs_stack', 'sample_stack'])
def expected_stack_util(request):
    prop = {'stack': request.getfixturevalue(request.param)}
    if request.param == "all_funcs_stack":
        prop["util"] = np.ones(13, bool)
    elif request.param == "sample_stack":
        prop["util"] = [True, True, False, True, True]
    return prop


def test_cpp_backend_could_be_imported():
    assert CPP_LOADED


@pytest.mark.parametrize("operator", range(13))
def test_backend_simplify_and_evaluate(backend, sample_agraph_values,
                                       operator, operator_evals_x0):
    expected_outcome = operator_evals_x0[operator]
    stack = np.array([[0, 0, 0],
                      [0, 1, 1],
                      [operator, 0, 0]])
    f_of_x = backend.simplify_and_evaluate(stack,
                                           sample_agraph_values.x,
                                           sample_agraph_values.constants)
    np.testing.assert_allclose(expected_outcome, f_of_x)


@pytest.mark.parametrize("operator", range(13))
# pylint: disable=invalid-name
def test_backend_simplify_and_evaluate_with_x_derivative(backend,
                                                         sample_agraph_values,
                                                         operator,
                                                         operator_x_derivs):
    expected_derivative = np.zeros(sample_agraph_values.x.shape)
    expected_derivative[:, 0] = operator_x_derivs[operator].flatten()
    stack = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 1, 1],
                      [operator, 0, 1]])
    _, df_dx = backend.simplify_and_evaluate_with_derivative(
        stack, sample_agraph_values.x, sample_agraph_values.constants,
        True)
    np.testing.assert_allclose(expected_derivative, df_dx)


@pytest.mark.parametrize("operator", range(13))
# pylint: disable=invalid-name
def test_backend_simplify_and_evaluate_with_c_derivative(backend,
                                                         sample_agraph_values,
                                                         operator,
                                                         operator_c_derivs):
    expected_derivative = np.zeros(sample_agraph_values.x.shape)
    expected_derivative[:, 1] = operator_c_derivs[operator].flatten()
    stack = np.array([[1, 1, 1],
                      [1, 1, 1],
                      [0, 1, 1],
                      [operator, 1, 0]])
    _, df_dx = backend.simplify_and_evaluate_with_derivative(
        stack, sample_agraph_values.x, sample_agraph_values.constants, False)
    np.testing.assert_allclose(expected_derivative, df_dx)


def test_agraph_get_utilized_commands(backend, expected_stack_util):
    np.testing.assert_array_equal(
        backend.get_utilized_commands(expected_stack_util["stack"]),
        expected_stack_util["util"])


@pytest.mark.parametrize("the_backend, expected", [
    (PythonBackend, False),
    pytest.param(CppBackend, True,
                 marks=pytest.mark.skipif(not CPP_LOADED,
                                          reason='BingoCpp import failure'))
])
def test_agraph_backend_identifiers(the_backend, expected):
    assert the_backend.is_cpp() == expected
