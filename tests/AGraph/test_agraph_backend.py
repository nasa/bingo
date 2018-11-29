# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
# pylint: disable=invalid-name
from collections import namedtuple
import pytest
import numpy as np

from bingo.AGraph import Backend as PythonBackend
from bingocpp.build import bingocpp as CppBackend

@pytest.fixture
def sample_values():
    values = namedtuple('Point', ['x', 'constants'])
    x = np.vstack((np.linspace(-1.0, 0.0, 11),
                   np.linspace(0.0, 1.0, 11))).transpose()
    constants = [10, 3.14]
    return values(x, constants)


@pytest.fixture
def operator_evals_x0(sample_values):
    x_0 = sample_values.x[:, 0].reshape((-1, 1))
    c_0 = np.full(x_0.shape, sample_values.constants[0])
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
def operator_x_derivs(sample_values):
    def last_nan(array):
        array[-1] = np.nan
        return array
    x_0 = sample_values.x[:, 0].reshape((-1, 1))
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
def operator_c_derivs(sample_values):
    c_1 = np.full((sample_values.x.shape[0], 1),
                  sample_values.constants[1])
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


@pytest.mark.parametrize("backend", [PythonBackend, CppBackend])
@pytest.mark.parametrize("operator", range(13))
def test_backend_simplify_and_evaluate(backend, sample_values, operator,
                                       operator_evals_x0):
    expected_outcome = operator_evals_x0[operator]
    stack = np.array([[0, 0, 0],
                      [0, 1, 1],
                      [operator, 0, 0]])
    f_of_x = backend.simplify_and_evaluate(stack,
                                           sample_values.x,
                                           sample_values.constants)
    np.testing.assert_allclose(expected_outcome, f_of_x)


@pytest.mark.parametrize("backend", [PythonBackend, CppBackend])
@pytest.mark.parametrize("operator", range(13))
def test_backend_simplify_and_evaluate_with_x_derivative(backend,
                                                         sample_values,
                                                         operator,
                                                         operator_x_derivs):
    expected_derivative = np.zeros(sample_values.x.shape)
    expected_derivative[:, 0] = operator_x_derivs[operator].flatten()
    stack = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 1, 1],
                      [operator, 0, 1]])
    _, df_dx = backend.simplify_and_evaluate_with_derivative(
        stack, sample_values.x, sample_values.constants, True)
    np.testing.assert_allclose(expected_derivative, df_dx)


@pytest.mark.parametrize("backend", [PythonBackend, CppBackend])
@pytest.mark.parametrize("operator", range(13))
def test_backend_simplify_and_evaluate_with_c_derivative(backend,
                                                         sample_values,
                                                         operator,
                                                         operator_c_derivs):
    expected_derivative = np.zeros(sample_values.x.shape)
    expected_derivative[:, 1] = operator_c_derivs[operator].flatten()
    stack = np.array([[1, 1, 1],
                      [1, 1, 1],
                      [0, 1, 1],
                      [operator, 1, 0]])
    _, df_dx = backend.simplify_and_evaluate_with_derivative(
        stack, sample_values.x, sample_values.constants, False)
    np.testing.assert_allclose(expected_derivative, df_dx)


@pytest.mark.parametrize("backend", [PythonBackend, CppBackend])
@pytest.mark.parametrize("stack,util_array", [
    (all_funcs_stack(), np.ones(13, bool)),
    (sample_stack(), [True, True, False, True, True]),
])
def test_agraph_get_utilized_commands(backend, stack, util_array):
    np.testing.assert_array_equal(backend.get_utilized_commands(stack),
                                  util_array)
