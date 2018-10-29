import sys
from collections import namedtuple

import numpy as np
import pytest

sys.path.append("..")

from bingo.AGraph import Backend as Backend


@pytest.fixture
def sample_values():
    values = namedtuple('Point', ['x', 'constants'])
    x = np.vstack((np.linspace(-1.0, 0.0, 11),
                   np.linspace(0.0, 1.0, 11))).transpose()
    constants = [10, 3.14]
    return values(x, constants)


@pytest.mark.parametrize("operator,eval_string", [
    (0, "x_0"),
    (1, "c_0"),
    (2, "x_0+x_0"),
    (3, "x_0-x_0"),
    (4, "x_0*x_0"),
    (5, "x_0/x_0"),
    (6, "np.sin(x_0)"),
    (7, "np.cos(x_0)"),
    (8, "np.exp(x_0)"),
    (9, "np.log(np.abs(x_0))"),
    (10, "np.power(np.abs(x_0), x_0)"),
    (11, "np.abs(x_0)"),
    (12, "np.sqrt(np.abs(x_0))"),
])
def test_backend_simplify_and_evaluate(sample_values, operator, eval_string):
    x_0 = sample_values.x[:, 0].reshape((-1, 1))
    c_0 = np.full(x_0.shape, sample_values.constants[0])
    expected_outcome = eval(eval_string)
    stack = np.array([[0, 0, 0],
                      [0, 1, 1],
                      [operator, 0, 0]])
    f_of_x = Backend.simplify_and_evaluate(stack,
                                           sample_values.x,
                                           sample_values.constants)
    np.testing.assert_allclose(expected_outcome, f_of_x)


@pytest.mark.parametrize("operator,eval_string", [
    (0, "np.ones(x_0.shape)"),
    (1, "np.zeros(x_0.shape)"),
    (2, "np.full(x_0.shape, 2.0)"),
    (3, "np.zeros(x_0.shape)"),
    (4, "2*x_0"),
    (5, "set_last_nan(np.zeros(x_0.shape))"),
    (6, "np.cos(x_0)"),
    (7, "-np.sin(x_0)"),
    (8, "np.exp(x_0)"),
    (9, "1.0 / x_0"),
    (10, "set_last_nan(np.power(np.abs(x_0), x_0)*(np.log(np.abs(x_0)) + 1))"),
    (11, "np.sign(x_0)"),
    (12, "0.5*np.sign(x_0) / np.sqrt(np.abs(x_0))"),
])
def test_backend_simplify_and_evaluate_with_x_derivative(sample_values,
                                                         operator,
                                                         eval_string):
    def set_last_nan(array):
        array[-1] = np.nan
        return array
    x_0 = sample_values.x[:, 0].reshape((-1, 1))
    expected_derivative = np.zeros(sample_values.x.shape)
    expected_derivative[:, 0] = eval(eval_string).flatten()
    stack = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 1, 1],
                      [operator, 0, 1]])
    _, df_dx = Backend.simplify_and_evaluate_with_derivative(
            stack, sample_values.x, sample_values.constants, True)
    np.testing.assert_allclose(expected_derivative, df_dx)


@pytest.mark.parametrize("operator,eval_string", [
    (0, "np.zeros(c_1.shape)"),
    (1, "np.ones(c_1.shape)"),
    (2, "np.full(c_1.shape, 2.0)"),
    (3, "np.zeros(c_1.shape)"),
    (4, "2*c_1"),
    (5, "np.zeros(c_1.shape)"),
    (6, "np.cos(c_1)"),
    (7, "-np.sin(c_1)"),
    (8, "np.exp(c_1)"),
    (9, "1.0 / c_1"),
    (10, "np.power(np.abs(c_1), c_1)*(np.log(np.abs(c_1)) + 1)"),
    (11, "np.sign(c_1)"),
    (12, "0.5*np.sign(c_1) / np.sqrt(np.abs(c_1))"),
])
def test_backend_simplify_and_evaluate_with_c_derivative(sample_values,
                                                         operator,
                                                         eval_string):
    c_1 = np.full((sample_values.x.shape[0], 1),
                  sample_values.constants[1])
    expected_derivative = np.zeros(sample_values.x.shape)
    expected_derivative[:, 1] = eval(eval_string).flatten()
    stack = np.array([[1, 1, 1],
                      [1, 1, 1],
                      [0, 1, 1],
                      [operator, 1, 0]])
    _, df_dx = Backend.simplify_and_evaluate_with_derivative(
            stack, sample_values.x, sample_values.constants, False)
    np.testing.assert_allclose(expected_derivative, df_dx)

