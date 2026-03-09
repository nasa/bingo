"""Tests for bingo.expressions.agraph.evaluation"""

import numpy as np
import pytest

from bingo.expressions.agraph.pyagraph.operators import (
    VARIABLE,
    CONSTANT,
    INTEGER,
    ADDITION,
    SUBTRACTION,
    MULTIPLICATION,
    SIN,
)
from bingo.expressions.agraph.pyagraph.evaluation import evaluation


@pytest.fixture
def simple_x():
    return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


class TestEvaluate:
    def test_single_variable(self, simple_x):
        # stack: X0
        stack = np.array([[VARIABLE, 0, 0]], dtype=np.uint8)
        result = evaluation.evaluate(stack, simple_x, (), ())
        np.testing.assert_array_almost_equal(result, simple_x[:, 0:1])

    def test_constant(self, simple_x):
        # stack: C0 = 3.14
        stack = np.array([[CONSTANT, 0, 0]], dtype=np.uint8)
        result = evaluation.evaluate(stack, simple_x, (3.14,), ())
        expected = np.full((3, 1), 3.14)
        np.testing.assert_array_almost_equal(result, expected)

    def test_integer(self, simple_x):
        # stack: I0 = 5
        stack = np.array([[INTEGER, 0, 0]], dtype=np.uint8)
        result = evaluation.evaluate(stack, simple_x, (), (5,))
        expected = np.full((3, 1), 5.0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_x0_plus_c0(self, simple_x):
        # stack: [X0, C0, X0 + C0]
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [CONSTANT, 0, 0],
                [ADDITION, 0, 1],
            ],
            dtype=np.uint8,
        )
        result = evaluation.evaluate(stack, simple_x, (10.0,), ())
        expected = simple_x[:, 0:1] + 10.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_x0_times_x1(self, simple_x):
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 1, 1],
                [MULTIPLICATION, 0, 1],
            ],
            dtype=np.uint8,
        )
        result = evaluation.evaluate(stack, simple_x, (), ())
        expected = (simple_x[:, 0] * simple_x[:, 1]).reshape(-1, 1)
        np.testing.assert_array_almost_equal(result, expected)

    def test_sin_x0(self, simple_x):
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [SIN, 0, 0],
            ],
            dtype=np.uint8,
        )
        result = evaluation.evaluate(stack, simple_x, (), ())
        expected = np.sin(simple_x[:, 0:1])
        np.testing.assert_array_almost_equal(result, expected)

    def test_integer_plus_variable(self, simple_x):
        # I0 + X0  where I0 = 7
        stack = np.array(
            [
                [INTEGER, 0, 0],
                [VARIABLE, 0, 0],
                [ADDITION, 0, 1],
            ],
            dtype=np.uint8,
        )
        result = evaluation.evaluate(stack, simple_x, (), (7,))
        expected = simple_x[:, 0:1] + 7.0
        np.testing.assert_array_almost_equal(result, expected)


class TestEvaluateWithDerivative:
    def test_x_gradient_of_x0(self, simple_x):
        # f(x) = X0  =>  df/dX0 = 1, df/dX1 = 0
        stack = np.array([[VARIABLE, 0, 0]], dtype=np.uint8)
        f, df_dx = evaluation.evaluate_with_derivative(stack, simple_x, (), (), True)
        np.testing.assert_array_almost_equal(f, simple_x[:, 0:1])
        expected_grad = np.zeros_like(simple_x)
        expected_grad[:, 0] = 1.0
        np.testing.assert_array_almost_equal(df_dx, expected_grad)

    def test_x_gradient_of_sum(self, simple_x):
        # f(x) = X0 + X1  =>  df/dX0 = 1, df/dX1 = 1
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 1, 1],
                [ADDITION, 0, 1],
            ],
            dtype=np.uint8,
        )
        f, df_dx = evaluation.evaluate_with_derivative(stack, simple_x, (), (), True)
        np.testing.assert_array_almost_equal(
            f, (simple_x[:, 0] + simple_x[:, 1]).reshape(-1, 1)
        )
        np.testing.assert_array_almost_equal(df_dx, np.ones_like(simple_x))

    def test_const_gradient(self, simple_x):
        # f(x) = C0 * X0  =>  df/dC0 = X0
        stack = np.array(
            [
                [CONSTANT, 0, 0],
                [VARIABLE, 0, 0],
                [MULTIPLICATION, 0, 1],
            ],
            dtype=np.uint8,
        )
        f, df_dc = evaluation.evaluate_with_derivative(
            stack, simple_x, (2.0,), (), False
        )
        expected_f = 2.0 * simple_x[:, 0:1]
        np.testing.assert_array_almost_equal(f, expected_f)
        np.testing.assert_array_almost_equal(df_dc, simple_x[:, 0:1])

    def test_x_gradient_of_subtraction(self, simple_x):
        # f(x) = X_0 - X_1  =>  df/dX_0 = 1, df/dX_1 = -1
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 1, 1],
                [SUBTRACTION, 0, 1],
            ],
            dtype=np.uint8,
        )
        f, df_dx = evaluation.evaluate_with_derivative(stack, simple_x, (), (), True)
        expected_grad = np.zeros_like(simple_x)
        expected_grad[:, 0] = 1.0
        expected_grad[:, 1] = -1.0
        np.testing.assert_array_almost_equal(df_dx, expected_grad)

    def test_x_gradient_sin(self, simple_x):
        # f(x) = sin(X_0)  =>  df/dX_0 = cos(X_0)
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [SIN, 0, 0],
            ],
            dtype=np.uint8,
        )
        f, df_dx = evaluation.evaluate_with_derivative(stack, simple_x, (), (), True)
        np.testing.assert_array_almost_equal(f, np.sin(simple_x[:, 0:1]))
        expected_grad = np.zeros_like(simple_x)
        expected_grad[:, 0] = np.cos(simple_x[:, 0])
        np.testing.assert_array_almost_equal(df_dx, expected_grad)
