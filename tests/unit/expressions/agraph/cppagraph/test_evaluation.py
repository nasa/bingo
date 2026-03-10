"""Tests for cppagraph evaluation engine — must match pyagraph results."""

import numpy as np
import pytest

from bingo.expressions.agraph.cppagraph import (
    VARIABLE,
    CONSTANT,
    INTEGER,
    ADDITION,
    SUBTRACTION,
    MULTIPLICATION,
    SIN,
    evaluate,
    evaluate_with_derivative,
)
from bingo.expressions.agraph.pyagraph.evaluation import evaluation as py_eval


@pytest.fixture
def simple_x():
    return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


# ================================================================== #
#  evaluate()                                                         #
# ================================================================== #


class TestEvaluate:
    def test_single_variable(self, simple_x):
        stack = np.array([[VARIABLE, 0, 0]], dtype=np.uint8)
        result = evaluate(stack, simple_x, (), ())
        np.testing.assert_array_almost_equal(result, simple_x[:, 0:1])

    def test_constant(self, simple_x):
        stack = np.array([[CONSTANT, 0, 0]], dtype=np.uint8)
        result = evaluate(stack, simple_x, (3.14,), ())
        expected = np.full((3, 1), 3.14)
        np.testing.assert_array_almost_equal(result, expected)

    def test_integer(self, simple_x):
        stack = np.array([[INTEGER, 0, 0]], dtype=np.uint8)
        result = evaluate(stack, simple_x, (), (5,))
        expected = np.full((3, 1), 5.0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_x0_plus_c0(self, simple_x):
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [CONSTANT, 0, 0],
                [ADDITION, 0, 1],
            ],
            dtype=np.uint8,
        )
        result = evaluate(stack, simple_x, (10.0,), ())
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
        result = evaluate(stack, simple_x, (), ())
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
        result = evaluate(stack, simple_x, (), ())
        expected = np.sin(simple_x[:, 0:1])
        np.testing.assert_array_almost_equal(result, expected)

    def test_integer_plus_variable(self, simple_x):
        stack = np.array(
            [
                [INTEGER, 0, 0],
                [VARIABLE, 0, 0],
                [ADDITION, 0, 1],
            ],
            dtype=np.uint8,
        )
        result = evaluate(stack, simple_x, (), (7,))
        expected = simple_x[:, 0:1] + 7.0
        np.testing.assert_array_almost_equal(result, expected)


# ================================================================== #
#  evaluate_with_derivative()                                         #
# ================================================================== #


class TestEvaluateWithDerivative:
    def test_x_gradient_of_x0(self, simple_x):
        stack = np.array([[VARIABLE, 0, 0]], dtype=np.uint8)
        f, df_dx = evaluate_with_derivative(stack, simple_x, (), (), True)
        np.testing.assert_array_almost_equal(f, simple_x[:, 0:1])
        expected_grad = np.zeros_like(simple_x)
        expected_grad[:, 0] = 1.0
        np.testing.assert_array_almost_equal(df_dx, expected_grad)

    def test_x_gradient_of_sum(self, simple_x):
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 1, 1],
                [ADDITION, 0, 1],
            ],
            dtype=np.uint8,
        )
        f, df_dx = evaluate_with_derivative(stack, simple_x, (), (), True)
        np.testing.assert_array_almost_equal(
            f, (simple_x[:, 0] + simple_x[:, 1]).reshape(-1, 1)
        )
        np.testing.assert_array_almost_equal(df_dx, np.ones_like(simple_x))

    def test_const_gradient(self, simple_x):
        stack = np.array(
            [
                [CONSTANT, 0, 0],
                [VARIABLE, 0, 0],
                [MULTIPLICATION, 0, 1],
            ],
            dtype=np.uint8,
        )
        f, df_dc = evaluate_with_derivative(stack, simple_x, (2.0,), (), False)
        expected_f = 2.0 * simple_x[:, 0:1]
        np.testing.assert_array_almost_equal(f, expected_f)
        np.testing.assert_array_almost_equal(df_dc, simple_x[:, 0:1])

    def test_x_gradient_of_subtraction(self, simple_x):
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 1, 1],
                [SUBTRACTION, 0, 1],
            ],
            dtype=np.uint8,
        )
        f, df_dx = evaluate_with_derivative(stack, simple_x, (), (), True)
        expected_grad = np.zeros_like(simple_x)
        expected_grad[:, 0] = 1.0
        expected_grad[:, 1] = -1.0
        np.testing.assert_array_almost_equal(df_dx, expected_grad)

    def test_x_gradient_sin(self, simple_x):
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [SIN, 0, 0],
            ],
            dtype=np.uint8,
        )
        f, df_dx = evaluate_with_derivative(stack, simple_x, (), (), True)
        np.testing.assert_array_almost_equal(f, np.sin(simple_x[:, 0:1]))
        expected_grad = np.zeros_like(simple_x)
        expected_grad[:, 0] = np.cos(simple_x[:, 0])
        np.testing.assert_array_almost_equal(df_dx, expected_grad)


# ================================================================== #
#  Cross-implementation: cppagraph vs pyagraph                        #
# ================================================================== #


class TestCrossImplementation:
    """Verify cppagraph evaluate matches pyagraph evaluate exactly."""

    @pytest.mark.parametrize(
        "stack, constants, integers, desc",
        [
            pytest.param(
                np.array([[VARIABLE, 0, 0]], dtype=np.uint8),
                (),
                (),
                "X0",
            ),
            pytest.param(
                np.array([[CONSTANT, 0, 0]], dtype=np.uint8),
                (3.14,),
                (),
                "C0",
            ),
            pytest.param(
                np.array([[INTEGER, 0, 0]], dtype=np.uint8),
                (),
                (7,),
                "I0",
            ),
            pytest.param(
                np.array(
                    [
                        [VARIABLE, 0, 0],
                        [CONSTANT, 0, 0],
                        [ADDITION, 0, 1],
                    ],
                    dtype=np.uint8,
                ),
                (10.0,),
                (),
                "X0+C0",
            ),
            pytest.param(
                np.array(
                    [
                        [VARIABLE, 0, 0],
                        [VARIABLE, 1, 1],
                        [MULTIPLICATION, 0, 1],
                    ],
                    dtype=np.uint8,
                ),
                (),
                (),
                "X0*X1",
            ),
            pytest.param(
                np.array(
                    [
                        [VARIABLE, 0, 0],
                        [SIN, 0, 0],
                    ],
                    dtype=np.uint8,
                ),
                (),
                (),
                "sin(X0)",
            ),
            pytest.param(
                np.array(
                    [
                        [VARIABLE, 0, 0],
                        [VARIABLE, 1, 1],
                        [SUBTRACTION, 0, 1],
                        [SIN, 2, 2],
                        [VARIABLE, 0, 0],
                        [MULTIPLICATION, 3, 4],
                    ],
                    dtype=np.uint8,
                ),
                (),
                (),
                "sin(X0-X1)*X0",
            ),
        ],
        ids=lambda x: x if isinstance(x, str) else "",
    )
    def test_evaluate_matches(self, stack, constants, integers, desc, simple_x):
        cpp_result = evaluate(stack, simple_x, constants, integers)
        py_result = py_eval.evaluate(stack, simple_x, constants, integers)
        np.testing.assert_array_almost_equal(cpp_result, py_result, err_msg=desc)

    @pytest.mark.parametrize("wrt_x", [True, False], ids=["wrt_x", "wrt_c"])
    def test_derivative_matches(self, wrt_x, simple_x):
        stack = np.array(
            [
                [CONSTANT, 0, 0],
                [VARIABLE, 0, 0],
                [MULTIPLICATION, 0, 1],
            ],
            dtype=np.uint8,
        )
        constants = (2.0,)
        cpp_f, cpp_d = evaluate_with_derivative(stack, simple_x, constants, (), wrt_x)
        py_f, py_d = py_eval.evaluate_with_derivative(
            stack, simple_x, constants, (), wrt_x
        )
        np.testing.assert_array_almost_equal(cpp_f, py_f)
        np.testing.assert_array_almost_equal(cpp_d, py_d)
