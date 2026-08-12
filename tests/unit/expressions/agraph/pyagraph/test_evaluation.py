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
    DIVISION,
    POWER,
    SAFE_POWER,
    SQUARE,
    CUBE,
    SQRT,
    ABS,
    EXPONENTIAL,
    LOGARITHM,
    SIN,
    COS,
    TAN,
    ARCSIN,
    ARCCOS,
    ARCTAN,
    SINH,
    COSH,
    TANH,
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


def _finite_difference_hessian(stack, x, constants, integers, step=1e-6):
    num_constants = len(constants)
    hessian = np.empty((x.shape[0], num_constants, num_constants))
    for const_index in range(num_constants):
        constants_plus = list(constants)
        constants_minus = list(constants)
        constants_plus[const_index] += step
        constants_minus[const_index] -= step
        _, gradient_plus = evaluation.evaluate_with_derivative(
            stack, x, tuple(constants_plus), integers, False
        )
        _, gradient_minus = evaluation.evaluate_with_derivative(
            stack, x, tuple(constants_minus), integers, False
        )
        hessian[:, :, const_index] = (gradient_plus - gradient_minus) / (2.0 * step)
    return hessian


class TestEvaluateWithConstHessian:
    def test_constant_terminal(self, simple_x):
        stack = np.array([[CONSTANT, 0, 0]], dtype=np.uint8)

        f_of_x, gradient, hessian = evaluation.evaluate_with_const_hessian(
            stack, simple_x, (2.0,), ()
        )

        np.testing.assert_allclose(f_of_x, 2.0)
        np.testing.assert_allclose(gradient, 1.0)
        np.testing.assert_allclose(hessian, 0.0)

    def test_analytic_mixed_constant_hessian(self, simple_x):
        # f(C0, C1) = C0 * C1 + C0**2
        stack = np.array(
            [
                [CONSTANT, 0, 0],
                [CONSTANT, 1, 0],
                [MULTIPLICATION, 0, 1],
                [SQUARE, 0, 0],
                [ADDITION, 2, 3],
            ],
            dtype=np.uint8,
        )

        f_of_x, gradient, hessian = evaluation.evaluate_with_const_hessian(
            stack, simple_x, (2.0, 3.0), ()
        )

        np.testing.assert_allclose(f_of_x, 10.0)
        np.testing.assert_allclose(gradient, [[7.0, 2.0]] * len(simple_x))
        expected_hessian = np.array([[2.0, 1.0], [1.0, 0.0]])
        np.testing.assert_allclose(hessian, [expected_hessian] * len(simple_x))
        np.testing.assert_allclose(hessian, hessian.swapaxes(1, 2))

    def test_shared_subexpression_matches_finite_difference(self, simple_x):
        # f(C0, C1) = sin(C0 * C1) + C0 * C1
        stack = np.array(
            [
                [CONSTANT, 0, 0],
                [CONSTANT, 1, 0],
                [MULTIPLICATION, 0, 1],
                [SIN, 2, 0],
                [ADDITION, 3, 2],
            ],
            dtype=np.uint8,
        )
        constants = (0.8, 1.1)

        _, gradient, hessian = evaluation.evaluate_with_const_hessian(
            stack, simple_x, constants, ()
        )
        _, expected_gradient = evaluation.evaluate_with_derivative(
            stack, simple_x, constants, (), False
        )
        expected_hessian = _finite_difference_hessian(stack, simple_x, constants, ())

        np.testing.assert_allclose(gradient, expected_gradient)
        np.testing.assert_allclose(hessian, expected_hessian, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize(
        "operator, constant",
        [
            (SQUARE, 1.3),
            (CUBE, 1.3),
            (SQRT, -1.3),
            (ABS, -1.3),
            (EXPONENTIAL, 0.4),
            (LOGARITHM, -1.3),
            (SIN, 0.4),
            (COS, 0.4),
            (TAN, 0.4),
            (ARCSIN, 0.4),
            (ARCCOS, 0.4),
            (ARCTAN, 0.4),
            (SINH, 0.4),
            (COSH, 0.4),
            (TANH, 0.4),
        ],
    )
    def test_unary_operators_match_finite_difference(
        self, simple_x, operator, constant
    ):
        stack = np.array(
            [
                [CONSTANT, 0, 0],
                [operator, 0, 0],
            ],
            dtype=np.uint8,
        )

        _, gradient, hessian = evaluation.evaluate_with_const_hessian(
            stack, simple_x, (constant,), ()
        )
        _, expected_gradient = evaluation.evaluate_with_derivative(
            stack, simple_x, (constant,), (), False
        )
        expected_hessian = _finite_difference_hessian(
            stack, simple_x, (constant,), ()
        )

        np.testing.assert_allclose(gradient, expected_gradient)
        np.testing.assert_allclose(hessian, expected_hessian, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize(
        "operator, constants",
        [
            (ADDITION, (1.3, -0.7)),
            (SUBTRACTION, (1.3, -0.7)),
            (MULTIPLICATION, (1.3, -0.7)),
            (DIVISION, (1.3, 0.7)),
            (POWER, (1.3, 0.7)),
            (SAFE_POWER, (-1.3, 0.7)),
        ],
    )
    def test_binary_operators_match_finite_difference(
        self, simple_x, operator, constants
    ):
        stack = np.array(
            [
                [CONSTANT, 0, 0],
                [CONSTANT, 1, 0],
                [operator, 0, 1],
            ],
            dtype=np.uint8,
        )

        _, gradient, hessian = evaluation.evaluate_with_const_hessian(
            stack, simple_x, constants, ()
        )
        _, expected_gradient = evaluation.evaluate_with_derivative(
            stack, simple_x, constants, (), False
        )
        expected_hessian = _finite_difference_hessian(stack, simple_x, constants, ())

        np.testing.assert_allclose(gradient, expected_gradient)
        np.testing.assert_allclose(hessian, expected_hessian, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(hessian, hessian.swapaxes(1, 2))

    def test_power_with_fixed_integer_exponent_has_finite_hessian(self, simple_x):
        stack = np.array(
            [
                [CONSTANT, 0, 0],
                [INTEGER, 0, 0],
                [POWER, 0, 1],
            ],
            dtype=np.uint8,
        )

        _, gradient, hessian = evaluation.evaluate_with_const_hessian(
            stack, simple_x, (-2.0,), (2,)
        )

        np.testing.assert_allclose(gradient, -4.0)
        np.testing.assert_allclose(hessian, 2.0)

    def test_constant_free_expression_has_empty_derivatives(self, simple_x):
        stack = np.array([[VARIABLE, 0, 0]], dtype=np.uint8)

        f_of_x, gradient, hessian = evaluation.evaluate_with_const_hessian(
            stack, simple_x, (), ()
        )

        np.testing.assert_allclose(f_of_x, simple_x[:, 0:1])
        assert gradient.shape == (len(simple_x), 0)
        assert hessian.shape == (len(simple_x), 0, 0)

    def test_integer_terminal_has_empty_derivatives(self, simple_x):
        stack = np.array([[INTEGER, 0, 0]], dtype=np.uint8)

        f_of_x, gradient, hessian = evaluation.evaluate_with_const_hessian(
            stack, simple_x, (), (7,)
        )

        np.testing.assert_allclose(f_of_x, 7.0)
        assert gradient.shape == (len(simple_x), 0)
        assert hessian.shape == (len(simple_x), 0, 0)

    def test_singular_power_preserves_nan_derivatives(self, simple_x):
        stack = np.array(
            [
                [CONSTANT, 0, 0],
                [CONSTANT, 1, 0],
                [POWER, 0, 1],
            ],
            dtype=np.uint8,
        )

        _, gradient, hessian = evaluation.evaluate_with_const_hessian(
            stack, simple_x, (0.0, 2.0), ()
        )

        assert np.isnan(gradient).all()
        assert np.isnan(hessian).all()

    def test_logarithm_at_zero_preserves_existing_nan_hessian(self, simple_x):
        stack = np.array(
            [
                [CONSTANT, 0, 0],
                [LOGARITHM, 0, 0],
            ],
            dtype=np.uint8,
        )

        _, gradient, hessian = evaluation.evaluate_with_const_hessian(
            stack, simple_x, (0.0,), ()
        )

        assert np.isposinf(gradient).all()
        assert np.isnan(hessian).all()

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
