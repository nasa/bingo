"""Tests for bingo.expressions.agraph.operator_eval"""

import numpy as np
import pytest

from bingo.expressions.agraph.operators import (
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
from bingo.expressions.agraph.evaluation.operator_eval import (
    FORWARD_EVAL_MAP,
    REVERSE_EVAL_MAP,
    forward_eval_function,
    reverse_eval_function,
)


@pytest.fixture
def sample_data():
    """Provide sample x, constants, integers, and precomputed forward values."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    constants = (2.5, -1.0)
    integers = (3, 7)
    # forward_eval[0] = x[:, 0], forward_eval[1] = x[:, 1]
    fwd = [
        x[:, 0].reshape(-1, 1),
        x[:, 1].reshape(-1, 1),
    ]
    return x, constants, integers, fwd


class TestForwardEvalMaps:
    def test_all_operators_have_forward(self):
        from bingo.expressions.agraph.operators import IS_TERMINAL_MAP

        for op in IS_TERMINAL_MAP:
            assert op in FORWARD_EVAL_MAP

    def test_all_operators_have_reverse(self):
        from bingo.expressions.agraph.operators import IS_TERMINAL_MAP

        for op in IS_TERMINAL_MAP:
            assert op in REVERSE_EVAL_MAP


class TestTerminalForwardEval:
    def test_variable_loads_column(self, sample_data):
        x, constants, integers, fwd = sample_data
        result = forward_eval_function(VARIABLE, 0, 0, x, constants, integers, fwd)
        np.testing.assert_array_equal(result, x[:, 0].reshape(-1, 1))

    def test_constant_loads_value(self, sample_data):
        x, constants, integers, fwd = sample_data
        result = forward_eval_function(CONSTANT, 0, 0, x, constants, integers, fwd)
        assert result == 2.5

    def test_integer_loads_value(self, sample_data):
        x, constants, integers, fwd = sample_data
        result = forward_eval_function(INTEGER, 1, 1, x, constants, integers, fwd)
        assert result == 7.0


class TestArithmeticForwardEval:
    def test_addition(self, sample_data):
        x, constants, integers, fwd = sample_data
        result = forward_eval_function(ADDITION, 0, 1, x, constants, integers, fwd)
        np.testing.assert_array_almost_equal(result, fwd[0] + fwd[1])

    def test_subtraction(self, sample_data):
        x, constants, integers, fwd = sample_data
        result = forward_eval_function(SUBTRACTION, 0, 1, x, constants, integers, fwd)
        np.testing.assert_array_almost_equal(result, fwd[0] - fwd[1])

    def test_multiplication(self, sample_data):
        x, constants, integers, fwd = sample_data
        result = forward_eval_function(
            MULTIPLICATION, 0, 1, x, constants, integers, fwd
        )
        np.testing.assert_array_almost_equal(result, fwd[0] * fwd[1])

    def test_division(self, sample_data):
        x, constants, integers, fwd = sample_data
        result = forward_eval_function(DIVISION, 0, 1, x, constants, integers, fwd)
        np.testing.assert_array_almost_equal(result, fwd[0] / fwd[1])


class TestUnaryForwardEval:
    @pytest.mark.parametrize(
        "op, np_fn",
        [
            (SIN, np.sin),
            (COS, np.cos),
            (TAN, np.tan),
            (SINH, np.sinh),
            (COSH, np.cosh),
            (TANH, np.tanh),
            (EXPONENTIAL, np.exp),
            (ARCSIN, np.arcsin),
            (ARCCOS, np.arccos),
            (ARCTAN, np.arctan),
        ],
    )
    def test_unary_function(self, op, np_fn, sample_data):
        x, constants, integers, _ = sample_data
        # Use small values to stay in domain
        small_x = np.array([[0.3, 0.5], [0.1, 0.2]])
        fwd = [small_x[:, 0].reshape(-1, 1), small_x[:, 1].reshape(-1, 1)]
        result = forward_eval_function(op, 0, 0, small_x, constants, integers, fwd)
        expected = np_fn(fwd[0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_square(self, sample_data):
        x, constants, integers, fwd = sample_data
        result = forward_eval_function(SQUARE, 0, 0, x, constants, integers, fwd)
        np.testing.assert_array_almost_equal(result, fwd[0] ** 2)

    def test_cube(self, sample_data):
        x, constants, integers, fwd = sample_data
        result = forward_eval_function(CUBE, 0, 0, x, constants, integers, fwd)
        np.testing.assert_array_almost_equal(result, fwd[0] ** 3)

    def test_abs(self, sample_data):
        x, constants, integers, _ = sample_data
        neg_x = np.array([[-1.0, -2.0], [-3.0, -4.0]])
        fwd = [neg_x[:, 0].reshape(-1, 1)]
        result = forward_eval_function(ABS, 0, 0, neg_x, constants, integers, fwd)
        np.testing.assert_array_almost_equal(result, np.abs(fwd[0]))

    def test_sqrt(self, sample_data):
        x, constants, integers, fwd = sample_data
        result = forward_eval_function(SQRT, 0, 0, x, constants, integers, fwd)
        np.testing.assert_array_almost_equal(result, np.sqrt(np.abs(fwd[0])))

    def test_log(self, sample_data):
        x, constants, integers, fwd = sample_data
        result = forward_eval_function(LOGARITHM, 0, 0, x, constants, integers, fwd)
        np.testing.assert_array_almost_equal(result, np.log(np.abs(fwd[0])))


class TestPowerForwardEval:
    def test_power(self, sample_data):
        x, constants, integers, fwd = sample_data
        result = forward_eval_function(POWER, 0, 1, x, constants, integers, fwd)
        np.testing.assert_array_almost_equal(result, np.power(fwd[0], fwd[1]))

    def test_safe_power(self, sample_data):
        x, constants, integers, _ = sample_data
        neg_x = np.array([[-2.0, 3.0], [-4.0, 2.0]])
        fwd = [neg_x[:, 0].reshape(-1, 1), neg_x[:, 1].reshape(-1, 1)]
        result = forward_eval_function(
            SAFE_POWER, 0, 1, neg_x, constants, integers, fwd
        )
        expected = np.power(np.abs(fwd[0]), fwd[1])
        np.testing.assert_array_almost_equal(result, expected)
