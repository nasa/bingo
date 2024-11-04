# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring

import numpy as np
import pytest

from onnx.reference import ReferenceEvaluator
from onnx.checker import check_model

from bingo.symbolic_regression.agraph.operator_definitions import (
    INTEGER,
    VARIABLE,
    CONSTANT,
    ADDITION,
    SUBTRACTION,
    MULTIPLICATION,
    DIVISION,
    SIN,
    COS,
    EXPONENTIAL,
    LOGARITHM,
    POWER,
    ABS,
    SQRT,
    SAFE_POWER,
    SINH,
    COSH,
)
from bingo.symbolic_regression.agraph.onnx_interface import make_onnx_model


@pytest.fixture
def sample_command_array(mocker):
    return np.array(
        [
            [VARIABLE, 0, 0],
            [CONSTANT, 1, 1],
            [COS, 1, 1],
            [MULTIPLICATION, 0, 2],
            [SIN, 3, 0],
        ]
    )  # sin(x0*cos(c1))


@pytest.fixture
def all_funcs_command_array():
    return np.array(
        [
            [INTEGER, 5, 5],
            [VARIABLE, 0, 0],
            [CONSTANT, 0, 0],
            [ADDITION, 1, 0],
            [SUBTRACTION, 2, 3],
            [MULTIPLICATION, 4, 1],
            [DIVISION, 5, 1],
            [SIN, 6, 0],
            [COS, 7, 0],
            [EXPONENTIAL, 8, 0],
            [LOGARITHM, 9, 0],
            [POWER, 10, 0],
            [ABS, 11, 0],
            [SQRT, 12, 0],
            [SAFE_POWER, 13, 1],
            [SINH, 14, 0],
            [COSH, 15, 0],
        ]
    )


def test_can_create_onnx_model_and_run_it(sample_command_array):
    constants = np.array([0.1, 2.0], dtype=np.float32)
    onnx_model = make_onnx_model(sample_command_array, constants)

    check_model(onnx_model)
    sess = ReferenceEvaluator(onnx_model)
    x = np.array([[0.2, 0.3], [0.5, 0.6]], dtype=np.float32)
    y = sess.run(None, {"X": x})[0]
    expected_y = np.sin(x[0:, 0].reshape(-1, 1) * np.cos(constants[1]))
    np.testing.assert_almost_equal(expected_y, y)


def test_can_create_model_with_all_bingo_operators(all_funcs_command_array):
    constants = np.array([0.1, 2.0], dtype=np.float32)
    onnx_model = make_onnx_model(all_funcs_command_array, constants)

    check_model(onnx_model)

    x = np.array([[0.2, 0.3], [0.5, 0.6]], dtype=np.float32)
    sess = ReferenceEvaluator(onnx_model)
    y = sess.run(None, {"X": x})[0]
    expected_y = np.array([[1.3887318], [1.2607156]])
    np.testing.assert_almost_equal(expected_y, y)


@pytest.mark.parametrize(
    "safe_op,x_val,expected_y",
    [(LOGARITHM, -np.e, 1), (SQRT, -1, 1), (SAFE_POWER, -1, 1)],
)
def test_operators_with_saftey(safe_op, x_val, expected_y):
    command_array = np.array([[VARIABLE, 0, 0], [INTEGER, 1, 1], [safe_op, 0, 1]])

    onnx_model = make_onnx_model(command_array, np.array([]))
    check_model(onnx_model)

    x = np.array([[x_val]], dtype=np.float32)
    sess = ReferenceEvaluator(onnx_model)
    y = sess.run(None, {"X": x})[0]
    expected_y = np.array([[expected_y]])
    np.testing.assert_almost_equal(expected_y, y)
