"""Tests for bingo.expressions.agraph.onnx_interface"""

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
onnxruntime = pytest.importorskip("onnxruntime")

from bingo.expressions.agraph.pyagraph.operators import (
    VARIABLE,
    CONSTANT,
    INTEGER,
    ADDITION,
    MULTIPLICATION,
    SIN,
    SQUARE,
)
from bingo.expressions.agraph.pyagraph.onnx_interface import make_onnx_model


@pytest.fixture
def sample_x():
    return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)


def _run_onnx(model, x):
    """Helper: run ONNX model with onnxruntime."""
    sess = onnxruntime.InferenceSession(model.SerializeToString())
    return sess.run(None, {"X": x})[0]


class TestOnnxModelGeneration:
    def test_single_variable(self, sample_x):
        stack = np.array([[VARIABLE, 0, 0]], dtype=np.uint8)
        model = make_onnx_model(stack, (), ())
        result = _run_onnx(model, sample_x)
        np.testing.assert_array_almost_equal(result.ravel(), sample_x[:, 0])

    def test_x0_plus_c0(self, sample_x):
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [CONSTANT, 0, 0],
                [ADDITION, 0, 1],
            ],
            dtype=np.uint8,
        )
        model = make_onnx_model(stack, (10.0,), ())
        result = _run_onnx(model, sample_x)
        expected = sample_x[:, 0] + 10.0
        np.testing.assert_array_almost_equal(result.ravel(), expected, decimal=4)

    def test_integer_node(self, sample_x):
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [INTEGER, 0, 0],
                [ADDITION, 0, 1],
            ],
            dtype=np.uint8,
        )
        model = make_onnx_model(stack, (), (5,))
        result = _run_onnx(model, sample_x)
        expected = sample_x[:, 0] + 5.0
        np.testing.assert_array_almost_equal(result.ravel(), expected, decimal=4)

    def test_square(self, sample_x):
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [SQUARE, 0, 0],
            ],
            dtype=np.uint8,
        )
        model = make_onnx_model(stack, (), ())
        result = _run_onnx(model, sample_x)
        expected = sample_x[:, 0] ** 2
        np.testing.assert_array_almost_equal(result.ravel(), expected, decimal=4)

    def test_sin(self, sample_x):
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [SIN, 0, 0],
            ],
            dtype=np.uint8,
        )
        model = make_onnx_model(stack, (), ())
        result = _run_onnx(model, sample_x)
        expected = np.sin(sample_x[:, 0])
        np.testing.assert_array_almost_equal(result.ravel(), expected, decimal=4)

    def test_model_has_expected_name(self):
        stack = np.array([[VARIABLE, 0, 0]], dtype=np.uint8)
        model = make_onnx_model(stack, (), (), name="test_model")
        assert model.graph.name == "test_model"
