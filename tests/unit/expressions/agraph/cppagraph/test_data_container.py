"""Tests for cppagraph DataContainer — must match pyagraph behavior."""

import numpy as np
import pytest

from bingo.expressions.agraph.cppagraph import DataContainer
from bingo.expressions.agraph.pyagraph.data_container import (
    DataContainer as PyDataContainer,
)


class TestConstruction:
    def test_2d_arrays(self):
        dc = DataContainer(np.ones((5, 2)), np.ones((5, 1)))
        assert dc.x.shape == (5, 2)
        assert dc.y.shape == (5, 1)

    def test_1d_auto_reshape(self):
        dc = DataContainer(np.ones(5), np.ones(5))
        assert dc.x.shape == (5, 1)
        assert dc.y.shape == (5, 1)

    def test_mismatched_rows_raises(self):
        with pytest.raises(ValueError, match="rows"):
            DataContainer(np.ones((3, 2)), np.ones((5, 1)))

    def test_from_lists(self):
        dc = DataContainer([1, 2, 3], [4, 5, 6])
        assert dc.x.shape == (3, 1)
        assert dc.y.shape == (3, 1)
        assert dc.x.dtype == float


class TestSlicing:
    def test_getitem_returns_data_container(self):
        dc = DataContainer(np.arange(10).reshape(5, 2), np.arange(5).reshape(5, 1))
        sliced = dc[1:3]
        assert isinstance(sliced, DataContainer)
        assert len(sliced) == 2
        np.testing.assert_array_equal(sliced.x, dc.x[1:3])
        np.testing.assert_array_equal(sliced.y, dc.y[1:3])

    def test_len(self):
        dc = DataContainer(np.ones((7, 3)), np.ones((7, 1)))
        assert len(dc) == 7


class TestCrossImplementation:
    """Verify cppagraph DataContainer matches pyagraph DataContainer."""

    def test_values_match(self):
        x = np.random.default_rng(42).standard_normal((10, 3))
        y = np.random.default_rng(42).standard_normal((10, 1))
        cpp_dc = DataContainer(x, y)
        py_dc = PyDataContainer(x, y)
        np.testing.assert_array_equal(cpp_dc.x, py_dc.x)
        np.testing.assert_array_equal(cpp_dc.y, py_dc.y)

    def test_slicing_matches(self):
        x = np.arange(20, dtype=float).reshape(5, 4)
        y = np.arange(5, dtype=float).reshape(5, 1)
        cpp_dc = DataContainer(x, y)
        py_dc = PyDataContainer(x, y)
        for s in [slice(0, 3), slice(2, 5), slice(1, 4)]:
            np.testing.assert_array_equal(cpp_dc[s].x, py_dc[s].x)
            np.testing.assert_array_equal(cpp_dc[s].y, py_dc[s].y)
