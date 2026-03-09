"""Tests for bingo.expressions.data_container"""

import numpy as np
import pytest

from bingo.expressions.agraph.pyagraph.data_container import DataContainer


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
