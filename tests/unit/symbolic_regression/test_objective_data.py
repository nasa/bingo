import numpy as np
import pytest

from bingo.symbolic_regression import ObjectiveData


def test_objective_data_exposes_aligned_arrays_and_indexes_them_together():
    data = ObjectiveData(np.arange(4), np.arange(8).reshape(4, 2))

    assert isinstance(data.arrays, tuple)
    np.testing.assert_array_equal(data.arrays[0], np.arange(4))
    np.testing.assert_array_equal(data.arrays[1], np.arange(8).reshape(4, 2))

    subset = data[[3, 1]]

    assert isinstance(subset, ObjectiveData)
    np.testing.assert_array_equal(subset.arrays[0], [3, 1])
    np.testing.assert_array_equal(subset.arrays[1], [[6, 7], [2, 3]])


def test_objective_data_rejects_misaligned_arrays():
    with pytest.raises(ValueError, match="equal first-axis length"):
        ObjectiveData(np.arange(2), np.arange(3))


def test_objective_data_scalar_index_retains_the_sample_axis():
    data = ObjectiveData(np.arange(4), np.arange(8).reshape(4, 2))

    subset = data[2]

    np.testing.assert_array_equal(subset.arrays[0], [2])
    np.testing.assert_array_equal(subset.arrays[1], [[4, 5]])
