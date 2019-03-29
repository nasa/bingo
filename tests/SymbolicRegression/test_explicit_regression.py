# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import warnings
import pytest
import numpy as np

from bingo.SymbolicRegression.ExplicitRegression import ExplicitRegression, ExplicitTrainingData


class SampleTrainingData:
    def __init__(self, x, y):
        self.x = x
        self.y = y


@pytest.fixture()
def dummy_training_data():
    x = np.linspace(0, 1, 50, endpoint=False).reshape((-1, 5))
    y = np.linspace(0.2, 4.7, 10).reshape((-1, 1))
    return SampleTrainingData(x, y)


def test_explicit_regression(dummy_sum_equation, dummy_training_data):
    regressor = ExplicitRegression(dummy_training_data)
    fitness = regressor(dummy_sum_equation)
    np.testing.assert_almost_equal(fitness, 0)


def test_explicit_regression_with_nan(dummy_sum_equation,
                                      dummy_training_data):
    dummy_training_data.x[0, 0] = np.nan
    regressor = ExplicitRegression(dummy_training_data)
    fitness = regressor(dummy_sum_equation)
    assert np.isnan(fitness)


def test_reshaping_of_training_data():
    one_dim_input = np.zeros(5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        training_data = ExplicitTrainingData(one_dim_input, one_dim_input)
    assert training_data.x.ndim == 2
    assert training_data.y.ndim == 2


def test_poorly_shaped_input_x_of_training_data():
    x = np.zeros((5, 3, 3))
    y = x.flatten()
    with pytest.raises(ValueError):
        _ = ExplicitTrainingData(x, y)


def test_poorly_shaped_input_y_of_training_data():
    y = np.zeros((5, 3, 3))
    x = y.flatten()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError):
            _ = ExplicitTrainingData(x, y)


def test_getting_subset_of_training_data():
    data_input = np.arange(5).reshape((-1, 1))
    training_data = ExplicitTrainingData(data_input, data_input)
    subset_training_data = training_data[[0, 2, 3]]

    expected_subset = np.array([[0], [2], [3]])
    np.testing.assert_array_equal(subset_training_data.x,
                                  expected_subset)
    np.testing.assert_array_equal(subset_training_data.y,
                                  expected_subset)


@pytest.mark.parametrize("input_size", [2, 5, 50])
def test_correct_training_data_length(input_size):
    data_input = np.arange(input_size).reshape((-1, 1))
    training_data = ExplicitTrainingData(data_input, data_input)
    assert len(training_data) == input_size
