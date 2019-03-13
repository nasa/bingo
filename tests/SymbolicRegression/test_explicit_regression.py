# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.ExplicitRegression import ExplicitRegression, ExplicitTrainingData


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
