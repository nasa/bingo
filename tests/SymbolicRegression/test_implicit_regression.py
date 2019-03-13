# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.ImplicitRegression import ImplicitRegression, \
                                     ImplicitRegressionSchmidt, \
                                     ImplicitTrainingData


class SampleTrainingData:
    def __init__(self, x, dx_dt):
        self.x = x
        self.dx_dt = dx_dt


@pytest.fixture()
def dummy_training_data():
    x = np.linspace(0, 1, 50, endpoint=False).reshape((-1, 5))
    dx_dt = np.ones(x.shape)
    dx_dt[:, [3, 4]] *= -1
    dx_dt[:, 2] = 0
    return SampleTrainingData(x, dx_dt)


@pytest.mark.parametrize("normalize_dot", [True, False])
def test_implicit_regression(dummy_sum_equation, dummy_training_data,
                             normalize_dot):
    regressor = ImplicitRegression(dummy_training_data,
                                   required_params=None,
                                   normalize_dot=normalize_dot)
    fitness = regressor(dummy_sum_equation)
    np.testing.assert_almost_equal(fitness, 0.14563031020)


@pytest.mark.parametrize("required_params, infinite_fitness_expected",
                         [(4, False), (5, True)])
def test_implicit_regression_no_normalization(dummy_sum_equation,
                                              dummy_training_data,
                                              required_params,
                                              infinite_fitness_expected):
    regressor = ImplicitRegression(dummy_training_data,
                                   required_params=required_params,
                                   normalize_dot=False)
    fitness = regressor(dummy_sum_equation)
    assert np.isinf(fitness) == infinite_fitness_expected


def test_schmidt_regression(dummy_sum_equation, dummy_training_data):
    regressor = ImplicitRegressionSchmidt(dummy_training_data)
    fitness = regressor(dummy_sum_equation)
    np.testing.assert_almost_equal(fitness, 0.44420421701352086)
