# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import warnings
import pytest
import numpy as np

from bingo.SymbolicRegression.ImplicitRegression import ImplicitRegression, \
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


def test_reshaping_of_training_data():
    x = np.zeros(5)
    dx_dt = np.zeros((5, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        training_data = ImplicitTrainingData(x, dx_dt)
    assert training_data.x.ndim == 2


def test_poorly_shaped_input_x_of_training_data():
    x = np.zeros((5, 1, 1))
    dx_dt = np.zeros((5, 1))
    with pytest.raises(ValueError):
        _ = ImplicitTrainingData(x, dx_dt)


def test_poorly_shaped_input_dx_dt_of_training_data():
    x = np.zeros((5, 1))
    dx_dt = np.zeros(5)
    with pytest.raises(ValueError):
        _ = ImplicitTrainingData(x, dx_dt)


def test_getting_subset_of_training_data():
    data_input = np.arange(5).reshape((5, 1))
    training_data = ImplicitTrainingData(data_input, data_input)
    subset_training_data = training_data[[0, 2, 3]]

    expected_subset = np.array([[0], [2], [3]])
    np.testing.assert_array_equal(subset_training_data.x,
                                  expected_subset)
    np.testing.assert_array_equal(subset_training_data.dx_dt,
                                  expected_subset)


@pytest.mark.parametrize("input_size", [2, 5, 50])
def test_correct_training_data_length(input_size):
    data_input = np.arange(input_size).reshape((-1, 1))
    training_data = ImplicitTrainingData(data_input, data_input)
    assert len(training_data) == input_size


def test_correct_partial_calculation_in_training_data():
    data_input = np.arange(20, dtype=float).reshape((20, 1))
    data_input = np.c_[data_input * 0,
                       data_input * 1,
                       data_input * 2]
    training_data = ImplicitTrainingData(data_input)
    expected_derivative = np.c_[np.ones(13) * 0,
                                np.ones(13) * 1,
                                np.ones(13) * 2]
    np.testing.assert_array_almost_equal(training_data.dx_dt,
                                         expected_derivative)


def test_correct_partial_calculation_in_training_data_2():
    data_input = np.arange(20, dtype=float).reshape((20, 1)) * 2.0
    data_input = np.vstack((data_input, [np.nan], data_input))
    training_data = ImplicitTrainingData(data_input)
    expected_derivative = np.full((26, 1), 2.0)
    np.testing.assert_array_almost_equal(training_data.dx_dt,
                                         expected_derivative)
