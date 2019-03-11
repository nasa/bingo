# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.Base.Equation import Equation
from bingo.EquationFitnessEvaluators import ExplicitRegression, \
                                            ImplicitRegression


class SumEqualtion(Equation):
    def evaluate_equation_at(self, x):
        return np.sum(x, axis=1).reshape((-1, 1))

    def evaluate_equation_with_x_gradient_at(self, x):
        x_sum = self.evaluate_equation_at(x)
        return x_sum, x

    def evaluate_equation_with_local_opt_gradient_at(self, x):
        pass

    def get_complexity(self):
        pass

    def get_latex_string(self):
        pass

    def get_console_string(self):
        pass

    def __str__(self):
        pass


class SampleTrainingData:
    def __init__(self, x=None, y=None, dx_dt=None):
        self.x = x
        self.y = y
        self.dx_dt = dx_dt


@pytest.fixture()
def dummy_sum_equation():
    return SumEqualtion()


@pytest.fixture()
def dummy_training_data():
    x = np.linspace(0, 1, 50, endpoint=False).reshape((-1, 5))
    y = np.linspace(0.2, 4.7, 10).reshape((-1, 1))
    dx_dt = np.ones(x.shape)
    dx_dt[:, [3, 4]] *= -1
    dx_dt[:, 2] = 0
    return SampleTrainingData(x=x, y=y, dx_dt=dx_dt)


@pytest.fixture()
def dummy_explicit_training_data_with_nan(dummy_training_data):
    return dummy_training_data


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


@pytest.mark.parametrize("normalize_dot", [True, False])
def test_implicit_regression(dummy_sum_equation, dummy_training_data,
                             normalize_dot):
    regressor = ImplicitRegression(dummy_training_data,
                                   required_params=None,
                                   normalize_dot=normalize_dot)
    fitness = regressor(dummy_sum_equation)
    np.testing.assert_almost_equal(fitness, .14563031020)


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
