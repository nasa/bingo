# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
from collections import namedtuple
import numpy as np
import pytest

from bingo.symbolic_regression import implicit_regression as pyimplicit
from bingo.symbolic_regression.equation import Equation as pyequation
try:
    from bingocpp.build import symbolic_regression as bingocpp
except ImportError:
    bingocpp = None


Implicit = namedtuple("Implicit", ["training_data", "regression", "equation"])

CPP_PARAM = pytest.param("cpp",
                         marks=pytest.mark.skipif(not bingocpp,
                                                  reason='BingoCpp import '
                                                         'failure'))


@pytest.fixture(params=["python", CPP_PARAM])
def implicit(request):
    if request.param == "python":
        return Implicit(pyimplicit.ImplicitTrainingData,
                        pyimplicit.ImplicitRegression,
                        pyequation)
    return Implicit(bingocpp.ImplicitTrainingData,
                    bingocpp.ImplicitRegression,
                    bingocpp.Equation)


@pytest.fixture
def sample_implicit(implicit):
    return _make_sample_implicit(implicit, required_params=None)


def _make_sample_implicit(implicit, required_params):
    x = np.arange(30, dtype=float).reshape((10, 3))
    dx_dt = np.array([[3, 2, 1]]*10, dtype=float)
    itd = implicit.training_data(x, dx_dt)
    if required_params is None:
        reg = implicit.regression(itd)
    else:
        reg = implicit.regression(itd, required_params=required_params)

    class SampleEqu(implicit.equation):
        def evaluate_equation_at(self, x):
            pass

        def evaluate_equation_with_x_gradient_at(self, x):
            sample_df_dx = np.array([[1, 0, -1]]*10)
            return np.ones((10, 3), dtype=float), sample_df_dx

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

        def distance(self, _chromosome):
            return 0

    return Implicit(itd, reg, SampleEqu())


@pytest.mark.parametrize("three_dim", ["x", "dx_dt"])
def test_raises_error_on_training_data_with_high_dims(implicit, three_dim):
    x = np.zeros(10)
    dx_dt = np.zeros((10, 1))
    if three_dim == "x":
        x = x.reshape((-1, 1, 1))
    else:
        dx_dt = dx_dt.reshape((-1, 1, 1))

    with pytest.raises(TypeError):
        implicit.training_data(x, dx_dt)


def test_training_data_xy(implicit):
    x = np.zeros(10)
    dx_dt = np.ones((10, 1))
    itd = implicit.training_data(x, dx_dt)
    np.testing.assert_array_equal(itd.x, np.zeros((10, 1)))
    np.testing.assert_array_equal(itd.dx_dt, np.ones((10, 1)))


def test_training_data_slicing(sample_implicit):
    indices = [2, 4, 6, 8]
    sliced_etd = sample_implicit.training_data[indices]
    expected_x = np.array([[i * 3, i * 3 + 1, i * 3 + 2] for i in indices])
    expected_dx_dt = np.array([[3, 2, 1]]*len(indices), dtype=float)
    np.testing.assert_array_equal(sliced_etd.x, expected_x)
    np.testing.assert_array_equal(sliced_etd.dx_dt, expected_dx_dt)


@pytest.mark.parametrize("num_elements", range(1, 4))
def test_training_data_len(implicit, num_elements):
    x = np.arange(num_elements)
    dx_dt = np.arange(num_elements).reshape((-1, 1))
    etd = implicit.training_data(x, dx_dt)
    assert len(etd) == num_elements


def test_correct_partial_calculation_in_training_data(implicit):
    data_input = np.arange(20, dtype=float).reshape((20, 1))
    data_input = np.c_[data_input * 0,
                       data_input * 1,
                       data_input * 2]
    training_data = implicit.training_data(data_input)

    expected_derivative = np.c_[np.ones(13) * 0,
                                np.ones(13) * 1,
                                np.ones(13) * 2]
    np.testing.assert_array_almost_equal(training_data.dx_dt,
                                         expected_derivative)


def test_correct_partial_calculation_in_training_data_2_sections(implicit):
    data_input = np.arange(20, dtype=float).reshape((20, 1)) * 2.0
    data_input = np.vstack((data_input, [np.nan], data_input))
    training_data = implicit.training_data(data_input)
    expected_derivative = np.full((26, 1), 2.0)
    np.testing.assert_array_almost_equal(training_data.dx_dt,
                                         expected_derivative)


@pytest.mark.parametrize("required_params, expected_fit",
                         [(None, 0.5), (2, 0.5), (3, np.inf)])
def test_implicit_regression(implicit, required_params, expected_fit):
    sample = _make_sample_implicit(implicit, required_params)
    fit_vec = sample.regression.evaluate_fitness_vector(sample.equation)
    expected_fit_vec = np.full((10,), expected_fit, dtype=float)
    np.testing.assert_array_almost_equal(fit_vec, expected_fit_vec)
