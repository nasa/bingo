# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
from collections import namedtuple
import numpy as np
import pytest

from bingo.symbolic_regression import explicit_regression as pyexplicit
from bingo.symbolic_regression.equation import Equation as pyequation
try:
    from bingocpp.build import symbolic_regression as bingocpp
except ImportError:
    bingocpp = None


Explicit = namedtuple("Explicit", ["training_data", "regression", "equation"])

CPP_PARAM = pytest.param("cpp",
                         marks=pytest.mark.skipif(not bingocpp,
                                                  reason='BingoCpp import '
                                                         'failure'))


@pytest.fixture(params=["python", CPP_PARAM])
def explicit(request):
    if request.param == "python":
        return Explicit(pyexplicit.ExplicitTrainingData,
                        pyexplicit.ExplicitRegression,
                        pyequation)
    return Explicit(bingocpp.ExplicitTrainingData,
                    bingocpp.ExplicitRegression,
                    bingocpp.Equation)


@pytest.fixture
def sample_explicit(explicit):
    return _make_sample_explicit(explicit, relative=False)


@pytest.fixture
def sample_explicit_relative(explicit):
    return _make_sample_explicit(explicit, relative=True)


def _make_sample_explicit(explicit, relative):
    x = np.arange(10, dtype=float)
    y = np.arange(1, 11, dtype=float)
    etd = explicit.training_data(x, y)
    reg = explicit.regression(etd, relative=relative)

    class SampleEqu(explicit.equation):
        def evaluate_equation_at(self, x):
            return np.ones((10, 1), dtype=float)

        def evaluate_equation_with_x_gradient_at(self, x):
            pass

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

    return Explicit(etd, reg, SampleEqu())


@pytest.mark.parametrize("three_dim", ["x", "y"])
def test_raises_error_on_training_data_with_high_dims(explicit, three_dim):
    x = np.zeros(10)
    y = np.zeros(10)
    if three_dim == "x":
        x = x.reshape((-1, 1, 1))
    else:
        y = y.reshape((-1, 1, 1))

    with pytest.raises(TypeError):
        explicit.training_data(x, y)


def test_training_data_xy(explicit):
    x = np.zeros(10)
    y = np.ones(10)
    etd = explicit.training_data(x, y)
    np.testing.assert_array_equal(etd.x, np.zeros((10, 1)))
    np.testing.assert_array_equal(etd.y, np.ones((10, 1)))


def test_training_data_slicing(sample_explicit):
    indices = [2, 4, 6, 8]
    sliced_etd = sample_explicit.training_data[indices]
    expected_x = np.array(indices).reshape((-1, 1))
    expected_y = np.array(indices).reshape((-1, 1)) + 1
    np.testing.assert_array_equal(sliced_etd.x, expected_x)
    np.testing.assert_array_equal(sliced_etd.y, expected_y)


@pytest.mark.parametrize("num_elements", range(1, 4))
def test_training_data_len(explicit, num_elements):
    x = np.arange(num_elements)
    y = np.arange(num_elements)
    etd = explicit.training_data(x, y)
    assert len(etd) == num_elements


def test_explicit_regression(sample_explicit):
    fit_vec = sample_explicit.regression.evaluate_fitness_vector(
        sample_explicit.equation)
    expected_fit_vec = 1 - np.arange(1, 11)
    np.testing.assert_array_equal(fit_vec, expected_fit_vec)


def test_explicit_regression_relative(sample_explicit_relative):
    fit_vec = sample_explicit_relative.regression.evaluate_fitness_vector(
        sample_explicit_relative.equation)
    expected_fit_vec = (1 - np.arange(1, 11)) / np.arange(1, 11)
    np.testing.assert_array_equal(fit_vec, expected_fit_vec)
