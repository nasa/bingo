# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np
import pytest
import dill

from bingo.symbolic_regression.explicit_regression \
    import ExplicitTrainingData as pyExplicitTrainingData, \
           ExplicitRegression as pyExplicitRegression
from bingo.symbolic_regression.equation import Equation as pyEquation
try:
    from bingocpp import ExplicitTrainingData as cppExplicitTrainingData, \
                         ExplicitRegression as cppExplicitRegression, \
                         Equation as cppEquation
    bingocpp = True
except ImportError:
    bingocpp = False

CPP_PARAM = pytest.param("Cpp",
                         marks=pytest.mark.skipif(not bingocpp,
                                                  reason='BingoCpp import '
                                                         'failure'))


@pytest.fixture(params=["Python", CPP_PARAM])
def engine(request):
    return request.param


@pytest.fixture
def explicit_training_data(engine):
    if engine == "Python":
        return pyExplicitTrainingData
    return cppExplicitTrainingData


@pytest.fixture
def explicit_regression(engine):
    if engine == "Python":
        return pyExplicitRegression
    return cppExplicitRegression


@pytest.fixture
def equation(engine):
    if engine == "Python":
        return pyEquation
    return cppEquation


@pytest.fixture
def sample_training_data(explicit_training_data):
    x = np.arange(10, dtype=float)
    y = np.arange(1, 11, dtype=float)
    return explicit_training_data(x, y)


@pytest.fixture
def sample_regression(sample_training_data, explicit_regression):
    return explicit_regression(sample_training_data, relative=False)


@pytest.fixture
def sample_regression_relative(sample_training_data, explicit_regression):
    return explicit_regression(sample_training_data, relative=True)


@pytest.fixture
def sample_equation(equation):

    class SampleEqu(equation):
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

    return SampleEqu()


@pytest.mark.parametrize("three_dim", ["x", "y"])
def test_raises_error_on_training_data_with_high_dims(explicit_training_data,
                                                      three_dim):
    x = np.zeros(10)
    y = np.zeros(10)
    if three_dim == "x":
        x = x.reshape((-1, 1, 1))
    else:
        y = y.reshape((-1, 1, 1))

    with pytest.raises(TypeError):
        explicit_training_data(x, y)


def test_training_data_xy(explicit_training_data):
    x = np.zeros(10)
    y = np.ones(10)
    etd = explicit_training_data(x, y)
    np.testing.assert_array_equal(etd.x, np.zeros((10, 1)))
    np.testing.assert_array_equal(etd.y, np.ones((10, 1)))


def test_training_data_slicing(sample_training_data):
    indices = [2, 4, 6, 8]
    sliced_etd = sample_training_data[indices]
    expected_x = np.array(indices).reshape((-1, 1))
    expected_y = np.array(indices).reshape((-1, 1)) + 1
    np.testing.assert_array_equal(sliced_etd.x, expected_x)
    np.testing.assert_array_equal(sliced_etd.y, expected_y)


@pytest.mark.parametrize("num_elements", range(1, 4))
def test_training_data_len(explicit_training_data, num_elements):
    x = np.arange(num_elements)
    y = np.arange(num_elements)
    etd = explicit_training_data(x, y)
    assert len(etd) == num_elements


def test_explicit_regression(sample_regression, sample_equation):
    fit_vec = sample_regression.evaluate_fitness_vector(sample_equation)
    expected_fit_vec = 1 - np.arange(1, 11)
    np.testing.assert_array_equal(fit_vec, expected_fit_vec)


def test_explicit_regression_relative(sample_regression_relative,
                                      sample_equation):
    fit_vec = sample_regression_relative.evaluate_fitness_vector(
        sample_equation)
    expected_fit_vec = (1 - np.arange(1, 11)) / np.arange(1, 11)
    np.testing.assert_array_equal(fit_vec, expected_fit_vec)


def test_can_pickle(sample_regression):
    _ = dill.loads(dill.dumps(sample_regression))
