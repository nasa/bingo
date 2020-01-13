# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import warnings
import pytest
import numpy as np

from bingo.symbolic_regression.explicit_regression import ExplicitRegression, ExplicitTrainingData
try:
    from bingocpp.build import bingocpp as bingocpp
except ImportError:
    bingocpp = None


class SampleTrainingData:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def init_x_and_y():
    x = np.linspace(0, 1, 50, endpoint=False).reshape((-1, 5))
    y = np.linspace(0.2, 4.7, 10).reshape((-1, 1))
    return (x, y)


@pytest.fixture()
def dummy_training_data():
    x, y = init_x_and_y()
    return SampleTrainingData(x, y)


@pytest.fixture()
def dummy_training_data_cpp():
    if bingocpp is None:
        return None
    x, y = init_x_and_y()
    return bingocpp.ExplicitTrainingData(x, y)


@pytest.fixture(params=[
    "python",
    pytest.param("cpp", marks=pytest.mark.skipif(not bingocpp,
                        reason='BingoCpp import failure'))
])
def explicit_data(request, dummy_training_data, dummy_training_data_cpp,
                           dummy_sum_equation, dummy_sum_equation_cpp):
    if request.param == "python":
        return (dummy_training_data, dummy_sum_equation)
    return (dummy_training_data_cpp, dummy_sum_equation_cpp)


@pytest.fixture(params=[
    "python",
    pytest.param("cpp", marks=pytest.mark.skipif(not bingocpp,
                        reason='BingoCpp import failure'))
])
def explicit_data_nan(request, dummy_sum_equation, dummy_sum_equation_cpp):
    x, y = init_x_and_y()
    x[0, 0] = np.nan
    if request.param == "python":
        return (SampleTrainingData(x, y), dummy_sum_equation)
    return (bingocpp.ExplicitTrainingData(x, y), dummy_sum_equation_cpp)


def test_explicit_regression(explicit_data):
    dummy_training_data = explicit_data[0]
    dummy_sum_equation = explicit_data[1]
    regressor = ExplicitRegression(dummy_training_data)
    fitness = regressor(dummy_sum_equation)
    np.testing.assert_almost_equal(fitness, 0)


def test_explicit_regression_with_nan(explicit_data_nan):
    dummy_training_data = explicit_data_nan[0]
    dummy_sum_equation = explicit_data_nan[1]
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


@pytest.mark.parametrize("python", 
    [True, pytest.param(False, marks = pytest.mark.skipif(not bingocpp,
                               reason = 'BingoCpp import failure'))])
def test_getting_subset_of_training_data(python):
    data_input = np.arange(5).reshape((-1, 1))
    training_data = ExplicitTrainingData(data_input, data_input) \
                    if python \
                    else bingocpp.ExplicitTrainingData(data_input, data_input)
    subset_training_data = training_data[[0, 2, 3]]

    expected_subset = np.array([[0], [2], [3]])
    np.testing.assert_array_equal(subset_training_data.x,
                                  expected_subset)
    np.testing.assert_array_equal(subset_training_data.y,
                                  expected_subset)


@pytest.mark.parametrize("python", 
    [True, pytest.param(False, marks = pytest.mark.skipif(not bingocpp,
                               reason = 'BingoCpp import failure'))])
@pytest.mark.parametrize("input_size", [2, 5, 50])
def test_correct_training_data_length(python, input_size):
    data_input = np.arange(input_size).reshape((-1, 1))
    training_data = ExplicitTrainingData(data_input, data_input)
    assert len(training_data) == input_size
