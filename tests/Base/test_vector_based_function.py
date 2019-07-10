# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.Base.FitnessFunction import VectorBasedFunction


class DummyVectorFunction(VectorBasedFunction):
    def evaluate_fitness_vector(self, individual):
        return individual


@pytest.mark.parametrize("metric, expected_value",
                         [("mean absolute error", 2.),
                          ("mae", 2.),
                          ("mean squared error", 14/3),
                          ("mse", 14/3),
                          ("root mean squared error", np.sqrt(14/3)),
                          ("rmse", np.sqrt(14/3))])
def test_using_metrics(metric, expected_value):
    fitness_funtion = DummyVectorFunction(metric=metric)
    assert fitness_funtion([1, 2, 3]) == expected_value


def test_invalid_metric():
    with pytest.raises(KeyError):
        _ = DummyVectorFunction(metric="non existent metric")