"""Tests for generic Python fitness aggregation."""

import numpy as np
import pytest

from bingo.evaluation.fitness_function import FitnessFunction, VectorBasedFunction
from bingo.evaluation.training_data import TrainingData


class _VectorFitness(VectorBasedFunction):
    def evaluate_fitness_vector(self, individual):
        return np.array([-2.0, -1.0, 0.0, 1.0, 2.0])


class _NanVectorFitness(VectorBasedFunction):
    def evaluate_fitness_vector(self, individual):
        return np.array([np.nan, -1.0, 0.0, 1.0, 2.0])


def test_fitness_function_cannot_be_instantiated():
    with pytest.raises(TypeError):
        FitnessFunction()


def test_fitness_function_stores_training_data(mocker):
    mocker.patch.object(FitnessFunction, "__abstractmethods__", new_callable=set)
    mocker.patch.object(TrainingData, "__abstractmethods__", new_callable=set)
    training_data = TrainingData()

    fitness = FitnessFunction(training_data)

    assert fitness.eval_count == 0
    assert fitness.training_data is training_data


@pytest.mark.parametrize(
    "metric, expected",
    [
        ("mae", 1.2),
        ("mean absolute error", 1.2),
        ("mse", 2.0),
        ("mean squared error", 2.0),
        ("rmse", np.sqrt(2.0)),
        ("root mean squared error", np.sqrt(2.0)),
    ],
)
def test_vector_fitness_aggregates_its_error_vector(metric, expected):
    assert _VectorFitness(metric=metric)(object()) == pytest.approx(expected)


def test_vector_fitness_rejects_unknown_metric():
    with pytest.raises(ValueError):
        _VectorFitness(metric="unknown")


@pytest.mark.parametrize(
    "metric",
    [
        "mae",
        "mean absolute error",
        "mse",
        "mean squared error",
        "rmse",
        "root mean squared error",
    ],
)
def test_vector_fitness_propagates_nan(metric):
    assert np.isnan(_NanVectorFitness(metric=metric)(object()))
