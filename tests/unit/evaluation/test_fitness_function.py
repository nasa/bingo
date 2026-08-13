"""Tests for the fitness-function base class."""
import pytest

from bingo.evaluation.fitness_function import FitnessFunction
from bingo.evaluation.training_data import TrainingData


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
