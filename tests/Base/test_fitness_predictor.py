# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np
from bingo.Base.FitnessFunction import FitnessFunction
from bingo.Base.FitnessPredictor import FitnessPredictorFitnessFunction, \
                                   FitnessPredictorIndexGenerator
from bingo.Base.MultipleValues import MultipleValueChromosome


class MinPlusMean(FitnessFunction):
    def __call__(self, individual):
        return min(individual.values) + np.mean(self.training_data)


@pytest.fixture
def training_data():
    return np.arange(10)


@pytest.fixture
def fitness_function(training_data):
    return MinPlusMean(training_data)


@pytest.fixture
def sample_population():
    return [MultipleValueChromosome(list(range(i, i + 10))) for i in range(10)]


@pytest.fixture
def predictor_fitness_function(training_data,
                               fitness_function,
                               sample_population):
    return FitnessPredictorFitnessFunction(training_data,
                                           fitness_function,
                                           sample_population, 4)


def test_raises_error_not_enough_valid_trainers(training_data,
                                                fitness_function,
                                                sample_population):
    with pytest.raises(RuntimeError):
        _ = FitnessPredictorFitnessFunction(training_data,
                                            fitness_function,
                                            sample_population, 11)


@pytest.mark.parametrize("predictor_values", [[0, 1],
                                              [2],
                                              [5, 5, 5],
                                              [9, 8, 0]])
def test_fitness_predictor_fitness_function_call(predictor_fitness_function,
                                                 predictor_values):
    predictor = MultipleValueChromosome(predictor_values)
    expected_fitness = abs(4.5 - np.mean(predictor_values))
    fitness = predictor_fitness_function(predictor)
    np.testing.assert_almost_equal(fitness, expected_fitness)


def test_adding_trainers_to_predictor_fitness_function(
        predictor_fitness_function):
    for i in range(10):
        trainer = MultipleValueChromosome([i])
        predictor_fitness_function.add_trainer(trainer)
        assert trainer not in predictor_fitness_function._trainers
        assert np.any([[i] == trainer.values
                       for trainer in predictor_fitness_function._trainers])


def test_predicted_fitness_for_trainer(predictor_fitness_function,
                                       sample_population):
    predictor = MultipleValueChromosome([0, 9])
    for i, trainer in enumerate(sample_population):
        prediction = \
            predictor_fitness_function.predict_fitness_for_trainer(predictor,
                                                                   trainer)
        expected_prediction = i + 4.5
        np.testing.assert_almost_equal(prediction, expected_prediction)


@pytest.mark.parametrize("maximum", [2, 20])
def test_index_generator(maximum):
    generator = FitnessPredictorIndexGenerator(maximum)
    indices = np.array([generator() for _ in range(100)])
    assert np.all(indices >= 0)
    assert np.all(indices < maximum)
