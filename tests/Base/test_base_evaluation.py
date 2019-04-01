# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest

from bingo.Base.Evaluation import Evaluation
from SingleValue import SingleValueFitnessFunction


@pytest.fixture
def fitness_function():
    return SingleValueFitnessFunction()


def test_evaluation_evaluates_all_individuals(single_value_population_of_4,
                                              fitness_function):
    evaluation = Evaluation(fitness_function)
    evaluation(single_value_population_of_4)
    assert evaluation.eval_count == 4
    for indv in single_value_population_of_4:
        assert indv.fit_set
        assert indv.fitness is not None


def test_evaluation_skips_already_calculated_fitnesses(
        single_value_population_of_4, fitness_function):
    evaluation = Evaluation(fitness_function)
    single_value_population_of_4[0].fitness = 1.0
    evaluation(single_value_population_of_4)
    assert evaluation.eval_count == 3
    for indv in single_value_population_of_4:
        assert indv.fit_set
        assert indv.fitness is not None


def test_setting_eval_count(single_value_population_of_4, fitness_function):
    evaluation = Evaluation(fitness_function)
    evaluation.eval_count = -4
    assert evaluation.eval_count == -4
    evaluation(single_value_population_of_4)
    assert evaluation.eval_count == 0
