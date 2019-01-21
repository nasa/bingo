# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest

from bingo.EA.SimpleEvaluation import SimpleEvaluation
from SingleValue import SingleValueFitnessEvaluator

@pytest.fixture
def fitness_evaluator():
    return SingleValueFitnessEvaluator()


def test_evaluation_evaluates_all_individuals(single_value_population_of_4,
                                              fitness_evaluator):
    evaluation = SimpleEvaluation(fitness_evaluator)
    evaluation(single_value_population_of_4)
    assert evaluation.eval_count == 4
    for indv in single_value_population_of_4:
        assert indv.fit_set
        assert indv.fitness is not None


def test_evaluation_skips_already_calculated_fitnesses(
        single_value_population_of_4, fitness_evaluator):
    evaluation = SimpleEvaluation(fitness_evaluator)
    single_value_population_of_4[0].fitness = 1.0
    evaluation(single_value_population_of_4)
    assert evaluation.eval_count == 3
    for indv in single_value_population_of_4:
        assert indv.fit_set
        assert indv.fitness is not None


def test_setting_eval_count(single_value_population_of_4, fitness_evaluator):
    evaluation = SimpleEvaluation(fitness_evaluator)
    evaluation.eval_count = -4
    assert evaluation.eval_count == -4
    evaluation(single_value_population_of_4)
    assert evaluation.eval_count == 0
