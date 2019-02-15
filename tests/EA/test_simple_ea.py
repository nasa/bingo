# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest

from bingo.Base.Variation import Variation
from bingo.Base.Evaluation import Evaluation
from bingo.Base.Selection import Selection
from bingo.EA.SimpleEa import SimpleEa

from SingleValue import SingleValueChromosome


class add_v_variation(Variation):
    def __call__(self, population, number_offspring):
        offspring = [parent.copy() for parent in population]
        for indv in offspring:
            indv.value += "v"
        return offspring


class add_e_evaluation(Evaluation):
    def __call__(self, population):
        for indv in population:
            indv.fitness = indv.value
            indv.value += "e"


class add_s_selection(Selection):
    def __call__(self, population, _target_population_size):
        for indv in population:
            indv.value += "s"
        return population


@pytest.fixture
def sample_ea():
    return SimpleEa(add_v_variation(), add_e_evaluation(), add_s_selection())


@pytest.fixture
def numbered_pop():
    return [SingleValueChromosome(str(i)) for i in range(10)]


def test_correct_order_of_algorithm_phases(sample_ea, numbered_pop):
    next_gen = sample_ea.generational_step(numbered_pop)
    for i, indv in enumerate(next_gen):
        assert indv.value == str(i) + "ves"
