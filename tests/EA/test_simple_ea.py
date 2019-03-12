# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest

from bingo.Base.Variation import Variation
from bingo.Base.Evaluation import Evaluation
from bingo.Base.Selection import Selection
from bingo.EA.SimpleEa import SimpleEa

from SingleValue import SingleValueChromosome


class VariationAddV(Variation):
    def __call__(self, population, number_offspring):
        offspring = [parent.copy() for parent in population]
        for indv in offspring:
            indv.value += "v"
        return offspring


class EvaluationAddE(Evaluation):
    def __call__(self, population):
        for indv in population:
            indv.fitness = indv.value
            indv.value += "e"


class SelectionAddS(Selection):
    def __call__(self, population, _target_population_size):
        for indv in population:
            indv.value += "s"
        return population


@pytest.fixture
def sample_ea():
    return SimpleEa(VariationAddV(), EvaluationAddE(), SelectionAddS())


@pytest.fixture
def numbered_pop():
    return [SingleValueChromosome(str(i)) for i in range(10)]


def test_correct_order_of_algorithm_phases(sample_ea, numbered_pop):
    next_gen = sample_ea.generational_step(numbered_pop)
    for i, indv in enumerate(next_gen):
        assert indv.value == str(i) + "ves"
