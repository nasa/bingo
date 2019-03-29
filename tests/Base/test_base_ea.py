# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest

from bingo.Base.EvolutionaryAlgorithm import EvolutionaryAlgorithm

from SingleValue import SingleValueChromosome


@pytest.fixture
def sample_ea(add_e_evaluation, add_v_variation, add_s_selection):
    return EvolutionaryAlgorithm(add_v_variation, add_e_evaluation,
                                 add_s_selection)


@pytest.fixture
def numbered_pop():
    return [SingleValueChromosome(str(i)) for i in range(10)]


def test_correct_order_of_algorithm_phases(sample_ea, numbered_pop):
    next_gen = sample_ea.generational_step(numbered_pop)
    for i, indv in enumerate(next_gen):
        assert indv.value == str(i) + "ves"
