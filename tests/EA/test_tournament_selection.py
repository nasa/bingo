# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest

from bingo.EA.TournamentSelection import Tournament
from SingleValue import SingleValueChromosome


@pytest.fixture()
def population_all_ones():
    pop = [SingleValueChromosome(),
           SingleValueChromosome(),
           SingleValueChromosome(),
           SingleValueChromosome()]
    for indv in pop:
        indv.fitness = 1
    return pop


@pytest.fixture(params=range(4))
def population_with_0(request):
    pop = [SingleValueChromosome(),
           SingleValueChromosome(),
           SingleValueChromosome(),
           SingleValueChromosome()]
    for indv in pop:
        indv.fitness = 1
    pop[request.param].fitness = 0
    return pop


@pytest.fixture()
def tournament_of_4():
    return Tournament(4)


def test_tournament_too_small():
    with pytest.raises(ValueError):
        Tournament(0)


def test_tournament_selects_best_indv(tournament_of_4, population_with_0):
    new_population = tournament_of_4(population_with_0, 1)
    assert new_population[0].fitness == 0


def test_tournament_selects_indv(tournament_of_4, population_all_ones):
    new_population = tournament_of_4(population_all_ones, 1)
    assert new_population[0].fitness == 1


@pytest.mark.parametrize("new_pop_size", range(6))
def test_tournament_returns_correct_size_population(tournament_of_4,
                                                    population_all_ones,
                                                    new_pop_size):
    new_population = tournament_of_4(population_all_ones, new_pop_size)
    assert len(new_population) == new_pop_size
