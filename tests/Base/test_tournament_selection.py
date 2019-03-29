# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest

from bingo.Base.TournamentSelection import Tournament


@pytest.fixture()
def population_all_ones(single_value_population_of_4):
    pop = single_value_population_of_4
    for indv in pop:
        indv.fitness = 1
    return pop


@pytest.fixture(params=range(4))
def population_with_0(request, single_value_population_of_4):
    pop = single_value_population_of_4
    for indv in pop:
        indv.fitness = 1
    pop[request.param].fitness = 0
    return pop


@pytest.fixture()
def tournament_of_4():
    return Tournament(4)


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


@pytest.mark.parametrize("tourn_size,expected_error", [
    (0, ValueError),
    ("string", TypeError)
])
def test_raises_error_invalid_tournament_size(tourn_size, expected_error):
    with pytest.raises(expected_error):
        _ = Tournament(tourn_size)


def test_no_repeats_in_selected_population(tournament_of_4, population_with_0):
    new_population = tournament_of_4(population_with_0, 4)
    for i, indv in enumerate(new_population[:-1]):
        assert indv not in new_population[i+1:]
