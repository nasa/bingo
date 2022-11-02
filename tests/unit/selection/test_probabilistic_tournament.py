# Ignoring some linting rules in tests
# pylint: disable=missing-docstring
import pytest
import numpy as np
from bingo.selection.probabilistic_tournament import ProbabilisticTournament
from bingo.chromosomes.chromosome import Chromosome


@pytest.fixture
def dummy_chromosome(mocker):
    mocker.patch("bingo.chromosomes.chromosome.Chromosome", autospec=True)
    mocker.patch.object(Chromosome, "__abstractmethods__", new_callable=set)
    return Chromosome


@pytest.fixture
def population_n012(dummy_chromosome):
    return [dummy_chromosome(fitness=float(-i)) for i in range(3)]


@pytest.fixture
def population_nans(dummy_chromosome):
    return [dummy_chromosome(fitness=np.nan) for i in range(3)]


@pytest.fixture
def population_012(dummy_chromosome):
    return [dummy_chromosome(fitness=float(i)) for i in range(3)]


@pytest.mark.parametrize(
    "tourn_size,expected_error", [(0, ValueError), ("string", TypeError)]
)
def test_raises_error_invalid_tournament_size(tourn_size, expected_error):
    with pytest.raises(expected_error):
        _ = ProbabilisticTournament(tourn_size)


def test_tournament_selection_nan_fitness(mocker, population_nans):
    mocker.patch(
        "bingo.selection.probabilistic_tournament.np.random.choice",
        return_value=population_nans,
    )
    tourn = ProbabilisticTournament(tournament_size=3)
    new_population = tourn(population_nans, 1)
    assert np.isnan(new_population[0].fitness)


@pytest.mark.parametrize(
    "rand_value, logscale, selected_fitness",
    [
        (0.0, False, 0),
        (0.33, False, -1),
        (0.34, False, -2),
        (1.0, False, -2),
        (0.0, True, 0),
        (0.09, True, 0),
        (0.091, True, -1),
        (0.334, True, -1),
        (0.335, True, -2),
        (1.0, True, -2),
    ],
)
def test_tournament_selection_negative_fitness(
    mocker, population_n012, rand_value, logscale, selected_fitness
):
    mocker.patch(
        "bingo.selection.probabilistic_tournament.np.random.random",
        return_value=rand_value,
    )
    mocker.patch(
        "bingo.selection.probabilistic_tournament.np.random.choice",
        return_value=population_n012,
    )
    tourn = ProbabilisticTournament(
        tournament_size=3, logscale=logscale, negative=True
    )
    new_population = tourn(population_n012, 1)
    assert new_population[0].fitness == selected_fitness


@pytest.mark.parametrize(
    "rand_value, logscale, selected_fitness",
    [
        (0.0, False, 0),
        (0.33, False, 1),
        (0.34, False, 2),
        (1.0, False, 2),
        (0.0, True, 0),
        (0.09, True, 0),
        (0.091, True, 1),
        (0.334, True, 1),
        (0.335, True, 2),
        (1.0, True, 2),
    ],
)
def test_tournament_selection_with_positive_fitness(
    mocker, population_012, rand_value, logscale, selected_fitness
):
    mocker.patch(
        "bingo.selection.probabilistic_tournament.np.random.random",
        return_value=rand_value,
    )
    mocker.patch(
        "bingo.selection.probabilistic_tournament.np.random.choice",
        return_value=population_012,
    )
    tourn = ProbabilisticTournament(
        tournament_size=3, logscale=logscale, negative=False
    )
    new_population = tourn(population_012, 1)
    assert new_population[0].fitness == selected_fitness
