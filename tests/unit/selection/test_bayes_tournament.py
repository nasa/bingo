# Ignoring some linting rules in tests
# pylint: disable=missing-docstring
import pytest
from bingo.selection.bayes_tournament import BayesianModelSelectionTournament
from bingo.chromosomes.chromosome import Chromosome


@pytest.fixture
def dummy_chromosome(mocker):
    mocker.patch('bingo.chromosomes.chromosome.Chromosome', autospec=True)
    mocker.patch.object(Chromosome, "__abstractmethods__",
                        new_callable=set)
    return Chromosome


@pytest.fixture
def population_n012(dummy_chromosome):
    return [dummy_chromosome(fitness=float(-i)) for i in range(3)]


@pytest.mark.parametrize("tourn_size,expected_error", [
    (0, ValueError),
    ("string", TypeError)
])
def test_raises_error_invalid_tournament_size(tourn_size, expected_error):
    with pytest.raises(expected_error):
        _ = BayesianModelSelectionTournament(tourn_size)


@pytest.mark.parametrize("rand_value, logscale, selected_fitness",
                         [(0.0, False, 0),
                          (0.33, False, -1),
                          (0.34, False, -2),
                          (1.0, False, -2),
                          (0.0, True, 0),
                          (0.09, True, 0),
                          (0.091, True, -1),
                          (0.334, True, -1),
                          (0.335, True, -2),
                          (1.0, True, -2)])
def test_tournament_selection(mocker, population_n012, rand_value, logscale,
                              selected_fitness):
    mocker.patch("bingo.selection.bayes_tournament.np.random.random",
                 return_value=rand_value)
    mocker.patch("bingo.selection.bayes_tournament.np.random.choice",
                 return_value=population_n012)
    tourn = BayesianModelSelectionTournament(tournament_size=3,
                                             logscale=logscale)
    new_population = tourn(population_n012, 1)
    assert new_population[0].fitness == selected_fitness
