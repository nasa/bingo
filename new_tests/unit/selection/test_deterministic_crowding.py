# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.chromosomes.chromosome import Chromosome
from bingo.selection.deterministic_crowding import DeterministicCrowding


@pytest.fixture
def dummy_chromosome(mocker):
    mocker.patch('bingo.chromosomes.chromosome.Chromosome', autospec=True)
    mocker.patch.object(Chromosome, "__abstractmethods__",
                        new_callable=set)
    return Chromosome


@pytest.fixture
def population_of_4(dummy_chromosome):
    return [dummy_chromosome(fitness=i) for i in [1, 2, 3, 0]]


def test_cannot_pass_odd_population_size(population_of_4):
    population = population_of_4[:3]
    selection = DeterministicCrowding()
    with pytest.raises(ValueError):
        _ = selection(population, target_population_size=2)


def test_cannot_pass_odd_target_population_size(population_of_4):
    selection = DeterministicCrowding()
    with pytest.raises(ValueError):
        _ = selection(population_of_4, target_population_size=1)


def test_cannot_pass_large_target_population_size(population_of_4):
    selection = DeterministicCrowding()
    with pytest.raises(ValueError):
        _ = selection(population_of_4, target_population_size=3)


@pytest.mark.parametrize("distances, expected_fitnesses",
                         [([1, 1, 0, 0], [0, 2]),
                          ([0, 0, 1, 1,], [1, 0])])
def test_return_most_fit(mocker, population_of_4, distances,
                         expected_fitnesses):
    mocker.patch.object(Chromosome, "distance", side_effect=distances)
    selection = DeterministicCrowding()
    new_pop = selection(population_of_4, target_population_size=2)
    fitnesses = [individual.fitness for individual in new_pop]
    assert fitnesses == expected_fitnesses
