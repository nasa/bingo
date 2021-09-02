# Ignoring some linting rules in tests
# pylint: disable=missing-docstring
import pytest
import numpy as np
from bingo.chromosomes.chromosome import Chromosome
from bingo.selection.generalized_crowding import GeneralizedCrowding
from bingo.selection.deterministic_crowding import DeterministicCrowding
from bingo.selection.bayes_crowding import BayesCrowding


@pytest.fixture
def dummy_crowding(mocker):
    mocker.patch('bingo.selection.generalized_crowding.GeneralizedCrowding',
                 autospec=True)
    mocker.patch.object(GeneralizedCrowding, "__abstractmethods__",
                        new_callable=set)
    return GeneralizedCrowding


@pytest.fixture
def dummy_chromosome(mocker):
    mocker.patch('bingo.chromosomes.chromosome.Chromosome', autospec=True)
    mocker.patch.object(Chromosome, "__abstractmethods__",
                        new_callable=set)
    return Chromosome


@pytest.fixture
def population_of_4(dummy_chromosome):
    return [dummy_chromosome(fitness=i) for i in [1, 2, 3, 0]]


@pytest.fixture
def population_of_n4(dummy_chromosome):
    return [dummy_chromosome(fitness=i) for i in [-1, -2, -3, 0]]


@pytest.fixture
def population_of_4_with_nans(dummy_chromosome):
    return [dummy_chromosome(fitness=i) for i in [1, np.nan, np.nan, 0]]


def test_cannot_pass_odd_population_size(population_of_4, dummy_crowding):
    population = population_of_4[:3]
    selection = dummy_crowding()
    with pytest.raises(ValueError):
        _ = selection(population, target_population_size=2)


def test_cannot_pass_odd_target_population_size(population_of_4,
                                                dummy_crowding):
    selection = dummy_crowding()
    with pytest.raises(ValueError):
        _ = selection(population_of_4, target_population_size=1)


def test_cannot_pass_large_target_population_size(population_of_4,
                                                  dummy_crowding):
    selection = dummy_crowding()
    with pytest.raises(ValueError):
        _ = selection(population_of_4, target_population_size=4)


@pytest.mark.parametrize("distances, expected_fitnesses",
                         [([1, 1, 0, 0], [0, 2]),
                          ([0, 0, 1, 1], [1, 0])])
def test_deterministic_crowding(mocker, population_of_4, distances,
                                expected_fitnesses):
    mocker.patch.object(Chromosome, "distance", side_effect=distances)
    selection = DeterministicCrowding()
    new_pop = selection(population_of_4, target_population_size=2)
    fitnesses = [individual.fitness for individual in new_pop]
    assert fitnesses == expected_fitnesses


@pytest.mark.parametrize("distances, expected_fitnesses",
                         [([0, 0, 1, 1], [1, 0])])
def test_deterministic_crowding_with_nan(mocker, population_of_4_with_nans,
                                         distances, expected_fitnesses):
    mocker.patch.object(Chromosome, "distance", side_effect=distances)
    selection = DeterministicCrowding()
    new_pop = selection(population_of_4_with_nans, target_population_size=2)
    fitnesses = [individual.fitness for individual in new_pop]
    assert fitnesses == expected_fitnesses


@pytest.mark.parametrize("rand_value, logscale, expected_fitnesses",
                         [(0.0, False, [1, 3]),
                          (0.59, False, [1, 3]),
                          (0.61, False, [1, 2]),
                          (1.0, False, [1, 2]),
                          (0.0, True, [0, 3]),
                          (0.731, True, [0, 2]),
                          (0.732, True, [1, 2]),
                          (0.268, True, [0, 3]),
                          (0.269, True, [0, 2]),
                          (1.0, True, [1, 2])
                          ])
def test_bayes_crowding(mocker, population_of_4, rand_value, logscale,
                        expected_fitnesses):
    # -1(p) vs 0(c) and -2(p) vs -3(c)
    mocker.patch.object(Chromosome, "distance", side_effect=[1, 1, 0, 0])
    mocker.patch("bingo.selection.bayes_crowding.np.random.random",
                 return_value=rand_value)
    selection = BayesCrowding(logscale=logscale)
    new_pop = selection(population_of_4, target_population_size=2)
    fitnesses = [individual.fitness for individual in new_pop]
    assert fitnesses == expected_fitnesses


@pytest.mark.parametrize("distances, expected_fitnesses",
                         [([0, 0, 1, 1], [1, 0])])
def test_bayes_crowding_with_nan(mocker, population_of_4_with_nans,
                                 distances, expected_fitnesses):
    mocker.patch.object(Chromosome, "distance", side_effect=distances)
    selection = BayesCrowding()
    new_pop = selection(population_of_4_with_nans, target_population_size=2)
    fitnesses = [individual.fitness for individual in new_pop]
    assert fitnesses == expected_fitnesses
