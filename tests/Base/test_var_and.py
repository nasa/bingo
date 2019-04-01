# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.Base.VarAnd import VarAnd
from SingleValue import SingleValueMutation, SingleValueNegativeMutation
from SingleValue import SingleValueCrossover, SingleValueNegativeCrossover


@pytest.fixture
def crossover():
    return SingleValueCrossover()


@pytest.fixture
def mutation():
    return SingleValueMutation()


@pytest.fixture
def crossover_n():
    return SingleValueNegativeCrossover()


@pytest.fixture
def mutation_n():
    return SingleValueNegativeMutation()


def test_var_and_stats_are_correct_size(single_value_population_of_4,
                                        crossover, mutation):
    variation = VarAnd(crossover=crossover, mutation=mutation,
                       crossover_probability=0.5,
                       mutation_probability=0.5)

    _ = variation(single_value_population_of_4, 4)

    assert len(variation.crossover_offspring) == 4
    assert len(variation.mutation_offspring) == 4


@pytest.mark.parametrize(
    "cross_prob,mut_prob,expected_cross,expected_mut",
    [(0.0, 0.0, False, False),
     (1.0, 0.0, True, False),
     (0.0, 1.0, False, True),
     (1.0, 1.0, True, True)])
def test_var_and_correct_stats_output(single_value_population_of_4,
                                      crossover, mutation,
                                      cross_prob, mut_prob,
                                      expected_cross, expected_mut):
    variation = VarAnd(crossover=crossover, mutation=mutation,
                       crossover_probability=cross_prob,
                       mutation_probability=mut_prob)

    _ = variation(single_value_population_of_4, 4)
    for cross, mut in zip(variation.crossover_offspring,
                          variation.mutation_offspring):
        assert cross == expected_cross
        assert mut == expected_mut


def test_var_and_replication(single_value_population_of_4, crossover_n,
                             mutation_n):
    variation = VarAnd(crossover=crossover_n, mutation=mutation_n,
                       crossover_probability=0.0,
                       mutation_probability=0.0)

    offspring = variation(single_value_population_of_4, 4)
    for off, indv in zip(offspring, single_value_population_of_4):
        assert off.value == indv.value


def test_var_and_crossover(single_value_population_of_4, crossover_n, mutation):
    variation = VarAnd(crossover=crossover_n, mutation=mutation,
                       crossover_probability=1.0,
                       mutation_probability=0.0)

    offspring = variation(single_value_population_of_4, 4)
    for off, indv in zip(offspring, single_value_population_of_4):
        assert off.value == -indv.value


def test_var_and_mutation(single_value_population_of_4, crossover, mutation_n):
    variation = VarAnd(crossover=crossover, mutation=mutation_n,
                       crossover_probability=0.0,
                       mutation_probability=1.0)

    offspring = variation(single_value_population_of_4, 4)
    for off, indv in zip(offspring, single_value_population_of_4):
        assert off.value == -indv.value


def test_var_and_crossover_and_mutation(single_value_population_of_4,
                                        crossover_n, mutation_n):
    variation = VarAnd(crossover=crossover_n, mutation=mutation_n,
                       crossover_probability=1.0,
                       mutation_probability=1.0)

    offspring = variation(single_value_population_of_4, 4)
    for off, indv in zip(offspring, single_value_population_of_4):
        assert off.value == indv.value


def test_var_and_approximate_probabilities(single_value_population_of_100,
                                           crossover, mutation):
    np.random.seed(1)
    variation = VarAnd(crossover=crossover, mutation=mutation,
                       crossover_probability=0.5,
                       mutation_probability=0.5)

    _ = variation(single_value_population_of_100, 100)
    num_cross = np.sum(variation.crossover_offspring)
    num_mut = np.sum(variation.mutation_offspring)
    num_both = np.sum(np.logical_and(variation.crossover_offspring,
                                     variation.mutation_offspring))
    num_rep = 100 - np.sum(np.logical_or(variation.crossover_offspring,
                                         variation.mutation_offspring))
    assert num_cross == 56
    assert num_mut == 44
    assert num_both == 22
    assert num_rep == 22


@pytest.mark.parametrize("num_offspring", [3, 4, 5, 6])
def test_var_and_for_correct_offspring_size(single_value_population_of_4,
                                            num_offspring,
                                            crossover, mutation):
    variation = VarAnd(crossover=crossover, mutation=mutation,
                       crossover_probability=0.5,
                       mutation_probability=0.5)

    offspring = variation(single_value_population_of_4, num_offspring)
    assert len(offspring) == num_offspring


@pytest.mark.parametrize("prob,expected_error", [
    (-0.5, ValueError),
    (2, ValueError),
    ("string", TypeError)
])
def test_raises_error_invalid_crossover_probability(crossover, mutation,
                                                    prob, expected_error):
    with pytest.raises(expected_error):
        _ = VarAnd(crossover=crossover, mutation=mutation,
                   crossover_probability=prob, mutation_probability=0.0)


@pytest.mark.parametrize("prob,expected_error", [
    (-0.5, ValueError),
    (2, ValueError),
    ("string", TypeError)
])
def test_raises_error_invalid_mutation_probability(crossover, mutation,
                                                   prob, expected_error):
    with pytest.raises(expected_error):
        _ = VarAnd(crossover=crossover, mutation=mutation,
                   crossover_probability=0.0, mutation_probability=prob)
