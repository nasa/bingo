# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.Base.MultipleValues import MultipleValueChromosomeGenerator, SinglePointCrossover, \
                                 SinglePointMutation
from bingo.Base.VarOr import VarOr


@pytest.fixture
def population():
    generator = MultipleValueChromosomeGenerator(mutation_function, 10)
    return [generator() for _ in range(25)]


@pytest.fixture
def var_or():
    crossover = SinglePointCrossover()
    mutation = SinglePointMutation(mutation_function)
    var_or_instance = VarOr(crossover, mutation, 0.2, 0.4)
    return var_or_instance


def mutation_function():
    return np.random.choice([True, False])


def test_invalid_probabilities():
    crossover = SinglePointCrossover()
    mutation = SinglePointMutation(mutation_function)
    with pytest.raises(ValueError):
        _ = VarOr(crossover, mutation, 0.6, 0.41)


def test_offspring_not_equals_parents(population, var_or):
    offspring = var_or(population, 25)
    for i, indv in enumerate(population):
        assert indv is not offspring[i]


def test_no_two_variations_at_once(population, var_or):
    _ = var_or(population, 25)
    for cross, mut in zip(var_or.crossover_offspring,
                          var_or.mutation_offspring):
        assert not (cross and mut)
    for i, indv in enumerate(var_or.crossover_offspring):
        assert not (indv and var_or.mutation_offspring[i])


def test_just_replication(population):
    crossover = SinglePointCrossover()
    mutation = SinglePointMutation(mutation_function)
    var_or_instance = VarOr(crossover, mutation, 0.0, 0.0)
    _ = var_or_instance(population, 25)
    for cross, mut in zip(var_or_instance.crossover_offspring,
                          var_or_instance.mutation_offspring):
        assert not (cross or mut)
