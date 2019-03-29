# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.Base.MultipleValues import MultipleValueChromosome, \
                                 MultipleValueChromosomeGenerator, SinglePointCrossover,\
                                 SinglePointMutation


def mutation_onemax_specific():
    return np.random.choice([True, False])


@pytest.fixture
def sample_float_list_chromosome():
    chromosome = MultipleValueChromosome(
        [np.random.choice([1.0, 0.0]) for _ in range(10)])
    return chromosome


@pytest.fixture
def sample_int_list_chromosome():
    chromosome = MultipleValueChromosome(
        [np.random.choice([1, 0]) for _ in range(10)])
    return chromosome


@pytest.fixture
def sample_bool_list_chromosome():
    chromosome = MultipleValueChromosome(
        [np.random.choice([True, False]) for _ in range(10)])
    return chromosome


@pytest.fixture
def population():
    generator = MultipleValueChromosomeGenerator(mutation_onemax_specific, 10)
    return [generator() for _ in range(25)]


def test_length_of_list(sample_float_list_chromosome):
    assert len(sample_float_list_chromosome.values) == 10


def test_float_values_in_list(sample_float_list_chromosome):
    for i in range(10):
        assert isinstance(sample_float_list_chromosome.values[i],
                          float)


def test_int_values_in_list(sample_int_list_chromosome):
    for i in range(10):
        assert isinstance(sample_int_list_chromosome.values[i],
                          (int, np.integer))


def test_bool_values_in_list(sample_bool_list_chromosome):
    for i in range(10):
        assert sample_bool_list_chromosome.values[i] or \
               not sample_bool_list_chromosome.values[i]


def test_generator():
    generator = MultipleValueChromosomeGenerator(mutation_onemax_specific, 10)
    pop = [generator() for i in range(20)]
    assert len(pop) == 20
    assert len(pop[0].values) == 10


def test_crossover(population):
    crossover = SinglePointCrossover()
    child_1, child_2 = crossover(population[0], population[1])
    cross_pt = crossover._crossover_point
    assert child_1.values[:cross_pt] == \
        population[0].values[:cross_pt]

    assert child_2.values[:cross_pt] == \
        population[1].values[:cross_pt]

    assert child_1.values[cross_pt:] == \
        population[1].values[cross_pt:]

    assert child_2.values[cross_pt:] == \
        population[0].values[cross_pt:]


def test_mutation_is_single_point():
    mutator = SinglePointMutation(mutation_onemax_specific)
    parent = MultipleValueChromosome(
        [np.random.choice([True, False]) for _ in range(10)])
    child = mutator(parent)
    discrepancies = 0
    for i in range(len(parent.values)):
        if child.values[i] != parent.values[i]:
            discrepancies += 1

    assert discrepancies <= 1


def test_fitness_is_not_inherited_mutation():
    mutator = SinglePointMutation(mutation_onemax_specific)
    parent = MultipleValueChromosome(
        [np.random.choice([True, False]) for _ in range(10)])
    child = mutator(parent)
    assert not child.fit_set


def test_fitness_is_not_inherited_crossover():
    crossover = SinglePointCrossover()
    parent1 = MultipleValueChromosome(
        [np.random.choice([True, False]) for _ in range(10)])
    parent2 = MultipleValueChromosome(
        [np.random.choice([True, False]) for _ in range(10)])
    child1, child2 = crossover(parent1, parent2)
    assert not child1.fit_set
    assert not child2.fit_set


def test_genetic_age_is_oldest_parent():
    crossover = SinglePointCrossover()
    parent1 = MultipleValueChromosome([np.random.choice([True, False])
                                       for _ in range(10)])
    parent2 = MultipleValueChromosome([np.random.choice([True, False])
                                       for _ in range(10)])
    parent1.genetic_age = 8
    parent2.genetic_age = 4
    child1, child2 = crossover(parent1, parent2)
    assert child1.genetic_age == 8
    assert child2.genetic_age == 8


def test_distance(sample_bool_list_chromosome):
    chromosome = sample_bool_list_chromosome.copy()
    for i, indv in enumerate(sample_bool_list_chromosome.values):
        assert indv == chromosome.values[i]
    chromosome.values[0] = \
        (not sample_bool_list_chromosome.values[0])
    assert sample_bool_list_chromosome.distance(chromosome) == 1
