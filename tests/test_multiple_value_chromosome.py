import pytest
import numpy as np

from bingo.Base.Variation import Variation
from bingo.Base.Evaluation import Evaluation
from bingo.Base.Selection import Selection
from bingo.EA.SimpleEa import SimpleEa
from bingo.MultipleValues import *



def mutation_onemax_specific():
    return np.random.choice([True, False])

@pytest.fixture
def sample_float_list_chromosome():
	chromosome = MultipleValueChromosome( [np.random.choice([1.0, 0.0]) for i in range(10)])
	return chromosome

@pytest.fixture
def sample_int_list_chromosome():
	chromosome = MultipleValueChromosome([np.random.choice([1, 0]) for i in range(10)])
	return chromosome

@pytest.fixture
def sample_bool_list_chromosome():
	chromosome = MultipleValueChromosome([np.random.choice([True, False]) for i in range(10)])
	return chromosome

@pytest.fixture
def population():
    generator = MultipleValueGenerator(mutation_onemax_specific, 10)
    return [generator() for i in range(25)]

def test_length_of_list(sample_float_list_chromosome):
	assert len(sample_float_list_chromosome.list_of_values) == 10

def test_float_values_in_list(sample_float_list_chromosome):
	for i in range(10):
		assert isinstance(sample_float_list_chromosome.list_of_values[i], float)

def test_int_values_in_list(sample_int_list_chromosome):
	for i in range(10):
		assert isinstance(sample_int_list_chromosome.list_of_values[i], (int, np.integer))

def test_bool_values_in_list(sample_bool_list_chromosome):
	for i in range(10):
		assert sample_bool_list_chromosome.list_of_values[i] == True or sample_bool_list_chromosome.list_of_values[i] == False

def test_generator():
	generator = MultipleValueGenerator(mutation_onemax_specific, 10)
	pop = [generator() for i in range(20)]
	assert len(pop) == 20
	assert len(pop[0].list_of_values) == 10

def test_crossover(population):
	crossover = SinglePointCrossover()
	child_1, child_2 = crossover(population[0], population[1])
	cross_pt = crossover._crossover_point
	assert child_1.list_of_values[:cross_pt] == population[0].list_of_values[:cross_pt]
	assert child_2.list_of_values[:cross_pt] == population[1].list_of_values[:cross_pt]
	assert child_1.list_of_values[cross_pt :] == population[1].list_of_values[cross_pt :]
	assert child_2.list_of_values[cross_pt :] == population[0].list_of_values[cross_pt :]

def test_mutation_is_single_point():
    mutator = SinglePointMutation(mutation_onemax_specific)
    parent = MultipleValueChromosome([np.random.choice([True, False]) for i in range(10)])
    child = mutator(parent)
    discrepancies = 0
    for i in range(len(parent.list_of_values)):
        if child.list_of_values[i] != parent.list_of_values[i]:
            discrepancies += 1

    assert discrepancies <= 1

def test_fitness_is_not_inherited_mutation():
    mutator = SinglePointMutation(mutation_onemax_specific)
    parent = MultipleValueChromosome([np.random.choice([True, False]) for i in range(10)])
    child = mutator(parent)
    assert child.fit_set == False

def test_fitness_is_not_inherited_crossover():
    crossover = SinglePointCrossover()
    parent1 = MultipleValueChromosome([np.random.choice([True, False]) for i in range(10)])
    parent2 = MultipleValueChromosome([np.random.choice([True, False]) for i in range(10)])
    child1, child2 = crossover(parent1, parent2)
    assert child1.fit_set == False
    assert child2.fit_set == False
