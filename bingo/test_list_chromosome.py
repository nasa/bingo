import pytest
import numpy as np

from bingo.Base.Variation import Variation
from bingo.Base.Evaluation import Evaluation
from bingo.Base.Selection import Selection
from bingo.EA.SimpleEa import SimpleEa
from MultipleValues import *

# TODO: learn how to parametrize these fixtures 
@pytest.fixture
def sample_float_list_chromosome():
	chromosome = MultipleValueChromosome(10, 'float')
	return chromosome

@pytest.fixture
def sample_int_list_chromosome():
	chromosome = MultipleValueChromosome(10, 'int')
	return chromosome

@pytest.fixture
def sample_bool_list_chromosome():
	chromosome = MultipleValueChromosome(10, 'bool')
	return chromosome

@pytest.fixture
def numbered_pop():
	return [MultipleValueChromosome(10) for i in range(10)]

def test_length_of_list(sample_float_list_chromosome):
	assert len(sample_float_list_chromosome._list_of_values) == 10

def test_float_values_in_list(sample_float_list_chromosome):
	for i in range(10):
		assert isinstance(sample_float_list_chromosome._list_of_values[i], float)

def test_int_values_in_list(sample_int_list_chromosome):
	for i in range(10):
		assert isinstance(sample_int_list_chromosome._list_of_values[i], (int, np.integer))

def test_bool_values_in_list(sample_bool_list_chromosome):
	for i in range(10):
		assert sample_bool_list_chromosome._list_of_values[i] == True or sample_bool_list_chromosome._list_of_values[i] == False

def test_crossover(numbered_pop):
	crossover = MultipleValueCrossover()
	child_1, child_2 = crossover(numbered_pop[0], numbered_pop[1])
	cross_pt = crossover._crossover_point
	assert child_1._list_of_values[:cross_pt] == numbered_pop[0]._list_of_values[:cross_pt]
	assert child_2._list_of_values[:cross_pt] == numbered_pop[1]._list_of_values[:cross_pt]
	assert child_1._list_of_values[cross_pt :] == numbered_pop[1]._list_of_values[cross_pt :]
	assert child_2._list_of_values[cross_pt :] == numbered_pop[0]._list_of_values[cross_pt :]

# def test_negative_crossover(numbered_pop):
# 	crossover = MultipleValueNegativeCrossover()
# 	child_1, child_2 = crossover(numbered_pop[0], numbered_pop[1])
# 	for i in child_1._list_of_values:
# 		assert child_1._list_of_values[i] ^ numbered_pop[0]._list_of_values[i]








