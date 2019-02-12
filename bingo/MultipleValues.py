

import numpy as np

from Base.Chromosome import Chromosome
from Base.FitnessEvaluator import FitnessEvaluator
from Base.Mutation import Mutation
from Base.Crossover import Crossover
from Base.Generator import Generator


class MultipleValueChromosome(Chromosome):
	""" Multiple value individual 

	Parameters
	----------
	"""
	def __init__(self, number_of_values, type_of_value='bool'):
		super().__init__()
		self._type_of_value = type_of_value
		self._list_of_values = []
		if type_of_value == 'bool':
			for i in range(number_of_values):
				self._list_of_values.append(np.random.choice([True, False]))
		elif type_of_value == 'float':
			for i in range(number_of_values):
				self._list_of_values.append(np.random.choice([1.0, 0.0]))
		elif type_of_value == 'int':
			for i in range(number_of_values):
				self._list_of_values.append(np.random.choice([1, 0]))



	def __str__(self):
		return str(self._list_of_values)


class MultipleValueGenerator(Generator):
	"""Generation of a population of Multi-Value Chromosomes


	"""
	def __init__(self):
		super().__init__()
	

	def __call__(self, population_size=20, values_per_chromosome=10):
		return [MultipleValueChromosome(values_per_chromosome) for i in range(population_size)]


class MultipleValueMutation(Mutation):
	"""Mutation for multiple valued chromosomes


	"""
	def __init__(self, mutation_function):
		super().__init__()
		self._mutation = mutation_function

	def __call__(self, parent):
		child = parent.copy()
		child = self._mutation(child)
		return child

class MultipleValueCrossover(Crossover):
	"""Crossover for multiple valued chromosomes

	Crossover results in two individuals with single-point crossed-over lists 
	whose values are provided by the parents. Crossover point is a random integer
	produced by numpy.
	"""
	def __init__(self):
		super().__init__()
		self._crossover_point = 0

	def __call__(self, parent_1, parent_2):
		child_1 = parent_1.copy()
		child_2 = parent_2.copy()
		self._crossover_point = np.random.randint(len(parent_1._list_of_values))
		child_1._list_of_values = parent_1._list_of_values[:self._crossover_point] + parent_2._list_of_values[self._crossover_point:]
		child_2._list_of_values = parent_2._list_of_values[:self._crossover_point] + parent_1._list_of_values[self._crossover_point:]
		return child_1, child_2




