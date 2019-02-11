

import numpy as np

from Base.Chromosome import Chromosome
from Base.FitnessEvaluator import FitnessEvaluator
from Base.Mutation import Mutation
from Base.Crossover import Crossover


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



class MultipleValueMutation(Mutation):
	"""Mutation for multiple valued chromosomes

	Mutation results in new random value corresponding to the type 
	of elements in the list. If type is 'float', choice is between [0.0, 0.1]. 
	If type is 'int', choice is between [0, 1]. If type is 'bool' then [True, False]
	"""
# TODO: change to single point mutation
	def __call__(self, parent):
		child = parent.copy()
		length_of_list = len(child._list_of_values)
		for i in range(length_of_list):
			self.eval_count += 1
		return child

class MultipleValueCrossover(Crossover):
	"""Crossover for multiple valued chromosomes

	Crossover results in two individuals with single-point crossed-over lists 
	whose values are provided by the parents. Crossover point is random.
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


# TODO make this less gross, maybe make function to toggle value?
class MultipleValueNegativeCrossover(Crossover):
	"""Crossover for multiple valued chromosomes

	Crossover results in two individuals with 
	"""
	def __call__(self, parent_1, parent_2):
		child_1 = parent_1.copy()
		child_2 = parent_2.copy()
		# if parent_1._type_of_value == 'bool':
		# 	for i in child_1._list_of_values:
		# 		child_1._list_of_values[i] = not child_1._list_of_values[i]
		# else:
		# 	for i in child_1._list_of_values:
		# 		child_1._list_of_values[i] *= -1

		# if parent_2._type_of_value == 'bool':
		# 	for i in child_2._list_of_values:
		# 		child_2._list_of_values[i] = not child_2._list_of_values[i]
		# else:
		# 	for i in child_2._list_of_values:
		# 		child_2._list_of_values[i] *= -1

		return child_1, child_2



