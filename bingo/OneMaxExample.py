import numpy as np

from Base.Chromosome import Chromosome
from Base.FitnessEvaluator import FitnessEvaluator
from Base.Mutation import Mutation
from Base.Crossover import Crossover

# TODO: assessment for list of values in list chromosome
class MultipleValueFitnessEvaluator(FitnessEvaluator):
	"""Fitness for multiple value chromosomes

	Fitness equals the chromosome value
	"""
	def __init__(self):
		super().__init__()
		self.total_false_values = 0

	def __call__(self, individual):
		self.total_false_values = 0
		length_of_list = len(individual._list_of_values)
		for i in range(length_of_list):
			if individual._list_of_values[i] == False:
				self.total_false_values += 1
			self.eval_count += 1

		return self.total_false_values



	# TODO: make this something usable for all classes 
def _get_random_list_value(type_of_value='bool'):
	if type_of_value == 'bool':
		return np.random.choice([True, False])
	elif type_of_value == 'float':
		return np.random.choice([1.0, 0.0])
	elif type_of_value == 'int':
		return np.random.choice([1, 0])
	else:
		raise ValueError('Invalid type of list value')