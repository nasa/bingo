import numpy as np

from Base.Chromosome import Chromosome
from Base.FitnessEvaluator import FitnessEvaluator
from Base.Mutation import Mutation
from Base.Crossover import Crossover
from Base.Variation import Variation
from Base.Evaluation import Evaluation
from Base.Selection import Selection
from bingo.EA.SimpleEa import SimpleEa
from bingo.EA.VarAnd import VarAnd
from bingo.EA.TournamentSelection import Tournament
from bingo.EA.SimpleEvaluation import SimpleEvaluation
from MultipleValues import *


# TODO: assessment for list of values in list chromosome
class MultipleValueFitnessEvaluator(FitnessEvaluator):
	"""Fitness for multiple value chromosomes

	Fitness equals the number of true values in the chromosome's list of values 
	"""
	def __call__(self, individual):
		fitness = 0
		for val in individual._list_of_values:
			if val == True:
				fitness += 1

		self.eval_count += 1
		return fitness


def mutation_onemax_specific():
	return np.random.choice([True, False])

def population_input():
	while True:
		values_per_list  = input("Enter the number of values each List Chromosome will hold:\n")
		population_size = input("Enter the desired population size: \n")
		try:
			generator = MultipleValueGenerator()
			int_vals_per_list = int(values_per_list)
			int_pop_size = int(population_size)
			population = generator(population_size=int_pop_size, values_per_chromosome=int_vals_per_list)
			if int_vals_per_list <= 0 or int_pop_size <= 0:
				print("\nError: List length and population size must be positive integers")
				continue
			break
		except ValueError:
			print("\nValueError, please enter a number for list length and population size\n")

		except TypeError:
			print("\nTypeError, please enter a valid number for list length and population size\n")

	return population

def execute_generational_steps():
	population= population_input()
	selection = Tournament(10)
	crossover = MultipleValueCrossover()
	mutation = MultipleValueMutation(mutation_onemax_specific)
	fitness = MultipleValueFitnessEvaluator()
	evaluation = SimpleEvaluation(fitness)
	variation = VarAnd(crossover, mutation, 0.8, 0.8)
	ea = SimpleEa(variation, evaluation, selection)
	for i in range(10):
		next_gen = ea.generational_step(population)
		print("\nGeneration #", i)
		print("----------------------\n")
		report_max_min_mean_fitness(next_gen)
		print("population: \n")
		for indv in population:
			print(indv._list_of_values)
		print("next gen: \n")
		for indv in next_gen:
			print("Fitness: ", indv.fitness, "; Values: ", indv._list_of_values)
		population = next_gen

# TODO: idk if this needs to be a function but uhhh keep track of %true over time 
def prove_population_is_evolving_towards_true(population):
	do_something = 'this is pretty self explanatory i just didnt wanna forget'

def report_max_min_mean_fitness(population):
	fitness_evaluator = MultipleValueFitnessEvaluator()
	fitness = [fitness_evaluator(indv) for indv in population]
	print(fitness)
	print("Max fitness: \t", np.max(fitness))
	print("Min fitness: \t", np.min(fitness))
	print("Mean fitness: \t", np.mean(fitness))



execute_generational_steps()
# population= population_input()
# selection = Tournament(10)
# crossover = MultipleValueCrossover()
# mutation = MultipleValueMutation(mutation_onemax_specific)
# fitness_evaluator = MultipleValueFitnessEvaluator()
# evaluation = SimpleEvaluation(fitness_evaluator)
# variation = VarAnd(crossover, mutation, 0.8, 0.8)
# ea = SimpleEa(variation, evaluation, selection)
# for i in range(20):
# 	for indv in population:
# 		indv.fitness = fitness_evaluator(indv)
# 		#print("indv.fitness = ", indv.fitness)
# 	next_gen = ea.generational_step(population)
# 	print("Generation #", i)
# 	print("----------------------\n")
# 	report_max_min_mean_fitness(next_gen)
# 	print("population: \n")
# 	for indv in population:
# 		print(indv._list_of_values)
# 	print("next gen: \n")
# 	for indv in next_gen:
# 		print("Fitness: ", indv.fitness, indv._list_of_values)
# 	population = next_gen














