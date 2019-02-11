import pytest
import numpy as np

from bingo.Base.Variation import Variation
from bingo.Base.Evaluation import Evaluation
from bingo.Base.Selection import Selection
from bingo.EA.SimpleEa import SimpleEa
from bingo.EA.SimpleEvaluation import SimpleEvaluation
from MultipleValues import MultipleValueChromosome
from OneMaxExample import *


@pytest.fixture
def fitness_evaluator():
	return MultipleValueFitnessEvaluator()

@pytest.fixture
def sample_bool_list_chromosome():
	chromosome = MultipleValueChromosome(10, 'bool')
	return chromosome

@pytest.fixture
def numbered_pop():
	return [MultipleValueChromosome(10) for i in range(10)]

def test_fitness_evaluation_false_value_count(sample_bool_list_chromosome, fitness_evaluator):
	number_false_values = fitness_evaluator(sample_bool_list_chromosome)
	assert number_false_values >= 0

def test_fitness_evaluation_eval_count(sample_bool_list_chromosome, fitness_evaluator):
	number_false_values = fitness_evaluator(sample_bool_list_chromosome)
	assert fitness_evaluator.eval_count == 10

def test_evaluation_evaluates_all_list_values_per_individual(numbered_pop, fitness_evaluator):
	evaluation = SimpleEvaluation(fitness_evaluator)
	evaluation(numbered_pop)
	assert evaluation.eval_count == 100
	for indv in numbered_pop:
		assert indv.fit_set
		assert indv.fitness is not None

def test_evaluation_skips_already_calculated_fitnesses(numbered_pop, fitness_evaluator):
	evaluation = SimpleEvaluation(fitness_evaluator)
	numbered_pop[0].fitness = 1.0
	evaluation(numbered_pop)
	assert evaluation.eval_count == 90
	for indv in numbered_pop:
		assert indv.fit_set
		assert indv.fitness is not None

def test_fitness_equals_false_value_count(fitness_evaluator, numbered_pop):
	evaluation = SimpleEvaluation(fitness_evaluator)
	evaluation(numbered_pop)
	for indv in numbered_pop:
		assert indv.fitness == fitness_evaluator(indv)



	

