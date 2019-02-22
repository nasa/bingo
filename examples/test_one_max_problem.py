import pytest
import numpy as np

from bingo.Base.Variation import Variation
from bingo.Base.Evaluation import Evaluation
from bingo.Base.Selection import Selection
from bingo.EA.SimpleEa import SimpleEa
from bingo.EA.SimpleEvaluation import SimpleEvaluation
from bingo.MultipleValues import *
from OneMaxExample import *

@pytest.fixture
def fitness_evaluator():
    return MultipleValueFitnessEvaluator()

@pytest.fixture
def sample_bool_list_chromosome():
    chromosome = MultipleValueChromosome([np.random.choice([True, False]) for i in range(10)])
    return chromosome

@pytest.fixture
def population():
    generator = MultipleValueGenerator(mutation_onemax_specific, 10)
    return [generator() for i in range(25)]

def test_fitness_evaluation_true_value_count_nonnegative(sample_bool_list_chromosome, fitness_evaluator):
    number_true_values = fitness_evaluator(sample_bool_list_chromosome)
    assert number_true_values >= 0

def test_fitness_evaluation_eval_count(sample_bool_list_chromosome, fitness_evaluator):
    number_true_values = fitness_evaluator(sample_bool_list_chromosome)
    assert fitness_evaluator.eval_count == 1

def test_evaluation_evaluates_all_list_values_per_individual(population, fitness_evaluator):
    evaluation = SimpleEvaluation(fitness_evaluator)
    evaluation(population)
    assert evaluation.eval_count == 25
    for indv in population:
        assert indv.fit_set
        assert indv.fitness is not None

def test_evaluation_skips_already_calculated_fitnesses(population, fitness_evaluator):
    evaluation = SimpleEvaluation(fitness_evaluator)
    population[0].fitness = 1.0
    evaluation(population)
    assert evaluation.eval_count == 24
    for indv in population:
        assert indv.fit_set
        assert indv.fitness is not None

def test_fitness_equals_true_value_count(fitness_evaluator, population):
    evaluation = SimpleEvaluation(fitness_evaluator)
    evaluation(population)

    for indv in population:
        fitness = 0
        for val in indv.list_of_values:
            if val == False:
                fitness += 1
        assert indv.fitness == fitness_evaluator(indv)
        assert fitness == indv.fitness
