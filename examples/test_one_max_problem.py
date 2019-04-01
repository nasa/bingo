import pytest
import numpy as np

from bingo.Base.Evaluation import Evaluation
from bingo.Base.MultipleValues import  MultipleValueChromosome, \
                                  MultipleValueChromosomeGenerator
from OneMaxExample import MultipleValueFitnessFunction, \
                          mutation_onemax_specific

@pytest.fixture
def fitness_function():
    return MultipleValueFitnessFunction()

@pytest.fixture
def sample_bool_list_chromosome():
    chromosome = MultipleValueChromosome([np.random.choice([True, False]) for _ in range(10)])
    return chromosome

@pytest.fixture
def population():
    generator = MultipleValueChromosomeGenerator(mutation_onemax_specific, 10)
    return [generator() for i in range(25)]

def test_fitness_evaluation_true_value_count_nonnegative(sample_bool_list_chromosome, fitness_function):
    number_true_values = fitness_function(sample_bool_list_chromosome)
    assert number_true_values >= 0

def test_fitness_evaluation_eval_count(sample_bool_list_chromosome, fitness_function):
    number_true_values = fitness_function(sample_bool_list_chromosome)
    assert fitness_function.eval_count == 1

def test_evaluation_evaluates_all_list_values_per_individual(population, fitness_function):
    evaluation = Evaluation(fitness_function)
    evaluation(population)
    assert evaluation.eval_count == 25
    for indv in population:
        assert indv.fit_set
        assert indv.fitness is not None

def test_evaluation_skips_already_calculated_fitnesses(population, fitness_function):
    evaluation = Evaluation(fitness_function)
    population[0].fitness = 1.0
    evaluation(population)
    assert evaluation.eval_count == 24
    for indv in population:
        assert indv.fit_set
        assert indv.fitness is not None

def test_fitness_equals_true_value_count(fitness_function, population):
    evaluation = Evaluation(fitness_function)
    evaluation(population)

    for indv in population:
        fitness = 0
        for val in indv.values:
            if val == False:
                fitness += 1
        assert indv.fitness == fitness_function(indv)
        assert fitness == indv.fitness
