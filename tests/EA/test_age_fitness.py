import pytest
import numpy as np

from bingo.Base.FitnessEvaluator import FitnessEvaluator
from bingo.Base.Mutation import Mutation
from bingo.Base.Crossover import Crossover
from bingo.Base.Variation import Variation
from bingo.Base.Evaluation import Evaluation
from bingo.Base.Selection import Selection
from bingo.EA.SimpleEvaluation import SimpleEvaluation
from bingo.MultipleValues import MultipleValueGenerator, MultipleValueChromosome
from bingo.AgeFitness import AgeFitness

INITIAL_POP_SIZE = 10
TARGET_POP_SIZE = 5
SIMPLE_INDV_SIZE = 1
COMPLEX_INDV_SIZE = 2

class MultipleValueFitnessEvaluator(FitnessEvaluator):
    def __call__(self, individual):
        fitness = np.count_nonzero(individual.list_of_values)
        self.eval_count += 1
        return len(individual.list_of_values) - fitness

def return_true():
    return True

def return_false():
    return False

@pytest.fixture
def weak_indvidual():
    generator = MultipleValueGenerator(return_false, SIMPLE_INDV_SIZE)
    indv = generator()
    indv.genetic_age = 100
    return indv

@pytest.fixture
def fit_individual():
    generator = MultipleValueGenerator(return_true, SIMPLE_INDV_SIZE)
    indv = generator()
    return indv

@pytest.fixture
def strong_population():
    generator = MultipleValueGenerator(return_true, SIMPLE_INDV_SIZE)
    return [generator() for i in range(INITIAL_POP_SIZE)]

@pytest.fixture
def non_dominated_population():
    young_weak = MultipleValueChromosome([False, False])
    middle_average = MultipleValueChromosome([False, True])
    middle_average.genetic_age = 1
    old_fit = MultipleValueChromosome([True, True])
    old_fit.genetic_age = 2
    return [young_weak, middle_average, old_fit]

@pytest.fixture
def all_dominated_population(non_dominated_population):
    non_dominated_population[0].genetic_age = 2
    non_dominated_population[2].genetic_age = 0
    return non_dominated_population


def test_age_fitness_selection_remove_one_weak(strong_population, weak_indvidual):
    age_fitness_selection = AgeFitness(remove_equals=False)
    indv_population = strong_population + [weak_indvidual]
    fitness = MultipleValueFitnessEvaluator()
    evaluator = SimpleEvaluation(fitness)
    evaluator(indv_population)

    new_population = age_fitness_selection(indv_population, TARGET_POP_SIZE)

    weak_indv_removed = True
    for i, indv in enumerate(new_population):
        if False in indv.list_of_values:
            weak_indv_removed = False
            break
    assert weak_indv_removed
    assert len(new_population) == len(strong_population)

def test_target_population_size_is_valid(strong_population):
    age_fitness_selection = AgeFitness()
    with pytest.raises(ValueError):
        age_fitness_selection(strong_population, len(strong_population) + 1)

def test_none_selected_for_removal(non_dominated_population):
    age_fitness_selection = AgeFitness()
    fitness = MultipleValueFitnessEvaluator()
    evaluator = SimpleEvaluation(fitness)
    evaluator(non_dominated_population)

    new_population = age_fitness_selection(non_dominated_population, 1)
    assert len(new_population) == len(non_dominated_population)

def test_all_but_one_removed(all_dominated_population):
    age_fitness_selection = AgeFitness()
    fitness = MultipleValueFitnessEvaluator()
    evaluator = SimpleEvaluation(fitness)
    evaluator(all_dominated_population)

    new_population = age_fitness_selection(all_dominated_population, 1)
    assert len(new_population) == 1

def test_all_but_one_removed_large_selection_size(strong_population, weak_indvidual):
    population = strong_population +[weak_indvidual]
    age_fitness_selection = AgeFitness(selection_size=10)
    fitness = MultipleValueFitnessEvaluator()
    evaluator = SimpleEvaluation(fitness)
    evaluator(population)

    new_population = age_fitness_selection(population, 1)
    assert len(new_population) == 1
    assert new_population[0].list_of_values == [True]
    assert age_fitness_selection._selection_attempts == 2

def test_one_iteration(weak_indvidual, fit_individual):
    population = [weak_indvidual for i in range(10)] + [fit_individual]
    age_fitness_selection = AgeFitness(selection_size=len(population))
    fitness = MultipleValueFitnessEvaluator()
    evaluator = SimpleEvaluation(fitness)
    evaluator(population)

    new_population = age_fitness_selection(population, 1)
    assert len(new_population) == 1
    assert new_population[0].list_of_values == [True]
    assert age_fitness_selection._selection_attempts == 1

if __name__ == '__main__':
    test_all_but_one_removed_large_selection_size(strong_population(), weak_indvidual())