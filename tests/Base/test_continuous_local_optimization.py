import pytest
import numpy as np

from bingo.Base.FitnessEvaluator import FitnessEvaluator, VectorBasedEvaluator
from bingo.Base.ContinuousLocalOptimization import ContinuousLocalOptimization
from bingo.MultipleFloatChromosome import MultipleFloatChromosome

NUM_VALS = 10
NUM_OPT = 3

class MultipleFloatValueFitnessEvaluator(FitnessEvaluator):
    def __call__(self, individual):
        print(individual)
        return np.linalg.norm(individual.list_of_values)

class FloatVectorFitnessEvaluator(VectorBasedEvaluator):
    def _evaluate_fitness_vector(self, individual):
        vals = individual.list_of_values
        return [x - 0 for x in vals]

@pytest.fixture
def opt_individual():
    vals = [1. for _ in range(NUM_VALS)]
    return MultipleFloatChromosome(vals, [1, 3, 4])

@pytest.fixture
def reg_individual():
    vals = [1. for _ in range(NUM_VALS)]
    return MultipleFloatChromosome(vals)

def test_optimize_params(opt_individual, reg_individual):
    fitness_function = MultipleFloatValueFitnessEvaluator()
    local_opt_fitness_function = ContinuousLocalOptimization(fitness_function)
    opt_indv_fitness = local_opt_fitness_function(opt_individual)
    reg_indv_fitness = local_opt_fitness_function(reg_individual)
    assert opt_indv_fitness == pytest.approx(np.sqrt(NUM_VALS - NUM_OPT))
    assert reg_indv_fitness == pytest.approx(np.sqrt(NUM_VALS))

def test_optimize_fitness_vector(opt_individual, reg_individual):
    reg_list = [1. for _ in range(NUM_VALS)]
    opt_list = [1. for _ in range(NUM_VALS)]
    opt_list[:3] = [0., 0., 0.]
    fitness_function = FloatVectorFitnessEvaluator()
    local_opt_fitness_function = ContinuousLocalOptimization(fitness_function,
                                                             algorithm='lm')
    opt_indv_fitness = local_opt_fitness_function(opt_individual)
    reg_indv_fitness = local_opt_fitness_function(reg_individual)
    assert opt_indv_fitness == pytest.approx(np.mean(opt_list))
    assert reg_indv_fitness == pytest.approx(np.mean(reg_list))

def test_valid_fitness_function():
    fitness_function = MultipleFloatValueFitnessEvaluator()
    with pytest.raises(TypeError):
        ContinuousLocalOptimization(fitness_function, algorithm='lm')
