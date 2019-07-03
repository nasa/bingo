# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.Base.FitnessFunction import FitnessFunction, VectorBasedFunction
from bingo.Base.ContinuousLocalOptimization import ContinuousLocalOptimization
from bingo.Base.MultipleFloats import MultipleFloatChromosome

NUM_VALS = 10
NUM_OPT = 3


class MultipleFloatValueFitnessFunction(FitnessFunction):
    def __call__(self, individual):
        print(individual)
        return np.linalg.norm(individual.values)


class FloatVectorFitnessFunction(VectorBasedFunction):
    def evaluate_fitness_vector(self, individual):
        vals = individual.values
        return [x - 0 for x in vals]


@pytest.fixture
def opt_individual():
    vals = [1. for _ in range(NUM_VALS)]
    return MultipleFloatChromosome(vals, [1, 3, 4])


@pytest.fixture
def reg_individual():
    vals = [1. for _ in range(NUM_VALS)]
    return MultipleFloatChromosome(vals)


@pytest.mark.parametrize("algorithm", [
    'Nelder-Mead',
    'Powell',
    'CG',
    'BFGS',
    # 'Newton-CG',
    'L-BFGS-B',
    # 'TNC',
    # 'COBYLA',
    'SLSQP'
    # 'trust-constr'
    # 'dogleg',
    # 'trust-ncg',
    # 'trust-exact',
    # 'trust-krylov'
])
def test_optimize_params(opt_individual, reg_individual, algorithm):
    fitness_function = MultipleFloatValueFitnessFunction()
    local_opt_fitness_function = ContinuousLocalOptimization(
        fitness_function, algorithm)
    opt_indv_fitness = local_opt_fitness_function(opt_individual)
    reg_indv_fitness = local_opt_fitness_function(reg_individual)
    assert opt_indv_fitness == pytest.approx(np.sqrt(NUM_VALS - NUM_OPT),
                                             rel=5.e-6)
    assert reg_indv_fitness == pytest.approx(np.sqrt(NUM_VALS))


@pytest.mark.parametrize("algorithm", [
    # 'hybr',
    'lm'
    # 'broyden1',
    # 'broyden2',
    # 'anderson',
    # 'linearmixing',
    # 'diagbroyden',
    # 'excitingmixing',
    # 'krylov',
    # 'df-sane'
])
def test_optimize_fitness_vector(opt_individual, reg_individual, algorithm):
    reg_list = [1. for _ in range(NUM_VALS)]
    opt_list = [1. for _ in range(NUM_VALS)]
    opt_list[:3] = [0., 0., 0.]
    fitness_function = FloatVectorFitnessFunction()
    local_opt_fitness_function = ContinuousLocalOptimization(
        fitness_function, algorithm)
    opt_indv_fitness = local_opt_fitness_function(opt_individual)
    reg_indv_fitness = local_opt_fitness_function(reg_individual)
    assert opt_indv_fitness == pytest.approx(np.mean(opt_list))
    assert reg_indv_fitness == pytest.approx(np.mean(reg_list))


def test_valid_fitness_function():
    fitness_function = MultipleFloatValueFitnessFunction()
    with pytest.raises(TypeError):
        ContinuousLocalOptimization(fitness_function, algorithm='lm')


def test_not_valid_algorithm():
    fitness_function = MultipleFloatValueFitnessFunction()
    with pytest.raises(KeyError):
        ContinuousLocalOptimization(fitness_function,
                                    algorithm='Dwayne - The Rock - Johnson')


def test_get_eval_count_pass_through():
    fitness_function = MultipleFloatValueFitnessFunction()
    fitness_function.eval_count = 123
    local_opt_fitness_function = \
        ContinuousLocalOptimization(fitness_function, "Powell")
    assert local_opt_fitness_function.eval_count == 123


def test_set_eval_count_pass_through():
    fitness_function = MultipleFloatValueFitnessFunction()
    local_opt_fitness_function = \
        ContinuousLocalOptimization(fitness_function, "Powell")
    local_opt_fitness_function.eval_count = 123
    assert fitness_function.eval_count == 123


def test_get_training_data_pass_through():
    fitness_function = MultipleFloatValueFitnessFunction()
    fitness_function.training_data = 123
    local_opt_fitness_function = \
        ContinuousLocalOptimization(fitness_function, "Powell")
    assert local_opt_fitness_function.training_data == 123


def test_set_training_data_pass_through():
    fitness_function = MultipleFloatValueFitnessFunction()
    local_opt_fitness_function = \
        ContinuousLocalOptimization(fitness_function, "Powell")
    local_opt_fitness_function.training_data = 123
    assert fitness_function.training_data == 123
