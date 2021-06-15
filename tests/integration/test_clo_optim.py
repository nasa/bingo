# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.evaluation.fitness_function \
    import FitnessFunction, VectorBasedFunction
from bingo.local_optimizers.continuous_local_opt \
    import ContinuousLocalOptimization
from bingo.chromosomes.multiple_floats import MultipleFloatChromosome

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
