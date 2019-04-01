# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.Base.DeterministicCrowdingEA import DeterministicCrowdingEA
from bingo.Base.MultipleValues import SinglePointCrossover, SinglePointMutation, \
                                 MultipleValueChromosomeGenerator


@pytest.fixture
def dc_ea(onemax_evaluator):
    crossover = SinglePointCrossover()
    mutation = SinglePointMutation(return_true)
    return DeterministicCrowdingEA(onemax_evaluator, crossover, mutation, 0.2,
                                   0.2)


@pytest.fixture
def unfit_generator():
    return MultipleValueChromosomeGenerator(return_false, 10)


@pytest.fixture
def fit_generator():
    return MultipleValueChromosomeGenerator(return_true, 10)


@pytest.fixture
def unfit_pop(unfit_generator):
    return [unfit_generator() for _ in range(10)]


@pytest.fixture
def fit_pop(fit_generator):
    return [fit_generator() for _ in range(10)]


def mutation_function():
    return np.random.choice([True, False])


def return_false():
    return False


def return_true():
    return True


def return_total_fitness(population, evaluation):
    evaluation(population)
    return sum(indv.fitness for indv in population)


def test_next_gen_fitness_lower_than_parents(unfit_pop, dc_ea,
                                             onemax_evaluator):
    np.random.seed(0)
    parent_fitness = return_total_fitness(unfit_pop, onemax_evaluator)
    next_gen = dc_ea.generational_step(unfit_pop)
    next_gen_fitness = return_total_fitness(next_gen, onemax_evaluator)
    assert len(next_gen) == len(unfit_pop)
    assert next_gen_fitness < parent_fitness


def test_fit_parents_chosen_for_next_gen(fit_pop, dc_ea, onemax_evaluator):
    parent_fitness = return_total_fitness(fit_pop, onemax_evaluator)
    next_gen = dc_ea.generational_step(fit_pop)
    next_gen_fitness = return_total_fitness(next_gen, onemax_evaluator)
    assert parent_fitness == next_gen_fitness
    for old, new in zip(fit_pop, next_gen):
        assert old == new
