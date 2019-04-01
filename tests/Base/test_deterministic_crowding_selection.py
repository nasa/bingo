# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.Base.DeterministicCrowdingSelection import \
    DeterministicCrowdingSelection
from bingo.Base.MultipleValues import MultipleValueChromosomeGenerator


@pytest.fixture
def selection():
    return DeterministicCrowdingSelection()


@pytest.fixture
def unfit_generator():
    return MultipleValueChromosomeGenerator(return_false, 10)


@pytest.fixture
def mixed_fit_generator():
    return MultipleValueChromosomeGenerator(mutation_function, 10)


@pytest.fixture
def unfit_pop(onemax_evaluator, unfit_generator):
    pop = [unfit_generator() for _ in range(10)]
    onemax_evaluator(pop)
    return pop


@pytest.fixture
def fit_pop(onemax_evaluator, mixed_fit_generator):
    pop = [mixed_fit_generator() for _ in range(10)]
    onemax_evaluator(pop)
    return pop


def mutation_function():
    return np.random.choice([True, False])


def return_false():
    return False


def test_cannot_pass_odd_population_size(unfit_generator, selection):
    population = [unfit_generator() for _ in range(15)]
    with pytest.raises(ValueError):
        _ = selection(population, 10)


def test_target_pop_size_not_greater_than_pop_length(fit_pop, selection):
    with pytest.raises(ValueError):
        _ = selection(fit_pop, 12)


def test_fit_offspring_chosen_over_unfit_parents(unfit_pop, fit_pop,
                                                 onemax_evaluator, selection):
    next_gen = selection(unfit_pop + fit_pop, 10)
    onemax_evaluator(next_gen)
    for old, new in zip(fit_pop, next_gen):
        assert old.fitness == new.fitness
        assert old == new


def test_fit_parents_chosen_over_unfit_offspring(fit_pop, unfit_pop,
                                                 onemax_evaluator, selection):
    next_gen = selection(fit_pop + unfit_pop, 10)
    onemax_evaluator(next_gen)
    for old, new in zip(fit_pop, next_gen):
        assert old.fitness == new.fitness
        assert old == new


def test_unfit_parents_chosen_over_unfit_offspring(unfit_pop, onemax_evaluator,
                                                   selection):
    unfit_kids = [indv.copy() for indv in unfit_pop]
    next_gen = selection(unfit_pop + unfit_kids, 10)
    onemax_evaluator(next_gen)
    for old, new in zip(unfit_pop, next_gen):
        assert old.fitness == new.fitness
        assert old == new
