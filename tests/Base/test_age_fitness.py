# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import collections

import pytest

from bingo.Base.MultipleValues import MultipleValueChromosomeGenerator, \
                                 MultipleValueChromosome
from bingo.Base.Mutation import Mutation
from bingo.Base.Crossover import Crossover
from bingo.Base.AgeFitnessSelection import AgeFitness
from bingo.Base.AgeFitnessEA import AgeFitnessEA

INITIAL_POP_SIZE = 10
TARGET_POP_SIZE = 5
SIMPLE_INDV_SIZE = 1
COMPLEX_INDV_SIZE = 6


class DumbyCrossover(Crossover):
    def __call__(self, parent1, parent2):
        pass


class DumbyMutation(Mutation):
    def __call__(self, parent1):
        pass


def return_true():
    return True


def return_false():
    return False


@pytest.fixture
def fit_individual():
    generator = MultipleValueChromosomeGenerator(return_true, SIMPLE_INDV_SIZE)
    indv = generator()
    return indv


@pytest.fixture
def strong_population():
    generator = MultipleValueChromosomeGenerator(return_true, SIMPLE_INDV_SIZE)
    return [generator() for _ in range(INITIAL_POP_SIZE)]


@pytest.fixture
def weak_individual():
    generator = MultipleValueChromosomeGenerator(return_false, SIMPLE_INDV_SIZE)
    indv = generator()
    indv.genetic_age = 100
    return indv


@pytest.fixture
def weak_population():
    generator = MultipleValueChromosomeGenerator(return_false, 2 * SIMPLE_INDV_SIZE)
    return [generator() for _ in range(INITIAL_POP_SIZE)]


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


@pytest.fixture
def pareto_front_population():
    size_of_list = 6
    population = []
    for i in range(size_of_list+1):
        values = [False]*(size_of_list - i) + [True]*i
        indv = MultipleValueChromosome(values)
        indv.genetic_age = i
        population.append(indv)
    return population


@pytest.fixture
def selected_indiviudals(pareto_front_population):
    list_size = len(pareto_front_population[0].values)
    list_one = [False]*int(list_size/2)+[True]*int((list_size+1)/2)
    list_two = [False]*int((list_size+1)/2)+[True]*int(list_size/2)

    selected_indv_one = MultipleValueChromosome(list_one)
    selected_indv_two = MultipleValueChromosome(list_two)

    selected_indv_one.genetic_age = list_size
    selected_indv_two.genetic_age = list_size + 1
    return [selected_indv_one, selected_indv_two]


def test_target_population_size_is_valid(strong_population):
    age_fitness_selection = AgeFitness()
    with pytest.raises(ValueError):
        age_fitness_selection(strong_population, len(strong_population) + 1)


def test_none_selected_for_removal(non_dominated_population, onemax_evaluator):
    age_fitness_selection = AgeFitness()
    onemax_evaluator(non_dominated_population)

    target_pop_size = 1
    new_population = age_fitness_selection(non_dominated_population,
                                           target_pop_size)
    assert len(new_population) == len(non_dominated_population)


def test_all_but_one_removed(all_dominated_population, onemax_evaluator):
    age_fitness_selection = AgeFitness()
    onemax_evaluator(all_dominated_population)

    target_pop_size = 1
    new_population = age_fitness_selection(all_dominated_population,
                                           target_pop_size)
    assert len(new_population) == target_pop_size


def test_all_but_one_removed_large_selection_size(strong_population,
                                                  weak_individual,
                                                  onemax_evaluator):
    population = strong_population +[weak_individual]
    onemax_evaluator(population)

    age_fitness_selection = AgeFitness(selection_size=len(strong_population))

    target_pop_size = 1
    new_population = age_fitness_selection(population, target_pop_size)

    assert len(new_population) == target_pop_size
    assert new_population[0].values == [True]
    assert age_fitness_selection._selection_attempts == 2


def test_all_removed_in_one_iteration(weak_individual,
                                      fit_individual,
                                      onemax_evaluator):
    population = [weak_individual for _ in range(10)] + [fit_individual]
    onemax_evaluator(population)

    age_fitness_selection = AgeFitness(selection_size=len(population))

    target_pop_size = 1
    new_population = age_fitness_selection(population, target_pop_size)

    assert len(new_population) == target_pop_size
    assert new_population[0].values == [True]
    assert age_fitness_selection._selection_attempts == 1


def test_selection_size_larger_than_population(weak_population, fit_individual,
                                               onemax_evaluator):
    population = weak_population + [fit_individual]
    onemax_evaluator(population)

    age_fitness_selection = AgeFitness(selection_size=(len(population)+100))

    target_pop_size = 2
    new_population = age_fitness_selection(population, target_pop_size)

    assert len(new_population) == target_pop_size
    assert age_fitness_selection._selection_attempts == 1
    count = 1
    for indv in new_population:
        if not any(indv.values):
            count *= 2
        elif all(indv.values):
            count *= 3

    assert count == 6


def test_keep_pareto_front_miss_target_pop_size(pareto_front_population,
                                                onemax_evaluator,
                                                selected_indiviudals):
    selected_indv_one = selected_indiviudals[0]
    selected_indv_two = selected_indiviudals[1]
    population = pareto_front_population + selected_indiviudals
    onemax_evaluator(population)

    age_fitness_selection = AgeFitness(selection_size=len(population))
    new_population = age_fitness_selection(population, TARGET_POP_SIZE)

    assert len(new_population) == len(pareto_front_population)

    selected_indvs_removed = True
    for indv in new_population:
        if (indv.genetic_age == selected_indv_one and
                indv.values == selected_indv_one.values) or \
               (indv.genetic_age == selected_indv_two and
                indv.values == selected_indv_two.values):
            selected_indvs_removed = False
            break
    assert selected_indvs_removed


def test_age_fitness_ea_step(pareto_front_population, onemax_evaluator,
                             selected_indiviudals):
    population = pareto_front_population + selected_indiviudals
    mutation = DumbyMutation()
    crossover = DumbyCrossover()
    generator = MultipleValueChromosomeGenerator(return_false, COMPLEX_INDV_SIZE)
    evo_alg = AgeFitnessEA(onemax_evaluator, generator, crossover, mutation,
                           0, 0, len(pareto_front_population),
                           selection_size=2*len(population))
    new_population = evo_alg.generational_step(population)
    assert len(new_population) == len(population)


def test_get_pareto_front(pareto_front_population,
                          selected_indiviudals,
                          onemax_evaluator):
    selection = AgeFitness()
    population = pareto_front_population + selected_indiviudals
    onemax_evaluator(population)
    new_population = selection.select_pareto_front(population)
    assert collections.Counter(new_population) == \
    collections.Counter(pareto_front_population)
