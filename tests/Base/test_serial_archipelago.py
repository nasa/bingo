# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import random

import pytest
import numpy as np

from bingo.Base.MultipleValues import SinglePointCrossover, \
                                      SinglePointMutation, \
                                      MultipleValueChromosomeGenerator
from bingo.Base.Island import Island
from bingo.Base.MuPlusLambdaEA import MuPlusLambda
from bingo.Base.TournamentSelection import Tournament
from bingo.Base.Evaluation import Evaluation
from bingo.Base.FitnessFunction import FitnessFunction
from bingo.Base.SerialArchipelago import SerialArchipelago


POP_SIZE = 5
SELECTION_SIZE = 10
VALUE_LIST_SIZE = 10
OFFSPRING_SIZE = 20
ERROR_TOL = 10e-6


class MultipleValueFitnessFunction(FitnessFunction):
    def __call__(self, individual):
        fitness = np.count_nonzero(individual.values)
        self.eval_count += 1
        return fitness


def generate_three():
    return 3


def generate_two():
    return 2


def generate_one():
    return 1


def generate_zero():
    return 0


def mutation_function():
    return np.random.choice([False, True])


@pytest.fixture
def evol_alg():
    crossover = SinglePointCrossover()
    mutation = SinglePointMutation(mutation_function)
    selection = Tournament(SELECTION_SIZE)
    fitness = MultipleValueFitnessFunction()
    evaluator = Evaluation(fitness)
    return MuPlusLambda(evaluator, selection, crossover, mutation,
                        0.2, 0.4, OFFSPRING_SIZE)


@pytest.fixture
def zero_island(evol_alg):
    generator = MultipleValueChromosomeGenerator(generate_zero,
                                                 VALUE_LIST_SIZE)
    return Island(evol_alg, generator, POP_SIZE)


@pytest.fixture
def one_island(evol_alg):
    generator = MultipleValueChromosomeGenerator(generate_one,
                                                 VALUE_LIST_SIZE)
    return Island(evol_alg, generator, POP_SIZE)


@pytest.fixture
def two_island(evol_alg):
    generator = MultipleValueChromosomeGenerator(generate_two,
                                                 VALUE_LIST_SIZE)
    return Island(evol_alg, generator, POP_SIZE)


@pytest.fixture
def three_island(evol_alg):
    generator = MultipleValueChromosomeGenerator(generate_three,
                                                 VALUE_LIST_SIZE)
    return Island(evol_alg, generator, POP_SIZE)


@pytest.fixture
def island_list(zero_island, one_island, two_island, three_island):
    return [zero_island, one_island, two_island, three_island]


@pytest.fixture
def island(evol_alg):
    generator = MultipleValueChromosomeGenerator(mutation_function,
                                                 VALUE_LIST_SIZE)
    return Island(evol_alg, generator, POP_SIZE)


def test_archipelago_generated(island):
    archipelago = SerialArchipelago(island, num_islands=3)
    assert len(archipelago._islands) == 3
    for island_i in archipelago._islands:
        assert island_i != island
        assert island_i._population_size == island._population_size


# def test_generational_step_executed(island):
#     random.seed(0)
#     archipelago = SerialArchipelago(island, num_islands=3)
#     archipelago._step_through_generations(1)
#     for island_i in archipelago._islands:
#         assert island_i.get_best_individual()


def test_island_migration(one_island, island_list):
    archipelago = SerialArchipelago(one_island, num_islands=4)
    archipelago._islands = island_list

    archipelago._coordinate_migration_between_islands()

    migration_count = 0
    for i, island in enumerate(archipelago._islands):
        initial_individual_values = [i]*VALUE_LIST_SIZE
        for individual in island.population:
            if initial_individual_values != individual.values:
                migration_count += 1
                break
    assert len(island_list) == migration_count


def test_convergence_of_archipelago(one_island, island_list):
    archipelago = SerialArchipelago(one_island, num_islands=4)
    archipelago._islands = island_list

    assert archipelago.get_best_fitness() <= 0


def test_convergence_of_archipelago_unconverged(one_island):
    archipelago = SerialArchipelago(one_island, num_islands=6)
    assert archipelago.get_best_fitness() > 0


def test_best_individual_returned(one_island):
    archipelago = SerialArchipelago(one_island)
    generator = MultipleValueChromosomeGenerator(generate_zero, VALUE_LIST_SIZE)
    best_indv = generator()
    archipelago._islands[0].load_population([best_indv], replace=False)
    assert archipelago.get_best_individual().fitness == 0


def test_best_fitness_eval_count(one_island):
    num_islands = 4
    archipelago = SerialArchipelago(one_island,
                                    num_islands=num_islands)
    assert archipelago.get_fitness_evaluation_count() == 0
    archipelago.evolve(1)
    expected_evaluations = num_islands * (POP_SIZE+OFFSPRING_SIZE)
    assert archipelago.get_fitness_evaluation_count() == expected_evaluations


def test_archipelago_runs(one_island, two_island, three_island):
    max_generations = 100
    min_generations = 20
    error_tol = 0
    generation_step_report = 10
    archipelago = SerialArchipelago(one_island, num_islands=4)
    archipelago._islands = [one_island, two_island, three_island, three_island]
    result = archipelago.evolve_until_convergence(max_generations,
                                                  error_tol,
                                                  generation_step_report,
                                                  min_generations)
    assert result.success


def test_total_population_not_affected_by_migration(one_island):
    archipelago = SerialArchipelago(one_island, num_islands=4)
    total_pop_before = sum([len(i.population) for i in archipelago._islands])
    archipelago._coordinate_migration_between_islands()
    total_pop_after = sum([len(i.population) for i in archipelago._islands])
    assert total_pop_after == total_pop_before


def test_potential_hof_members(mocker, one_island):
    island_a = mocker.Mock(hall_of_fame=['a'])
    island_b = mocker.Mock(hall_of_fame=['b'])
    island_c = mocker.Mock(hall_of_fame=['c'])
    archipelago = SerialArchipelago(one_island, num_islands=3)
    archipelago._islands = [island_a, island_b, island_c]
    assert archipelago._get_potential_hof_members() == ['a', 'b', 'c']
