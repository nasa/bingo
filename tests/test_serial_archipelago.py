# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import random

import pytest
import numpy as np

from bingo.MultipleValues import SinglePointCrossover, SinglePointMutation, \
                                 MultipleValueGenerator
from bingo.Island import Island
from bingo.EA.MuPlusLambda import MuPlusLambda
from bingo.EA.TournamentSelection import Tournament
from bingo.Base.Evaluation import Evaluation
from bingo.Base.FitnessFunction import FitnessFunction
from bingo.SerialArchipelago import SerialArchipelago

POP_SIZE = 25
SELECTION_SIZE = 10
VALUE_LIST_SIZE = 10
OFFSPRING_SIZE = 20

class MultipleValueFitnessFunction(FitnessFunction):
    def __call__(self, individual):
        fitness = np.count_nonzero(individual.list_of_values)
        self.eval_count += 1
        return len(individual.list_of_values) - fitness


@pytest.fixture
def island():
    crossover = SinglePointCrossover()
    mutation = SinglePointMutation(mutation_function)
    selection = Tournament(SELECTION_SIZE)
    fitness = MultipleValueFitnessFunction()
    evaluator = Evaluation(fitness)
    ev_alg = MuPlusLambda(evaluator, selection, crossover, mutation,
                          0.2, 0.4, OFFSPRING_SIZE)
    generator = MultipleValueGenerator(mutation_function, VALUE_LIST_SIZE)
    return Island(ev_alg, generator, POP_SIZE)


def mutation_function():
    return np.random.choice([True, False])


def test_archipelago_generated(island):
    archipelago = SerialArchipelago(island, num_islands=3)
    assert len(archipelago._islands) == 3
    for island_i in archipelago._islands:
        assert island_i != island
        assert island_i._population_size == island._population_size


def test_generational_step_executed(island):
    random.seed(0)
    archipelago = SerialArchipelago(island, num_islands=3)
    archipelago.step_through_generations(1)
    for island_i in archipelago._islands:
        assert island_i.best_individual()
