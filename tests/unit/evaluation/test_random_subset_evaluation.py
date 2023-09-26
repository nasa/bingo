# Ignoring some linting rules in tests
# pylint: disable=missing-docstring
from multiprocessing import pool
import numpy as np
import pytest

from bingo.chromosomes.chromosome import Chromosome
from bingo.evaluation.random_subset_evaluation import RandomSubsetEvaluation
from bingo.evaluation.fitness_function import FitnessFunction


class SubsetFitness(FitnessFunction):
    def __call__(self, indv):
        self.eval_count += 1
        return self.training_data


class DummyIndv(Chromosome):
    def __str__(self):
        return ""

    def distance(self, other):
        return 0


def test_subset_size_must_nonzero():
    training_data = [1, 2, 3, 4, 5]
    fitness_function = SubsetFitness(training_data)

    with pytest.raises(ValueError):
        _ = RandomSubsetEvaluation(fitness_function, subset_size=0)


def test_subset_of_training_data_is_used():
    training_data = np.arange(5, dtype=int)
    fitness_function = SubsetFitness(training_data)
    population = [DummyIndv(), DummyIndv()]

    evaluation = RandomSubsetEvaluation(fitness_function, subset_size=3)

    evaluation(population)

    assert len(population[0].fitness) == 3
    assert tuple(population[0].fitness) == tuple(population[1].fitness)


def test_subsets_are_different_in_each_evaluation():
    training_data = np.arange(25, dtype=int)
    fitness_function = SubsetFitness(training_data)

    evaluation = RandomSubsetEvaluation(fitness_function, subset_size=3)

    population_1 = [DummyIndv()]
    population_2 = [DummyIndv()]
    evaluation(population_1)
    evaluation(population_2)

    assert tuple(population_1[0].fitness) != tuple(population_2[0].fitness)
