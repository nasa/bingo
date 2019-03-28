# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from SingleValue import SingleValueChromosome
from bingo.Base.Variation import Variation
from bingo.Base.Evaluation import Evaluation
from bingo.Base.Selection import Selection
from bingo.Base.FitnessFunction import FitnessFunction


@pytest.fixture
def single_value_population_of_4():
    return [SingleValueChromosome(),
            SingleValueChromosome(),
            SingleValueChromosome(),
            SingleValueChromosome()]


@pytest.fixture
def single_value_population_of_100():
    return [SingleValueChromosome() for _ in range(100)]


class VariationAddV(Variation):
    def __call__(self, population, number_offspring):
        offspring = [population[i % len(population)].copy()
                     for i in range(number_offspring)]
        for indv in offspring:
            indv.value += "v"
        return offspring


class EvaluationAddE(Evaluation):
    def __init__(self):
        super().__init__(None)

    def __call__(self, population):
        for indv in population:
            indv.fitness = indv.value
            indv.value += "e"


class SelectionAddS(Selection):
    def __call__(self, population, _target_population_size):
        for indv in population:
            indv.value += "s"
        return population


@pytest.fixture()
def add_e_evaluation():
    return EvaluationAddE()


@pytest.fixture()
def add_s_selection():
    return SelectionAddS()


@pytest.fixture()
def add_v_variation():
    return VariationAddV()


class MultipleValueFitnessFunction(FitnessFunction):
    def __call__(self, individual):
        fitness = np.count_nonzero(individual.values)
        self.eval_count += 1
        return len(individual.values) - fitness


@pytest.fixture
def onemax_fitness():
    return MultipleValueFitnessFunction()


@pytest.fixture
def onemax_evaluator(onemax_fitness):
    return Evaluation(onemax_fitness)
