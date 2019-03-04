import pytest
import numpy as np

from bingo.Base.Variation import Variation
from bingo.MultipleValues import SinglePointMutation, SinglePointCrossover
from bingo.Base.Evaluation import Evaluation
from bingo.EA.MuCommaLambda import MuCommaLambda
from bingo.Base.Selection import Selection

from SingleValue import SingleValueChromosome

class add_v_variation(Variation):
    def __call__(self, population, number_offspring):
        offspring = [parent.copy() for parent in population]
        for indv in offspring:
            indv.value += "v"
        return offspring

class add_e_evaluation(Evaluation):
    def __call__(self, population):
        for indv in population:
            indv.fitness = indv.value
            indv.value += "e"

class add_s_selection(Selection):
    def __call__(self, population, _target_population_size):
        for indv in population:
            indv.value += "s"
        return population

@pytest.fixture
def population():
    return [SingleValueChromosome(str(i)) for i in range(10)]

@pytest.fixture
def ea():
    crossover = SinglePointCrossover()
    mutation = SinglePointMutation(mutation_function)
    ea = MuCommaLambda(add_e_evaluation(), add_s_selection(), crossover, mutation, 0.2, 0.4, 20)
    ea._variation = add_v_variation()
    return ea

def mutation_function():
    return np.random.choice([True, False])

def test_basic_functionality(population, ea):
    offspring = ea.generational_step(population)
    for i, indv in enumerate(offspring):
        assert indv.value == str(i) + "ves"
