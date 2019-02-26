import pytest
import numpy as np

from bingo.MultipleValues import *
from bingo.EA.VarOr import VarOr
from bingo.EA.BasicIsland import Island
from bingo.EA.MuPlusLambda import MuPlusLambda
from bingo.EA.TournamentSelection import Tournament
from bingo.EA.SimpleEvaluation import SimpleEvaluation
from examples.OneMaxExample import MultipleValueFitnessEvaluator

@pytest.fixture
def island():
    crossover = SinglePointCrossover()
    mutation = SinglePointMutation(mutation_function)
    var_or = VarOr(crossover, mutation, 0.2, 0.4)
    selection = Tournament(10)
    fitness = MultipleValueFitnessEvaluator()
    evaluator = SimpleEvaluation(fitness)
    ea = MuPlusLambda(var_or, evaluator, selection)
    generator = MultipleValueGenerator(mutation_function, 10)
    return Island(ea, generator, 25)

def mutation_function():
    return np.random.choice([True, False])

def test_no_best_individual_unless_evaluated(island):
    with pytest.raises(ValueError):
        island.best_individual()

def test_generational_steps_change_population(island):
    population = island.pop
    offspring = island.execute_generational_step()
    assert population != offspring
    offspring_2 = island.execute_generational_step()
    assert offspring != offspring_2

def test_best_individual(island):
    population = island.execute_generational_step()
    fitness = [indv.fitness for indv in population]
    best = island.best_individual()
    assert best.fitness == min(fitness)
