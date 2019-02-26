import pytest
import numpy as np

from bingo.Base.Variation import Variation
from bingo.Util.ArgumentValidation import argument_validation
from bingo.Base.EvolutionaryAlgorithm import EvolutionaryAlgorithm
from bingo.MultipleValues import *
from bingo.EA.VarOr import VarOr
from bingo.EA.MuPlusLambda import MuPlusLambda
from bingo.EA.TournamentSelection import Tournament
from bingo.EA.SimpleEvaluation import SimpleEvaluation
from examples.OneMaxExample import MultipleValueFitnessEvaluator

@pytest.fixture
def population():
    generator = MultipleValueGenerator(mutation_function, 10)
    return [generator() for i in range(25)]

@pytest.fixture
def ea():
    crossover = SinglePointCrossover()
    mutation = SinglePointMutation(mutation_function)
    var_or = VarOr(crossover, mutation, 0.2, 0.4)
    selection = Tournament(10)
    fitness = MultipleValueFitnessEvaluator()
    evaluator = SimpleEvaluation(fitness)
    return MuPlusLambda(var_or, evaluator, selection)

def mutation_function():
    return np.random.choice([True, False])


def test_basic_functionality(population, ea):
    offspring = ea.generational_step(population)
    assert offspring != population
