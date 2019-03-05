import pytest
import numpy as np

from bingo.Base.Variation import Variation
from bingo.MultipleValues import MultipleValueGenerator, SinglePointCrossover, SinglePointMutation
from bingo.EA.VarOr import VarOr

@pytest.fixture
def population():
    generator = MultipleValueGenerator(mutation_function, 10)
    return [generator() for i in range(25)]

@pytest.fixture
def var_or():
    crossover = SinglePointCrossover()
    mutation = SinglePointMutation(mutation_function)
    var_Or = VarOr(crossover, mutation, 0.2, 0.4)
    return var_Or

def mutation_function():
    return np.random.choice([True, False])

def test_invalid_probabilities():
    crossover = SinglePointCrossover()
    mutation = SinglePointMutation(mutation_function)
    with pytest.raises(ValueError):
        var_Or = VarOr(crossover, mutation, 0.6, 0.4)

def test_offspring_not_equals_parents(population, var_or):
    offspring = var_or(population, 25)
    for i, indv in enumerate(population):
        assert indv is not offspring[i]
        
def test_no_two_variations_at_once(population, var_or):
    offspring = var_or(population, 25)
    for i, indv in enumerate(var_or.crossover_offspring):
        assert not (indv and var_or.mutation_offspring[i])
