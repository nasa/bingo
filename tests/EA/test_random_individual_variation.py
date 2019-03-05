import pytest

from bingo.MultipleValues import MultipleValueGenerator,\
                                    SinglePointCrossover,\
                                    SinglePointMutation
from bingo.EA.VarOr import VarOr
from bingo.Base.Variation import Variation
from bingo.EA.RandomIndividualVariation import RandomIndividualVariation

POP_SIZE = 25
SIMPLE_INDV_SIZE = 1
COMPLEX_INDV_SIZE = 2

class ReplicationVariation(Variation):
    def __init__(self):
        super().__init__()
    def __call__(self, population, number_offspring):
        return population[0:number_offspring]


def false_variation_function():
    return False

def true_variation_function():
    return True

@pytest.fixture
def weak_population():
    generator = MultipleValueGenerator(false_variation_function,
                                       SIMPLE_INDV_SIZE)
    return [generator() for i in range(25)]

@pytest.fixture
def true_chromosome_generator():
    return MultipleValueGenerator(true_variation_function, SIMPLE_INDV_SIZE)
    
@pytest.fixture
def init_replication_variation():
    return ReplicationVariation() 

def test_random_individual_added_to_pop(init_replication_variation, 
                                        true_chromosome_generator,
                                        weak_population):
    rand_indv_var_or = RandomIndividualVariation(init_replication_variation,
                                                 true_chromosome_generator)
    offspring = rand_indv_var_or(weak_population, POP_SIZE)
    success = False
    count = 0
    for indv in offspring:
        if True in indv.list_of_values:
            success = True
            count += 1
    assert success and count == 1

