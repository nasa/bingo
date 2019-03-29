# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest

from bingo.Base.MultipleValues import MultipleValueChromosomeGenerator
from bingo.Base.Variation import Variation
from bingo.Base.AddRandomIndividualVariation import AddRandomIndividualVariation

POP_SIZE = 25
SIMPLE_INDV_SIZE = 1
COMPLEX_INDV_SIZE = 2


class ReplicationVariation(Variation):
    def __call__(self, population, number_offspring):
        return population[0:number_offspring]


def false_variation_function():
    return False


def true_variation_function():
    return True


def true_multiple_variation_function():
    return [True]*COMPLEX_INDV_SIZE


@pytest.fixture
def weak_population():
    generator = MultipleValueChromosomeGenerator(false_variation_function,
                                                 SIMPLE_INDV_SIZE)
    return [generator() for i in range(25)]


@pytest.fixture
def weaker_population():
    generator = MultipleValueChromosomeGenerator(false_variation_function,
                                                 COMPLEX_INDV_SIZE)
    return [generator() for i in range(25)]


@pytest.fixture
def true_chromosome_generator():
    return MultipleValueChromosomeGenerator(true_variation_function,
                                            SIMPLE_INDV_SIZE)


@pytest.fixture
def init_replication_variation():
    return ReplicationVariation()


def test_random_individual_added_to_pop(init_replication_variation,
                                        true_chromosome_generator,
                                        weak_population):
    indvs_added = 1
    rand_indv_var_or = AddRandomIndividualVariation(init_replication_variation,
                                                    true_chromosome_generator,
                                                    num_rand_indvs=indvs_added)
    offspring = rand_indv_var_or(weak_population, POP_SIZE)
    count = 0
    for indv in offspring:
        if True in indv.values:
            count += 1
    assert count == indvs_added


def test_multiple_indviduals_added_to_pop(init_replication_variation,
                                          weaker_population):
    indvs_added = 2
    generator = MultipleValueChromosomeGenerator(true_multiple_variation_function,
                                                 COMPLEX_INDV_SIZE)
    rand_indv_var_or = AddRandomIndividualVariation(init_replication_variation,
                                                    generator,
                                                    num_rand_indvs=indvs_added)
    offspring = rand_indv_var_or(weaker_population, POP_SIZE)
    count = 0
    for indv in offspring:
        if all(indv.values):
            count += 1
    assert count == indvs_added
