# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.Base.MultipleValues import SinglePointMutation, SinglePointCrossover
from bingo.Base.MuPlusLambdaEA import MuPlusLambda
from bingo.Base.MuCommaLambdaEA import MuCommaLambda

from SingleValue import SingleValueChromosome


@pytest.fixture
def population():
    return [SingleValueChromosome(str(i)) for i in range(10)]


def mutation_function():
    return np.random.choice([True, False])


@pytest.fixture
def plus_algo(add_e_evaluation, add_v_variation, add_s_selection):
    crossover = SinglePointCrossover()
    mutation = SinglePointMutation(mutation_function)
    evo_alg = MuPlusLambda(add_e_evaluation, add_s_selection, crossover,
                           mutation, 0.2, 0.4, 20)
    evo_alg.variation = add_v_variation
    return evo_alg


@pytest.fixture
def comma_algo(add_e_evaluation, add_v_variation, add_s_selection):
    crossover = SinglePointCrossover()
    mutation = SinglePointMutation(mutation_function)
    evo_alg = MuCommaLambda(add_e_evaluation, add_s_selection, crossover,
                            mutation, 0.2, 0.4, 20)
    evo_alg.variation = add_v_variation
    return evo_alg


def test_mu_comma_lambda_is_offspring_only(population, comma_algo):
    offspring = comma_algo.generational_step(population)
    for i, indv in enumerate(offspring):
        assert indv.value.endswith("ves")


def test_mu_plus_lambda_is_combined_population(population, plus_algo):
    new_population = plus_algo.generational_step(population)
    num_offspring = 0
    num_original_population = 0
    for indv in new_population:
        print(indv.value)
        assert indv.value.endswith("es")
        if "v" in indv.value:
            num_offspring += 1
        else:
            num_original_population += 1
    assert num_offspring == 20
    assert num_original_population == 10
