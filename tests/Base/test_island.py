# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.Base.MultipleValues import SinglePointCrossover, SinglePointMutation, \
                                 MultipleValueChromosomeGenerator
from bingo.Base.Island import Island
from bingo.Base.MuPlusLambdaEA import MuPlusLambda
from bingo.Base.TournamentSelection import Tournament
from bingo.Base.Evaluation import Evaluation
from bingo.Base.FitnessFunction import FitnessFunction


class MultipleValueFitnessFunction(FitnessFunction):
    def __call__(self, individual):
        fitness = np.count_nonzero(individual.values)
        self.eval_count += 1
        return len(individual.values) - fitness


@pytest.fixture
def island():
    crossover = SinglePointCrossover()
    mutation = SinglePointMutation(mutation_function)
    selection = Tournament(10)
    fitness = MultipleValueFitnessFunction()
    evaluator = Evaluation(fitness)
    ev_alg = MuPlusLambda(evaluator, selection, crossover, mutation,
                          0.2, 0.4, 20)
    generator = MultipleValueChromosomeGenerator(mutation_function, 10)
    return Island(ev_alg, generator, 25)


def mutation_function():
    return np.random.choice([True, False])


def test_manual_evaluation(island):
    island.evaluate_population()
    for indv in island.get_population():
        assert indv.fit_set


def test_generational_steps_change_population_age(island):
    for indv in island.get_population():
        assert indv.genetic_age == 0
    island.execute_generational_step()
    for indv in island.population:
        assert indv.genetic_age > 0


def test_generational_age_increases(island):
    island.execute_generational_step()
    assert island.generational_age == 1
    island.execute_generational_step()
    assert island.generational_age == 2


def test_best_individual(island):
    island.execute_generational_step()
    fitness = [indv.fitness for indv in island.get_population()]
    best = island.best_individual()
    assert best.fitness == min(fitness)


def test_pareto_front_sorted_by_fitness(island):
    island.execute_generational_step()
    island.update_pareto_front()
    pareto_front = island.get_pareto_front()
    assert all(pareto_front[i].fitness <= pareto_front[i+1].fitness \
               for i in range(len(pareto_front)-1))

