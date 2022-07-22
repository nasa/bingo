import numpy as np
import pytest

from mpi4py import MPI

from bingo.evolutionary_optimizers.parallel_archipelago import \
    ParallelArchipelago
from bingo.evolutionary_algorithms.mu_plus_lambda import MuPlusLambda
from bingo.evolutionary_optimizers.island import Island
from bingo.selection.tournament import Tournament
from bingo.chromosomes.multiple_values import SinglePointCrossover, \
    SinglePointMutation, MultipleValueChromosomeGenerator
from bingo.evaluation.fitness_function import FitnessFunction
from bingo.evaluation.evaluation import Evaluation

POPULATION_SIZE = 100
N_PROC = 2


class MagnitudeFitness(FitnessFunction):
    def __call__(self, individual):
        self.eval_count += 1
        return np.linalg.norm(individual.values)


@pytest.fixture
def evaluation():
    return Evaluation(MagnitudeFitness(), multiprocess=N_PROC)


@pytest.fixture
def ea():
    selection = Tournament(5)
    crossover = SinglePointCrossover()
    mutation = SinglePointMutation(np.random.random)
    return MuPlusLambda(evaluation, selection, crossover,
                        mutation, 0.4, 0.4, POPULATION_SIZE)


@pytest.fixture
def generator():
    return MultipleValueChromosomeGenerator(np.random.randint, 5)


@pytest.fixture
def multi_process_parallel_archipelago(ea, generator):
    island = Island(ea, generator)
    return ParallelArchipelago(island)


def get_total_population(parallel_archipelago, comm):
    island_population = parallel_archipelago.island.population
    total_population = comm.allgather(island_population)
    return total_population


def test_parallel_archipelago_and_multiprocessing_eval(
        multi_process_parallel_archipelago):
    np.random.seed(7)
    comm = MPI.COMM_WORLD
    archipelago = multi_process_parallel_archipelago

    for indv in get_total_population(archipelago, comm):
        assert not indv.fit_set
    assert archipelago.get_fitness_evaluation_count() == 0

    archipelago.evolve(1)

    for indv in get_total_population(archipelago, comm):
        assert indv.fit_set
    assert archipelago.get_fitness_evaluation_count() == POPULATION_SIZE
