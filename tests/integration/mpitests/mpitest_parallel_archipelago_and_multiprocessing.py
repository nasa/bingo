# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np

from mpi4py import MPI
from mpitest_util import mpi_assert_true, run_t_in_module

from bingo.evolutionary_optimizers.parallel_archipelago import \
    ParallelArchipelago
from bingo.evolutionary_algorithms.mu_plus_lambda import MuPlusLambda
from bingo.evolutionary_optimizers.island import Island
from bingo.selection.tournament import Tournament
from bingo.chromosomes.multiple_values import SinglePointCrossover, \
    SinglePointMutation, MultipleValueChromosomeGenerator
from bingo.evaluation.fitness_function import FitnessFunction
from bingo.evaluation.evaluation import Evaluation

COMM = MPI.COMM_WORLD
COMM_RANK = COMM.Get_rank()
COMM_SIZE = COMM.Get_size()

POPULATION_SIZE = 100
OFFSPRING_SIZE = 100
N_PROC = 2


class MagnitudeFitness(FitnessFunction):
    def __call__(self, individual):
        self.eval_count += 1
        return np.linalg.norm(individual.values)


def evaluation():
    return Evaluation(MagnitudeFitness(), multiprocess=N_PROC)


def evo_alg():
    selection = Tournament(5)
    crossover = SinglePointCrossover()
    mutation = SinglePointMutation(np.random.random)
    return MuPlusLambda(evaluation(), selection, crossover,
                        mutation, 0.4, 0.4, OFFSPRING_SIZE)


def generator():
    return MultipleValueChromosomeGenerator(np.random.random, 5)


def multi_process_parallel_archipelago():
    island = Island(evo_alg(), generator(), POPULATION_SIZE)
    return ParallelArchipelago(island, non_blocking=False)


def get_total_population(parallel_archipelago):
    island_population = parallel_archipelago.island.population
    total_population = COMM.allgather(island_population)
    return np.array(total_population).flatten()


def test_parallel_archipelago_and_multiprocessing_eval():
    assertions = []

    np.random.seed(7)
    n_islands = COMM.Get_size()
    archipelago = multi_process_parallel_archipelago()

    for indv in get_total_population(archipelago):
        assertions.append(not indv.fit_set)
    assertions.append(archipelago.get_fitness_evaluation_count() == 0)
    assertions.append(archipelago.generational_age == 0)

    archipelago.evolve(1)

    for indv in get_total_population(archipelago):
        assertions.append(indv.fit_set)
    assertions.append(archipelago.get_fitness_evaluation_count() ==
                      n_islands * (POPULATION_SIZE + OFFSPRING_SIZE))
    assertions.append(archipelago.generational_age == 1)
    return mpi_assert_true(all(assertions))


if __name__ == "__main__":
    run_t_in_module(__name__)
