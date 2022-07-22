import inspect
import numpy as np
import sys

from bingo.evolutionary_optimizers.parallel_archipelago import \
    ParallelArchipelago
from bingo.evolutionary_algorithms.mu_plus_lambda import MuPlusLambda
from bingo.evolutionary_optimizers.island import Island
from bingo.selection.tournament import Tournament
from bingo.chromosomes.multiple_values import SinglePointCrossover, \
    SinglePointMutation, MultipleValueChromosomeGenerator
from bingo.evaluation.fitness_function import FitnessFunction
from bingo.evaluation.evaluation import Evaluation

from mpi4py import MPI
COMM = MPI.COMM_WORLD
COMM_RANK = COMM.Get_rank()
COMM_SIZE = COMM.Get_size()

POPULATION_SIZE = 100
OFFSPRING_SIZE = 100
N_PROC = 2


def mpi_assert_true(value):
    if not value:
        message = "\tproc {}: False, expected True\n".format(COMM_RANK)
    else:
        message = ""
    all_values = COMM.allgather(value)
    all_messages = COMM.allreduce(message, op=MPI.SUM)
    return all(all_values), all_messages


class MagnitudeFitness(FitnessFunction):
    def __call__(self, individual):
        self.eval_count += 1
        return np.linalg.norm(individual.values)


def evaluation():
    return Evaluation(MagnitudeFitness(), multiprocess=N_PROC)


def ea():
    selection = Tournament(5)
    crossover = SinglePointCrossover()
    mutation = SinglePointMutation(np.random.random)
    return MuPlusLambda(evaluation(), selection, crossover,
                        mutation, 0.4, 0.4, OFFSPRING_SIZE)


def generator():
    return MultipleValueChromosomeGenerator(np.random.random, 5)


def multi_process_parallel_archipelago():
    island = Island(ea(), generator(), POPULATION_SIZE)
    # TODO: is it intended that non_blocking steps by sync_frequency?
    #       what if n_steps < sync_frequency?
    #       also < vs. <= in avg_age vs. target_age
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


def run_t(test_name, test_func):
    if COMM_RANK == 0:
        print(test_name, end=" ")
    COMM.barrier()
    success, message = test_func()
    COMM.barrier()
    if success:
        if COMM_RANK == 0:
            print(".")
    else:
        if COMM_RANK == 0:
            print("F")
            print(message, end=" ")
    return success


def driver():
    results = []
    tests = [(name, func)
             for name, func in inspect.getmembers(sys.modules[__name__],
                                                  inspect.isfunction)
             if "test" in name]
    if COMM_RANK == 0:
        print("========== collected", len(tests), "items ==========")

    for name, func in tests:
        results.append(run_t(name, func))

    num_success = sum(results)
    num_failures = len(results) - num_success
    if COMM_RANK == 0:
        print("==========", end="  ")
        if num_failures > 0:
            print(num_failures, "failed,", end=" ")
        print(num_success, "passed ==========")

    if num_failures > 0:
        exit(-1)


if __name__ == "__main__":
    driver()
