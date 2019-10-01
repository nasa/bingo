# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np
from mpi4py import MPI

from bingo.symbolic_regression.agraph.crossover import AGraphCrossover
from bingo.symbolic_regression.agraph.mutation import AGraphMutation
from bingo.symbolic_regression.agraph.generator import AGraphGenerator
from bingo.symbolic_regression.agraph.component_generator import ComponentGenerator
from bingo.symbolic_regression.explicit_regression import ExplicitRegression, ExplicitTrainingData

from bingo.evolutionary_algorithms import AgeFitnessEA
from bingo.evolutionary_optimizers import parallel_archipelago
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers import island
from bingo.local_optimizers import ContinuousLocalOptimization

POP_SIZE = 100
STACK_SIZE = 10

def init_x_vals(start, stop, num_points):
    return np.linspace(start, stop, num_points).reshape([-1, 1])

def equation_eval(x):
    return x**2 + 3.5*x**3

def execute_generational_steps():
    communicator = MPI.COMM_WORLD
    rank = MPI.COMM_WORLD.Get_rank()

    x = None
    y = None

    if rank == 0:
        x = init_x_vals(-10, 10, 100)
        y = equation_eval(x)

    x = MPI.COMM_WORLD.bcast(x, root=0)
    y = MPI.COMM_WORLD.bcast(y, root=0)

    training_data = ExplicitTrainingData(x, y)

    component_generator = ComponentGenerator(x.shape[1])
    component_generator.add_operator(2)
    component_generator.add_operator(3)
    component_generator.add_operator(4)

    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_generator)

    agraph_generator = AGraphGenerator(STACK_SIZE, component_generator)

    fitness = ExplicitRegression(training_data=training_data)
    local_opt_fitness = ContinuousLocalOptimization(fitness, algorithm='lm')
    evaluator = Evaluation(local_opt_fitness)

    ea = AgeFitnessEA(evaluator, agraph_generator, crossover,
                      mutation, 0.4, 0.4, POP_SIZE)


    island = island(ea, agraph_generator, POP_SIZE)

    archipelago = parallel_archipelago(island)

    if archipelago.run_islands(2000, 1000, 1000):
        if rank == 0:
            print("best: ", archipelago.get_best_individual())


def main():
    execute_generational_steps()

if __name__ == '__main__':
    main()
    

