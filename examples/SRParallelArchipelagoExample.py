"""Evolve a symbolic-regression expression across MPI processes."""

import numpy as np
from mpi4py import MPI

from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_optimizers.island import Island
from bingo.evolutionary_optimizers.parallel_archipelago import ParallelArchipelago
from bingo.expressions import (
    AGraphCrossover,
    AGraphGenerator,
    AGraphMutation,
    ComponentGenerator,
)
from bingo.symbolic_regression import ExplicitRegression


def main():
    communicator = MPI.COMM_WORLD
    rank = communicator.Get_rank()
    x = np.linspace(-10, 10, 100).reshape(-1, 1) if rank == 0 else None
    y = x**2 + 3.5 * x**3 if rank == 0 else None
    x = communicator.bcast(x, root=0)
    y = communicator.bcast(y, root=0)

    component_generator = ComponentGenerator(x.shape[1], random_state=rank)
    for operator in ("+", "-", "*"):
        component_generator.add_operator(operator)
    generator = AGraphGenerator(10, 10, component_generator, random_state=rank)
    crossover = AGraphCrossover(10, 10, random_state=rank)
    mutation = AGraphMutation(component_generator, random_state=rank)
    evaluator = Evaluation(ExplicitRegression(x, y))
    ea = AgeFitnessEA(evaluator, generator, crossover, mutation, 0.4, 0.4, 100)
    archipelago = ParallelArchipelago(Island(ea, generator, 100))

    result = archipelago.evolve_until_convergence(
        max_generations=500, fitness_threshold=1.0e-4
    )
    best_individual = archipelago.get_best_individual()
    if result.success and rank == 0:
        print("Best:", best_individual.expression)


if __name__ == "__main__":
    main()
