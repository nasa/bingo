"""Evolve an expression for x**2 + 3.5 * x**3."""

import numpy as np

from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_optimizers.island import Island
from bingo.expressions import (
    AGraphCrossover,
    AGraphGenerator,
    AGraphMutation,
    ComponentGenerator,
)
from bingo.symbolic_regression import ExplicitRegression


POP_SIZE = 128
STACK_SIZE = 10


def init_island():
    np.random.seed(4)
    x = np.linspace(-10, 10, 100).reshape(-1, 1)
    y = x**2 + 3.5 * x**3

    component_generator = ComponentGenerator(x.shape[1], random_state=4)
    for operator in ("+", "-", "*"):
        component_generator.add_operator(operator)

    generator = AGraphGenerator(
        STACK_SIZE, STACK_SIZE, component_generator, random_state=4
    )
    crossover = AGraphCrossover(STACK_SIZE, STACK_SIZE, random_state=4)
    mutation = AGraphMutation(component_generator, random_state=4)
    evaluator = Evaluation(ExplicitRegression(x, y))
    ea = AgeFitnessEA(evaluator, generator, crossover, mutation, 0.4, 0.4, POP_SIZE)
    return Island(ea, generator, POP_SIZE)


def main():
    island = init_island()
    island.evolve_until_convergence(max_generations=1000, fitness_threshold=1e-6)
    print("Best individual:", island.get_best_individual())
    print("Best fitness:", island.get_best_fitness())
    print("Fitness evaluations:", island.get_fitness_evaluation_count())


if __name__ == "__main__":
    main()
