"""Evolve a symbolic-regression expression on a serial archipelago."""

import numpy as np

from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_optimizers.island import Island
from bingo.evolutionary_optimizers.serial_archipelago import SerialArchipelago
from bingo.expressions import (
    AGraphCrossover,
    AGraphGenerator,
    AGraphMutation,
    ComponentGenerator,
)
from bingo.symbolic_regression import ExplicitRegression


def main():
    x = np.linspace(-10, 10, 100).reshape(-1, 1)
    y = x**2 + 3.5 * x**3
    component_generator = ComponentGenerator(x.shape[1], random_state=7)
    for operator in ("+", "-", "*"):
        component_generator.add_operator(operator)

    generator = AGraphGenerator(10, 10, component_generator, random_state=7)
    crossover = AGraphCrossover(10, 10, random_state=7)
    mutation = AGraphMutation(component_generator, random_state=7)
    evaluator = Evaluation(ExplicitRegression(x, y))
    ea = AgeFitnessEA(evaluator, generator, crossover, mutation, 0.4, 0.4, 100)
    archipelago = SerialArchipelago(Island(ea, generator, 100))

    result = archipelago.evolve_until_convergence(
        max_generations=500, fitness_threshold=1.0e-4
    )
    if result.success:
        print(archipelago.get_best_individual().expression)
    else:
        print("Failed.")
    print(result.ea_diagnostics)


if __name__ == "__main__":
    main()
