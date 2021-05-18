# TODO remove

# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
from copy import deepcopy

import numpy as np

from bingo.local_optimizers.continuous_local_opt \
    import ContinuousLocalOptimization
from bingo.symbolic_regression import ComponentGenerator, \
    AGraphGenerator, \
    ExplicitRegression, \
    ExplicitTrainingData

POP_SIZE = 128
STACK_SIZE = 10
MUTATION_PROBABILITY = 0.4
CROSSOVER_PROBABILITY = 0.4
NUM_POINTS = 100
START = -10
STOP = 10
ERROR_TOLERANCE = 1e-6


def init_x_vals(start, stop, num_points):
    return np.linspace(start, stop, num_points).reshape([-1, 1])


def equation_eval(x):
    return x**2 + 3.5*x**3


def main():
    x = init_x_vals(START, STOP, NUM_POINTS)
    y = equation_eval(x)
    training_data = ExplicitTrainingData(x, y)

    component_generator = ComponentGenerator(x.shape[1])
    component_generator.add_operator("+")
    component_generator.add_operator("-")
    component_generator.add_operator("*")

    agraph_generator = AGraphGenerator(STACK_SIZE, component_generator)

    fitness = ExplicitRegression(training_data=training_data, metric="mse")
    nelder_mead_optimizer = ContinuousLocalOptimization(fitness)
    grad_algorithm = 'CG'
    grad_optimizer = ContinuousLocalOptimization(fitness, algorithm=grad_algorithm)

    # np.random.seed(16)
    np.random.seed(17)
    normal_agraph = agraph_generator()
    grad_agraph = deepcopy(normal_agraph)

    print("Optimization using Nelder-Mead:")
    print("Before local optimization: f(X_0) = ", normal_agraph)
    print("                          fitness = ", fitness(normal_agraph))
    _ = nelder_mead_optimizer(normal_agraph)
    print("After local optimization:  f(X_0) = ", normal_agraph)
    print("                          fitness = ", fitness(normal_agraph))

    print("\nOptimization using {}:".format(grad_algorithm))
    print("Before local optimization: f(X_0) = ", grad_agraph)
    print("                          fitness = ", fitness(grad_agraph))
    _ = grad_optimizer(grad_agraph)
    print("After local optimization:  f(X_0) = ", grad_agraph)
    print("                          fitness = ", fitness(grad_agraph))


if __name__ == '__main__':
    main()
