"""Benchmark a single expression-island evolutionary generation."""

import timeit

import numpy as np

from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_optimizers.island import Island
from bingo.expressions.agraph import (
    AGraphCrossover,
    AGraphGenerator,
    AGraphMutation,
    ComponentGenerator,
)
from bingo.symbolic_regression.explicit_regression import ExplicitRegression


POPULATION_SIZE = 32
STACK_SIZE = 24
TIMING_NUMBER = 3
TIMING_REPEATS = 5


class StatsPrinter:
    """Format generation timings as milliseconds per individual."""

    def __init__(self, times):
        milliseconds = np.asarray(times) * 1e3 / (TIMING_NUMBER * POPULATION_SIZE)
        self._minimum = milliseconds.min()
        self._maximum = milliseconds.max()
        self._mean = milliseconds.mean()

    def print(self):
        print("\nEVOLUTION BENCHMARK")
        print(f"{'name':<28} {'mean (ms)':>12} {'min (ms)':>12} {'max (ms)':>12}")
        print(
            f"{'expression island generation':<28} {self._mean:>12.4f} "
            f"{self._minimum:>12.4f} {self._maximum:>12.4f}"
        )


def _make_island():
    """Construct an expression-backed island without local optimization."""
    x = np.linspace(-1.0, 1.0, 64).reshape(-1, 1)
    y = x[:, 0] ** 2 + x[:, 0]
    component_generator = ComponentGenerator(
        input_x_dimension=1,
        num_initial_load_statements=2,
        terminal_probability=0.25,
        constant_probability=0.0,
        random_state=1,
    )
    component_generator.add_operator("+")
    component_generator.add_operator("*")
    generator = AGraphGenerator(
        STACK_SIZE,
        STACK_SIZE,
        component_generator,
        simplification="reduce",
        random_state=2,
    )
    crossover = AGraphCrossover(STACK_SIZE, STACK_SIZE, random_state=3)
    mutation = AGraphMutation(component_generator, random_state=4)
    evaluation = Evaluation(ExplicitRegression(x, y))
    algorithm = AgeFitnessEA(
        evaluation,
        generator,
        crossover,
        mutation,
        crossover_probability=0.4,
        mutation_probability=0.4,
        population_size=POPULATION_SIZE,
    )
    return Island(algorithm, generator, POPULATION_SIZE)


def do_benchmarking():
    """Run the expression-island evolution benchmark."""
    def evolve_generation():
        island = _make_island()
        island._execute_generational_step()

    times = timeit.repeat(
        evolve_generation,
        number=TIMING_NUMBER,
        repeat=TIMING_REPEATS,
    )
    return StatsPrinter(times)