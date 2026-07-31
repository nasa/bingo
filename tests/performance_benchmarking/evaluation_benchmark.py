"""Benchmark expression prediction and input-gradient evaluation."""

import timeit

import numpy as np

import bingo.expressions.agraph as agraph


NUM_EXPRESSIONS = 32
NUM_SAMPLES = 128
TIMING_NUMBER = 25
TIMING_REPEATS = 5


class StatsPrinter:
    """Format timing samples as milliseconds per expression."""

    def __init__(self):
        self._rows = []

    def add_stats(self, name, times):
        milliseconds = np.asarray(times) * 1e3 / (TIMING_NUMBER * NUM_EXPRESSIONS)
        self._rows.append(
            (name, milliseconds.min(), milliseconds.max(), milliseconds.mean())
        )

    def print(self):
        print("\nEVALUATION BENCHMARKS")
        print(f"{'name':<28} {'mean (ms)':>12} {'min (ms)':>12} {'max (ms)':>12}")
        for name, minimum, maximum, mean in self._rows:
            print(f"{name:<28} {mean:>12.4f} {minimum:>12.4f} {maximum:>12.4f}")


def _expressions():
    """Build a repeatable set of expressions using the active backend."""
    equations = (
        "x0 + x1",
        "sin(x0) + x1 * x1",
        "x0 * x0 + 3.0 * x1",
        "exp(x0) / (1.0 + x1 * x1)",
    )
    return [agraph.AGraphExpression(equations[index % len(equations)])
            for index in range(NUM_EXPRESSIONS)]


def _benchmark_backend(backend, x):
    """Return prediction and gradient timing samples for one backend."""
    agraph.set_backend(backend)
    expressions = _expressions()

    def predict():
        for expression in expressions:
            expression.predict(x)

    def gradient():
        for expression in expressions:
            expression.gradient(x)

    return (
        timeit.repeat(predict, number=TIMING_NUMBER, repeat=TIMING_REPEATS),
        timeit.repeat(gradient, number=TIMING_NUMBER, repeat=TIMING_REPEATS),
    )


def do_benchmarking():
    """Run the available expression evaluation backends."""
    x = np.linspace(-1.0, 1.0, NUM_SAMPLES * 2).reshape(NUM_SAMPLES, 2)
    printer = StatsPrinter()
    for backend in ("python", "cpp"):
        try:
            prediction_times, gradient_times = _benchmark_backend(backend, x)
        except ImportError:
            continue
        printer.add_stats(f"{backend}: predict", prediction_times)
        printer.add_stats(f"{backend}: input gradient", gradient_times)
    agraph.set_backend("auto")
    return printer