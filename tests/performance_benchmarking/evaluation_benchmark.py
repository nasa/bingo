"""Benchmark expression prediction and input-gradient evaluation."""

import timeit

import numpy as np

import bingo.expressions.agraph as agraph


NUM_EXPRESSIONS = 32
NUM_SAMPLES = 128
TIMING_NUMBER = 25
TIMING_REPEATS = 5
BATCH_TIMING_NUMBER = 5
BATCH_SIZES = (1, 8, 32, 128)


class StatsPrinter:
    """Format timing samples as milliseconds per expression."""

    def __init__(self):
        self._rows = []
        self._speedups = []

    def add_stats(self, name, times, timing_number=TIMING_NUMBER):
        milliseconds = np.asarray(times) * 1e3 / (timing_number * NUM_EXPRESSIONS)
        self._rows.append(
            (name, milliseconds.min(), milliseconds.max(), milliseconds.mean())
        )

    def add_speedup(self, name, repeated_times, batched_times):
        speedups = np.asarray(repeated_times) / np.asarray(batched_times)
        self._speedups.append((name, speedups.mean()))

    def print(self):
        print("\nEVALUATION BENCHMARKS")
        print(f"{'name':<28} {'mean (ms)':>12} {'min (ms)':>12} {'max (ms)':>12}")
        for name, minimum, maximum, mean in self._rows:
            print(f"{name:<28} {mean:>12.4f} {minimum:>12.4f} {maximum:>12.4f}")
        if self._speedups:
            print("\nBATCHING SPEEDUPS")
            for name, speedup in self._speedups:
                print(f"{name:<28} {speedup:>12.2f}x")


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


def _batch_expressions():
    """Build constant-bearing expressions with varied operator costs."""
    equations = (
        "x0 * 1.0 + 2.0",
        "sin(x0 * 1.0) + 2.0",
        "exp(x0 * 1.0) / (2.0 + x1 * x1)",
        "sqrt(x0 * x0 + 1.0) + 2.0",
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


def _benchmark_constant_batch(backend, x, batch_size):
    """Compare one batched call with repeated scalar predictions."""
    agraph.set_backend(backend)
    expressions = _batch_expressions()
    constant_batches = []
    for expression in expressions:
        n_constants = len(expression.constants)
        values = np.linspace(0.5, 1.5, max(1, n_constants * batch_size))
        constant_batches.append(values[: n_constants * batch_size].reshape(
            n_constants, batch_size
        ))

    def batched():
        for expression, constants in zip(expressions, constant_batches):
            expression.predict(x, constants=constants)

    def repeated():
        for expression, constants in zip(expressions, constant_batches):
            for column in range(batch_size):
                expression.predict(x, constants=constants[:, column])

    return (
        timeit.repeat(
            batched, number=BATCH_TIMING_NUMBER, repeat=TIMING_REPEATS
        ),
        timeit.repeat(
            repeated, number=BATCH_TIMING_NUMBER, repeat=TIMING_REPEATS
        ),
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
        for batch_size in BATCH_SIZES:
            batched_times, repeated_times = _benchmark_constant_batch(
                backend, x, batch_size
            )
            label = f"{backend}: constants B={batch_size}"
            printer.add_stats(
                f"{label} batched", batched_times, BATCH_TIMING_NUMBER
            )
            printer.add_stats(
                f"{label} repeated", repeated_times, BATCH_TIMING_NUMBER
            )
            printer.add_speedup(label, repeated_times, batched_times)
    agraph.set_backend("auto")
    return printer
