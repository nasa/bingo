# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import tests.evaluation_benchmark as evaluation_benchmark

if __name__ == '__main__':
    evaluation_benchmark.do_benchmarking()
