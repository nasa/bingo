# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import tests.evaluation_benchmark as evaluation_benchmark
import tests.fitness_benchmark as fitness_benchmark
import tests.continous_local_opt_benchmarks as clo_benchmark

if __name__ == '__main__':
    print("------------------USING AGRAPH PROBLEM SET # 2-----------------------")
    evaluation_benchmark.do_benchmarking()
    fitness_benchmark.do_benchmarking()
    clo_benchmark.do_benchmarking(debug=True)

