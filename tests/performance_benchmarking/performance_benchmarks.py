# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import tests.performance_benchmarking.evaluation_benchmark as evaluation_benchmark
import tests.performance_benchmarking.fitness_benchmark as fitness_benchmark
import tests.performance_benchmarking.continous_local_opt_benchmarks as clo_benchmark

PROBLEM_SET_VERSION = 2


def print_stats(printer_list):
    for printer in printer_list:
        printer.print()


if __name__ == '__main__':
    title = 'USING AGRAPH PROBLEM SET # '
    title += str(PROBLEM_SET_VERSION)
    num_stars_left_side = int((80 - len(title))/2)
    num_stars_right_side = int((80 - len(title) + 1) / 2)

    printer_list = [evaluation_benchmark.do_benchmarking()]
    printer_list += fitness_benchmark.do_benchmarking()
    printer_list += clo_benchmark.do_benchmarking(debug=False)

    print('\n\n' + '*'*num_stars_left_side + title + '*'*num_stars_right_side)
    print("Note: Times are in milliseconds per individual\n")
    print_stats(printer_list)

