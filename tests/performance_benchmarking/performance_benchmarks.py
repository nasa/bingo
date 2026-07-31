# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import evaluation_benchmark
import evolution_benchmark


def print_stats(printer_list):
    for printer in printer_list:
        printer.print()


if __name__ == '__main__':
    TITLE = 'EXPRESSION PERFORMANCE BENCHMARKS'
    NUM_STARS_LEFT_SIDE = int((80 - len(TITLE)) / 2)
    NUM_STARS_RIGHT_SIDE = int((80 - len(TITLE) + 1) / 2)

    PRINTER_LIST = [
        evaluation_benchmark.do_benchmarking(),
        evolution_benchmark.do_benchmarking(),
    ]

    print('\n\n' + '*' * NUM_STARS_LEFT_SIDE + TITLE + 
          '*' * NUM_STARS_RIGHT_SIDE)
    print("Note: Times are milliseconds per expression or evolved individual.\n")
    print_stats(PRINTER_LIST)

