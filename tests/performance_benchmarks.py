import timeit
import numpy as np


def test():
    """Stupid test function"""
    L = []
    for i in range(100):
        L.append(i)


class StatsPrinter:
    def __init__(self):
        self._header_format_string = \
            "{:<25}   {:>10} +- {:<10}   {:^10}   {:^10}"
        self._format_string = \
            "{:<25}   {:>10.4f} +- {:<10.4f}   {:^10.4f}   {:^10.4f}"
        self._output = ["-"*23+":::: PERFORMANCE BENCHMARKS ::::" + "-"*23,
                        self._header_format_string.format("NAME", "MEAN",
                                                          "STD", "MIN", "MAX"),
                        "-"*78]

    def add_stats(self, name, times):
        std_time = np.std(times)
        mean_time = np.mean(times)
        max_time = np.max(times)
        min_time = np.min(times)

        self._output.append(self._format_string.format(name, mean_time,
                                                       std_time, min_time,
                                                       max_time))

    def print(self):
        for line in self._output:
            print(line)


if __name__ == '__main__':

    printer = StatsPrinter()
    printer.add_stats("test_function",
                      timeit.repeat(test, number=100000, repeat=5))
    printer.print()
