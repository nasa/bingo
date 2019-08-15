"""
This Benchmark Suite module contains symbolic regression benchmarks
drawn from 'Genetic Programming Needs Better Benchmarks', (McDermott et al.).
"""
from . import BenchmarkDefinitions


class BenchmarkSuite:
    """Contains 9 Benchmarks (listed above) to measure
    performance of Bingo

    Parameters
    ----------
    include : list of str
        The indices of which Benchmarks to include
    exclude : list of str
        The indices of which Benchmarks to exclude. Default is none

    Attributes
    ----------
    benchmarks_dict : dictionary
        Contains the names, objective functions, and training/testing sets
        of each benchmark.
    benchmarks : list of Benchmarks
        Contains whichever Benchmarks are specified in
        `include` and/or `exclude`

    """
    def __init__(self, inclusive_terms=None, exclusive_terms=None):
        self._benchmarks = self._find_all_benchmarks()
        if inclusive_terms is not None:
            self._filter_inclusive(inclusive_terms)
        if exclusive_terms is not None:
            self._filter_exclusive(exclusive_terms)

    @staticmethod
    def _find_all_benchmarks():
        all_benchmarks = \
            [f() for name, f in BenchmarkDefinitions.__dict__.items()
             if callable(f) and name.startswith("bench")]
        return all_benchmarks

    def _filter_inclusive(self, terms):
        new_benchmark_list = [bench for bench in self._benchmarks
                              if BenchmarkSuite._has_terms(bench.name, terms)]
        self._benchmarks = new_benchmark_list

    @staticmethod
    def _has_terms(name, terms):
        for term in terms:
            if term not in name:
                return False
        return True

    def _filter_exclusive(self, terms):
        new_benchmark_list = [bench for bench in self._benchmarks
                              if BenchmarkSuite._hasnt_terms(bench.name,
                                                               terms)]
        self._benchmarks = new_benchmark_list

    @staticmethod
    def _hasnt_terms(name, terms):
        for term in terms:
            if term in name:
                return False
        return True

    def __len__(self):
        return len(self._benchmarks)

    def __getitem__(self, i):
        return self._benchmarks[i]

    def __iter__(self):
        return iter(self._benchmarks)
