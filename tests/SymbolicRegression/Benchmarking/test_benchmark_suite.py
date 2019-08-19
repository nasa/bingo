# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest

from bingo.SymbolicRegression.Benchmarking.BenchmarkSuite import BenchmarkSuite


def test_benchmark_finds_all_benchmark_definitions():
    suite = BenchmarkSuite()
    assert len(suite) == 3


@pytest.mark.parametrize("inclusive_terms, expected_names",
                         [(["Koza"], ["Koza-1", "Koza-2", "Koza-3"]),
                          (["Koza", "1"], ["Koza-1"])])
def test_benchmark_inclusive(inclusive_terms, expected_names):
    suite = BenchmarkSuite(inclusive_terms=inclusive_terms)
    names = [s.name for s in suite]
    assert set(names) == set(expected_names)


def test_benchmark_exclusive():
    suite = BenchmarkSuite(inclusive_terms=["Koza"],
                           exclusive_terms=["2", "3"])
    names = [s.name for s in suite]
    assert set(names) == {"Koza-1"}


def test_benchmark_get_item():
    suite = BenchmarkSuite(inclusive_terms=["Koza-1"])
    assert suite[0].name == "Koza-1"
