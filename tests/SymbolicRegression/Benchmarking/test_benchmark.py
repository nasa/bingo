import pytest

from bingo.SymbolicRegression.Benchmarking.Benchmark import Benchmark


@pytest.fixture
def benchmark():
    return Benchmark("koza_1", "x**4 + x**3 + x**2 + x", [-10, 10, 1000])

def test_initialize_benchmark_name(benchmark):
    assert benchmark.name == "koza_1"
    assert benchmark.objective_function == "x**4 + x**3 + x**2 + x"

def test_objective_function():
    benchmark = Benchmark("nguyen_8", "np.sqrt(x)", [0, 10, 1000], has_test_set=True)
    x = 4
    assert benchmark.equation_eval(x) == 2

def test_training_set(benchmark):
    x = 2
    assert benchmark.equation_eval(x) == 30
    assert benchmark.train_set.__len__() == 1000
    assert benchmark.test_set.__len__() == 0

def test_testing_set():
    benchmark = Benchmark("koza_1", "x**4 + x**3 + x**2 + x", [-10, 10, 1000], has_test_set=True)
    print("X: ", benchmark.test_set.x)
    print("Y: ", benchmark.test_set.y)
    assert benchmark.train_set.__len__() == 800
    assert benchmark.test_set.__len__() == 200


