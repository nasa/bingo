import pytest 

from bingo.SymbolicRegression.Benchmarking.BenchmarkSuite import BenchmarkSuite

@pytest.fixture
def benchmark_suite():
    return BenchmarkSuite()

def test_init_benchmark_suite(benchmark_suite):
    assert benchmark_suite.include == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert benchmark_suite.exclude == []

def test_benchmarks_dictionary(benchmark_suite):
    assert benchmark_suite.benchmarks_dict[0]['name'] == 'koza_1'
    assert benchmark_suite.benchmarks_dict[0]['function'] == 'x**4 + x**3 + x**2 + x'
    assert benchmark_suite.benchmarks_dict[7]['name'] == 'nguyen_6'
    assert benchmark_suite.benchmarks_dict[7]['function'] == 'np.sin(x) + np.sin(x + x**2)'

def test_includes_correct_benchmarks():
    bench_suite = BenchmarkSuite(include=[1, 3, 4])
    assert bench_suite.include == [1, 3, 4]
    for benchmark, i in zip(bench_suite.benchmarks, bench_suite.include):
        assert benchmark.name == bench_suite.benchmarks_dict[i]['name']
        assert benchmark.objective_function == bench_suite.benchmarks_dict[i]['function']

def test_excludes_correct_benchmarks():
    bench_suite = BenchmarkSuite(exclude=[1, 3, 4])
    assert bench_suite.include == [0, 2, 5, 6, 7, 8]
    for benchmark, i in zip(bench_suite.benchmarks, bench_suite.include):
        assert benchmark.name == bench_suite.benchmarks_dict[i]['name']
        assert benchmark.objective_function == bench_suite.benchmarks_dict[i]['function']

def test_benchmark_test_sets(benchmark_suite):
    for benchmark in benchmark_suite.benchmarks:
        if benchmark._has_test_set:
                assert benchmark.test_set.__len__() > 0
        else:
                assert benchmark.test_set.__len__() == 0

def test_benchmark_data_sets_nonempty(benchmark_suite):
    for benchmark in benchmark_suite.benchmarks:
        if benchmark._has_test_set == False:
                assert benchmark.test_set.__len__() == 0
        assert benchmark.train_set.__len__() + benchmark.test_set.__len__() == 100
 