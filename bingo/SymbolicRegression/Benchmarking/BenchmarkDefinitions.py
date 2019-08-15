from .Benchmark import AnalyticBenchmark


def bench_koza_1():
    name = "Koza-1"
    description = "A polynomial"
    source = "J.R. Koza. Genetic Programming: On the Programming of " + \
             "Computers by Means of Natural Selection. MIT Press 1992"
    x_dim = 1

    def eval_func(x):
        return x**4 + x**3 + x**2 + x

    train_dist = ("U", -1, 1, 20)
    test_dist = ("U", -1, 1, 20)
    extra_info = {"const_dim": 0}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_koza_2():
    name = "Koza-2"
    description = "A polynomial"
    source = "J.R. Koza. Genetic Programming: On the Programming of " + \
             "Computers by Means of Natural Selection. MIT Press 1992"
    x_dim = 1

    def eval_func(x):
        return x**5 - 2*x**3 + x

    train_dist = ("U", -1, 1, 20)
    test_dist = ("U", -1, 1, 20)
    extra_info = {"const_dim": 0}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_koza_3():
    name = "Koza-3"
    description = "A polynomial"
    source = "J.R. Koza. Genetic Programming: On the Programming of " + \
             "Computers by Means of Natural Selection. MIT Press 1992"
    x_dim = 1

    def eval_func(x):
        return x**6 - 2*x**4 + x**2

    train_dist = ("U", -1, 1, 20)
    test_dist = ("U", -1, 1, 20)
    extra_info = {"const_dim": 0}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)
