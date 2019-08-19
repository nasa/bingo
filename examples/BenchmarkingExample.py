import numpy as np

from bingo.SymbolicRegression.Benchmarking.BenchmarkSuite import BenchmarkSuite
from bingo.SymbolicRegression.Benchmarking.BenchmarkTest import BenchmarkTest
from bingo.SymbolicRegression.AGraph.ComponentGenerator import ComponentGenerator
from bingo.SymbolicRegression.AGraph.AGraphGenerator import AGraphGenerator
from bingo.SymbolicRegression.AGraph.AGraphCrossover import AGraphCrossover
from bingo.SymbolicRegression.AGraph.AGraphMutation import AGraphMutation
from bingo.SymbolicRegression.ExplicitRegression import ExplicitRegression
from bingo.Base.ContinuousLocalOptimization import ContinuousLocalOptimization
from bingo.Base.Evaluation import Evaluation
from bingo.Base.AgeFitnessEA import AgeFitnessEA
from bingo.Base.DeterministicCrowdingEA import DeterministicCrowdingEA
from bingo.Base.Island import Island


def training_function(training_data, ea_choice):
    component_generator = \
        ComponentGenerator(input_x_dimension=training_data.x.shape[1])
    component_generator.add_operator("+")
    component_generator.add_operator("-")
    component_generator.add_operator("*")

    agraph_generator = AGraphGenerator(agraph_size=32,
                                       component_generator=component_generator)

    crossover = AGraphCrossover(component_generator)
    mutation = AGraphMutation(component_generator)

    fitness = ExplicitRegression(training_data=training_data)
    local_opt_fitness = ContinuousLocalOptimization(fitness, algorithm='lm')
    evaluator = Evaluation(local_opt_fitness)

    POPULATION_SIZE = 32
    MUTATION_PROBABILITY = 0.1
    CROSSOVER_PROBABILITY = 0.7

    if ea_choice == "age_fitness":
        ea = AgeFitnessEA(evaluator, agraph_generator, crossover, mutation,
                          MUTATION_PROBABILITY, CROSSOVER_PROBABILITY,
                          POPULATION_SIZE)
    else:
        ea = DeterministicCrowdingEA(evaluator, crossover, mutation,
                                     MUTATION_PROBABILITY,
                                     CROSSOVER_PROBABILITY)

    island = Island(ea, agraph_generator, POPULATION_SIZE)
    opt_result = island.evolve_until_convergence(max_generations=MAX_GENERATIONS,
                                                 fitness_threshold=1e-6)

    return island.get_best_individual(), opt_result


def scoring_function(equation, scoring_data, opt_result):
    mae_function = ExplicitRegression(training_data=scoring_data)
    mae = mae_function(equation)
    return mae, opt_result.success


def parse_results(train_results, test_results):
    train_array = np.array(train_results)
    test_array = np.array(test_results)
    mae_train = np.mean(train_array, axis=1)[:, 0]
    mae_test = np.mean(test_array, axis=1)[:, 0]
    success_rate = np.mean(train_array, axis=1)[:, 1]
    return mae_train, mae_test, success_rate


def print_results(title, af_res, dc_res, bench_names):
    print("\n----------::", title, "::-------------")
    print("              {:^10}{:^10}{:^10}".format(bench_names[0],
                                                    bench_names[1],
                                                    bench_names[2]))
    print("age-fitness   {:^10.2e}{:^10.2e}{:^10.2e}".format(af_res[0],
                                                             af_res[1],
                                                             af_res[2]))
    print("det. crowding {:^10.2e}{:^10.2e}{:^10.2e}".format(dc_res[0],
                                                             dc_res[1],
                                                             dc_res[2]))


def run_benchmark_comparison():
    suite = BenchmarkSuite(inclusive_terms=["Koza"])
    age_fitness_strategy = \
        BenchmarkTest(lambda x: training_function(x, "age_fitness"),
                      scoring_function)
    deterministic_crowding_strategy = \
        BenchmarkTest(lambda x: training_function(x, "deterministic_crowding"),
                      scoring_function)

    train_scores_af, test_scores_af = \
        suite.run_benchmark_test(age_fitness_strategy, repeats=NUM_REPEATS)
    train_scores_dc, test_scores_dc = \
        suite.run_benchmark_test(deterministic_crowding_strategy,
                                 repeats=NUM_REPEATS)

    mae_train_af, mae_test_af, success_rate_af = \
        parse_results(train_scores_af, test_scores_af)
    mae_train_dc, mae_test_dc, success_rate_dc = \
        parse_results(train_scores_dc, test_scores_dc)
    benchmark_names = [benchmark.name for benchmark in suite]

    print_results("MAE (Train)", mae_train_af, mae_train_dc, benchmark_names)
    print_results("MAE (Test)", mae_test_af, mae_test_dc, benchmark_names)
    print_results("Success Rate", success_rate_af, success_rate_dc,
                  benchmark_names)


if __name__ == "__main__":
    MAX_GENERATIONS = 1000
    NUM_REPEATS = 10
    run_benchmark_comparison()
