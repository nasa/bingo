from functools import partial
import numpy as np
import pytest

from bingo.symbolic_regression.symbolic_regressor import SymbolicRegressor
from bingo.symbolic_regression.agraph.operator_definitions import OPERATOR_NAMES
from bingo.evolutionary_optimizers.evolutionary_optimizer import EvolutionaryOptimizer
from bingo.evolutionary_optimizers.island import Island
from bingo.evolutionary_optimizers.fitness_predictor_island import FitnessPredictorIsland
from bingo.symbolic_regression.agraph.agraph import AGraph
from bingo.symbolic_regression.agraph.generator import AGraphGenerator


@pytest.fixture
def basic_data():
    X = np.linspace(-10, 10, num=100).reshape((-1, 1))
    y = np.sin(X) + 3.0
    return X, y


# do we care about this?
def test_constructor_default():
    regr = SymbolicRegressor()
    assert regr.population_size == 500
    # TODO ...


@pytest.mark.parametrize("invalid_size", [0, -1])
def test_population_size_invalid(invalid_size, basic_data):
    regr = SymbolicRegressor(population_size=invalid_size)
    X, y = basic_data

    with pytest.raises(ValueError):
        regr.fit(X, y)


@pytest.mark.parametrize("invalid_size", [0, -1])
def test_stack_size_invalid(invalid_size, basic_data):
    regr = SymbolicRegressor(stack_size=invalid_size)
    X, y = basic_data

    with pytest.raises(ValueError):
        regr.fit(X, y)


def test_operator_none_default_set():
    regr = SymbolicRegressor(operators=None)
    assert regr.operators == {"+", "-", "*", "/"}


@pytest.mark.parametrize("invalid_operators", [{"abc"}, {"3"}, {""},
                                               "sin"  # needs to be list-like
                                               ])     # of operators
def test_operator_invalid(invalid_operators, basic_data):
    regr = SymbolicRegressor(operators=invalid_operators)
    X, y = basic_data

    with pytest.raises(ValueError):
        regr.fit(X, y)


@pytest.mark.parametrize("invalid_value, expected_err",
                         [(-1, ValueError), (1.1, ValueError),
                          ("a", TypeError), (True, ValueError)])
def test_crossover_prob_invalid(invalid_value, basic_data, expected_err):
    regr = SymbolicRegressor(crossover_prob=invalid_value)
    X, y = basic_data

    with pytest.raises(expected_err):
        regr.fit(X, y)


@pytest.mark.parametrize("invalid_value, expected_err",
                         [(-1, ValueError), (1.1, ValueError),
                          ("a", TypeError), (True, ValueError)])
def test_mutation_prob_invalid(invalid_value, basic_data, expected_err):
    regr = SymbolicRegressor(mutation_prob=invalid_value)
    X, y = basic_data

    with pytest.raises(expected_err):
        regr.fit(X, y)


@pytest.mark.parametrize("invalid_metric", ["aaa", 1, False])
def test_metric_invalid(invalid_metric, basic_data):
    regr = SymbolicRegressor(metric=invalid_metric)
    X, y = basic_data

    with pytest.raises(ValueError):
        regr.fit(X, y)


@pytest.mark.parametrize("invalid_alg", ["not_an_alg", 1, False])
def test_clo_alg_invalid(invalid_alg, basic_data):
    regr = SymbolicRegressor(clo_alg=invalid_alg)
    X, y = basic_data

    with pytest.raises(KeyError):
        regr.fit(X, y)


@pytest.mark.timeout(3)
@pytest.mark.parametrize("invalid_n_gens", [0, -1])
def test_n_gens_invalid(invalid_n_gens, basic_data):
    regr = SymbolicRegressor(generations=invalid_n_gens)
    X, y = basic_data

    with pytest.raises(ValueError):
        regr.fit(X, y)

# TODO make sure constructor args get to respective objects
#      e.g. if metric = "mse", make sure fitness gets "mse"

def constructor_params():
    return ["population_size", "stack_size", "operators",
            "use_simplification", "crossover_prob",
            "mutation_prob", "metric", "parallel",
            "clo_alg", "generations",
            "fitness_threshold", "max_time", "max_evals",
            "evolutionary_algorithm", "clo_threshold",
            "scale_max_evals", "random_state"]


# TODO need set_param tests? should be same as sklearn
@pytest.mark.parametrize("param", constructor_params())
def test_can_set_all_params(mocker, param):
    mocked_value = mocker.Mock(spec=object)  # spec=object so get_params()
    # doesn't try to treat the mock like a dictionary

    regr = SymbolicRegressor()
    output = regr.set_params(**{param: mocked_value})

    assert regr.get_params()[param] == mocked_value
    assert output is regr


def test_setting_param_doesnt_reset_others(mocker):
    regr = SymbolicRegressor()
    mocked_ea = mocker.Mock(spec=object)
    regr.set_params(evolutionary_algorithm=mocked_ea)
    regr.set_params(population_size=1)

    # setting population_size shouldn't reset
    # evolutionary_algorithm back to default
    assert regr.evolutionary_algorithm == mocked_ea
    assert regr.get_params()["evolutionary_algorithm"] == mocked_ea

    assert regr.population_size == 1
    assert regr.get_params()["population_size"] == 1


def test_setting_invalid_param_raises_value_error():
    regr = SymbolicRegressor()
    with pytest.raises(ValueError):
        regr.set_params(**{"not_a_valid_param": False})


def test_get_best_individual(mocker):
    regr = SymbolicRegressor()
    best_ind = mocker.Mock()
    regr.best_ind = best_ind

    assert regr.get_best_individual() == best_ind


def test_get_best_individual_without_fit_raises_error():
    regr = SymbolicRegressor()
    with pytest.raises(ValueError) as exc_info:
        regr.get_best_individual()
    assert str(exc_info.value) == "Best individual not set"


@pytest.mark.parametrize("data_size", [10, int(1e4), int(1e6)])
def test_fit_calls_evo_opt(mocker, data_size):
    # testing different data sizes in case evo opt changes per data size
    X = np.linspace(-10, 10, data_size).reshape((data_size, 1))
    y = np.linspace(10, -10, data_size).reshape((data_size, 1))

    from bingo.local_optimizers.continuous_local_opt import ContinuousLocalOptimization
    mocker.patch.object(ContinuousLocalOptimization, "__call__", return_value=0.01)

    mocked_evo = mocker.patch.object(EvolutionaryOptimizer, "evolve_until_convergence")

    regr = SymbolicRegressor()
    regr.fit(X, y)

    # can't use assert_called_with bc there might be extra args
    assert mocked_evo.call_args.kwargs["max_generations"] == regr.generations
    assert mocked_evo.call_args.kwargs["fitness_threshold"] == \
           regr.fitness_threshold
    assert mocked_evo.call_args.kwargs["max_fitness_evaluations"] == \
           regr.max_evals
    assert mocked_evo.call_args.kwargs["max_time"] == regr.max_time


def get_linear_agraph(slope, intercept):
    agraph = AGraph()
    agraph.command_array = np.array([[0, 0, 0],  # X_0
                                     [1, 0, 0],  # C_0
                                     [1, 1, 1],  # C_1
                                     [4, 0, 1],  # X_0 * C_0
                                     [2, 2, 3]  # X_0 * C_0 + C_1
                                     ], dtype=int)
    agraph.set_local_optimization_params([slope, intercept])
    return agraph


def test_fit_finds_sol_in_initial_pop(mocker):
    X = np.linspace(-10, 10).reshape((-1, 1))
    y = -2.3 * X + 3.5

    sol_agraph = get_linear_agraph(-2.3, 3.5)
    mocker.patch.object(AGraphGenerator, "__call__", return_value=sol_agraph)

    regr = SymbolicRegressor()
    regr.fit(X, y)

    assert regr.get_best_individual() == sol_agraph
    assert regr.archipelago.generational_age == 0


def test_fit_sets_random_seed(mocker):
    X = np.linspace(-10, 10).reshape((-1, 1))
    y = -2.3 * X + 3.5
    np_random_seed = mocker.patch("bingo.symbolic_regression.symbolic_regressor.np.random.seed")
    random_seed = mocker.patch("bingo.symbolic_regression.symbolic_regressor.random.seed")
    random_state = mocker.Mock()

    regr = SymbolicRegressor(random_state=random_state)
    regr.fit(X, y)

    np_random_seed.assert_called_with(random_state)
    random_seed.assert_called_with(random_state)


"""
def test_fit_finds_sol_normal(mocker):
    X = np.linspace(-10, 10).reshape((-1, 1))
    y = 5.0 * X

    regr = SymbolicRegressor()
    regr.fit(X, y)

    assert regr.get_best_individual().fitness <= regr.fitness_threshold
"""

# TODO do rest of validation tests
# TODO validation tests on fit
# TODO check to see which param's don't give descriptive exceptions
# TODO get_best_individual normal test
# TODO test fit returns self
# TODO test diverse population (works and doesn't error out when not possible)?

