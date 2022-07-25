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


# TODO argument validation using fit instead of constructor
# # do we care about this?
# def test_constructor_default():
#     regr = SymbolicRegressor()
#     assert regr.population_size == 500
#     # TODO ...
#
#
# @pytest.mark.parametrize("valid_size", [1, 1000])
# def test_constructor_population_size_valid(valid_size):
#     regr = SymbolicRegressor(population_size=valid_size)
#     assert regr.population_size == valid_size
#
#
# @pytest.mark.parametrize("invalid_size", [0, -1])
# def test_constructor_population_size_invalid(invalid_size):
#     with pytest.raises(ValueError) as exc_info:
#         SymbolicRegressor(population_size=invalid_size)
#
#     assert str(exc_info.value) == "Invalid population size"
#
#
# @pytest.mark.parametrize("valid_size", [1, 100])
# def test_constructor_stack_size_valid(valid_size):
#     regr = SymbolicRegressor(stack_size=valid_size)
#     assert regr.stack_size == valid_size
#
#
# @pytest.mark.parametrize("invalid_size", [0, -1])
# def test_constructor_stack_size_invalid(invalid_size):
#     with pytest.raises(ValueError) as exc_info:
#         SymbolicRegressor(stack_size=invalid_size)
#     assert str(exc_info.value) == "Invalid stack size"
#
#
# def valid_operators():
#     operators = [None]
#     for names in OPERATOR_NAMES.values():
#         operators.extend(names)
#     return operators
#
#
# @pytest.mark.parametrize("valid_operator", valid_operators())
# def test_constructor_operator_valid(valid_operator):
#     if valid_operator is None:
#         operators = {"+", "-", "*", "/"}
#     else:
#         operators = set(valid_operator)
#     regr = SymbolicRegressor(operators=operators)
#     assert regr.operators == operators
#
#
# @pytest.mark.parametrize("invalid_operator", [{"abc"}, {"3"}, {""},
#                                               "sin"  # needs to be iterable of
#                                                      # operators
#                                               ])
# def test_constructor_operator_invalid(invalid_operator):
#     with pytest.raises(ValueError) as exc_info:
#         SymbolicRegressor(operators=set(invalid_operator))
#     assert str(exc_info.value) == "Invalid operators"


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

    from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
    mocker.patch.object(LocalOptFitnessFunction, "__call__", return_value=0.01)

    mocked_evo = mocker.patch.object(EvolutionaryOptimizer,
                                     "evolve_until_convergence")

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

