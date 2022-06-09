import pytest

from bingo.symbolic_regression.symbolic_regressor import SymbolicRegressor
from bingo.symbolic_regression.agraph.operator_definitions import OPERATOR_NAMES


# do we care about this?
def test_constructor_default():
    regr = SymbolicRegressor()
    assert regr.population_size == 500
    # TODO ...


@pytest.mark.parametrize("valid_size", [1, 1000])
def test_constructor_population_size_valid(valid_size):
    regr = SymbolicRegressor(population_size=valid_size)
    assert regr.population_size == valid_size


@pytest.mark.parametrize("invalid_size", [0, -1])
def test_constructor_population_size_invalid(invalid_size):
    with pytest.raises(ValueError) as exc_info:
        SymbolicRegressor(population_size=invalid_size)

    assert str(exc_info.value) == "Invalid population size"


@pytest.mark.parametrize("valid_size", [1, 100])
def test_constructor_stack_size_valid(valid_size):
    regr = SymbolicRegressor(stack_size=valid_size)
    assert regr.stack_size == valid_size


@pytest.mark.parametrize("invalid_size", [0, -1])
def test_constructor_stack_size_invalid(invalid_size):
    with pytest.raises(ValueError) as exc_info:
        SymbolicRegressor(stack_size=invalid_size)
    assert str(exc_info.value) == "Invalid stack size"


def valid_operators():
    operators = [None]
    for names in OPERATOR_NAMES.values():
        operators.extend(names)
    return operators


@pytest.mark.parametrize("valid_operator", valid_operators())
def test_constructor_operator_valid(valid_operator):
    if valid_operator is None:
        operators = {"+", "-", "*", "/"}
    else:
        operators = set(valid_operator)
    regr = SymbolicRegressor(operators=operators)
    assert regr.operators == operators


@pytest.mark.parametrize("invalid_operator", [{"abc"}, {"3"}, {""},
                                              "sin"  # needs to be iterable of
                                                     # operators
                                              ])
def test_constructor_operator_invalid(invalid_operator):
    with pytest.raises(ValueError) as exc_info:
        SymbolicRegressor(operators=set(invalid_operator))
    assert str(exc_info.value) == "Invalid operators"


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


def test_get_best_individual_without_fit_raises_error():
    regr = SymbolicRegressor()
    with pytest.raises(ValueError) as exc_info:
        regr.get_best_individual()
    assert str(exc_info.value) == "Best individual not set"


# TODO do rest of validation tests
# TODO validation tests on fit
# TODO check to see which param's don't give descriptive exceptions
# TODO get_best_individual normal test


