# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np
import pytest

from bingo.symbolic_regression.equation_regressor import (
    EquationRegressor,
    INF_REPLACEMENT,
)
from bingo.symbolic_regression.agraph.agraph import AGraph
from bingo.symbolic_regression.agraph.generator import AGraphGenerator


def get_equ_reg_import(class_or_method_name):
    return "bingo.symbolic_regression.equation_regressor." + class_or_method_name


def patch_import(mocker, class_name):
    return mocker.patch(get_equ_reg_import(class_name))


class OptIndividual:
    def __init__(self):
        self._needs_opt = True
        self.constants = [float("inf")]
        self.fitness = float("inf")

    def needs_local_optimization(self):
        return self._needs_opt

    def get_number_local_optimization_params(self):
        return 1

    def set_local_optimization_params(self, params):
        self._needs_opt = False
        self.constants = params


def get_dummy_data(mocker):
    len = 250
    X = mocker.MagicMock()
    X.__len__.return_value = len
    X.shape = (len, mocker.Mock())

    y = mocker.MagicMock()
    y.__len__.return_value = len
    y.shape = (len, mocker.Mock())

    return X, y


@pytest.mark.parametrize("fit_retries", [4, 5])
def test_refit_best_individual(mocker, fit_retries):
    mocked_optimizer = patch_import(mocker, "ScipyOptimizer")
    mocked_regression = patch_import(mocker, "ExplicitRegression")
    mocked_training_data = patch_import(mocker, "ExplicitTrainingData")
    mocked_fit_func = patch_import(mocker, "LocalOptFitnessFunction")

    constant_iter = iter([5, 3, 4, 1, 2, 6])

    def mocked_clo(indv):
        if indv.needs_local_optimization():
            next_constant = next(constant_iter)
            indv.set_local_optimization_params([next_constant])
            indv.fitness = next_constant
        return indv.fitness

    clo = mocker.Mock(side_effect=mocked_clo)
    mocked_fit_func.return_value = clo

    equ = OptIndividual()

    X, y = get_dummy_data(mocker)

    regr = EquationRegressor(equ, fit_retries=fit_retries)
    regr.fit(X, y)

    assert clo.call_count == fit_retries + 1
    assert equ.fitness == 1
    assert equ.constants == (1,)

    mocked_optimizer.assert_called_once()
    mocked_training_data.assert_called_once_with(X, y)
    mocked_regression.assert_called_once()


def test_predict_bad_output(mocker):
    equ = mocker.Mock()
    equ.evaluate_equation_at = mocker.Mock(
        side_effect=lambda x: [0, -np.inf, np.inf, np.nan, 1, 2, 3]
    )
    regr = EquationRegressor(equ)
    expected_output = [0, -INF_REPLACEMENT, INF_REPLACEMENT, 0, 1, 2, 3]

    np.testing.assert_array_equal(regr.predict(mocker.Mock()), expected_output)
