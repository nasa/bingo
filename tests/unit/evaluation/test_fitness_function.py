# Ignoring some linting rules in tests
# pylint: disable=missing-docstring

import pytest
import numpy as np
from bingo.evaluation.fitness_function \
    import FitnessFunction, VectorBasedFunction


def test_fitness_function_cant_be_instanced():
    with pytest.raises(TypeError):
        _ = FitnessFunction()


def test_fitness_function_has_eval_count_and_data(mocker):
    mocker.patch.object(FitnessFunction, "__abstractmethods__",
                        new_callable=set)
    training_data = mocker.Mock()
    fit_func = FitnessFunction(training_data)

    assert fit_func.eval_count == 0
    assert fit_func.training_data is training_data


@pytest.mark.parametrize("metric, expected_fit", [("mae", 1.2),
                                                  ("mse", 2.0),
                                                  ("rmse", np.sqrt(2.0))])
def test_vector_based_function_metrics(mocker, metric, expected_fit):
    mocker.patch.object(VectorBasedFunction, "__abstractmethods__",
                        new_callable=set)
    mocker.patch.object(VectorBasedFunction, "evaluate_fitness_vector",
                        return_value=[-2, -1, 0, 1, 2])
    fit_func = VectorBasedFunction(metric=metric)
    dummy_indv = None
    
    assert fit_func(dummy_indv) == pytest.approx(expected_fit)
    fit_func.evaluate_fitness_vector.assert_called_once_with(dummy_indv)


@pytest.mark.parametrize("metric, expected_fit_grad", [("mae", [-1/3, 2/3]),
                                                       ("mse", [-4/3, 8/3]),
                                                       ("rmse", [np.sqrt(3/8) * -2/3, np.sqrt(3/8) * 4/3])])
def test_vector_based_function_gradient_metrics(mocker, metric, expected_fit_grad):
    mocker.patch.object(VectorBasedFunction, "__abstractmethods__",
                        new_callable=set)
    mocker.patch.object(VectorBasedFunction, "evaluate_fitness_vector",
                        return_value=np.array([-2, 0, 2]))
    mocker.patch.object(VectorBasedFunction, "evaluate_fitness_derivative",
                        return_value=np.array([[0.5, 1, -0.5], [1, 2, 3]]).transpose())
    fit_func = VectorBasedFunction(metric=metric)
    dummy_indv = None

    np.testing.assert_array_equal(fit_func.get_gradient(dummy_indv), expected_fit_grad)
    fit_func.evaluate_fitness_derivative.assert_called_once_with(dummy_indv)


def test_vector_based_function_invalid_metric(mocker):
    mocker.patch.object(VectorBasedFunction, "__abstractmethods__",
                        new_callable=set)
    mocker.patch.object(VectorBasedFunction, "evaluate_fitness_vector",
                        return_value=[-2, -1, 0, 1, 2])
    with pytest.raises(KeyError):
        _ = VectorBasedFunction(metric="invalid metric")


@pytest.mark.parametrize("metric", ["mae", "mse", "rmse"])
def test_vector_based_function_with_nan(mocker, metric):
    mocker.patch.object(VectorBasedFunction, "__abstractmethods__",
                        new_callable=set)
    mocker.patch.object(VectorBasedFunction, "evaluate_fitness_vector",
                        return_value=[np.nan, -1, 0, 1, 2])
    fit_func = VectorBasedFunction(metric=metric)
    dummy_indv = None

    assert np.isnan(fit_func(dummy_indv))
