# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.evaluation.fitness_function \
    import FitnessFunction, VectorBasedFunction
from bingo.evaluation.gradient_mixin import GradientMixin, VectorGradientMixin
from bingo.local_optimizers.continuous_local_opt \
    import ContinuousLocalOptimization, ChromosomeInterface, \
           ROOT_SET, MINIMIZE_SET


class DummyLocalOptIndividual(ChromosomeInterface):
    def needs_local_optimization(self):
        return True

    def get_number_local_optimization_params(self):
        return 1

    def set_local_optimization_params(self, params):
        try:
            self.param = params[0]
        except IndexError:  # for issue with powell
            self.param = params
            
    def get_local_optimization_params(self):
        return (self.param, )


@pytest.mark.parametrize("fit_func_type, raises_error",
                         [(FitnessFunction, True),
                          (VectorBasedFunction, False)])
def test_valid_fitness_function(mocker, fit_func_type, raises_error):
    mocked_fitness_function = mocker.create_autospec(fit_func_type)
    if raises_error:
        with pytest.raises(TypeError):
            _ = ContinuousLocalOptimization(mocked_fitness_function,
                                            algorithm='lm')
    else:
        _ = ContinuousLocalOptimization(mocked_fitness_function,
                                        algorithm='lm')


def test_invalid_algorithm(mocker):
    mocked_fitness_function = mocker.Mock()
    with pytest.raises(KeyError):
        ContinuousLocalOptimization(mocked_fitness_function,
                                    algorithm='Dwayne - The Rock - Johnson')


def test_get_eval_count_pass_through(mocker):
    fitness_function = mocker.Mock()
    fitness_function.eval_count = 123
    local_opt_fitness_function = \
        ContinuousLocalOptimization(fitness_function, "Powell")
    assert local_opt_fitness_function.eval_count == 123


def test_set_eval_count_pass_through(mocker):
    fitness_function = mocker.Mock()
    local_opt_fitness_function = \
        ContinuousLocalOptimization(fitness_function, "Powell")
    local_opt_fitness_function.eval_count = 123
    assert fitness_function.eval_count == 123


def test_get_training_data_pass_through(mocker):
    fitness_function = mocker.Mock()
    fitness_function.training_data = 123
    local_opt_fitness_function = \
        ContinuousLocalOptimization(fitness_function, "Powell")
    assert local_opt_fitness_function.training_data == 123


def test_set_training_data_pass_through(mocker):
    fitness_function = mocker.Mock()
    local_opt_fitness_function = \
        ContinuousLocalOptimization(fitness_function, "Powell")
    local_opt_fitness_function.training_data = 123
    assert fitness_function.training_data == 123


@pytest.mark.parametrize("algorithm", MINIMIZE_SET)
def test_optimize_params_minimize_without_gradient(mocker, algorithm):
    fitness_function = mocker.create_autospec(FitnessFunction)
    fitness_function.side_effect = lambda individual: 1 + individual.param**2

    local_opt_fitness_function = ContinuousLocalOptimization(
        fitness_function, algorithm)

    individual = DummyLocalOptIndividual()
    opt_indv_fitness = local_opt_fitness_function(individual)
    assert opt_indv_fitness == pytest.approx(1, rel=0.05)


class GradientFitnessFunction(GradientMixin, FitnessFunction):
    def __call__(self, individual):
        pass

    def get_fitness_and_gradient(self, individual):
        pass


@pytest.mark.parametrize("algorithm", MINIMIZE_SET)
def test_optimize_params_minimize_with_gradient(mocker, algorithm):
    fitness_function = mocker.create_autospec(GradientFitnessFunction)
    fitness_function.side_effect = lambda individual: 1 + individual.param**2
    fitness_function.get_fitness_and_gradient = lambda individual: (1 + individual.param**2, 2 * individual.param)

    local_opt_fitness_function = ContinuousLocalOptimization(
        fitness_function, algorithm)

    individual = DummyLocalOptIndividual()
    opt_indv_fitness = local_opt_fitness_function(individual)
    assert opt_indv_fitness == pytest.approx(1, rel=0.05)


class JacobianVectorFitnessFunction(VectorGradientMixin, VectorBasedFunction):
    def evaluate_fitness_vector(self, individual):
        pass

    def get_fitness_vector_and_jacobian(self, individual):
        pass


@pytest.mark.parametrize("algorithm", ROOT_SET)
def test_optimize_params_root_without_jacobian(mocker, algorithm):
    fitness_function = mocker.create_autospec(VectorBasedFunction)
    fitness_function.evaluate_fitness_vector = lambda x: 1 + np.abs([x.param])

    local_opt_fitness_function = ContinuousLocalOptimization(
        fitness_function, algorithm)

    individual = DummyLocalOptIndividual()
    _ = local_opt_fitness_function(individual)
    opt_indv_fitness = fitness_function.evaluate_fitness_vector(individual)
    assert opt_indv_fitness[0] == pytest.approx(1, rel=0.05)


@pytest.mark.parametrize("algorithm", ROOT_SET)
def test_optimize_params_root_with_jacobian(mocker, algorithm):
    fitness_function = mocker.create_autospec(JacobianVectorFitnessFunction)
    fitness_function.evaluate_fitness_vector = lambda x: 1 + np.abs([x.param])
    fitness_function.get_fitness_vector_and_jacobian = lambda x: (1 + np.abs([x.param]), np.sign([x.param]))

    local_opt_fitness_function = ContinuousLocalOptimization(
        fitness_function, algorithm)

    individual = DummyLocalOptIndividual()
    _ = local_opt_fitness_function(individual)
    opt_indv_fitness = fitness_function.evaluate_fitness_vector(individual)
    assert opt_indv_fitness[0] == pytest.approx(1, rel=0.05)
