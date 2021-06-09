import pytest
import numpy as np

from bingo.evaluation.fitness_function import VectorBasedFunction
from bingo.evaluation.gradient_mixin import GradientMixin, VectorGradientMixin


def test_gradient_mixin_cant_be_instanced():
    with pytest.raises(TypeError):
        _ = GradientMixin()


def test_vector_gradient_mixin_cant_be_instanced():
    with pytest.raises(TypeError):
        _ = VectorGradientMixin()


def test_vector_gradient_mixin_cant_be_instanced_without_vector_based_function():
    class MixinWithoutVectorBasedFunction(VectorGradientMixin):
        def get_jacobian(self, individual):
            pass

    with pytest.raises(TypeError):
        _ = MixinWithoutVectorBasedFunction()


class VectorFitnessFunction(VectorGradientMixin, VectorBasedFunction):
    pass


def test_vector_gradient_mixin_invalid_metric(mocker):
    # patch __abstractmethods__ so we can instantiate VectorFitnessFunction for this test
    mocker.patch.object(VectorFitnessFunction, "__abstractmethods__", new_callable=set)

    with pytest.raises(KeyError):
        _ = VectorFitnessFunction(metric="invalid metric")


def test_gradient_mixin_get_gradient_raises_not_implemented_error(mocker):
    mocker.patch.object(GradientMixin, "__abstractmethods__", new_callable=set)

    gradient_mixin = GradientMixin()
    with pytest.raises(NotImplementedError):
        gradient_mixin.get_gradient(None)


def test_vector_gradient_mixin_get_jacobian_raises_not_implemented_error(mocker):
    mocker.patch.object(VectorFitnessFunction, "__abstractmethods__", new_callable=set)

    vector_fitness_function = VectorFitnessFunction()
    with pytest.raises(NotImplementedError):
        vector_fitness_function.get_jacobian(None)


class VectorGradFitnessFunction(VectorGradientMixin, VectorBasedFunction):
    def __init__(self, metric):
        super().__init__(metric=metric)

    def evaluate_fitness_vector(self, individual):
        return np.array([-2, 0, 2])

    def get_jacobian(self, individual):
        return np.array([[0.5, 1, -0.5], [1, 2, 3]]).transpose()


@pytest.mark.parametrize("metric, expected_fit_grad", [("mae", [-1/3, 2/3]),
                                                       ("mse", [-4/3, 8/3]),
                                                       ("rmse", [np.sqrt(3/8) * -2/3, np.sqrt(3/8) * 4/3])])
def test_vector_gradient(metric, expected_fit_grad):
    vector_function = VectorGradFitnessFunction(metric)
    np.testing.assert_array_equal(vector_function.get_gradient(None), expected_fit_grad)
