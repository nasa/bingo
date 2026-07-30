"""Tests for generic Python gradient-based fitness aggregation."""

import numpy as np
import pytest

from bingo.evaluation.fitness_function import VectorBasedFunction
from bingo.evaluation.gradient_mixin import GradientMixin, VectorGradientMixin


class _Individual:
    def get_number_local_optimization_params(self):
        return 2


class _GradientFitness(VectorGradientMixin, VectorBasedFunction):
    def get_fitness_vector_and_jacobian(self, individual):
        return np.array([-2.0, 0.0, 2.0]), np.array(
            [[0.5, 1.0], [1.0, 2.0], [-0.5, 3.0]]
        )

    def evaluate_fitness_vector(self, individual):
        return self.get_fitness_vector_and_jacobian(individual)[0]


def test_gradient_mixin_cannot_be_instantiated():
    with pytest.raises(TypeError):
        GradientMixin()


def test_vector_gradient_mixin_requires_vector_fitness_base():
    class _InvalidGradientFitness(VectorGradientMixin):
        def get_fitness_vector_and_jacobian(self, individual):
            return None

    with pytest.raises(TypeError):
        _InvalidGradientFitness()


@pytest.mark.parametrize(
    "metric, expected_fitness, expected_gradient",
    [
        ("mae", 4 / 3, [-1 / 3, 2 / 3]),
        ("mean absolute error", 4 / 3, [-1 / 3, 2 / 3]),
        ("mse", 8 / 3, [-4 / 3, 8 / 3]),
        ("mean squared error", 8 / 3, [-4 / 3, 8 / 3]),
        ("rmse", np.sqrt(8 / 3), [-np.sqrt(3 / 8) * 2 / 3, np.sqrt(3 / 8) * 4 / 3]),
        ("root mean squared error", np.sqrt(8 / 3), [-np.sqrt(3 / 8) * 2 / 3, np.sqrt(3 / 8) * 4 / 3]),
        ("negative nmll laplace", 3.244922013421868, [-0.3169873, 0.6339746]),
        ("bic", 14.751955824267544, [-1.5, 3.0]),
    ],
)
def test_vector_gradient_fitness_aggregates_vector_and_jacobian(
    metric, expected_fitness, expected_gradient
):
    fitness, gradient = _GradientFitness(metric=metric).get_fitness_and_gradient(
        _Individual()
    )

    assert fitness == pytest.approx(expected_fitness)
    np.testing.assert_allclose(gradient, expected_gradient)


def test_vector_gradient_mixin_rejects_unknown_metric():
    with pytest.raises(ValueError):
        _GradientFitness(metric="unknown")
