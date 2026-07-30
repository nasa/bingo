# pylint: disable=missing-docstring
import numpy as np
import pytest

from bingo.chromosomes.chromosome import Chromosome
from bingo.evaluation.fitness_function import FitnessFunction
from bingo.evaluation.random_subset_evaluation import RandomSubsetEvaluation
from bingo.expressions.agraph.evolvable import EvolvableExpression
from bingo.expressions.agraph.pyagraph.expression import AGraphExpression
from bingo.expressions.agraph.pyagraph.operators import VARIABLE
from bingo.symbolic_regression.explicit_regression import ExplicitRegression
from bingo.symbolic_regression.implicit_regression import ImplicitRegression


class ObjectiveData:
    def __init__(self, values):
        self.values = np.asarray(values)

    def __getitem__(self, items):
        return ObjectiveData(self.values[items])

    def __len__(self):
        return len(self.values)


class SubsetFitness(FitnessFunction):
    def __init__(self, values):
        super().__init__()
        self._objective_data = ObjectiveData(values)

    def __call__(self, indv):
        self.eval_count += 1
        return self._objective_data.values


class DummyIndv(Chromosome):
    def __str__(self):
        return ""

    def distance(self, other):
        return 0


def _variable_individual():
    expression = AGraphExpression()
    expression.raw_command_array = np.array([[VARIABLE, 0, 0]], dtype=np.uint8)
    return EvolvableExpression(expression)


def test_subset_size_must_nonzero():
    with pytest.raises(ValueError):
        RandomSubsetEvaluation(SubsetFitness([1, 2, 3, 4, 5]), subset_size=0)


def test_subset_of_private_objective_data_is_used_for_every_individual():
    fitness_function = SubsetFitness(np.arange(5, dtype=int))
    population = [DummyIndv(), DummyIndv()]

    RandomSubsetEvaluation(fitness_function, subset_size=3)(population)

    assert len(population[0].fitness) == 3
    np.testing.assert_array_equal(population[0].fitness, population[1].fitness)


def test_subsets_are_different_in_each_evaluation():
    fitness_function = SubsetFitness(np.arange(25, dtype=int))
    evaluation = RandomSubsetEvaluation(fitness_function, subset_size=3)
    population_1 = [DummyIndv()]
    population_2 = [DummyIndv()]

    evaluation(population_1)
    evaluation(population_2)

    assert tuple(population_1[0].fitness) != tuple(population_2[0].fitness)


def test_subset_evaluation_slices_aligned_explicit_objective_data(mocker):
    x = np.arange(5.0).reshape(-1, 1)
    objective = ExplicitRegression(x, x.ravel())
    mocker.patch(
        "bingo.evaluation.random_subset_evaluation.np.random.choice",
        return_value=np.array([1, 3]),
    )

    RandomSubsetEvaluation(objective, subset_size=2)(
        [_variable_individual(), _variable_individual()]
    )

    np.testing.assert_array_equal(objective._objective_data.X, [[1.0], [3.0]])
    np.testing.assert_array_equal(objective._objective_data.y, [1.0, 3.0])


def test_subset_evaluation_slices_aligned_implicit_objective_data(mocker):
    x = np.arange(5.0).reshape(-1, 1)
    dx_dt = np.arange(10.0, 15.0).reshape(-1, 1)
    objective = ImplicitRegression(x, dx_dt)
    mocker.patch(
        "bingo.evaluation.random_subset_evaluation.np.random.choice",
        return_value=np.array([0, 4]),
    )

    RandomSubsetEvaluation(objective, subset_size=2)(
        [_variable_individual(), _variable_individual()]
    )

    np.testing.assert_array_equal(objective._objective_data.X, [[0.0], [4.0]])
    np.testing.assert_array_equal(objective._objective_data.dx_dt, [[10.0], [14.0]])
