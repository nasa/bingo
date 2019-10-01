# Ignoring some linting rules in tests
# pylint: disable=missing-docstring
import pytest

from bingo.chromosomes import chromosome
from bingo.symbolic_regression.equation import Equation
from bingo.chromosomes import crossover
from bingo.chromosomes.mutation import Mutation
from bingo.chromosomes import generator
from bingo.selection.selection import Selection
from bingo.variation import variation
from bingo.evaluation.fitness_function import FitnessFunction, VectorBasedFunction
from bingo.evolutionary_algorithms import evolutionary_algorithm
from bingo.local_optimizers.continuous_local_opt import ChromosomeInterface
from bingo.evaluation.training_data import TrainingData


@pytest.mark.parametrize("base_class", [chromosome, Equation, generator,
                                        crossover, Mutation, Selection,
                                        variation,
                                        FitnessFunction, VectorBasedFunction,
                                        evolutionary_algorithm,
                                        ChromosomeInterface, TrainingData])
def test_raises_error_construct_base_classes(base_class):
    with pytest.raises(TypeError):
        _ = base_class()
