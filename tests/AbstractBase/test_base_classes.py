# Ignoring some linting rules in tests
# pylint: disable=missing-docstring
import pytest

from bingo.Base.Chromosome import Chromosome
from bingo.SymbolicRegression.Equation import Equation
from bingo.Base.Crossover import Crossover
from bingo.Base.Mutation import Mutation
from bingo.Base.Generator import Generator
from bingo.Base.Selection import Selection
from bingo.Base.Variation import Variation
from bingo.Base.FitnessFunction import FitnessFunction, VectorBasedFunction
from bingo.Base.EvolutionaryAlgorithm import EvolutionaryAlgorithm
from bingo.Base.ContinuousLocalOptimization import ChromosomeInterface
from bingo.Base.TrainingData import TrainingData


@pytest.mark.parametrize("base_class", [Chromosome, Equation, Generator,
                                        Crossover, Mutation, Selection,
                                        Variation,
                                        FitnessFunction, VectorBasedFunction,
                                        EvolutionaryAlgorithm,
                                        ChromosomeInterface, TrainingData])
def test_raises_error_construct_base_classes(base_class):
    with pytest.raises(TypeError):
        _ = base_class()
