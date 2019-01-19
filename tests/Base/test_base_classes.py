# Ignoring some linting rules in tests
# pylint: disable=missing-docstring
import pytest

from bingo.Base.Chromosome import Chromosome
from bingo.Base.Equation import Equation
from bingo.Base.Crossover import Crossover
from bingo.Base.Mutation import Mutation
from bingo.Base.Generator import Generator
from bingo.Base.Selection import Selection
from bingo.Base.Variation import Variation
from bingo.Base.Evaluation import Evaluation
from bingo.Base.EvolutionaryAlgorithm import EvolutionaryAlgorithm
from bingo.Base.ContinuousLocalOptimization import ChromosomeInterface


@pytest.mark.parametrize("base_class", [Chromosome, Equation, Generator,
                                        Crossover, Mutation, Selection,
                                        Variation, Evaluation,
                                        EvolutionaryAlgorithm,
                                        ChromosomeInterface])
def test_raises_error_construct_base_classes(base_class):
    with pytest.raises(TypeError):
        _ = base_class()
