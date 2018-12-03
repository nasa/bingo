# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.GeneticIndividual import GeneticIndividual, EquationIndividual


def test_raises_error_construct_genetic_individual():
    with pytest.raises(TypeError):
        _ = GeneticIndividual()
