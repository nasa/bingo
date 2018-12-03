# Ignoring some linting rules in tests
# pylint: disable=missing-docstring
import pytest

from bingo.GeneticIndividual import GeneticIndividual, EquationIndividual


def test_raises_error_construct_genetic_individual():
    with pytest.raises(TypeError):
        _ = GeneticIndividual()


def test_raises_error_construct_equation_individual():
    with pytest.raises(TypeError):
        _ = EquationIndividual()
