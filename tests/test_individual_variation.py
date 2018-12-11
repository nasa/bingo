# Ignoring some linting rules in tests
# pylint: disable=missing-docstring
import pytest

from bingo import IndividualVariation


@pytest.mark.parametrize("variation_class", [IndividualVariation.Generation,
                                             IndividualVariation.Crossover,
                                             IndividualVariation.Mutation])
def test_raises_error_construct_individual_variation(variation_class):
    with pytest.raises(TypeError):
        _ = variation_class()
