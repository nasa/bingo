# Ignoring some linting rules in tests
# pylint: disable=missing-docstring
import pytest

from bingo import IndividualVariation as IV


@pytest.mark.parametrize("variation_class", [IV.IndividualGeneration,
                                             IV.IndividualCrossover,
                                             IV.IndividualMutation])
def test_raises_error_construct_individual_variation(variation_class):
    with pytest.raises(TypeError):
        _ = variation_class()
