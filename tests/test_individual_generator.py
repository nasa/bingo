# Ignoring some linting rules in tests
# pylint: disable=missing-docstring
import pytest

from bingo.IndividualGenerator import IndividualGenerator


def test_raises_error_construct_individual_generator():
    with pytest.raises(TypeError):
        _ = IndividualGenerator()
