# Ignoring some linting rules in tests
# pylint: disable=missing-docstring
import pytest
from bingo.variation.variation import Variation


def test_variation_cant_be_instanced():
    with pytest.raises(TypeError):
        _ = Variation()