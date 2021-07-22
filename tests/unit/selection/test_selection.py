# Ignoring some linting rules in tests
# pylint: disable=missing-docstring

import pytest
from bingo.selection.selection import Selection


def test_selection_cant_be_instanced():
    with pytest.raises(TypeError):
        _ = Selection()
