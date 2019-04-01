# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.SymbolicRegression.AGraph.AGraphGenerator import AGraphGenerator


@pytest.mark.parametrize("agraph_size,expected_error", [
    (0, ValueError),
    ("string", TypeError)
])
def test_raises_error_invalid_agraph_size_gen(agraph_size,
                                              expected_error,
                                              sample_component_generator):
    with pytest.raises(expected_error):
        _ = AGraphGenerator(agraph_size, sample_component_generator)


def test_generate(sample_component_generator):
    np.random.seed(0)
    expected_command_array = np.array([[0, 1, 0],
                                       [0, 1, 1],
                                       [6, 0, 1],
                                       [6, 2, 1],
                                       [6, 0, 1],
                                       [2, 1, 4]], dtype=int)
    generate_agraph = AGraphGenerator(6, sample_component_generator)
    agraph = generate_agraph()
    np.testing.assert_array_equal(agraph.command_array,
                                  expected_command_array)
