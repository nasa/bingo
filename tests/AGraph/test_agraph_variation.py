# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.AGraph.AGraphGeneration import AGraphGeneration
from bingo.AGraph.ComponentGenerator import ComponentGenerator


@pytest.fixture
def sample_component_generator():
    generator = ComponentGenerator(input_x_dimension=2,
                                   num_initial_load_statements=2,
                                   terminal_probability=0.4,
                                   constant_probability=0.5)
    generator.add_operator(2)
    generator.add_operator(6)
    return generator


@pytest.mark.parametrize("agraph_size,expected_error", [
    (0, ValueError),
    ("string", TypeError)
])
def test_raises_error_invalid_agraph_size(agraph_size, expected_error,
                                          sample_component_generator):
    with pytest.raises(expected_error):
        _ = AGraphGeneration(agraph_size, sample_component_generator)


def test_generate(sample_component_generator):
    np.random.seed(0)
    expected_command_array = np.array([[0, 1, 0],
                                       [0, 1, 1],
                                       [6, 0, 1],
                                       [6, 2, 1],
                                       [6, 0, 1],
                                       [2, 1, 4]])
    generate_agraph = AGraphGeneration(6, sample_component_generator)
    agraph = generate_agraph()
    np.testing.assert_array_equal(agraph.command_array,
                                  expected_command_array)
