# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.symbolic_regression.agraph.generator \
    import AGraphGenerator, bingocpp


@pytest.mark.parametrize("agraph_size,expected_error", [
    (0, ValueError),
    ("string", TypeError)
])
def test_raises_error_invalid_agraph_size_gen(agraph_size,
                                              expected_error,
                                              sample_component_generator):
    with pytest.raises(expected_error):
        _ = AGraphGenerator(agraph_size, sample_component_generator)


@pytest.mark.parametrize("cpp_backend", [
    False,
    pytest.param(True,
                 marks=pytest.mark.skipif(bingocpp is None,
                                          reason="failed bingocpp import"))])
def test_return_correct_agraph_backend(cpp_backend,
                                       sample_component_generator):
    generate_agraph = AGraphGenerator(6, sample_component_generator,
                                      cpp_backend)
    agraph = generate_agraph()
    assert agraph.is_cpp() == cpp_backend


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
