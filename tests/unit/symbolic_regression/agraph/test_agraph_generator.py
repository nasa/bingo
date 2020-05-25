# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.symbolic_regression.agraph.generator \
    import AGraphGenerator, BINGOCPP


@pytest.mark.parametrize("agraph_size,expected_error", [
    (0, ValueError),
    ("string", TypeError)
])
def test_raises_error_invalid_agraph_size_gen(agraph_size,
                                              expected_error,
                                              sample_component_generator):
    with pytest.raises(expected_error):
        _ = AGraphGenerator(agraph_size, sample_component_generator)


@pytest.mark.parametrize("python_backend", [
    True,
    pytest.param(False,
                 marks=pytest.mark.skipif(not BINGOCPP,
                                          reason="failed bingocpp import"))])
def test_return_correct_agraph_backend(python_backend,
                                       sample_component_generator):
    generate_agraph = AGraphGenerator(6, sample_component_generator,
                                      python_backend)
    agraph = generate_agraph()
    assert agraph.is_cpp() != python_backend


def test_generate(mocker):
    expected_command_array = np.arange(30).reshape((10, 3), dtype=int)
    mocked_component_generator = mocker.Mock()
    mocked_component_generator.random_command.side_effect = \
        [list(row) for row in expected_command_array]

    generate_agraph = AGraphGenerator(10, mocked_component_generator)
    agraph = generate_agraph()
    np.testing.assert_array_equal(agraph.command_array, expected_command_array)
