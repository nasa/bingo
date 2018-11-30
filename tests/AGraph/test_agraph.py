# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.AGraph.AGraph import AGraph


@pytest.fixture
def sample_agraph():
    test_graph = AGraph()
    test_graph.genetic_age = 10
    test_graph._command_array = np.array([[0, 0, 0],  # sin(X_0) + 1.0
                                          [1, 0, 0],
                                          [2, 0, 1],
                                          [6, 0, 2],
                                          [2, 3, 1]])
    test_graph._constants = [1.0, ]
    return test_graph


@pytest.fixture
def all_funcs_agraph():
    test_graph = AGraph()
    test_graph.genetic_age = 10
    test_graph._command_array = np.array([[0, 0, 0],
                                          [1, 0, 0],
                                          [2, 1, 0],
                                          [3, 2, 0],
                                          [4, 3, 0],
                                          [5, 4, 0],
                                          [6, 5, 0],
                                          [7, 6, 0],
                                          [8, 7, 0],
                                          [9, 8, 0],
                                          [10, 9, 0],
                                          [11, 10, 0],
                                          [12, 11, 0]])
    test_graph._constants = [1.0, ]
    return test_graph


def test_agraph_for_proper_super_init(sample_agraph):
    member_vars = vars(sample_agraph)
    assert 'genetic_age' in member_vars
    assert 'fitness' in member_vars
    assert 'fit_set' in member_vars


def test_deep_copy_agraph(sample_agraph):
    agraph_copy = sample_agraph.copy()
    sample_agraph._command_array[1, 1] = 100
    sample_agraph._constants[0] = 100.0

    assert 10 == agraph_copy.genetic_age
    assert 0 == agraph_copy._command_array[1, 1]
    assert 1.0 == pytest.approx(agraph_copy._constants[0])


@pytest.mark.parametrize("agraph,expected_latex_string", [
    (all_funcs_agraph(),
     "\\sqrt{ |(log{ exp{ cos{ sin{ \\frac{ (1.0 + X_0 - (X_0))(X_0)" +\
     " }{ X_0 } } } } })^{ (X_0) }| }"),
    (sample_agraph(),
     "sin{ X_0 } + 1.0"),
])
def test_agraph_latex_print(agraph, expected_latex_string):
    assert expected_latex_string == agraph.get_latex_string()


@pytest.mark.parametrize("agraph,expected_console_string", [
    (all_funcs_agraph(),
     "sqrt(|(log(exp(cos(sin(((1.0 + X_0 - (X_0))(X_0))/(X_0) )))))^(X_0)|)"),
    (sample_agraph(),
     "sin(X_0) + 1.0"),
])
def test_agraph_console_print(agraph, expected_console_string):
    assert expected_console_string == agraph.get_console_string()


def test_agraph_stack_print(sample_agraph):
    expected_str = "---full stack---\n" +\
                   "(0) <= X_0\n" +\
                   "(1) <= C_0 = 1.0\n" +\
                   "(2) <= (0) + (1)\n" +\
                   "(3) <= sin (0)\n" +\
                   "(4) <= (3) + (1)\n" +\
                   "---small stack---\n" +\
                   "(0) <= X_0\n" +\
                   "(1) <= C_0 = 1.0\n" +\
                   "(3) <= sin (0)\n" +\
                   "(4) <= (3) + (1)\n"
    assert sample_agraph.__str__() == expected_str


@pytest.mark.parametrize("agraph,expected_complexity", [
    (all_funcs_agraph(), 13),
    (sample_agraph(), 4),
])
def test_agraph_get_complexity(agraph, expected_complexity):
    assert agraph.get_complexity() == expected_complexity
