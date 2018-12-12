# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
from collections import namedtuple
import pytest
import numpy as np

from bingo.AGraph import AGraph
from bingo.AGraph import Backend as py_backend

AGraph.Backend = py_backend


@pytest.fixture
def invalid_agraph():
    test_graph = AGraph.AGraph()
    test_graph.command_array = np.array([[0, 0, 0],  # sin(X_0) + 1.0
                                         [1, 0, 0],
                                         [2, 0, 1],
                                         [6, 0, 2],
                                         [2, 3, 1]])
    return test_graph


@pytest.fixture
def sample_agraph():
    test_graph = AGraph.AGraph()
    test_graph.command_array = np.array([[0, 0, 0],  # sin(X_0) + 1.0
                                         [1, 0, 0],
                                         [2, 0, 1],
                                         [6, 0, 2],
                                         [2, 3, 1]])
    test_graph.genetic_age = 10
    test_graph.set_local_optimization_params([1.0, ])
    return test_graph


@pytest.fixture
def sample_values():
    values = namedtuple('Data', ['x', 'f_of_x', 'grad_x', 'grad_c'])
    x = np.vstack((np.linspace(-1.0, 0.0, 11),
                   np.linspace(0.0, 1.0, 11))).transpose()
    f_of_x = (np.sin(x[:, 0]) + 1.0).reshape((-1, 1))
    grad_x = np.zeros(x.shape)
    grad_x[:, 0] = np.cos(x[:, 0])
    grad_c = np.ones((x.shape[0], 1))
    return values(x, f_of_x, grad_x, grad_c)


@pytest.fixture
def all_funcs_agraph():
    test_graph = AGraph.AGraph()
    test_graph.genetic_age = 10
    test_graph.command_array = np.array([[0, 0, 0],
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
    test_graph.set_local_optimization_params([1.0, ])
    return test_graph


def test_agraph_for_proper_super_init(sample_agraph):
    member_vars = vars(sample_agraph)
    assert 'genetic_age' in member_vars
    assert '_fitness' in member_vars
    assert 'fit_set' in member_vars


def test_deep_copy_agraph(sample_agraph):
    agraph_copy = sample_agraph.copy()
    sample_agraph.command_array[1, 1] = 100
    sample_agraph.set_local_optimization_params([100.0, ])

    assert agraph_copy.genetic_age == 10
    assert agraph_copy.command_array[1, 1] == 0
    assert pytest.approx(agraph_copy._constants[0]) == 1.0


@pytest.mark.parametrize("agraph,expected_latex_string", [
    (all_funcs_agraph(),
     "\\sqrt{ |(log{ exp{ cos{ sin{ \\frac{ (1.0 + X_0 - (X_0))(X_0)" +\
     " }{ X_0 } } } } })^{ (X_0) }| }"),
    (sample_agraph(),
     "sin{ X_0 } + 1.0"),
    (invalid_agraph(),
     "sin{ X_0 } + ?"),
])
def test_agraph_latex_print(agraph, expected_latex_string):
    assert expected_latex_string == agraph.get_latex_string()


@pytest.mark.parametrize("agraph,expected_console_string", [
    (all_funcs_agraph(),
     "sqrt(|(log(exp(cos(sin(((1.0 + X_0 - (X_0))(X_0))/(X_0) )))))^(X_0)|)"),
    (sample_agraph(),
     "sin(X_0) + 1.0"),
    (invalid_agraph(),
     "sin(X_0) + ?"),
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


def test_invalid_agraph_stack_print(invalid_agraph):
    expected_str = "---full stack---\n" +\
                   "(0) <= X_0\n" +\
                   "(1) <= C\n" +\
                   "(2) <= (0) + (1)\n" +\
                   "(3) <= sin (0)\n" +\
                   "(4) <= (3) + (1)\n" +\
                   "---small stack---\n" +\
                   "(0) <= X_0\n" +\
                   "(1) <= C\n" +\
                   "(3) <= sin (0)\n" +\
                   "(4) <= (3) + (1)\n"
    assert invalid_agraph.__str__() == expected_str


@pytest.mark.parametrize("agraph,expected_complexity", [
    (all_funcs_agraph(), 13),
    (sample_agraph(), 4),
])
def test_agraph_get_complexity(agraph, expected_complexity):
    assert agraph.get_complexity() == expected_complexity


def test_evaluate_agraph(sample_agraph, sample_values):
    np.testing.assert_allclose(
        sample_agraph.evaluate_equation_at(sample_values.x),
        sample_values.f_of_x)


def test_evaluate_agraph_x_gradient(sample_agraph, sample_values):
    f_of_x, df_dx = \
        sample_agraph.evaluate_equation_with_x_gradient_at(sample_values.x)
    np.testing.assert_allclose(f_of_x, sample_values.f_of_x)
    np.testing.assert_allclose(df_dx, sample_values.grad_x)


def test_evaluate_agraph_c_gradient(sample_agraph, sample_values):
    f_of_x, df_dc = sample_agraph.evaluate_equation_with_local_opt_gradient_at(
        sample_values.x)
    np.testing.assert_allclose(f_of_x, sample_values.f_of_x)
    np.testing.assert_allclose(df_dc, sample_values.grad_c)


def test_raises_error_evaluate_invalid_agraph(invalid_agraph, sample_values):
    with pytest.raises(RuntimeError):
        _ = invalid_agraph.evaluate_equation_at(sample_values.x)


def test_raises_error_x_gradient_invalid_agraph(invalid_agraph, sample_values):
    with pytest.raises(RuntimeError):
        _ = invalid_agraph.evaluate_equation_with_x_gradient_at(
            sample_values.x)


def test_raises_error_c_gradient_invalid_agraph(invalid_agraph, sample_values):
    with pytest.raises(RuntimeError):
        _ = invalid_agraph.evaluate_equation_with_local_opt_gradient_at(
            sample_values.x)


def test_invalid_agraph_needs_optimization(invalid_agraph):
    assert invalid_agraph.needs_local_optimization()


def test_get_number_optimization_params(invalid_agraph):
    assert invalid_agraph.get_number_local_optimization_params() == 1


def test_set_optimization_params(invalid_agraph, sample_agraph, sample_values):
    invalid_agraph.set_local_optimization_params([1.0])

    assert not invalid_agraph.needs_local_optimization()
    np.testing.assert_allclose(
        invalid_agraph.evaluate_equation_at(sample_values.x),
        sample_agraph.evaluate_equation_at(sample_values.x))


def test_setting_fitness_updates_fit_set(sample_agraph):
    assert not sample_agraph.fit_set
    sample_agraph.fitness = 0
    assert sample_agraph.fit_set


def test_setting_command_array_unsets_fitness(sample_agraph):
    sample_agraph.fitness = 0
    assert sample_agraph.fit_set
    sample_agraph.command_array = np.ones((1, 3))
    assert not sample_agraph.fit_set
