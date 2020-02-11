# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
from collections import namedtuple
import pytest
import numpy as np

from bingo.symbolic_regression.agraph import agraph, backend as py_backend
try:
    from bingocpp.build import bingocpp as bingocpp
    cpp_agraph = bingocpp.AGraph()
    CPP_LOADED = True
except ImportError:
    bingocpp = None
    CPP_LOADED = False
    cpp_agraph = None


agraph.Backend = py_backend
EVALUATE = "bingo.symbolic_regression.agraph.agraph.Backend.evaluate"
EVALUATE_WTIH_DERIV = ("bingo.symbolic_regression.agraph.agraph.Backend."
                       "evaluate_with_derivative")


@pytest.fixture
def sample_agraph_1_values():
    values = namedtuple('Data', ['x', 'f_of_x', 'grad_x', 'grad_c'])
    x = np.vstack((np.linspace(-1.0, 0.0, 11),
                   np.linspace(0.0, 1.0, 11))).transpose()
    f_of_x = (np.sin(x[:, 0] + 2.0) + 2.0).reshape((-1, 1))
    grad_x = np.zeros(x.shape)
    grad_x[:, 0] = np.cos(x[:, 0] + 2.0)
    grad_c = (np.cos(x[:, 0] + 2.0) + 1.0).reshape((-1, 1))
    return values(x, f_of_x, grad_x, grad_c)


@pytest.fixture
def all_funcs_agraph():
    test_graph = agraph.AGraph()
    return _set_all_funcs_agraph_data(test_graph) 


@pytest.fixture
def all_funcs_agraph_cpp():
    if bingocpp == None:
        return None
    test_graph = bingocpp.AGraph()
    return _set_all_funcs_agraph_data(test_graph)


def _set_all_funcs_agraph_data(test_graph):
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
    _ = test_graph.needs_local_optimization()
    test_graph.set_local_optimization_params([2.0, ])
    return test_graph


@pytest.fixture(params=['all_funcs_agraph', 'sample_agraph_1'])
def expected_agraph_behavior(request):
    prop = {'agraph': request.getfixturevalue(request.param)}
    if request.param == "all_funcs_agraph":
        prop["latex string"] = "\\sqrt{ |(log{ exp{ cos{ sin{ \\frac{ (2.0" + \
                               " + X_0 - (X_0))(X_0) }{ X_0 } } } } })^{ (" + \
                               "X_0) }| }"
        prop["console string"] = "sqrt(|(log(exp(cos(sin(((2.0 + X_0 - (X_" + \
                                 "0))(X_0))/(X_0) )))))^(X_0)|)"
        prop["complexity"] = 13
    elif request.param == "sample_agraph_1":
        prop["latex string"] = "sin{ X_0 + 2.0 } + 2.0"
        prop["console string"] = "sin(X_0 + 2.0) + 2.0"
        prop["complexity"] = 5
    return prop


@pytest.fixture(params=['all_funcs_agraph_cpp', 'sample_agraph_1_cpp'])
def expected_agraph_behavior_cpp(request):
    prop = {'agraph': request.getfixturevalue(request.param)}
    if request.param == "all_funcs_agraph_cpp":
        prop["latex string"] = "\\sqrt{ |(log{ exp{ cos{ sin{ \\frac{ (2.000000" + \
                               " + X_0 - (X_0))(X_0) }{ X_0 } } } } })^{ (" + \
                               "X_0) }| }"
        prop["console string"] = "sqrt(|(log(exp(cos(sin(((2.000000 + X_0 - (X_" + \
                                 "0))(X_0))/(X_0))))))^(X_0)|)"
        prop["complexity"] = 13
    elif request.param == "sample_agraph_1_cpp":
        prop["latex string"] = "sin{ X_0 + 2.000000 } + 2.000000"
        prop["console string"] = "sin(X_0 + 2.000000) + 2.000000"
        prop["complexity"] = 5
    return prop


def test_cpp_agraph_could_be_imported():
    assert CPP_LOADED and bingocpp.AGraph() is not None


def test_agraph_for_proper_super_init(sample_agraph_1):
    member_vars = vars(sample_agraph_1)
    assert '_genetic_age' in member_vars
    assert '_fitness' in member_vars
    assert '_fit_set' in member_vars


def test_deep_copy_agraph(sample_agraph_1_list):
    agraph_copy = sample_agraph_1_list.copy()
    sample_agraph_1_list.mutable_command_array[1, 1] = 100

    assert agraph_copy.genetic_age == 10
    assert agraph_copy.command_array[1, 1] == 0


def test_agraph_latex_print(expected_agraph_behavior):
    assert expected_agraph_behavior["latex string"] == \
           expected_agraph_behavior["agraph"].get_latex_string()


def test_agraph_console_print(expected_agraph_behavior):
    assert expected_agraph_behavior["console string"] == \
           expected_agraph_behavior["agraph"].__str__()


def test_agraph_complexity(expected_agraph_behavior):
    assert expected_agraph_behavior["complexity"] == \
           expected_agraph_behavior["agraph"].get_complexity()


@pytest.mark.skipif(not bingocpp, reason='BingoCpp import failure')
def test_agraph_latex_print_Cpp(expected_agraph_behavior_cpp):
    assert expected_agraph_behavior_cpp["latex string"] == \
           expected_agraph_behavior_cpp["agraph"].get_latex_string()


@pytest.mark.skipif(not bingocpp, reason='BingoCpp import failure')
def test_agraph_console_print_cpp(expected_agraph_behavior_cpp):
    assert expected_agraph_behavior_cpp["console string"] == \
           expected_agraph_behavior_cpp["agraph"].__str__()


@pytest.mark.skipif(not bingocpp, reason='BingoCpp import failure')
def test_agraph_complexity_cpp(expected_agraph_behavior_cpp):
    assert expected_agraph_behavior_cpp["complexity"] == \
           expected_agraph_behavior_cpp["agraph"].get_complexity()


def test_agraph_stack_print(sample_agraph_1):
    expected_str = "---full stack---\n" +\
                    "(0) <= X_0\n" +\
                    "(1) <= C\n" +\
                    "(2) <= (0) + (1)\n" +\
                    "(3) <= sin (2)\n" +\
                    "(4) <= (0) + (1)\n" +\
                    "(5) <= (3) + (1)\n" +\
                    "---small stack---\n" +\
                    "(0) <= X_0\n" +\
                    "(1) <= C_0 = 2.0\n" +\
                    "(2) <= (0) + (1)\n" +\
                    "(3) <= sin (2)\n" +\
                    "(4) <= (3) + (1)\n"
    assert sample_agraph_1.get_stack_string() == expected_str


@pytest.mark.skipif(not bingocpp, reason='BingoCpp import failure')
def test_agraph_stack_print_cpp(sample_agraph_1_cpp):
    expected_str = "---full stack---\n" +\
                "(0) <= X_0\n" +\
                "(1) <= C\n" +\
                "(2) <= (0) + (1)\n" +\
                "(3) <= sin (2)\n" +\
                "(4) <= (0) + (1)\n" +\
                "(5) <= (3) + (1)\n" +\
                "---small stack---\n" +\
                "(0) <= X_0\n" +\
                "(1) <= C_0 = 2.000000\n" +\
                "(2) <= (0) + (1)\n" +\
                "(3) <= sin (2)\n" +\
                "(4) <= (3) + (1)\n"

    assert sample_agraph_1_cpp.get_stack_string() == expected_str


def test_evaluate_agraph(sample_agraph_1_list, sample_agraph_1_values):
    np.testing.assert_allclose(
        sample_agraph_1_list.evaluate_equation_at(sample_agraph_1_values.x),
        sample_agraph_1_values.f_of_x)


def test_evaluate_agraph_x_gradient(sample_agraph_1_list,
                                    sample_agraph_1_values):
    f_of_x, df_dx = \
        sample_agraph_1_list.evaluate_equation_with_x_gradient_at(
            sample_agraph_1_values.x)
    np.testing.assert_allclose(f_of_x, sample_agraph_1_values.f_of_x)
    np.testing.assert_allclose(df_dx, sample_agraph_1_values.grad_x)


def test_evaluate_agraph_c_gradient(sample_agraph_1_list,
                                    sample_agraph_1_values):
    f_of_x, df_dc = \
        sample_agraph_1_list.evaluate_equation_with_local_opt_gradient_at(
            sample_agraph_1_values.x)
    np.testing.assert_allclose(f_of_x, sample_agraph_1_values.f_of_x)
    np.testing.assert_allclose(df_dc, sample_agraph_1_values.grad_c)


def test_setting_fitness_updates_fit_set():
    sample_agraph = agraph.AGraph()
    assert not sample_agraph.fit_set
    sample_agraph.fitness = 0
    assert sample_agraph.fit_set




@pytest.mark.parametrize(
    'agraph',
    (
            agraph.AGraph(),
            pytest.param(cpp_agraph, marks=pytest.mark.skipif(
                                        not bingocpp,
                                        reason='BingoCpp import failure'))
    )
)
def test_setting_fitness_updates_fit_set_cpp(agraph):
    sample_agraph = agraph
    assert not sample_agraph.fit_set
    sample_agraph.fitness = 0
    assert sample_agraph.fit_set


def test_mutable_access_to_command_array_unsets_fitness(sample_agraph_1_list):
    assert sample_agraph_1_list.fit_set
    _ = sample_agraph_1_list.mutable_command_array
    assert not sample_agraph_1_list.fit_set


def test_setting_command_array_unsets_fitness(sample_agraph_1_list):
    sample_agraph_1_list.fitness = 0
    assert sample_agraph_1_list.fit_set
    sample_agraph_1_list.command_array = np.ones((1, 3))
    assert not sample_agraph_1_list.fit_set


def test_evaluate_overflow_exception(mocker,
                                     sample_agraph_1,
                                     sample_agraph_1_values):
    mocker.patch(EVALUATE)
    agraph.Backend.evaluate.side_effect = OverflowError

    values = sample_agraph_1.evaluate_equation_at(sample_agraph_1_values.x)
    assert np.isnan(values).all()


def test_evaluate_gradient_overflow_exception(mocker,
                                              sample_agraph_1,
                                              sample_agraph_1_values):
    mocker.patch(EVALUATE_WTIH_DERIV)
    agraph.Backend.evaluate_with_derivative.side_effect = OverflowError

    values = sample_agraph_1.evaluate_equation_with_x_gradient_at(
        sample_agraph_1_values.x)
    assert np.isnan(values).all()


def test_evaluate_local_opt_gradient_overflow_exception(mocker,
                                                        sample_agraph_1,
                                                        sample_agraph_1_values):
    mocker.patch(EVALUATE_WTIH_DERIV)
    agraph.Backend.evaluate_with_derivative.side_effect = OverflowError

    values = sample_agraph_1.evaluate_equation_with_local_opt_gradient_at(
        sample_agraph_1_values.x)
    assert np.isnan(values).all()


def test_distance_between_graphs(sample_agraph_1_list):
    assert sample_agraph_1_list.distance(sample_agraph_1_list) == 0
    other_agraph = sample_agraph_1_list.copy()
    other_agraph.mutable_command_array[2] = np.array([6, 1, 0])
    assert sample_agraph_1_list.distance(other_agraph) == 3