# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
from itertools import cycle
import numpy as np
import pytest

from bingo.symbolic_regression.agraph.operator_definitions import *
from bingo.symbolic_regression.agraph.agraph import AGraph
from bingo.symbolic_regression.agraph.mutation import AGraphMutation
from bingo.symbolic_regression.agraph.component_generator \
    import ComponentGenerator


@pytest.fixture
def sample_agraph(mocker):
    stack = np.array([[VARIABLE, 0, 0],
                      [CONSTANT, 1, 1],
                      [COS, 1, 1],
                      [MULTIPLICATION, 0, 2],
                      [ADDITION, 0, 1],
                      [SIN, 3, 0]])
    sample = mocker.create_autospec(AGraph)
    type(sample).command_array = mocker.PropertyMock(return_value=stack.copy())
    type(sample).mutable_command_array = \
        mocker.PropertyMock(return_value=stack.copy())
    sample.get_utilized_commands.return_value = np.array([1, 1, 1, 1, 0, 1],
                                                         dtype=bool)
    sample.copy.return_value = sample
    return sample


@pytest.fixture
def no_param_mut_or_prune_agraph(mocker):
    stack = np.array([[VARIABLE, 0, 0],
                      [ADDITION, 0, 0]])
    sample = mocker.create_autospec(AGraph)
    type(sample).command_array = mocker.PropertyMock(return_value=stack.copy())
    type(sample).mutable_command_array = \
        mocker.PropertyMock(return_value=stack.copy())
    sample.get_utilized_commands.return_value = np.array([1, 1], dtype=bool)
    sample.copy.return_value = sample
    return sample


@pytest.fixture
def single_variable_agraph(mocker):
    stack = np.array([[VARIABLE, 0, 0],
                      [VARIABLE, 0, 0]])
    sample = mocker.create_autospec(AGraph)
    type(sample).command_array = mocker.PropertyMock(return_value=stack.copy())
    type(sample).mutable_command_array = \
        mocker.PropertyMock(return_value=stack.copy())
    sample.get_utilized_commands.return_value = np.array([0, 1], dtype=bool)
    sample.copy.return_value = sample
    return sample


@pytest.fixture
def fork_agraph():
    test_graph = AGraph()
    test_graph.command_array = np.array([[CONSTANT, -1, -1],  # sin(sin(X_0))
                                         [VARIABLE, 0, 0],
                                         [SUBTRACTION, 1, 1],
                                         [SUBTRACTION, 1, 1],
                                         [SIN, 1, 1],
                                         [SIN, 4, 4]], dtype=int)
    test_graph.genetic_age = 1
    test_graph.set_local_optimization_params([])
    return test_graph


@pytest.fixture
def spaced_fork_agraph():
    test_graph = AGraph()
    test_graph.command_array = np.array([[CONSTANT, -1, -1],  # sin(sin(X_0))
                                         [VARIABLE, 0, 0],
                                         [SUBTRACTION, 1, 1],
                                         [SIN, 1, 1],
                                         [SIN, 3, 3]], dtype=int)
    test_graph.genetic_age = 1
    test_graph.set_local_optimization_params([])
    return test_graph


@pytest.fixture
def not_enough_unutil_fork_agraph():
    test_graph = AGraph()
    test_graph.command_array = np.array([[VARIABLE, 0, 0],  # sin(sin(X_0))
                                         [SUBTRACTION, 0, 0],
                                         [SIN, 0, 0],
                                         [SIN, 2, 2]], dtype=int)
    test_graph.genetic_age = 1
    test_graph.set_local_optimization_params([])
    return test_graph


@pytest.fixture
def many_unutil_fork_agraph():
    test_graph = AGraph()
    test_graph.command_array = np.array([[VARIABLE, 0, 0],  # sin(sin(X_0))
                                         [SUBTRACTION, 0, 0],
                                         [SUBTRACTION, 0, 0],
                                         [SUBTRACTION, 0, 0],
                                         [SUBTRACTION, 0, 0],
                                         [SUBTRACTION, 0, 0],
                                         [SIN, 0, 0],
                                         [SIN, 6, 6]], dtype=int)
    test_graph.genetic_age = 1
    test_graph.set_local_optimization_params([])
    return test_graph


@pytest.fixture
def unutil_after_mutation_location_agraph():
    test_graph = AGraph()
    test_graph.command_array = np.array([[VARIABLE, 0, 0],  # sin(sin(X_0))
                                         [SUBTRACTION, 0, 0],
                                         [SIN, 0, 0],
                                         [SUBTRACTION, 0, 0],
                                         [SIN, 2, 2]], dtype=int)
    test_graph.genetic_age = 1
    test_graph.set_local_optimization_params([])
    return test_graph


@pytest.fixture
def arity_one_op_with_two_different_params_agraph():
    test_graph = AGraph()
    test_graph.command_array = np.array([[VARIABLE, 0, 0],  # sin(X_0)
                                         [SUBTRACTION, 0, 0],
                                         [SUBTRACTION, 0, 0],
                                         [SIN, 0, 2]], dtype=int)
    test_graph.genetic_age = 1
    test_graph.set_local_optimization_params([])
    return test_graph


@pytest.fixture
def sample_component_generator(mocker):
    sample = mocker.create_autospec(ComponentGenerator)
    random_commands = cycle([[CONSTANT, -1, -1],
                             [VARIABLE, 0, 0],
                             [ADDITION, 6, 6],
                             [SUBTRACTION, 7, 8],
                             [VARIABLE, 1, 1]])
    sample.random_command.side_effect = random_commands
    sample.get_number_of_terminals.return_value = 2
    sample.get_number_of_operators.return_value = 2
    sample.random_terminal.side_effect = cycle([CONSTANT, VARIABLE])
    sample.random_terminal_parameter.return_value = 2
    sample.random_operator.side_effect = cycle([ADDITION, SUBTRACTION])
    sample.random_operator_parameter.return_value = 10
    type(sample).input_x_dimension = mocker.PropertyMock(return_value=2)
    return sample


@pytest.fixture
def fork_mutation_component_generator():
    generator = ComponentGenerator(input_x_dimension=2,
                                   num_initial_load_statements=2,
                                   terminal_probability=0.4,
                                   constant_probability=0.5)
    generator.add_operator(2)
    generator.add_operator(6)
    return generator


@pytest.fixture
def fork_mutation(fork_mutation_component_generator):
    mutation = AGraphMutation(fork_mutation_component_generator,
                              command_probability=0.0,
                              node_probability=0.0,
                              parameter_probability=0.0,
                              prune_probability=0.0,
                              fork_probability=1.0)
    return mutation


@pytest.mark.parametrize("prob,expected_error", [
    (-1, ValueError),
    (2.5, ValueError),
    ("string", TypeError)
])
@pytest.mark.parametrize("prob_index", range(5))
def test_raises_error_invalid_mutation_probability(mocker, prob,
                                                   expected_error,
                                                   prob_index):
    mocked_component_generator = mocker.Mock()
    input_probabilities = [0.20]*5
    input_probabilities[prob_index] = prob
    with pytest.raises(expected_error):
        _ = AGraphMutation(mocked_component_generator, *input_probabilities)


@pytest.mark.parametrize("repeats", range(5))
@pytest.mark.parametrize("algo_index", range(3))
def test_single_point_mutations(sample_agraph, sample_component_generator,
                                algo_index, repeats):
    input_probabilities = [0.0] * 5
    input_probabilities[algo_index] = 1.0
    mutation = AGraphMutation(sample_component_generator, *input_probabilities)

    child = mutation(sample_agraph)
    p_stack = sample_agraph.command_array
    c_stack = child.mutable_command_array
    changed_commands = 0
    for p, c in zip(p_stack, c_stack):
        if (p != c).any():
            if p[0] != 1 or c[0] != 1:
                changed_commands += 1
    if changed_commands != 1:
        print("parent\n", p_stack)
        print("child\n", c_stack)
    assert changed_commands == 1


@pytest.mark.parametrize("repeats", range(5))
@pytest.mark.parametrize("algo_index, expected_node_mutation", [
    (1, True),
    (2, False),
    (3, False)
])
def test_mutation_of_nodes(sample_agraph, sample_component_generator,
                           algo_index, expected_node_mutation, repeats):
    input_probabilities = [0.0] * 5
    input_probabilities[algo_index] = 1.0
    mutation = AGraphMutation(sample_component_generator, *input_probabilities)

    child = mutation(sample_agraph)
    p_stack = sample_agraph.command_array
    c_stack = child.mutable_command_array
    changed_columns = np.sum(p_stack != c_stack, axis=0)

    if expected_node_mutation:
        assert changed_columns[0] == 1
    else:
        assert changed_columns[0] == 0


@pytest.mark.parametrize("repeats", range(5))
@pytest.mark.parametrize("algo_index", [2, 3])
def test_mutation_of_parameters(sample_agraph, sample_component_generator,
                                algo_index, repeats):
    input_probabilities = [0.0] * 5
    input_probabilities[algo_index] = 1.0
    mutation = AGraphMutation(sample_component_generator, *input_probabilities)

    child = mutation(sample_agraph)
    p_stack = sample_agraph.command_array
    c_stack = child.mutable_command_array
    changed_columns = np.sum(p_stack != c_stack, axis=0)

    assert sum(changed_columns[1:]) > 0


@pytest.mark.parametrize("repeats", range(5))
def test_pruning_mutation(sample_agraph, sample_component_generator, repeats):
    mutation = AGraphMutation(sample_component_generator,
                              command_probability=0.0,
                              node_probability=0.0,
                              parameter_probability=0.0,
                              prune_probability=1.0,
                              fork_probability=0.0)
    child = mutation(sample_agraph)
    p_stack = sample_agraph.command_array
    c_stack = child.mutable_command_array
    changes = p_stack != c_stack

    p_changes = p_stack[changes]
    c_changes = c_stack[changes]
    if p_changes.size > 0:
        np.testing.assert_array_equal(p_changes,
                                      np.full(p_changes.shape,
                                              p_changes[0]))
        np.testing.assert_array_equal(c_changes,
                                      np.full(c_changes.shape,
                                              c_changes[0]))
        assert c_changes[0] < p_changes[0]


@pytest.mark.parametrize("algo_index", [2, 3])
def test_impossible_param_or_prune_mutation(mocker, algo_index,
                                            no_param_mut_or_prune_agraph,
                                            sample_component_generator):
    type(sample_component_generator).input_x_dimension = \
        mocker.PropertyMock(return_value=1)
    input_probabilities = [0.0] * 5
    input_probabilities[algo_index] = 1.0
    mutation = AGraphMutation(sample_component_generator, *input_probabilities)

    child = mutation(no_param_mut_or_prune_agraph)
    p_stack = no_param_mut_or_prune_agraph.command_array
    c_stack = child.mutable_command_array

    np.testing.assert_array_equal(c_stack, p_stack)


def test_mutate_variable(single_variable_agraph, sample_component_generator):
    mutation = AGraphMutation(sample_component_generator,
                              command_probability=0.0,
                              node_probability=0.0,
                              parameter_probability=1.0,
                              prune_probability=0.0,
                              fork_probability=0.0)
    child = mutation(single_variable_agraph)
    p_stack = single_variable_agraph.command_array
    c_stack = child.mutable_command_array

    assert p_stack[-1, 1] != c_stack[-1, 1]
    assert p_stack[-1, 2] != c_stack[-1, 2]


def command_array_is_valid(individual):
    for i, (command, op1, op2) in enumerate(individual.command_array):
        if not IS_TERMINAL_MAP[command]:
            if op1 >= i or op2 >= i:
                return False
    return True


@pytest.mark.parametrize("repeats", range(5))
def test_fork_mutation(fork_agraph, fork_mutation, repeats):
    # np.random.seed(10)
    parent = fork_agraph
    child = fork_mutation(parent)
    print("parent:", parent)
    print("child:", child)

    assert parent.get_complexity() < child.get_complexity()
    assert command_array_is_valid(child)


@pytest.mark.parametrize("mutation_location", [1, 3, 4])
def test_fork_mutation_spaced_util_commands(mocker, spaced_fork_agraph, fork_mutation, mutation_location):
    mocker.patch.object(np.random, "choice", return_value=mutation_location)
    parent = spaced_fork_agraph
    child = fork_mutation(parent)
    print("parent:", parent)
    print("child:", child)

    assert parent.get_complexity() < child.get_complexity()
    assert command_array_is_valid(child)


def test_fork_mutation_not_enough_unutilized_commands(not_enough_unutil_fork_agraph, fork_mutation):
    parent = not_enough_unutil_fork_agraph
    child = fork_mutation(parent)
    print("parent:", parent)
    print("child:", child)

    np.testing.assert_array_equal(not_enough_unutil_fork_agraph.command_array, child.command_array)


@pytest.mark.parametrize("fork_size", [2, 3, 4])
def test_fork_mutation_many_unutilized_commands(
        mocker, many_unutil_fork_agraph, fork_mutation, fork_size):
    mocker.patch.object(np.random, "randint", return_value=fork_size)
    parent = many_unutil_fork_agraph
    child = fork_mutation(parent)
    print("parent:", parent)
    print("child:", child)

    assert parent.get_complexity() == child.get_complexity() - fork_size
    assert command_array_is_valid(child)


@pytest.mark.parametrize("mutation_location", [0, 2])
def test_fork_mutation_unutil_after_mutation_location(
        mocker, unutil_after_mutation_location_agraph, fork_mutation, mutation_location):
    mocker.patch.object(np.random, "choice", return_value=mutation_location)
    parent = unutil_after_mutation_location_agraph
    child = fork_mutation(parent)
    print("parent:", parent)
    print("child:", child)

    assert parent.get_complexity() < child.get_complexity()
    assert command_array_is_valid(child)


def test_fork_mutation_arity_one_operator_linked_to_unutilized_command(
        mocker, arity_one_op_with_two_different_params_agraph, fork_mutation):
    mocker.patch.object(np.random, "choice", return_value=3)
    parent = arity_one_op_with_two_different_params_agraph
    child = fork_mutation(parent)
    print("parent:", parent)
    print("child:", child)

    assert parent.get_complexity() < child.get_complexity()
    assert command_array_is_valid(child)
