# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np
import pytest

import bingo.SymbolicRegression.AGraph.AGraph as AcyclicGraph
from bingo.SymbolicRegression.AGraph.AGraph import AGraph
from bingo.SymbolicRegression.AGraph.AGraphMutation import AGraphMutation
from bingo.SymbolicRegression.AGraph.ComponentGenerator \
    import ComponentGenerator

NODE_TYPE = 0
PARAM_1 = 1
PARAM_2 = 2

@pytest.fixture
def terminal_only_agraph():
    test_graph = AGraph()
    test_graph.command_array = np.array([[0, 1, 3],  # X_0
                                         [1, 1, 2],
                                         [3, 1, 1],
                                         [4, 0, 2],
                                         [0, 0, 0]], dtype=int)
    test_graph.genetic_age = 1
    test_graph.set_local_optimization_params([1.0, 1.0])
    return test_graph


@pytest.fixture(params=['sample_agraph_1', 'sample_agraph_2'])
def mutation_parent(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize("prob,expected_error", [
    (-1, ValueError),
    (2.5, ValueError),
    ("string", TypeError)
])
@pytest.mark.parametrize("prob_index", range(4))
def test_raises_error_invalid_mutation_probability(prob,
                                                   expected_error,
                                                   prob_index,
                                                   sample_component_generator):
    input_probabilities = [0.25]*4
    input_probabilities[prob_index] = prob
    with pytest.raises(expected_error):
        _ = AGraphMutation(sample_component_generator, *input_probabilities)


def test_mutation_genetic_age(mutation_parent, sample_component_generator):
    mutation = AGraphMutation(sample_component_generator)
    child = mutation(mutation_parent)
    assert child.genetic_age == mutation_parent.genetic_age


def test_mutation_resets_fitness(mutation_parent, sample_component_generator):
    assert mutation_parent.fit_set

    mutation = AGraphMutation(sample_component_generator)
    child = mutation(mutation_parent)
    assert not child.fit_set
    assert child.fitness is None


@pytest.mark.parametrize("algo_index", range(3))
def test_single_point_mutations(mutation_parent, algo_index,
                                sample_component_generator):
    np.random.seed(0)
    input_probabilities = [0.0]*4
    input_probabilities[algo_index] = 1.0
    mutation = AGraphMutation(sample_component_generator, *input_probabilities)

    for _ in range(5):
        child = mutation(mutation_parent)
        p_stack = mutation_parent.command_array
        c_stack = child.command_array
        changed_commands = 0
        for p, c in zip(p_stack, c_stack):
            if (p != c).any():
                if p[0] != 1 or c[0] != 1:
                    changed_commands += 1
        if changed_commands != 1:
            print("parent\n", p_stack)
            print("child\n", c_stack)
        assert changed_commands == 1


@pytest.mark.parametrize("algo_index,expected_node_mutation", [
    (1, True),
    (2, False),
    (3, False)
])
def test_mutation_of_nodes(mutation_parent, sample_component_generator,
                           algo_index, expected_node_mutation):
    np.random.seed(0)
    input_probabilities = [0.0] * 4
    input_probabilities[algo_index] = 1.0
    mutation = AGraphMutation(sample_component_generator, *input_probabilities)

    for _ in range(5):
        child = mutation(mutation_parent)
        p_stack = mutation_parent.command_array
        c_stack = child.command_array
        changed_columns = np.sum(p_stack != c_stack, axis=0)

        if expected_node_mutation:
            assert changed_columns[0] == 1
        else:
            assert changed_columns[0] == 0


@pytest.mark.parametrize("algo_index", [2, 3])
def test_mutation_of_parameters(mutation_parent, sample_component_generator,
                                algo_index):
    np.random.seed(0)
    input_probabilities = [0.0] * 4
    input_probabilities[algo_index] = 1.0
    mutation = AGraphMutation(sample_component_generator, *input_probabilities)

    for _ in range(5):
        child = mutation(mutation_parent)
        p_stack = mutation_parent.command_array
        c_stack = child.command_array
        changed_columns = np.sum(p_stack != c_stack, axis=0)

        assert sum(changed_columns[1:]) > 0


def test_pruning_mutation(mutation_parent, sample_component_generator):
    np.random.seed(10)
    mutation = AGraphMutation(sample_component_generator,
                              command_probability=0.0,
                              node_probability=0.0,
                              parameter_probability=0.0,
                              prune_probability=1.0)
    for _ in range(5):
        child = mutation(mutation_parent)
        p_stack = mutation_parent.command_array
        c_stack = child.command_array
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


def test_pruning_mutation_on_unprunable_agraph(terminal_only_agraph,
                                               sample_component_generator):
    np.random.seed(10)
    mutation = AGraphMutation(sample_component_generator,
                              command_probability=0.0,
                              node_probability=0.0,
                              parameter_probability=0.0,
                              prune_probability=1.0)
    for _ in range(5):
        child = mutation(terminal_only_agraph)
        p_stack = terminal_only_agraph.command_array
        c_stack = child.command_array
        np.testing.assert_array_equal(p_stack, c_stack)


def test_mutation_creates_valid_parameters(sample_agraph_1):
    comp_generator = ComponentGenerator(input_x_dimension=2,
                                        num_initial_load_statements=2,
                                        terminal_probability=0.4,
                                        constant_probability=0.5)
    for operator in range(2, 13):
        comp_generator.add_operator(operator)
    np.random.seed(0)
    mutation = AGraphMutation(comp_generator,
                              command_probability=0.0,
                              node_probability=0.0,
                              parameter_probability=1.0,
                              prune_probability=0.0)
    for _ in range(20):
        child = mutation(sample_agraph_1)
        for row, operation in enumerate(child.command_array):
            if not AcyclicGraph.IS_TERMINAL_MAP[operation[NODE_TYPE]]:
                assert operation[PARAM_1] < row
                assert operation[PARAM_2] < row
