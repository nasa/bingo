# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.AGraph.AGraph import AGraph
from bingo.AGraph import AGraphVariation
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


@pytest.fixture
def sample_agraph_1():
    test_graph = AGraph()
    test_graph.command_array = np.array([[0, 0, 0],  # sin(X_0) + 1.0
                                         [1, 0, 0],
                                         [2, 0, 1],
                                         [6, 2, 2],
                                         [2, 3, 1]])
    test_graph.genetic_age = 10
    test_graph.set_local_optimization_params([1.0, ])
    return test_graph


@pytest.fixture
def sample_agraph_2():
    test_graph = AGraph()
    test_graph.command_array = np.array([[0, 1, 3],  # sin((c_1-c_1)*X_1)
                                         [1, 1, 2],
                                         [3, 1, 1],
                                         [4, 0, 2],
                                         [6, 3, 0]], dtype=int)
    test_graph.genetic_age = 20
    test_graph.set_local_optimization_params([1.0, 1.0])
    return test_graph


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


@pytest.mark.parametrize("agraph_size,expected_error", [
    (0, ValueError),
    ("string", TypeError)
])
def test_raises_error_invalid_agraph_size_gen(agraph_size,
                                              expected_error,
                                              sample_component_generator):
    with pytest.raises(expected_error):
        _ = AGraphVariation.Generation(agraph_size, sample_component_generator)


def test_generate(sample_component_generator):
    np.random.seed(0)
    expected_command_array = np.array([[0, 1, 0],
                                       [0, 1, 1],
                                       [6, 0, 1],
                                       [6, 2, 1],
                                       [6, 0, 1],
                                       [2, 1, 4]], dtype=int)
    generate_agraph = AGraphVariation.Generation(6, sample_component_generator)
    agraph = generate_agraph()
    np.testing.assert_array_equal(agraph.command_array,
                                  expected_command_array)


@pytest.mark.parametrize("parent_1,parent_2", [
    (sample_agraph_1(), sample_agraph_2()),
    (sample_agraph_2(), sample_agraph_1()),
])
def test_crossover_is_single_point(parent_1, parent_2):
    np.random.seed(0)
    crossover = AGraphVariation.Crossover()
    child_1, child_2 = crossover(parent_1, parent_2)

    crossover_point_reached = False
    for c_1, c_2, p_1, p_2 in zip(child_1.command_array,
                                  child_2.command_array,
                                  parent_1.command_array,
                                  parent_2.command_array):
        if not crossover_point_reached:
            if np.array_equal(c_1, p_1):
                np.testing.assert_array_equal(c_2, p_2)
            elif np.array_equal(c_1, p_2):
                crossover_point_reached = True
            else:
                raise RuntimeError("Genes do not match either parent!")

        if crossover_point_reached:
            np.testing.assert_array_equal(c_1, p_2)
            np.testing.assert_array_equal(c_2, p_1)


@pytest.mark.parametrize("parent_1,parent_2", [
    (sample_agraph_1(), sample_agraph_2()),
    (sample_agraph_2(), sample_agraph_1()),
])
def test_crossover_genetic_age(parent_1, parent_2):
    crossover = AGraphVariation.Crossover()
    child_1, child_2 = crossover(parent_1, parent_2)

    oldest_paraent_age = max(parent_1.genetic_age,
                             parent_2.genetic_age)

    assert child_1.genetic_age == oldest_paraent_age
    assert child_2.genetic_age == oldest_paraent_age


@pytest.mark.parametrize("parent_1,parent_2", [
    (sample_agraph_1(), sample_agraph_2()),
    (sample_agraph_2(), sample_agraph_1()),
])
def test_crossover_resets_fitness(parent_1, parent_2):
    parent_1.fitness = 1
    parent_2.fitness = 2
    assert parent_1.fit_set
    assert parent_2.fit_set

    crossover = AGraphVariation.Crossover()
    child_1, child_2 = crossover(parent_1, parent_2)
    assert not child_1.fit_set
    assert not child_2.fit_set
    assert child_1.fitness is None
    assert child_2.fitness is None


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
        _ = AGraphVariation.Mutation(sample_component_generator,
                                     *input_probabilities)


@pytest.mark.parametrize("parent", [sample_agraph_1(),
                                    sample_agraph_2()])
def test_mutation_genetic_age(parent, sample_component_generator):
    mutation = AGraphVariation.Mutation(sample_component_generator)
    child = mutation(parent)
    assert child.genetic_age == parent.genetic_age


@pytest.mark.parametrize("parent", [sample_agraph_1(),
                                    sample_agraph_2()])
def test_mutation_resets_fitness(parent, sample_component_generator):
    parent.fitness = 1
    assert parent.fit_set

    mutation = AGraphVariation.Mutation(sample_component_generator)
    child = mutation(parent)
    assert not child.fit_set
    assert child.fitness is None


@pytest.mark.parametrize("parent", [sample_agraph_1(),
                                    sample_agraph_2()])
@pytest.mark.parametrize("algo_index", range(3))
def test_single_point_mutations(parent, algo_index,
                                sample_component_generator):
    np.random.seed(0)
    input_probabilities = [0.0]*4
    input_probabilities[algo_index] = 1.0
    mutation = AGraphVariation.Mutation(sample_component_generator,
                                        *input_probabilities)

    for _ in range(5):
        child = mutation(parent)
        p_stack = parent.command_array
        c_stack = child.command_array
        changed_commands = np.sum(np.max(p_stack != c_stack, axis=1))

        assert changed_commands == 1


@pytest.mark.parametrize("parent", [sample_agraph_1(),
                                    sample_agraph_2()])
@pytest.mark.parametrize("algo_index,expected_node_mutation", [
    (1, True),
    (2, False),
    (3, False)
])
def test_mutation_of_nodes(parent, sample_component_generator, algo_index,
                           expected_node_mutation):
    np.random.seed(0)
    input_probabilities = [0.0] * 4
    input_probabilities[algo_index] = 1.0
    mutation = AGraphVariation.Mutation(sample_component_generator,
                                        *input_probabilities)

    for _ in range(5):
        child = mutation(parent)
        p_stack = parent.command_array
        c_stack = child.command_array
        changed_columns = np.sum(p_stack != c_stack, axis=0)

        if expected_node_mutation:
            assert changed_columns[0] == 1
        else:
            assert changed_columns[0] == 0


@pytest.mark.parametrize("parent", [sample_agraph_1(),
                                    sample_agraph_2()])
@pytest.mark.parametrize("algo_index", [2, 3])
def test_mutation_of_parameters(parent, sample_component_generator,
                                algo_index):
    np.random.seed(0)
    input_probabilities = [0.0] * 4
    input_probabilities[algo_index] = 1.0
    mutation = AGraphVariation.Mutation(sample_component_generator,
                                        *input_probabilities)

    for _ in range(5):
        child = mutation(parent)
        p_stack = parent.command_array
        c_stack = child.command_array
        changed_columns = np.sum(p_stack != c_stack, axis=0)

        assert sum(changed_columns[1:]) > 0


@pytest.mark.parametrize("parent", [sample_agraph_1(),
                                    sample_agraph_2()])
def test_pruning_mutation(parent, sample_component_generator):
    np.random.seed(10)
    mutation = AGraphVariation.Mutation(sample_component_generator,
                                        command_probability=0.0,
                                        node_probability=0.0,
                                        parameter_probability=0.0,
                                        prune_probability=1.0)
    for _ in range(5):
        child = mutation(parent)
        p_stack = parent.command_array
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
    mutation = AGraphVariation.Mutation(sample_component_generator,
                                        command_probability=0.0,
                                        node_probability=0.0,
                                        parameter_probability=0.0,
                                        prune_probability=1.0)
    for _ in range(5):
        child = mutation(terminal_only_agraph)
        p_stack = terminal_only_agraph.command_array
        c_stack = child.command_array
        np.testing.assert_array_equal(p_stack, c_stack)
