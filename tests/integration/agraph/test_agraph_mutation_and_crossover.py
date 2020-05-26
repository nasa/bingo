# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np
import pytest

from bingo.symbolic_regression.agraph.operator_definitions import *
from bingo.symbolic_regression.agraph.agraph import AGraph as pyagraph
from bingo.symbolic_regression.agraph.mutation import AGraphMutation
from bingo.symbolic_regression.agraph.crossover import AGraphCrossover
from bingo.symbolic_regression.agraph.component_generator \
    import ComponentGenerator

try:
    from bingocpp.build import symbolic_regression as bingocpp
except ImportError:
    bingocpp = None

CPP_PARAM = pytest.param("cpp",
                         marks=pytest.mark.skipif(not bingocpp,
                                                  reason='BingoCpp import '
                                                         'failure'))


@pytest.fixture(params=["python", CPP_PARAM])
def agraph_implementation(request):
    if request.param == "python":
        return pyagraph
    return bingocpp.AGraph


@pytest.fixture
def sample_component_generator():
    generator = ComponentGenerator(input_x_dimension=2,
                                   num_initial_load_statements=2,
                                   terminal_probability=0.4,
                                   constant_probability=0.5)
    generator.add_operator(2)
    generator.add_operator(6)
    return generator


def _sample_agraph_1(test_graph):  # sin(X_0 + 2.0) + 2.0
    test_graph.command_array = np.array([[VARIABLE, 0, 0],
                                         [CONSTANT, 0, 0],
                                         [ADDITION, 0, 1],
                                         [SIN, 2, 2],
                                         [ADDITION, 0, 1],
                                         [ADDITION, 3, 1]], dtype=int)
    test_graph.genetic_age = 10
    _ = test_graph.needs_local_optimization()
    test_graph.set_local_optimization_params([2.0, ])
    test_graph.fitness = 1
    return test_graph


def _sample_agraph_2(test_graph):  # sin((c_1-c_1)*X_1)
    test_graph.command_array = np.array([[VARIABLE, 1, 3],
                                         [CONSTANT, 1, 1],
                                         [SUBTRACTION, 1, 1],
                                         [MULTIPLICATION, 0, 2],
                                         [ADDITION, 0, 1],
                                         [SIN, 3, 0]], dtype=int)
    test_graph.genetic_age = 20
    _ = test_graph.needs_local_optimization()
    test_graph.set_local_optimization_params([1.0])
    test_graph.fitness = 2
    return test_graph


@pytest.fixture(params=[1, 2])
def mutation_parent(request, agraph_implementation):
    test_graph = agraph_implementation()
    if request.param == 1:
        return _sample_agraph_1(test_graph)
    return _sample_agraph_2(test_graph)


@pytest.fixture(params=[1, 2])
def crossover_parents(request, agraph_implementation):
    parent_1 = agraph_implementation()
    parent_2 = agraph_implementation()
    if request.param == 1:
        return _sample_agraph_1(parent_1), _sample_agraph_2(parent_2)
    return _sample_agraph_2(parent_1), _sample_agraph_1(parent_2)


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


def test_mutation_creates_valid_parameters(mutation_parent):
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
        child = mutation(mutation_parent)
        for row, operation in enumerate(child.command_array):
            if not IS_TERMINAL_MAP[operation[0]]:
                assert operation[1] < row
                assert operation[2] < row


def test_crossover_resets_fitness(sample_component_generator,
                                  crossover_parents):
    assert crossover_parents[0].fit_set
    assert crossover_parents[1].fit_set

    crossover = AGraphCrossover()
    child_1, child_2 = crossover(crossover_parents[0], crossover_parents[1])
    assert not child_1.fit_set
    assert not child_2.fit_set
    assert child_1.fitness is None
    assert child_2.fitness is None

