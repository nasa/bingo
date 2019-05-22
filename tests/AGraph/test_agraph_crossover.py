# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np
import pytest

from bingo.SymbolicRegression.AGraph.AGraph import AGraph
from bingo.SymbolicRegression.AGraph.ComponentGenerator import ComponentGenerator
from bingo.SymbolicRegression.AGraph.AGraphCrossover import AGraphCrossover


@pytest.fixture
def manual_constants_crossover():
    generator = ComponentGenerator(input_x_dimension=2,
                                   num_initial_load_statements=2,
                                   terminal_probability=0.4,
                                   constant_probability=0.5,
                                   automatic_constant_optimization=False)
    generator.add_operator(2)
    generator.add_operator(6)
    return AGraphCrossover(component_generator=generator)


@pytest.fixture
def manual_constants_parents():
    parent_1 = AGraph()
    parent_1.command_array = np.array([[1, 0, 0],  # sin(5.0)
                                       [1, -1, -1],
                                       [1, -1, -1],
                                       [6, 0, 0]])
    parent_1.set_local_optimization_params([5.0, ])
    parent_2 = AGraph()
    parent_2.command_array = np.array([[0, 0, 0],  # cos(x_0 + 3.0)
                                       [1, 0, 0],
                                       [2, 0, 1],
                                       [7, 2, 2]])
    parent_2.set_local_optimization_params([3.0, ])
    return parent_1, parent_2


@pytest.fixture(params=[('sample_agraph_1', 'sample_agraph_2'),
                        ('sample_agraph_2', 'sample_agraph_1')])
def crossover_parents(request):
    return (request.getfixturevalue(request.param[0]),
            request.getfixturevalue(request.param[1]))


def test_crossover_is_single_point(sample_component_generator,
                                   crossover_parents):
    np.random.seed(0)
    crossover = AGraphCrossover(sample_component_generator)
    child_1, child_2 = crossover(crossover_parents[0], crossover_parents[1])

    crossover_point_reached = False
    for c_1, c_2, p_1, p_2 in zip(child_1.command_array,
                                  child_2.command_array,
                                  crossover_parents[0].command_array,
                                  crossover_parents[1].command_array):
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


def test_crossover_genetic_age(sample_component_generator, crossover_parents):
    crossover = AGraphCrossover(sample_component_generator)
    child_1, child_2 = crossover(crossover_parents[0], crossover_parents[1])

    oldest_paraent_age = max(crossover_parents[0].genetic_age,
                             crossover_parents[1].genetic_age)

    assert child_1.genetic_age == oldest_paraent_age
    assert child_2.genetic_age == oldest_paraent_age


def test_crossover_resets_fitness(sample_component_generator,
                                  crossover_parents):
    assert crossover_parents[0].fit_set
    assert crossover_parents[1].fit_set

    crossover = AGraphCrossover(sample_component_generator)
    child_1, child_2 = crossover(crossover_parents[0], crossover_parents[1])
    assert not child_1.fit_set
    assert not child_2.fit_set
    assert child_1.fitness is None
    assert child_2.fitness is None


def test_crossover_keeps_correct_manual_constants(manual_constants_crossover,
                                                  manual_constants_parents):
    np.random.seed(0)
    parent_1, parent_2 = manual_constants_parents
    child_1, child_2 = manual_constants_crossover(parent_1, parent_2)
    np.testing.assert_array_almost_equal(child_1.constants, [5.0, 3.0])
    assert not child_2.constants


def test_crossover_makes_new_manual_constants(manual_constants_crossover,
                                              manual_constants_parents):
    np.random.seed(1)
    parent_1, parent_2 = manual_constants_parents
    child_1, child_2 = manual_constants_crossover(parent_1, parent_2)
    np.testing.assert_array_almost_equal(child_1.constants, [5.0,
                                                             99.4369621877737])
    assert not child_2.constants
