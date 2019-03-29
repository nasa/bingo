# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np
import pytest

from bingo.SymbolicRegression.AGraph.AGraphCrossover import AGraphCrossover


@pytest.fixture(params=[('sample_agraph_1', 'sample_agraph_2'),
                        ('sample_agraph_2', 'sample_agraph_1')])
def crossover_parents(request):
    return (request.getfixturevalue(request.param[0]),
            request.getfixturevalue(request.param[1]))


def test_crossover_is_single_point(crossover_parents):
    np.random.seed(0)
    crossover = AGraphCrossover()
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


def test_crossover_genetic_age(crossover_parents):
    crossover = AGraphCrossover()
    child_1, child_2 = crossover(crossover_parents[0], crossover_parents[1])

    oldest_paraent_age = max(crossover_parents[0].genetic_age,
                             crossover_parents[1].genetic_age)

    assert child_1.genetic_age == oldest_paraent_age
    assert child_2.genetic_age == oldest_paraent_age


def test_crossover_resets_fitness(crossover_parents):
    assert crossover_parents[0].fit_set
    assert crossover_parents[1].fit_set

    crossover = AGraphCrossover()
    child_1, child_2 = crossover(crossover_parents[0], crossover_parents[1])
    assert not child_1.fit_set
    assert not child_2.fit_set
    assert child_1.fitness is None
    assert child_2.fitness is None
