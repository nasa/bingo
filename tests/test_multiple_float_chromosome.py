# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest

from bingo.MultipleFloats import MultipleFloatChromosome,\
                                 MultipleFloatChromosomeGenerator

@pytest.fixture
def list_of_floats():
    return [float(i) for i in range(10)]

@pytest.fixture
def opt_individual(list_of_floats):
    return MultipleFloatChromosome(list_of_floats, [1, 2, 3])

@pytest.fixture
def individual(list_of_floats):
    return MultipleFloatChromosome(list_of_floats)

def test_only_accepts_floats_for_value_list():
    list_of_values = [str(i) for i in range (10)]
    with pytest.raises(ValueError):
        chromosome = MultipleFloatChromosome(list_of_values)

def test_accpets_valid_indicies(list_of_floats):
    with pytest.raises(ValueError):
        chromosome = MultipleFloatChromosome(list_of_floats, [6, 1, 22])
    with pytest.raises(ValueError):
        chromosome = MultipleFloatChromosome(list_of_floats, ['a', 2, 3])

# TODO: test for replicate indicies (possibly use set to raise error or convert iterables to set)

def test_generator_function_produces_floats():
    def bad_function():
        return "This is not good"
    with pytest.raises(ValueError):
        generator = MultipleFloatChromosomeGenerator(bad_function, 10)

def test_needs_local_optimization(opt_individual, individual):
    assert not individual.needs_local_optimization()
    assert opt_individual.needs_local_optimization()

def test_get_local_optimization_params(opt_individual):
    assert opt_individual.get_number_local_optimization_params() == 3

def test_set_local_optimization_params(opt_individual):
    params = [0, 0, 0]
    opt_individual.set_local_optimization_params(params)
    assert opt_individual.list_of_values[1:4] == params
