# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest

from bingo.chromosomes.multiple_floats import MultipleFloatChromosome,\
                                              MultipleFloatChromosomeGenerator

DUMMY_VALUE = 999


def test_multiple_float_needs_local_optimization():
    chromosome_with_opt = MultipleFloatChromosome([1, 2, 3], [1])
    chromosome_without_opt = MultipleFloatChromosome([1, 2, 3])

    assert chromosome_with_opt. needs_local_optimization()
    assert not chromosome_without_opt. needs_local_optimization()


@pytest.mark.parametrize("num_params", range(3))
def test_getting_number_of_optimization_params(num_params):
    needs_opt_list = list(range(num_params))
    chromosome = MultipleFloatChromosome([1, 2, 3], needs_opt_list)
    assert chromosome.get_number_local_optimization_params() == num_params


def test_setting_optimization_params():
    needs_opt_list = [1, 3, 5]
    chromosome = MultipleFloatChromosome([0] * 6, needs_opt_list)
    chromosome.set_local_optimization_params([1, 1, 1])
    for i in needs_opt_list:
        assert chromosome.values[i] == 1


@pytest.mark.parametrize("bad_opt_list", [[-1, 0],
                                          [0, 4],
                                          [0, 0.5]])
def test_generator_errors_with_bad_opt_list(mocker, bad_opt_list):
    with pytest.raises(ValueError):
        _ = MultipleFloatChromosomeGenerator(mocker.Mock(),
                                             values_per_chromosome=4,
                                             needs_opt_list=bad_opt_list)


def test_generator():
    def dummy_function():
        return DUMMY_VALUE
    needs_opt_list = [1, 3, 5]
    generator = MultipleFloatChromosomeGenerator(dummy_function,
                                                 values_per_chromosome=6,
                                                 needs_opt_list=needs_opt_list)
    chromosome = generator()
    chromosome.set_local_optimization_params([1, 1, 1])
    for i in needs_opt_list:
        assert chromosome.values[i] == 1


# LIST_SIZE = 10
# OPT_INDEX_START = 1
# OPT_INDEX_STOP = 3
#
# @pytest.fixture
# def list_of_floats():
#     return [float(i) for i in range(10)]
#
# @pytest.fixture
# def opt_individual(list_of_floats):
#     opt_list = [i for i in range(OPT_INDEX_START, OPT_INDEX_STOP+1)]
#     return MultipleFloatChromosome(list_of_floats, opt_list)
#
# @pytest.fixture
# def individual(list_of_floats):
#     return MultipleFloatChromosome(list_of_floats)
#
# def return_float_one():
#     return 1.0
#
# def random_float_function():
#     return np.random.random()*10
#
# def test_accpets_valid_indicies(list_of_floats):
#     with pytest.raises(ValueError):
#         MultipleFloatChromosomeGenerator(return_float_one,
#                                          LIST_SIZE, [6, 1, 22])
#     with pytest.raises(ValueError):
#         MultipleFloatChromosomeGenerator(return_float_one,
#                                          LIST_SIZE, ['a', 2, 3])
#
# def test_removes_duplicates():
#     list_of_dupes = [6, 1, 1, 2, 3, 4, 5, 5, 5, 0, 5, 5, 5, 5, 5, 9, 8, 7, 5]
#     generator = MultipleFloatChromosomeGenerator(return_float_one,
#                                                  LIST_SIZE,
#                                                  list_of_dupes)
#     assert generator._needs_opt_list == [i for i in range(LIST_SIZE)]
#
# def test_generator_function_produces_floats():
#     def bad_function():
#         return "This is not good"
#     with pytest.raises(ValueError):
#         MultipleFloatChromosomeGenerator(bad_function, LIST_SIZE)
#
# def test_needs_local_optimization(opt_individual, individual):
#     assert not individual.needs_local_optimization()
#     assert opt_individual.needs_local_optimization()
#
# def test_get_local_optimization_params(opt_individual):
#     assert opt_individual.get_number_local_optimization_params() == 3
#
# def test_set_local_optimization_params(opt_individual):
#     params = [0 for _ in range(OPT_INDEX_START, OPT_INDEX_STOP+1)]
#     opt_individual.set_local_optimization_params(params)
#     new_values = opt_individual.values[OPT_INDEX_START:OPT_INDEX_STOP + 1]
#     assert new_values == params
#
# def test_generate_individual_with_opt_list():
#     opt_list = [1, 2, 3]
#     generator = MultipleFloatChromosomeGenerator(random_float_function,
#                                                  LIST_SIZE,
#                                                  opt_list)
#     individual = generator()
#     assert individual._needs_opt_list == opt_list
#
# def test_generate_individual_with_no_opt_list():
#     generator = MultipleFloatChromosomeGenerator(random_float_function,
#                                                  LIST_SIZE)
#     individual = generator()
#     assert not individual._needs_opt_list
