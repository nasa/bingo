# Ignoring some linting rules in tests
# pylint: disable=missing-docstring
# pylint: disable=abstract-class-instantiated

import pytest
from bingo.chromosomes.chromosome import Chromosome


@pytest.fixture
def individual(mocker):
    mocker.patch.object(Chromosome, "__abstractmethods__",
                        new_callable=set)
    return Chromosome()


def test_base_chromosome_instantiation_fails():
    with pytest.raises(TypeError):
        _ = Chromosome()


def test_setting_fitness(individual):
    assert not individual.fit_set
    individual.fitness = 1
    assert individual.fit_set
    assert individual.fitness == 1


def test_copy_is_a_copy(individual):
    assert individual.copy() is not individual


def test_genetic_age_starts_at_zero(individual):
    assert individual.genetic_age == 0
    individual.genetic_age = 10
    assert individual.genetic_age == 10


def test_optimization_interface_methods_raise_not_implemented(mocker):
    mocker.patch.object(Chromosome, "__abstractmethods__", new_callable=set)

    expected_exception_str = "This Chromosome cannot be used in local " \
                             "optimization until its local optimization " \
                             "interface has been implemented"

    with pytest.raises(NotImplementedError) as exc_info:
        Chromosome().needs_local_optimization()
    assert expected_exception_str == str(exc_info.value)

    with pytest.raises(NotImplementedError) as exc_info:
        Chromosome().get_number_local_optimization_params()
    assert expected_exception_str == str(exc_info.value)

    with pytest.raises(NotImplementedError) as exc_info:
        Chromosome().set_local_optimization_params(mocker.Mock())
    assert expected_exception_str == str(exc_info.value)
