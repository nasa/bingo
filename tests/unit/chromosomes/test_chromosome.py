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

    with pytest.raises(NotImplementedError):
        Chromosome().needs_local_optimization()

    with pytest.raises(NotImplementedError):
        Chromosome().get_number_local_optimization_params()

    with pytest.raises(NotImplementedError):
        Chromosome().set_local_optimization_params(mocker.Mock())
