import pytest

from bingo.chromosomes.chromosome import Chromosome


def test_init(mocker):
    mocker.patch.object(Chromosome, "__abstractmethods__", new_callable=set)

    chromosome = Chromosome(genetic_age=10, fitness=1e-5, fit_set=True)
    assert chromosome.genetic_age == 10
    assert chromosome.fitness == 1e-5
    assert chromosome.fit_set


def test_get_and_set_fitness(mocker):
    mocker.patch.object(Chromosome, "__abstractmethods__", new_callable=set)

    chromosome = Chromosome()
    assert not chromosome.fit_set

    chromosome.fitness = 1
    # setting fitness should also set fit_set to True
    assert chromosome.fitness == 1
    assert chromosome.fit_set


def test_get_and_set_genetic_age(mocker):
    mocker.patch.object(Chromosome, "__abstractmethods__", new_callable=set)

    chromosome = Chromosome()
    chromosome.genetic_age = 3
    assert chromosome.genetic_age == 3


def test_get_and_set_fit_set(mocker):
    mocker.patch.object(Chromosome, "__abstractmethods__", new_callable=set)

    chromosome = Chromosome()
    chromosome.fit_set = True
    assert chromosome.fit_set


def test_copy(mocker):
    mocker.patch.object(Chromosome, "__abstractmethods__", new_callable=set)

    normal_chromosome = Chromosome(genetic_age=10, fitness=1e-5, fit_set=True)
    copied_chromosome = normal_chromosome.copy()

    # shouldn't be same object
    assert copied_chromosome is not normal_chromosome
    # should have same properties
    assert copied_chromosome.__dict__ == normal_chromosome.__dict__


def test_optimization_interface_methods_raise_not_implemented(mocker):
    mocker.patch.object(Chromosome, "__abstractmethods__", new_callable=set)

    with pytest.raises(NotImplementedError):
        Chromosome().needs_local_optimization()

    with pytest.raises(NotImplementedError):
        Chromosome().get_number_local_optimization_params()

    with pytest.raises(NotImplementedError):
        Chromosome().set_local_optimization_params(mocker.Mock())
