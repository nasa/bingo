# Ignoring some linting rules in tests
# pylint: disable=missing-docstring
import pytest

from multiprocessing import Pool

from bingo.evaluation.evaluation import Evaluation
from bingo.evaluation.fitness_function import FitnessFunction
from bingo.symbolic_regression import AGraph


def test_evaluation_has_accessor_to_fitness_function_eval_count(mocker):
    mocked_fit_function = mocker.Mock()
    mocked_fit_function.eval_count = 10
    evaluation = Evaluation(mocked_fit_function)

    assert evaluation.eval_count == 10

    evaluation.eval_count = 100
    assert mocked_fit_function.eval_count == 100


# NOTE (David Randall): multiprocess uses pickle which hates Mock objects,
# using other objects in place of mocked objects where needed
class FitnessInc(FitnessFunction):
    def __call__(self, indv):
        return indv.fitness + 1


@pytest.mark.parametrize("n_proc", [False, 2])
def test_evaluation_finds_fitness_for_individuals_that_need_it(n_proc):
    population = [AGraph() for _ in range(10)]
    for i, indv in enumerate(population):
        indv.fitness = i
        if i in [2, 4, 6, 8]:
            indv.fit_set = False
        else:
            indv.fit_set = True

    evaluation = Evaluation(FitnessInc(), multiprocess=n_proc)
    evaluation(population)

    for i, indv in enumerate(population):
        if i in [2, 4, 6, 8]:
            assert indv.fitness == i + 1
        else:
            assert indv.fitness == i


@pytest.mark.parametrize("n_proc", [False, 2])
def test_evaluation_redundant_evaluation(n_proc):
    population = [AGraph() for _ in range(5)]
    for i, indv in enumerate(population):
        indv.fitness = i
        if i in [2, 3]:
            indv.fit_set = False
        else:
            indv.fit_set = True
    evaluation = Evaluation(FitnessInc(), redundant=True, multiprocess=n_proc)

    evaluation(population)

    for i, indv in enumerate(population):
        assert population[i].fitness == i + 1


@pytest.mark.parametrize("n_proc", [False, 2])
def test_evaluation_multiprocessing(mocker, n_proc):
    population = [AGraph() for _ in range(10)]
    for i, indv in enumerate(population):
        indv.fitness = i
        indv.fit_set = False  # IMPORTANT: have to do this after setting fitness
    pool = mocker.patch("bingo.evaluation.evaluation.Pool",
                        new=mocker.Mock(wraps=Pool))
    evaluation = Evaluation(FitnessInc(), multiprocess=n_proc)

    evaluation(population)

    if n_proc:
        pool.assert_called_with(processes=n_proc)
    else:
        assert not pool.called

    for i, indv in enumerate(population):
        assert population[i].fitness == i + 1
