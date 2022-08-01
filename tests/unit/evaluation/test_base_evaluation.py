# Ignoring some linting rules in tests
# pylint: disable=missing-docstring
from multiprocessing import Pool

import pytest

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
        self.eval_count += 1
        return indv.fitness + 1


@pytest.mark.parametrize("n_proc", [False, 2])
def test_evaluation_finds_fitness_for_individuals_that_need_it(n_proc):
    fit_not_set_idx = [2, 4, 6, 8]
    population = [AGraph() for _ in range(10)]
    for i, indv in enumerate(population):
        indv.fitness = i
        if i in fit_not_set_idx:
            indv.fit_set = False
        else:
            indv.fit_set = True
    fitness_fn = FitnessInc()
    evaluation = Evaluation(fitness_fn, multiprocess=n_proc)

    evaluation(population)

    for i, indv in enumerate(population):
        if i in fit_not_set_idx:
            assert indv.fitness == i + 1
        else:
            assert indv.fitness == i
    assert evaluation.eval_count == len(fit_not_set_idx)
    assert fitness_fn.eval_count == len(fit_not_set_idx)


@pytest.mark.parametrize("n_proc", [False, 2])
def test_evaluation_redundant_evaluation(n_proc):
    n_indv = 5
    population = [AGraph() for _ in range(n_indv)]
    for i, indv in enumerate(population):
        indv.fitness = i
        if i in [2, 3]:
            indv.fit_set = False
        else:
            indv.fit_set = True
    fitness_fn = FitnessInc()
    evaluation = Evaluation(fitness_fn, redundant=True, multiprocess=n_proc)

    evaluation(population)

    for i, indv in enumerate(population):
        assert indv.fitness == i + 1
    assert evaluation.eval_count == n_indv
    assert fitness_fn.eval_count == n_indv


@pytest.mark.parametrize("n_proc", [False, 2])
def test_evaluation_multiprocessing(mocker, n_proc):
    n_indv = 10
    population = [AGraph() for _ in range(n_indv)]
    for i, indv in enumerate(population):
        indv.fitness = i
        indv.fit_set = False  # IMPORTANT: have to do this after setting fitness
    pool = mocker.patch("bingo.evaluation.evaluation.Pool",
                        new=mocker.Mock(wraps=Pool))
    fitness_fn = FitnessInc()
    evaluation = Evaluation(fitness_fn, multiprocess=n_proc)

    evaluation(population)

    if n_proc:
        pool.assert_called_with(processes=n_proc)
    else:
        assert not pool.called

    for i, indv in enumerate(population):
        assert indv.fitness == i + 1
    assert evaluation.eval_count == n_indv
    assert fitness_fn.eval_count == n_indv
