# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import os
from collections import namedtuple
from time import sleep, time
import numpy as np

from bingo.evolutionary_optimizers.evolutionary_optimizer import (
    EvolutionaryOptimizer,
    load_evolutionary_optimizer_from_file,
)
from bingo.stats.hall_of_fame import HallOfFame


DummyIndv = namedtuple(
    "DummyIndv",
    [
        "fitness",
    ],
)


class DummyDiagnostics:
    def __init__(self):
        self.summary = 0

    def get_log_header(self):
        return "a"

    def get_log_stats(self):
        return [0]


class DummyEO(EvolutionaryOptimizer):
    def __init__(self, convergence_rate, hall_of_fame=None, test_function=None):
        self.best_fitness = 1.0
        self.convergence_rate = convergence_rate
        super().__init__(hall_of_fame, test_function)

    def _do_evolution(self, num_generations):
        self.best_fitness *= self.convergence_rate
        self.generational_age += num_generations

    def get_best_fitness(self):
        return self.best_fitness

    def get_best_individual(self):
        return DummyIndv(self.best_fitness)

    def get_fitness_evaluation_count(self):
        return self.generational_age * 2

    def get_ea_diagnostic_info(self):
        return DummyDiagnostics()

    def _get_potential_hof_members(self):
        return [DummyIndv(self.best_fitness)]


def get_gamma_params(mean, std):
    scale = std**2 / mean
    shape = mean**2 / std**2
    return shape, scale


class DummyVariableTimeEO(EvolutionaryOptimizer):
    def __init__(
        self,
        evo_time_mean,
        evo_time_std,
        evo_time_slowdown,
        hall_of_fame=None,
        test_function=None,
    ):
        self._evo_mean = evo_time_mean
        self._evo_std = evo_time_std
        self._evo_slowdown = evo_time_slowdown
        super().__init__(hall_of_fame, test_function)

    def _do_evolution(self, num_generations):
        wait_time = 0
        for _ in range(num_generations):
            wait_time += np.random.gamma(
                *get_gamma_params(self._evo_mean, self._evo_std)
            )
            self._evo_mean += self._evo_slowdown
            self.generational_age += 1
        sleep(wait_time)

    def get_best_fitness(self):
        return 1.0

    def get_best_individual(self):
        return DummyIndv(1.0)

    def get_fitness_evaluation_count(self):
        return 0

    def get_ea_diagnostic_info(self):
        return DummyDiagnostics()

    def _get_potential_hof_members(self):
        return [DummyIndv(1.0)]


@pytest.fixture
def converging_eo():
    return DummyEO(0.5)


@pytest.fixture
def stale_eo():
    return DummyEO(1.0)


def test_run_until_absolute_convergence(converging_eo):
    optimization_result = converging_eo.evolve_until_convergence(
        max_generations=10,
        convergence_check_frequency=1,
        fitness_threshold=0.126,
        stagnation_generations=10,
    )
    assert optimization_result.success
    assert optimization_result.status == 0
    assert optimization_result.ngen == 3
    assert optimization_result.fitness == pytest.approx(0.125)


def test_run_until_time_limit(converging_eo):
    optimization_result = converging_eo.evolve_until_convergence(
        max_generations=10,
        convergence_check_frequency=1,
        fitness_threshold=0.126,
        stagnation_generations=10,
        max_time=0.0,
    )
    assert not optimization_result.success
    assert optimization_result.status == 4
    assert optimization_result.ngen == 0


def test_run_min_generations_converge(converging_eo):
    optimization_result = converging_eo.evolve_until_convergence(
        max_generations=10,
        min_generations=25,
        convergence_check_frequency=1,
        fitness_threshold=0.126,
        stagnation_generations=10,
    )
    assert optimization_result.status == 0
    assert optimization_result.ngen == 25


def test_run_min_generations_stagnate(stale_eo):
    optimization_result = stale_eo.evolve_until_convergence(
        max_generations=10,
        min_generations=25,
        convergence_check_frequency=1,
        fitness_threshold=0.126,
        stagnation_generations=10,
    )
    assert optimization_result.status == 1
    assert optimization_result.ngen == 25


def test_num_gens_taken_in_optimization(converging_eo):
    optimization_result = converging_eo.evolve_until_convergence(
        max_generations=10,
        convergence_check_frequency=1,
        fitness_threshold=0.26,
        stagnation_generations=10,
    )
    assert optimization_result.ngen == 2
    optimization_result = converging_eo.evolve_until_convergence(
        max_generations=10,
        convergence_check_frequency=1,
        fitness_threshold=0.126,
        stagnation_generations=10,
    )
    assert optimization_result.ngen == 1


def test_run_convergence_check_chunks(converging_eo):
    optimization_result = converging_eo.evolve_until_convergence(
        max_generations=100,
        convergence_check_frequency=5,
        fitness_threshold=0.126,
        stagnation_generations=10,
    )
    assert optimization_result.ngen == 15


def test_run_until_stagnation(stale_eo):
    optimization_result = stale_eo.evolve_until_convergence(
        max_generations=10,
        convergence_check_frequency=1,
        fitness_threshold=0.126,
        stagnation_generations=5,
    )
    assert not optimization_result.success
    assert optimization_result.status == 1
    assert optimization_result.ngen == 5
    assert optimization_result.fitness == pytest.approx(1.0)


def test_run_until_max_steps(converging_eo):
    optimization_result = converging_eo.evolve_until_convergence(
        max_generations=2,
        convergence_check_frequency=1,
        fitness_threshold=0.126,
    )
    assert not optimization_result.success
    assert optimization_result.status == 2
    assert optimization_result.ngen == 2
    assert optimization_result.fitness == pytest.approx(0.25)


@pytest.mark.parametrize("min_generations", [0, 2])
def test_run_until_max_fitness_evaluations(converging_eo, min_generations):
    optimization_result = converging_eo.evolve_until_convergence(
        max_generations=10,
        min_generations=min_generations,
        convergence_check_frequency=1,
        fitness_threshold=0.126,
        max_fitness_evaluations=4,
    )
    assert not optimization_result.success
    assert optimization_result.status == 3
    assert optimization_result.ngen == 2
    assert optimization_result.fitness == pytest.approx(0.25)


@pytest.mark.parametrize(
    "invalid_arg_dict",
    [
        {"max_generations": 0},
        {"convergence_check_frequency": 0},
        {"min_generations": -1},
    ],
)
def test_raises_error_on_invalid_input(converging_eo, invalid_arg_dict):
    arg_dict = {
        "max_generations": 2,
        "convergence_check_frequency": 1,
        "absolute_error_threshold": 0.126,
        "stagnation_generations": 10,
        "min_generations": 0,
    }
    arg_dict.update(invalid_arg_dict)
    with pytest.raises(ValueError):
        _ = converging_eo.evolve_until_convergence(**arg_dict)


def test_hof_update(mocker):
    hof = mocker.Mock()
    mocker.spy(hof, "update")
    eo = DummyEO(1.0, hof)
    eo.evolve(1)
    eo.evolve(1)
    eo.evolve(1)
    assert hof.update.call_count == 3
    for call in hof.update.call_args_list:
        called_with_population = call[0][0]
        assert len(called_with_population) == 1
        assert called_with_population[0].fitness == 1.0


def test_dump_then_load(converging_eo):
    converging_eo.evolve(1)
    converging_eo.dump_to_file("testing_dump_and_load.pkl")
    converging_eo.evolve(1)
    converging_eo = load_evolutionary_optimizer_from_file("testing_dump_and_load.pkl")

    assert 1 == converging_eo.generational_age
    converging_eo.evolve(1)
    assert 2 == converging_eo.generational_age

    os.remove("testing_dump_and_load.pkl")


def test_dynamic_checkpointing(converging_eo):
    _ = converging_eo.evolve_until_convergence(
        max_generations=10,
        convergence_check_frequency=1,
        fitness_threshold=0.126,
        stagnation_generations=5,
        checkpoint_base_name="test_dynamic_checkpoint",
    )
    for i in range(4):
        expected_file_name = "test_dynamic_checkpoint_{}.pkl".format(i)
        assert os.path.isfile(expected_file_name)
        os.remove(expected_file_name)


def test_limited_checkpoint_num(converging_eo):
    _ = converging_eo.evolve_until_convergence(
        max_generations=10,
        convergence_check_frequency=1,
        fitness_threshold=0.126,
        stagnation_generations=5,
        checkpoint_base_name="test_limited_checkpoint_num",
        num_checkpoints=2,
    )

    for i in range(4):
        expected_file_name = "test_limited_checkpoint_num_{}.pkl".format(i)
        if i < 2:
            assert not os.path.isfile(expected_file_name)
        else:
            assert os.path.isfile(expected_file_name)
            os.remove(expected_file_name)


def test_hof_integration(converging_eo):
    hof = HallOfFame(1)
    eo_w_hof = DummyEO(0.5, hof)
    _ = eo_w_hof.evolve_until_convergence(
        max_generations=10,
        convergence_check_frequency=1,
        fitness_threshold=0.126,
        stagnation_generations=10,
    )
    assert hof[0].fitness == 0.125


def test_that_test_function_is_used(mocker):
    mocked_test_function = mocker.Mock(return_value=1.0)
    convergeing_eo_with_test_func = DummyEO(0.5, test_function=mocked_test_function)
    _ = convergeing_eo_with_test_func.evolve_until_convergence(
        max_generations=2,
        convergence_check_frequency=1,
        fitness_threshold=0.126,
    )
    assert mocked_test_function.call_count == 3


def test_strict_time_limit():
    np.random.seed(0)
    eo = DummyVariableTimeEO(0.01, 0.02, 0)
    max_time = 1.0
    t0 = time()
    optim_result = eo.evolve_until_convergence(
        max_generations=9e99,
        convergence_check_frequency=20,
        fitness_threshold=0,
        max_time=max_time,
    )
    elapsed_time = time() - t0

    assert elapsed_time < max_time
    assert optim_result.status == 5
