# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring

import pytest
from collections import namedtuple

try:
    from bingo.evolutionary_optimizers.parallel_archipelago \
        import ParallelArchipelago, load_parallel_archipelago_from_file
    PAR_ARCH_LOADED = True
except ImportError:
    PAR_ARCH_LOADED = False


DummyChromosome = namedtuple("DummyChromosome", ["fitness"])


@pytest.mark.parametrize("non_blocking", [True, False])
@pytest.mark.parametrize("sync_frequency", [10, 12])
@pytest.mark.parametrize("n_steps", [120, 5, 1])
def test_step_through_generations(mocker, non_blocking, sync_frequency, n_steps):
    expected_sync_freq = sync_frequency
    if n_steps < sync_frequency:
        expected_sync_freq = 1
    mocked_island = mocker.Mock()
    type(mocked_island).generational_age = \
        mocker.PropertyMock(side_effect=list(range(expected_sync_freq, 200,
                                                    expected_sync_freq)))
    arch = ParallelArchipelago(mocked_island, non_blocking=non_blocking,
                               sync_frequency=sync_frequency)

    arch._step_through_generations(n_steps)

    if non_blocking:
        assert mocked_island.evolve.call_count == n_steps // expected_sync_freq
    else:
        mocked_island.evolve.assert_called_once_with(n_steps,
                                                     hall_of_fame_update=False,
                                                     suppress_logging=True)


def test_best_individual(mocker):
    best_indv = DummyChromosome(fitness=0)
    mocked_island_with_best = mocker.Mock()
    mocked_island_with_best.get_best_individual.return_value = best_indv
    arch = ParallelArchipelago(mocked_island_with_best)
    assert arch.get_best_individual() == best_indv


def test_fitness_eval_count(mocker):
    num_evals = 3
    mocked_island = mocker.Mock()
    mocked_island.get_fitness_evaluation_count.return_value = num_evals
    arch = ParallelArchipelago(mocked_island)
    assert arch.get_fitness_evaluation_count() == num_evals


def test_ea_diagnostics(mocker):
    diagnostics = 3
    mocked_island = mocker.Mock()
    mocked_island.get_ea_diagnostic_info.return_value = diagnostics
    arch = ParallelArchipelago(mocked_island)
    assert arch.get_ea_diagnostic_info() == diagnostics
