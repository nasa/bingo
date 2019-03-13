# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.AtomicPotentialRegression import PairwiseAtomicPotential, \
                                            PairwiseAtomicTrainingData


class SampleTrainingData:
    def __init__(self, r, potential_energy, config_lims_r):
        self.r = r
        self.potential_energy = potential_energy
        self.config_lims_r = config_lims_r


@pytest.fixture()
def dummy_training_data():
    r = np.ones((10, 1))
    potential_energy = np.arange(1, 5)
    config_lims_r = [0, 1, 3, 6, 10]
    return SampleTrainingData(r, potential_energy, config_lims_r)


def test_pairwise_potential_regression(dummy_sum_equation,
                                       dummy_training_data):
    regressor = PairwiseAtomicPotential(dummy_training_data)
    fitness = regressor(dummy_sum_equation)
    np.testing.assert_almost_equal(fitness, 0)
