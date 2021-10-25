"""The "Deterministic Crowding" selection

This module defines the basis of the "deterministic crowding" selection
algorithm in bingo analyses. The next generation is selected by pairing parents
with their offspring and selecting the most fit of the two.
"""
import numpy as np
from .generalized_crowding import GeneralizedCrowding


# pylint: disable=too-few-public-methods
class DeterministicCrowding(GeneralizedCrowding):
    """The class that performs deterministic crowding selection on a population
    """
    @staticmethod
    def _return_most_fit(child, parent, idx):
        import pdb;pdb.set_trace()
        if np.isnan(parent.fitness[idx]):
            return child
        if np.isnan(child.fitness[idx]):
            return parent
        return child if child.fitness[idx] < parent.fitness[idx] else parent
