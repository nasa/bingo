"""The "Probabilistic Crowding" selection

This module defines the basis of the "probabilistic crowding" selection
algorithm in bingo analyses. The next generation is selected by pairing parents
with their offspring and selecting the child with a probility that it related
to the fitness of the paired child and parent.
"""
import numpy as np
from .generalized_crowding import GeneralizedCrowding


# pylint: disable=too-few-public-methods
class ProbabilisticCrowding(GeneralizedCrowding):
    """The class that performs deterministic crowding selection on a population
    """
    @staticmethod
    def _return_most_fit(child, parent):
        if np.isnan(parent.fitness):
            return child
        if np.isnan(child.fitness):
            return parent

        prob = parent.fitness / (child.fitness + parent.fitness)
        return child if np.random.random() < prob else parent

