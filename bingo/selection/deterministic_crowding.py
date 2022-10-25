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

        Parameters
        ----------
        population : list of Chromosome
            The population on which to perform selection. This population
            includes both the parent and child populations, with the parents in
            the first half and the children in the latter half
        target_population_size : int
            The size of the next generation

        Returns
        -------
        population : list of Chromosome
            The newly selected generation of chromosomes
    """
    @staticmethod
    def _return_most_fit(child, parent):
        if np.isnan(parent.fitness):
            return child
        if np.isnan(child.fitness):
            return parent
        return child if child.fitness < parent.fitness else parent
