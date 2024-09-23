"""The "Probabilistic Crowding" selection

This module defines the basis of the probabilistic crowding selection
algorithm in bingo analyses. The next generation is selected by pairing parents
with their offspring and selecting the child with a probility that is related
to the fitness of the paired child and parent.
"""

import numpy as np
from .generalized_crowding import GeneralizedCrowding


# pylint: disable=too-few-public-methods
class ProbabilisticCrowding(GeneralizedCrowding):
    """Crowding using probabilistic model selection

    Fitness of individuals are assumed to be a measure of model evidence, such
    that a ratio between two fitness values gives the Bayes Factor.

    Parameters
    ----------
    log_scale : bool
        Whether fitnesses of the individuals is in log space. Default True.
    negative : bool
        Whether to invert the fitness of the individual (before log). Default
        True.
    """

    def __init__(self, log_scale=True, negative=False):
        self._log_scale = log_scale
        self._negative = negative
        super().__init__()

    def _return_most_fit(self, child, parent):
        p_fit = parent.fitness
        if np.isnan(p_fit):
            return child
        c_fit = child.fitness
        if np.isnan(c_fit):
            return parent

        if self._negative:
            p_fit = -p_fit
            c_fit = -c_fit

        if self._log_scale:
            prob = np.exp(c_fit - p_fit)
            prob = prob / (prob + 1)
        else:
            prob = c_fit / (p_fit + c_fit)
        return child if np.random.random() < prob else parent
