import numpy as np

from bingo.selection.deterministic_crowding import DeterministicCrowding


class NondeterministicCrowding(DeterministicCrowding):

    def __init__(self, logscale=True):
        self._logscale = logscale

    def _return_most_fit(self, child, parent):
        if np.isnan(parent.fitness):
            return child
        if np.isnan(child.fitness):
            return parent

        if self._logscale:
            prob = np.exp(parent.fitness - child.fitness)
            prob = prob / (prob + 1)
        else:
            prob = child.fitness / (parent.fitness + child.fitness)

        return child if np.random.random() < prob else parent