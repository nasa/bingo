"""
A set of classes for simple EA tests
"""
import numpy as np

from bingo.Base.Chromosome import Chromosome
from bingo.Base.Evaluation import Evaluation
from bingo.Base.Mutation import Mutation
from bingo.Base.Crossover import Crossover


class SingleValueChromosome(Chromosome):
    """Single value individual

    Parameters
    ----------
    value : float
            value to set for the individual. Default is random [0, 1)
    """
    def __init__(self, value=None):
        super().__init__()
        if value is None:
            self.value = np.random.random()
        else:
            self.value = value

    def __str__(self):
        return str(self.value)


class SingleValueEvaluation(Evaluation):
    """Evaluation for single valued chromosomes

    Fitness equals the chromosomes value.
    """
    def __call__(self, individual):
        return individual.value


class SingleValueMutation(Mutation):
    """Mutation for single valued chromosomes

    Mutation results in a new random value [0, 1).
    """
    def __call__(self, parent):
        child = parent.copy()
        child.value = np.random.random()
        return child


class SingleValueCrossover(Crossover):
    """Crossover for single valued chromosomes

    Crossover results in two individuals with skewed averages of the parents.
    """
    def __call__(self, parent_1, parent_2):
        child_1 = parent_1.copy()
        child_2 = parent_2.copy()
        child_1.value = 0.6*parent_1.value + 0.4*parent_2.value
        child_2.value = 0.6*parent_2.value + 0.4*parent_1.value
        return child_1, child_2
