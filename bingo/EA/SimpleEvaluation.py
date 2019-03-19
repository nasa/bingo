"""Simple form of the  genetic operation of Evaluation.

This module defines a simple (relatively standard) form of the evaluation phase
of a bingo evolutionary algorithms.
"""

from ..Base.Evaluation import Evaluation


class SimpleEvaluation(Evaluation):
    """Simple form of the EA phase calculating fitness of a population.

    All individuals in the population are evaluated with a fitness evaluator
    unless their fitness has already been set.

    Parameters
    ----------
    fitness_function : FitnessFunction
                        The function class that is used to calculate fitnesses
                        of individuals in the population.
    """
    def __init__(self, fitness_function):
        super().__init__()
        self._fitness_function = fitness_function

    @property
    def eval_count(self):
        """int : the number of evaluations that have been performed"""
        return self._fitness_function.eval_count

    @eval_count.setter
    def eval_count(self, value):
        self._fitness_function.eval_count = value

    def __call__(self, population):
        """Evaluates the fitness of an individual

        Parameters
        ----------
        population : list of Chromosome
                     population for which fitness should be calculated
        """
        for indv in population:
            if not indv.fit_set:
                indv.fitness = self._fitness_function(indv)