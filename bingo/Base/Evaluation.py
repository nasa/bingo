"""The genetic operation of Evaluation.

This module defines the a basic form of the evaluation phase of bingo
evolutionary algorithms.
"""


class Evaluation:
    """Base phase for calculating fitness of a population.

    A base class for the fitness evaluation of populations of genetic
    individuals (list of Chromosomes) in bingo.  All individuals in the
    population are evaluated with a fitness function unless their fitness has
    already been set.

    Parameters
    ----------
    fitness_function : FitnessFunction
                        The function class that is used to calculate fitnesses
                        of individuals in the population.

    Attributes
    ----------
    fitness_function : FitnessFunction
                        The function class that is used to calculate fitnesses
                        of individuals in the population.
    eval_count : int
                 the number of fitness function evaluations that have occurred
    """
    def __init__(self, fitness_function):
        self.fitness_function = fitness_function

    @property
    def eval_count(self):
        """int : the number of evaluations that have been performed"""
        return self.fitness_function.eval_count

    @eval_count.setter
    def eval_count(self, value):
        self.fitness_function.eval_count = value

    def __call__(self, population):
        """Evaluates the fitness of an individual

        Parameters
        ----------
        population : list of Chromosome
                     population for which fitness should be calculated
        """
        for indv in population:
            if not indv.fit_set:
                indv.fitness = self.fitness_function(indv)
