"""
This module contains the code for an island in an island-based GA optimization

it is general enough to work on any representation/fitness
"""
import logging
import numpy as np
from bingo.Util.ArgumentValidation import argument_validation

LOGGER = logging.getLogger(__name__)

class Island(object):
    """
    Island: code for island of genetic algorithm
    """
    @argument_validation(population_size={">=": 0})
    def __init__(self, evolution_algorithm, generator, population_size):
        """Initialization of island

        Parameters
        ----------
        evolution_algorithm : EvolutionaryAlgorithm
                              The desired algorithm to use in assessing the population
        generator : Generator
                    The generator class that returns an instance of a chromosome
        population_size : int
                          The desired size of the population

        Attributes
        ----------
        num_generations : int
                          The number of generational steps that have been executed
        """
        self.pop = [generator() for i in range(population_size)]
        self._ea = evolution_algorithm
        self._population_size = population_size
        self._num_generations = 0

    def execute_generational_step(self):
        """Executes a single generational step using the provided evolutionary algorithm

        Returns
        -------
        population : list of Chromosomes
                     The offspring generation yielded from the generational step
        """
        self._num_generations += 1
        return self._ea.generational_step(self.pop)

    def best_individual(self):
        """Finds the individual with the lowest fitness in a population

        Returns
        -------
        best : Chromosome
               The Chromosome with the lowest fitness value
        """
        if self._num_generations < 1:
            raise ValueError('ValueError: Must execute at least one generational step \
             before finding the best individual')
        best = self.pop[0]
        for indv in self.pop[1:]:
            if indv.fitness < best.fitness or np.isnan(best.fitness).any():
                best = indv
        return best
