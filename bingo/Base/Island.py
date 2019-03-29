"""
This module contains the code for an island in an island-based GA optimization

it is general enough to work on any representation/fitness
"""
import logging
import numpy as np
from ..Util.ArgumentValidation import argument_validation

LOGGER = logging.getLogger(__name__)


class Island:
    """
    Island: code for island of genetic algorithm
    """
    @argument_validation(population_size={">=": 0})
    def __init__(self, evolution_algorithm, generator, population_size):
        """Initialization of island

        Parameters
        ----------
        evolution_algorithm : EvolutionaryAlgorithm
                              The desired algorithm to use in assessing the
                              population
        generator : Generator
                    The generator class that returns an instance of a
                    chromosome
        population_size : int
                          The desired size of the population

        Attributes
        ----------
        generational_age : int
                          The number of generational steps that have been
                          executed

        population : list of Chromosomes
                     The population that is evolving
        """
        self.population = [generator() for _ in range(population_size)]
        self._ea = evolution_algorithm
        self._population_size = population_size
        self.generational_age = 0

    def execute_generational_step(self):
        """Executes a single generational step using the provided evolutionary
        algorithm

        Returns
        -------
        population : list of Chromosomes
                     The offspring generation yielded from the generational
                     step
        """
        self.generational_age += 1
        self.population = self._ea.generational_step(self.population)
        for indv in self.population:
            indv.genetic_age += 1

    def evaluate_population(self):
        """Manually trigger evaluation of population"""
        self._ea.evaluation(self.population)

    def best_individual(self):
        """Finds the individual with the lowest fitness in a population

        Returns
        -------
        best : Chromosome
               The Chromosome with the lowest fitness value
        """
        self.evaluate_population()
        best = self.population[0]
        for indv in self.population:
            if indv.fitness < best.fitness or np.isnan(best.fitness).any():
                best = indv
        return best
