"""
This module contains the code for an island in an island-based GA optimization

it is general enough to work on any representation/fitness
"""
import logging

import numpy as np

from ..Util.ArgumentValidation import argument_validation
from .AgeFitnessSelection import AgeFitness

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
            The desired algorithm to use in assessing the population
        generator : Generator
            The generator class that returns an instance of a chromosome
        population_size : int
            The desired size of the population

        Attributes
        ----------
        generational_age : int
            The number of generational steps that have been executed

        population : list of Chromosomes
            The population that is evolvingj
            
        """
        self.population = [generator() for _ in range(population_size)]
        self.generational_age = 0
        self.pareto_front_selection = AgeFitness()
        self._ea = evolution_algorithm
        self._population_size = population_size
        self._pareto_front = []

    def execute_generational_step(self):
        """Executes a single generational step using the provided evolutionary
        algorithm
        
        """
        self.generational_age += 1
        self.population = self._ea.generational_step(self.population)
        for indv in self.population:
            indv.genetic_age += 1
        # self.update_pareto_front()

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

    def load_population(self, population, replace=True):
        """Loads population from a pickleable object

        Parameters
        ----------
        population: list of Chromosomes
            population which is loaded into island
        replace: boolean
            if true, value results in all of the population being
            loaded/replaced. False value means that the population in pop_list
            is appended to the current population
        """
        if replace:
            self.population = []
        self.population += population

    def get_population(self):
        """Getter for population

        Returns
        -------
        list of Chromosome:
            The list of Chromosomes in the island population
        """
        return self.population

    def update_pareto_front(self):
        """Updates a list of Chromosomes that form the pareto front based on 
            the new population.
        """
        self._pareto_front = self.pareto_front_selection.select_pareto_front(
            self._pareto_front + self.population)
        self._pareto_front.sort(key=lambda x: x.fitness)

    def get_pareto_front(self):
        """Getter for the pareto front

        Returns
        -------
        list of Chromsomes:
            The list of Chromosomes in the population that represent the 
            pareto front. The pareto front is returned in sorted order.
        """
        return self._pareto_front
        