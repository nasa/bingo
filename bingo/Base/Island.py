"""
This module contains the code for an island in an island-based GA optimization

it is general enough to work on any representation/fitness
"""
import logging

import numpy as np

from .EvolutionaryOptimizer import EvolutionaryOptimizer
from ..Util.ArgumentValidation import argument_validation

LOGGER = logging.getLogger(__name__)


# TODO add inherrited attributes in doc
class Island(EvolutionaryOptimizer):
    """
    Island: code for island of genetic algorithm
    """
    @argument_validation(population_size={">=": 0})
    def __init__(self, evolution_algorithm, generator, population_size,
                 hall_of_fame=None):
        """Initialization of island

        Parameters
        ----------
        evolution_algorithm : EvolutionaryAlgorithm
            The desired algorithm to use in assessing the population
        generator : Generator
            The generator class that returns an instance of a chromosome
        population_size : int
            The desired size of the population
        hall_of_fame : HallOfFame (optional)
            The hall of fame object to be used for storing best individuals

        Attributes
        ----------
        generational_age : int
            The number of generational steps that have been executed

        population : list of Chromosomes
            The population that is evolving
        """
        super().__init__(hall_of_fame)
        self.population = [generator() for _ in range(population_size)]
        self._ea = evolution_algorithm
        self._population_size = population_size

    def _do_evolution(self, num_generations):
        for _ in range(num_generations):
            self._execute_generational_step()

    def _execute_generational_step(self):
        self.generational_age += 1
        self.population = self._ea.generational_step(self.population)
        for indv in self.population:
            indv.genetic_age += 1

    def evaluate_population(self):
        """Manually trigger evaluation of population"""
        self._ea.evaluation(self.population)

    def get_best_individual(self):
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

    def get_best_fitness(self):
        """ finds the fitness value of the most fit individual

        Returns
        -------
         :
            Fitness of best individual
        """
        return self.get_best_individual().fitness

    def get_fitness_evaluation_count(self):
        """ Gets the number of fitness evaluations performed

        Returns
        -------
        int :
            number of fitness evaluations
        """
        return self._ea.evaluation.eval_count

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

    def _get_potential_hof_members(self):
        return self.population

    def dump_fraction_of_population(self, fraction):
        # TODO doc
        np.random.shuffle(self.population)
        index = int(round(fraction * len(self.population)))
        dumped_population = self.population[:index]
        self.population = self.population[index:]
        return dumped_population
