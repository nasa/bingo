"""Variation where crossover and mutation may co-occur

VarAnd.py is similar to the function of the same name in DEAP.  It allows for
definition of a variation by crossover and mutation probabilities. Offspring may
be the result of both crossover :and: mutation, hence the name.
"""
import numpy as np

from .Variation import Variation
from ..Util.ArgumentValidation import argument_validation


class VarAnd(Variation):
    """Variation where crossover and mutation may co-occur

    Parameters
    ----------
    crossover : Crossover
                Crossover function class used in the variation
    mutation : Mutation
               Mutation function class used in the variation
    crossover_probability : float
                            Probability that crossover will occur on an
                            individual
    mutation_probability : float
                           Probability that mutation will occur on an individual


    Attributes
    ----------
    crossover_offspring : array of bool
                          list indicating whether the corresponding member of
                          the last offspring was a result of crossover
    mutation_offspring : array of bool
                         list indicating whether the corresponding member of
                         the last offspring was a result of mutation
    """

    @argument_validation(crossover_probability={">=": 0, "<=": 1},
                         mutation_probability={">=": 0, "<=": 1})
    def __init__(self, crossover, mutation, crossover_probability,
                 mutation_probability):
        super().__init__()
        self._crossover = crossover
        self._mutation = mutation
        self._crossover_probability = crossover_probability
        self._mutation_probability = mutation_probability

    @argument_validation(number_offspring={">=": 0})
    def __call__(self, population, number_offspring):
        """Performs "And" variation on a population.

        Parameters
        ----------
        population : list of Chromosome
                     The population on which to perform selection
        number_offspring : int
                           number of offspring to produce

        Returns
        -------
        list of Chromosome :
            The offspring of the population
        """
        self.crossover_offspring = np.zeros(number_offspring, bool)
        self.mutation_offspring = np.zeros(number_offspring, bool)
        offspring = self._crossover_population(number_offspring, population)
        self._mutate_population(offspring)
        return offspring

    def _crossover_population(self, number_offspring, population):
        offspring = []
        for i in range(0, number_offspring - 1, 2):
            parent_index = i % len(population)
            if np.random.random() <= self._crossover_probability:
                child_1, child_2 = self._crossover(population[parent_index],
                                                   population[parent_index + 1])
                offspring.append(child_1)
                offspring.append(child_2)
                self.crossover_offspring[i:i + 2] = True
            else:
                offspring.append(population[parent_index].copy())
                offspring.append(population[parent_index + 1].copy())
        if len(offspring) < number_offspring:
            parent_index = (len(offspring) + 1) % len(population)
            offspring.append(population[parent_index].copy())
        return offspring

    def _mutate_population(self, offspring):
        for i, parent in enumerate(offspring):
            if np.random.random() <= self._mutation_probability:
                offspring[i] = self._mutation(parent)
                self.mutation_offspring[i] = True
