"""The base for generalized crowding selection

This module defines the basis of the generalized crowding selection
algorithm in bingo analyses. The next generation is selected by pairing parents
with their offspring and selecting the most fit of the two.
"""
from abc import abstractmethod
from .selection import Selection


class GeneralizedCrowding(Selection):
    """The class that performs generalized crowding selection on a population
    """
    def __call__(self, population, target_population_size, idx):
        """Performs selection on a population

        Parameters
        ----------
        population : list of chromosomes
            The population on which to perform selection. This population
            includes both the parent and child populations, with the parents in
            the first half and the children in the latter half
        target_population_size : int
            The size of the next generation

        Returns
        -------
        population : list of chromosomes
            The newly selected generation of chromosomes
        """
        if (len(population) % 2) > 0 or (target_population_size % 2) > 0:
            raise ValueError('Population must be of even length')

        half_pop_size = len(population) // 2
        if target_population_size > half_pop_size:
            raise ValueError('Target population size cannot be greater\
                 than the half of the population')

        offspring = population[half_pop_size:]
        population = population[:half_pop_size]

        for i in range(target_population_size // 2):
            parent_1 = population[i*2]
            parent_2 = population[i*2+1]
            child_1 = offspring[i*2]
            child_2 = offspring[i*2+1]

            dist_a = parent_1.distance(child_1) + parent_2.distance(child_2)
            dist_b = parent_1.distance(child_2) + parent_2.distance(child_1)
            if dist_a <= dist_b:
                population[i*2] = self._return_most_fit(child_1, parent_1, idx)
                population[i*2+1] = self._return_most_fit(child_2, parent_2, 
                                                                            idx)
            else:
                population[i*2] = self._return_most_fit(child_2, parent_1, idx)
                population[i*2+1] = self._return_most_fit(child_1, parent_2, idx)

        return population

    @staticmethod
    @abstractmethod
    def _return_most_fit(child, parent):
        raise NotImplementedError
