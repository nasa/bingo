"""The "Deterministic Crowding" selection

This module defines the basis of the "deterministic crowding"
selection algorithm in bingo analyses. The next generation
is selected by pairing parents with their offspring and
advancing the most fit of the two.
"""
from .Selection import Selection


class DeterministicCrowdingSelection(Selection):
    """The class that performs deterministic crowding selection on a population
    """
    def __call__(self, population, target_population_size):
        """Performs selection on a population

        Parameters
        ----------
        population : list of Chromosomes
                     The population on which to perform selection. This
                     population includes both the parent and child populations,
                     with the parents in the first half and the children in the
                     latter half
        target_population_size : int
                                 The size of the next generation

        Returns
        -------
        population : list of Chromosomes
                     The newly selected generation of chromosomes
        """
        if (len(population)%2) > 0 or (target_population_size%2) > 0:
            raise ValueError('Population must be of even length')

        if target_population_size != (len(population)/2):
            raise ValueError('Target population size cannot be greater\
                 than the length of the population')

        offspring = population[target_population_size:]
        population = population[:target_population_size]

        for i in range(target_population_size//2):
            parent_1 = population[i*2]
            parent_2 = population[i*2+1]
            child_1 = offspring[i*2]
            child_2 = offspring[i*2+1]

            dist_a = parent_1.distance(child_1) + parent_2.distance(child_2)
            dist_b = parent_1.distance(child_2) + parent_2.distance(child_1)
            if dist_a <= dist_b:
                population[i*2] = self._return_most_fit(child_1, parent_1)
                population[i*2+1] = self._return_most_fit(child_2, parent_2)
            else:
                population[i*2] = self._return_most_fit(child_2, parent_1)
                population[i*2+1] = self._return_most_fit(child_1, parent_2)

        return population

    @staticmethod
    def _return_most_fit(child, parent):
        return child if child.fitness < parent.fitness else parent
