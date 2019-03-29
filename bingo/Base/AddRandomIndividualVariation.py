"""Variation that adds random individual(s)

This module wraps a variation in order to supply random
individual(s) to the offspring after the variation is carried out.
"""

from .Variation import Variation
from ..Util.ArgumentValidation import argument_validation


class AddRandomIndividualVariation(Variation):
    """A Variation object that takes in an implementation of Variation
    that adds a random individual to the population before performing
    variation.

    Parameters
    ----------
    variation : Variation
        Variation object that performs the variation among individuals
    chromosome_generator : Generator
        Generator for random individual
    num_rand_indvs : int
        The number of random individuals to generate per call

    """
    def __init__(self, variation, chromosome_generator, num_rand_indvs=1):
        self._variation = variation
        self._chromosome_generator = chromosome_generator
        self._num_rand_indvs = num_rand_indvs

    @argument_validation(number_offspring={">=": 0})
    def __call__(self, population, number_offspring):
        """Generates a number of random indiviudals and adds the to the
        population then performs variation on the new population.

        Parameters
        ----------
        population : list of Chromosomes
            The population on which to perform variation
        number_offspring : int
            number of offspring to produce.

        Returns
        -------
        list of Chromosome:
            The offspring of the original population and the
            new random individuals
        """
        children = self._variation(population, number_offspring)
        return self._generate_new_pop(children)

    def _generate_new_pop(self, population):
        for _ in range(self._num_rand_indvs):
            random_indv = self._chromosome_generator()
            population.append(random_indv)
        return population
