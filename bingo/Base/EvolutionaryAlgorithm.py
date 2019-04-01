"""The base of evolutionary algorithm definition

This module defines the basis of evolutionary algorithms in bingo analyses.
An Base in bingo is defined by three phases: Variation, Evaluation, and
Selection.  These phases, when repeated, define the evolution of a population.
"""


class EvolutionaryAlgorithm():
    """The algorithm used to perform generational steps.

    The basic implementation used in this base Base implementation is a simple
    steady-state Base (akin to simpleEA in DEAP)

    Parameters
    ----------
    variation : `Variation`
                The phase of bingo EAs that is responsible for varying the
                population (usually through some form of crossover and/or
                mutation).
    evaluation : `Evaluation`
                 The phase in bingo EAs responsible for the evaluation of the
                 fitness of the individuals in the population.
    selection : `Selection`
                The phase of bingo EAs responsible for selecting the
                individuals in a population which survive into the next
                generation.

    Attributes
    ----------
    variation : `Variation`
                 Public to the variation phase of the Base
    evaluation : `Evaluation`
                 Public to the evaluation phase of the Base
    selection : `Selection`
                 Public to the selection phase of the Base
    """
    def __init__(self, variation, evaluation, selection):
        self.variation = variation
        self.evaluation = evaluation
        self.selection = selection

    def generational_step(self, population):
        """Performs a generational step on population.

        Parameters
        ----------
        population : list of Chromosome
                     The population at the start of the generational step

        Returns
        -------
        list of Chromosome :
            The next generation of the population
        """
        population_size = len(population)
        offspring = self.variation(population, population_size)
        self.evaluation(offspring)
        next_generation = self.selection(offspring, population_size)
        return next_generation

