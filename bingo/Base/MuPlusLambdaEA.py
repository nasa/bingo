"""The "Mu + Lambda"

This module defines the basis of the "mu plus lambda"
evolutionary algorithm in bingo analyses. The next generation
is evaluated and selected from both the parent and offspring
populations.
"""
from .EvolutionaryAlgorithm import EvolutionaryAlgorithm
from .VarOr import VarOr


class MuPlusLambda(EvolutionaryAlgorithm):
    """The algorithm used to perform generational steps.

    A class for the "mu plus lambda" evolutionary algorithm in bingo.

    Parameters
    ----------
    evaluation : Evaluation
        The evaluation algorithm that sets the fitness on the population.
    selection : Selection
                Selection instance to perform selection on a population
    crossover : Crossover
                The algorithm that performs crossover during variation.
    mutation : Mutation
               The algorithm that performs mutation during variation.
    crossover_probability : float
                            Probability that crossover will occur on an
                            individual.
    mutation_probability : float
                           Probability that mutation will occur on an
                           individual.
    number_offspring : int
                       The number of offspring produced from variation.

    Attributes
    ----------
    variation : VarOr
                VarOr variation to perform variation on a population
    evaluation : Evaluation
                 Evaluation instance to perform evaluation on a population
    selection : Selection
                Selection instance to perform selection on a population
    """
    def __init__(self, evaluation, selection, crossover, mutation,
                 crossover_probability, mutation_probability,
                 number_offspring):
        super().__init__(variation=VarOr(crossover, mutation,
                                         crossover_probability,
                                         mutation_probability),
                         evaluation=evaluation,
                         selection=selection)
        self._number_offspring = number_offspring

    def generational_step(self, population):
        """Performs selection on individuals.

        Parameters
        ----------
        population : list of Chromosome
                     The population at the start of the generational step

        Returns
        -------
        list of Chromosome :
            The next generation of the population
        """
        offspring = self.variation(population, self._number_offspring)
        self.evaluation(population + offspring)
        return self.selection(population + offspring, len(population))
