"""
The Age-Fitness Evolutionary Algorithm

This module defines the evolutionary algorithm that implements, ``VarOr``
variation on a population then performs Age-Fitness selection among
the variation result, the initial population, and a random chromosome.
"""

from .AgeFitnessSelection import AgeFitness
from .MuPlusLambdaEA import MuPlusLambda
from .AddRandomIndividualVariation import AddRandomIndividualVariation


class AgeFitnessEA(MuPlusLambda):
    """The algorithm used to perform generational steps.

    This class extends ``MuPlusLambda`` and executes ``generational_step``.

    Parameters
    ----------
    evaluation : Evaluation
        The evaluation algorithm that sets the fitness on the population.
    generator : Generator
        The individual generator for the random individual.
    crossover : Crossover
        The algorithm that performs crossover during variation.
    mutation : Mutation
        The algorithm that performs mutation during variation.
    crossover_probability : float
        Probability that crossover will occur on an individual.
    mutation_probability : float
        Probability that mutation will occur on an individual.
    population_size : int
        The targeted poulation size and the number of offspring produced from
        variation.
    selection_size : int
        The size of the group of individuals to be randomly
        compared. The size must be an integer greater than 1.

    Attributes
    ----------
    variation : `Variation`
                 Public access to the variation phase of the Base
    evaluation : `Evaluation`
                 Public access to the evaluation phase of the Base
    selection : `Selection`
                 Public access to the selection phase of the Base
    """
    def __init__(self, evaluation, generator, crossover, mutation,
                 crossover_probability, mutation_probability, population_size,
                 selection_size=2):
        self.selection = AgeFitness(selection_size=selection_size)
        super().__init__(evaluation, self.selection, crossover, mutation,
                         crossover_probability, mutation_probability,
                         number_offspring=population_size,
                         target_population_size=population_size)
        self.variation = AddRandomIndividualVariation(self.variation,
                                                      generator)
