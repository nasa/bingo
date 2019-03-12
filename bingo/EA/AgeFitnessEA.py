"""
The Age-Fitness Evolutionary Algorithm

This module defines the evolutionary algorithm that implements, ``VarOr``
variation on a population then performs Age-Fitness selection among 
the variation result, the initial population, and a random chromosome.
"""
import numpy as np

from bingo.AgeFitness import AgeFitness
from bingo.EA.MuPlusLambda import MuPlusLambda
from bingo.EA.VarOr import VarOr
from bingo.EA.RandomIndividualVariation import RandomIndividualVariation
from bingo.MultipleValues import MultipleValueGenerator
from bingo.Util.ArgumentValidation import argument_validation

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
    number_of_offspring : int
        The number of offspring produced from variation.
    selection_size : int
        The size of the group of individuals to be randomly
        compared. The size must be an integer greater than 1.
    """
    def __init__(self, evaluation, generator, crossover, mutation,
                 crossover_probability, mutation_probability, number_offspring,
                 selection_size=2):
        variation = VarOr(crossover, mutation, crossover_probability,
                          mutation_probability)
        self._variation = RandomIndividualVariation(variation, generator)
        self._evaluation = evaluation
        self._selection = AgeFitness(selection_size=selection_size)
        self._number_offspring = number_offspring
