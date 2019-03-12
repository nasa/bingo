import numpy as np

from bingo.AgeFitness import AgeFitness
from bingo.EA.MuPlusLambda import MuPlusLambda
from bingo.EA.VarOr import VarOr
from bingo.EA.RandomIndividualVariation import RandomIndividualVariation
from bingo.MultipleValues import MultipleValueGenerator
from bingo.Util.ArgumentValidation import argument_validation

class AgeFitnessEA(MuPlusLambda):

    def __init__(self, evaluation, generator, crossover, mutation,
                 crossover_probability, mutation_probability, number_offspring,
                 selection_size=2):
        variation = VarOr(crossover, mutation, crossover_probability,
                          mutation_probability)
        self._variation = RandomIndividualVariation(variation, generator)
        self._evaluation = evaluation
        self._selection = AgeFitness(selection_size=selection_size)
        self._number_offspring = number_offspring
