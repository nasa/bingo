from bingo.Base.Variation import Variation
from bingo.MultipleValues import MultipleValueGenerator
from bingo.Util.ArgumentValidation import argument_validation

class RandomIndividualVariation(Variation):

    def __init__(self, variation, chromosome_generator, num_rand_indvs=1):
        self._variation = variation
        self._chromosome_generator = chromosome_generator
        self._num_rand_indvs = num_rand_indvs

    @argument_validation(number_offspring={">=": 0})
    def __call__(self, population, number_offspring):
        new_population = self._generate_new_pop(population)
        return self._variation(new_population, number_offspring)
    
    def _generate_new_pop(self, population):
        random_indv = self._chromosome_generator()
        new_population = [random_indv] + population
        return new_population
