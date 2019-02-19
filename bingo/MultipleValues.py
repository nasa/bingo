

import numpy as np

from Base.Chromosome import Chromosome
from Base.FitnessEvaluator import FitnessEvaluator
from Base.Mutation import Mutation
from Base.Crossover import Crossover
from Base.Generator import Generator


class MultipleValueChromosome(Chromosome):
    """ Multiple value individual 

    Parameters
    ----------
    list_of_values : list of either integers, floats, or boolean values
                contains the list of values corresponding to the chromosome

    """
    def __init__(self, list_of_values):
        super().__init__()
        self._list_of_values = list_of_values

    def __str__(self):
        return str(self._list_of_values)


class MultipleValueGenerator(Generator):
    """Generation of a population of Multi-Value Chromosomes
    """
    def __call__(self, random_value_function, population_size=20, values_per_chromosome=10):
    
        """Generation of a population of size 'population_size' of Multi-Value Chromosomes 
        with lists that contain 'values_per_list' values 


        Parameters
        ----------
        random_value_function : user defined function
                a function that returns a list of randomly generated values. this list is then passed to the MultipleValueChromosome constructor
        popularion_size : int, default=20
                the size of the population to be generated
        values_per_chromosome : int, default=10
                the number of values that each chromosome will hold

        Returns
        -------
        out : a list of size "population size" of MultipleValueChromosomes of length "values_per_chromosome"
    
        """
        return [MultipleValueChromosome(random_value_function(values_per_chromosome)) for i in range(population_size)]


class MultipleValueMutation(Mutation):
    """Mutation for multiple valued chromosomes

    Parameters
    ----------
    mutation_function : user defined funxtion
                a function that returns a random value that will replace (or "mutate") a random value in a chromosome list.
    """
    def __init__(self, mutation_function):
        super().__init__()
        self._mutation_function = mutation_function

    def __call__(self, parent):
        """Performs single-point mutation using the user-defined mutation function passed to the constructor

        Parameters
        ----------
        parent : MultipleValueChromosome
                The parent chromosome that is copied to create the child that will undergo mutation

        Returns
        -------
        child : MultipleValueChromosome
                The child chromosome that has undergone mutation
        """
        child = parent.copy()
        child.fit_set = False
        mutation_point = np.random.randint(len(parent._list_of_values))
        child._list_of_values[mutation_point] = self._mutation_function()
        return child

class MultipleValueCrossover(Crossover):
    """Crossover for multiple valued chromosomes

    Crossover results in two individuals with single-point crossed-over lists 
    whose values are provided by the parents. Crossover point is a random integer
    produced by numpy.
    """
    def __init__(self):
        super().__init__()
        self._crossover_point = 0

    def __call__(self, parent_1, parent_2):
        """Performs single-point crossover of two parent chromosomes

        Parameters
        ----------
        parent_1 : MultipleValueChromosome
                the first parent to be used for crossover
        parent_2 : MultipleValueChromosome
                the second parent to be used for crossover

        Returns
        -------
        child_1, child_2 : tuple of MultiValueChromosome
                a tuple of the 2 children produced by crossover
        """
        child_1 = parent_1.copy()
        child_2 = parent_2.copy()
        child_1.fit_set = False
        child_2.fit_set = False
        self._crossover_point = np.random.randint(len(parent_1._list_of_values))
        child_1._list_of_values = parent_1._list_of_values[:self._crossover_point] + parent_2._list_of_values[self._crossover_point:]
        child_2._list_of_values = parent_2._list_of_values[:self._crossover_point] + parent_1._list_of_values[self._crossover_point:]
        return child_1, child_2

