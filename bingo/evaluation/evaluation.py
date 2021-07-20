"""The genetic operation of evaluation.

This module defines the a basic form of the evaluation phase of bingo
evolutionary algorithms.
"""

import multiprocessing

import bingo.util.global_imports as gi

num_cores = multiprocessing.cpu_count()

class Evaluation:
    """Base phase for calculating fitness of a population.

    A base class for the fitness evaluation of populations of genetic
    individuals (list of chromosomes) in bingo.  All individuals in the
    population are evaluated with a fitness function unless their fitness has
    already been set.

    Parameters
    ----------
    fitness_function : FitnessFunction
        The function class that is used to calculate fitnesses of individuals
        in the population.

    Attributes
    ----------
    fitness_function : FitnessFunction
        The function class that is used to calculate fitnesses of individuals
        in the population.
    eval_count : int
        the number of fitness function evaluations that have occurred
    """
    def __init__(self, fitness_function):
        self.fitness_function = fitness_function

    @property
    def eval_count(self):
        """int : the number of evaluations that have been performed"""
        return self.fitness_function.eval_count

    @eval_count.setter
    def eval_count(self, value):
        self.fitness_function.eval_count = value

    def __call__(self, population):
        """Evaluates the fitness of an individual

        Parameters
        ----------
        population : list of chromosomes
                     population for which fitness should be calculated
        """

        def eval_fitness(indv):
            if not indv.fit_set:
                return self.fitness_function(indv)

        if gi.USING_PARALLEL_CPU:
            fitnesses = gi.jl.Parallel(n_jobs=num_cores)(gi.jl.delayed(eval_fitness)(indv) for indv in population)
            print("hi")

            for i, indv in enumerate(population):
                if not indv.fit_set:
                    indv.fitness = fitnesses[i]

        else:
            for indv in population:
                if not indv.fit_set:
                    indv.fitness = self.fitness_function(indv)


