"""Evaluation phase with random subsampling

In cases where evaluation is expensive it is sometimes useful to evaluate
fitness only on a subset of the underlying training data.  This class does this
by randomly selecting subsets of the data for fitness evaluation. The random 
subset selection occurs each time this evaluation phase is called.
"""

from copy import deepcopy
import numpy as np

from ..util.argument_validation import argument_validation
from .evaluation import Evaluation


class RandomSubsetEvaluation(Evaluation):
    """Phase which evaluates population using random subsets of training data

    A class for fitness evaluation of populations using random subsamples of
    training data.  A random subset of the training data is chosen each call.
    All individuals in the population are then evaluated with a fitness
    function using that same random subset.

    Parameters
    ----------
    fitness_function : FitnessFunction
        The function class that is used to calculate fitnesses of individuals
        in the population.
    subset_size : int
        The size of the subset of training data that will be used for fitness
        evaluation.
    redundant : bool
        Whether to re-evaluate individuals that have been evaluated previously.
        Default True.  Using False may lead to unexpected results where
        individuals are compared on disimilar subsets of the training data.
    multiprocess : int or bool
        Number of processes to use in parallel evaluation
        or False for serial evaluation. If using multiple processes,
        individuals and fitness functions need to be pickle-able.
        Default False.

    Attributes
    ----------
    fitness_function : FitnessFunction
        The function class that is used to calculate fitnesses of individuals
        in the population.
    eval_count : int
        the number of fitness function evaluations that have occurred
    """

    @argument_validation(
        subset_size={">=": 1},
    )
    def __init__(
        self,
        fitness_function,
        subset_size,
        redundant=True,
        multiprocess=False,
    ):
        super().__init__(fitness_function, redundant, multiprocess)
        self._subset_size = subset_size
        self._full_training_data = deepcopy(fitness_function.training_data)

    def __call__(self, population):
        """Evaluates the fitness of a population using random subsampling

        Parameters
        ----------
        population : list of chromosomes
                     population for which fitness should be calculated
        """
        subset = np.random.choice(
            len(self._full_training_data),
            self._subset_size,
            replace=False,
        )
        self.fitness_function.training_data = self._full_training_data[subset]
        super().__call__(population)
