"""The definition of fitness evaluations for individuals."""

from abc import ABCMeta, abstractmethod


class FitnessFunction(metaclass=ABCMeta):
    """Fitness evaluation metric for individuals.

    An abstract base class for the fitness evaluation of genetic individuals
    (chromosomes) in bingo.

    Parameters
    ----------
    training_data : TrainingData
        (Optional) data that can be used in fitness evaluation

    Attributes
    ----------
    eval_count : int
        the number of evaluations that have been performed
    training_data : TrainingData
        (Optional) data that can be used in fitness evaluation
    """

    def __init__(self, training_data=None):
        self.eval_count = 0
        self.training_data = training_data

    @abstractmethod
    def __call__(self, individual):
        """Evaluates the fitness of an individual

        Parameters
        ----------
        individual : Chromosome
            individual for which fitness will be calculated

        Notes
        -----
        The eval_count should be incremented in a subclass' __call__ definition
        for accurate evaluation counting

        Returns
        -------
        fitness : numeric
            fitness of the individual
        """
        raise NotImplementedError
