"""Explicit Symbolic Regression

Explicit symbolic regression is the search for a function, f, such that
f(x) = y.

The classes in this module encapsulate the parts of bingo evolutionary analysis
that are unique to explicit symbolic regression. Namely, these classes are an
appropriate fitness evaluator and a corresponding training data container.
"""

import warnings
import logging

from ..Base.FitnessFunction import VectorBasedFunction
from ..Base.TrainingData import TrainingData

LOGGER = logging.getLogger(__name__)


class ExplicitRegression(VectorBasedFunction):
    """ Traditional fitness evaluation for symbolic regression

    fitness = y - f(x) where x and y are in the training_data (i.e.
    training_data.x and training_data.y) and the function f is defined by the
    input Equation individual.

    Parameters
    ----------
    training_data : ExplicitTrainingData
                    data that is used in fitness evaluation.
    metric : str
        String defining the measure of error to use. Available options are:
        'mean absolute error', 'mean squared error', and
        'root mean squared error'
    """
    def evaluate_fitness_vector(self, individual):
        self.eval_count += 1
        f_of_x = individual.evaluate_equation_at(self.training_data.x)
        return (f_of_x - self.training_data.y).flatten()


class ExplicitTrainingData(TrainingData):
    """
    ExplicitTrainingData: Training data of this type contains an input array of
    data (x)  and an output array of data (y).  Both must be 2 dimensional
    numpy arrays

    Parameters
    ----------
    x : 2D numpy array
        independent variable
    y : 2D numpy array
        dependent variable
    """
    def __init__(self, x, y):
        if x.ndim == 1:
            warnings.warn("Explicit training x should be 2 dim array, " +
                          "reshaping array")
            x = x.reshape([-1, 1])
        if x.ndim > 2:
            raise ValueError('Explicit training x should be 2 dim array')

        if y.ndim == 1:
            warnings.warn("Explicit training y should be 2 dim array, " +
                          "reshaping array")
            y = y.reshape([-1, 1])
        if y.ndim > 2:
            raise ValueError('Explicit training y should be 2 dim array')

        self.x = x
        self.y = y

    def __getitem__(self, items):
        """gets a subset of the ExplicitTrainingData

        Parameters
        ----------
        items : list or int
                index (or indices) of the subset

        Returns
        -------
        ExplicitTrainingData :
                                a Subset
        """
        temp = ExplicitTrainingData(self.x[items, :], self.y[items, :])
        return temp

    def __len__(self):
        """ gets the length of the first dimension of the data

        Returns
        -------
        int :
              index-able size
        """
        return self.x.shape[0]
