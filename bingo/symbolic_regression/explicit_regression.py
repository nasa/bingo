"""Explicit Symbolic Regression

Explicit symbolic regression is the search for a function, f, such that
f(x) = y.

The classes in this module encapsulate the parts of bingo evolutionary analysis
that are unique to explicit symbolic regression. Namely, these classes are an
appropriate fitness evaluator and a corresponding training data container.
"""

import logging
import numpy as np
from scipy.stats import linregress

from ..evaluation.fitness_function import VectorBasedFunction
from ..evaluation.gradient_mixin import VectorGradientMixin
from ..evaluation.training_data import TrainingData

LOGGER = logging.getLogger(__name__)


class ExplicitRegression(VectorGradientMixin, VectorBasedFunction):
    """ExplicitRegression

    The traditional fitness evaluation for symbolic regression.
    fitness = M(y - f(x)) where x and y are in the training_data (i.e.
    training_data.x and training_data.y) and the function f is defined by
    the input Equation individual.  M is an aggregation metric such as mean
    squared error.

    Parameters
    ----------
    training_data : ExplicitTrainingData
        data that is used in fitness evaluation.
    metric : str
        String defining the measure of error to use. Available options are:
        'mean absolute error', 'mean squared error', 'root mean squared error',
        and "negative nmll laplace"
    relative : bool
        Whether to use relative, pointwise normalization of errors. Default:
        False.
    use_linear_correction : bool
        Whether to adjust outputs of equations by a least squares linear correction. Default: False.
    """

    def __init__(
        self, training_data, metric="mae", relative=False, use_linear_correction=False
    ):
        super().__init__(training_data, metric)
        self._relative = relative
        self._linear_correction = use_linear_correction

    def evaluate_fitness_vector(self, individual):
        """Traditional fitness evaluation for symbolic regression

        fitness = y - f(x) where x and y are in the training_data (i.e.
        training_data.x and training_data.y) and the function f is defined by
        the input Equation individual.

        Parameters
        ----------
        individual : Equation
            individual whose fitness is evaluated on `training_data`

        Returns
        -------
        float
            the fitness of the input Equation individual
        """
        self.eval_count += 1
        f_of_x = individual.evaluate_equation_at(self.training_data.x)

        if self._linear_correction:
            try:
                slope, intercept, _, _, _ = linregress(
                    f_of_x.flatten(), self.training_data.y.flatten()
                )
                f_of_x = intercept + slope * f_of_x
            except ValueError:
                pass

        error = f_of_x - self.training_data.y
        if not self._relative:
            return np.squeeze(error)
        return np.squeeze(error / self.training_data.y)

    def get_fitness_vector_and_jacobian(self, individual):
        r"""Fitness and jacobian evaluation of individual

        fitness = y - f(x) where x and y are in the training_data (i.e.
        training_data.x and training_data.y) and the function f is defined by
        the input Equation individual.

        jacobian = [[:math:`df_1/dc_1`, :math:`df_1/dc_2`, ...],
                    [:math:`df_2/dc_1`, :math:`df_2/dc_2`, ...],
                    ...]
        where :math:`f_\#` is the fitness function corresponding with the
        #th fitness vector entry and :math:`c_\#` is the corresponding
        constant of the individual

        Parameters
        ----------
        individual : Equation
            individual whose fitness will be evaluated on `training_data`
            and whose constants will be used for evaluating the jacobian

        Returns
        -------
        fitness_vector, jacobian :
            the vectorized fitness of the individual and
            the partial derivatives of each fitness function with respect
            to the individual's constants
        """
        self.eval_count += 1
        f_of_x, df_dc = individual.evaluate_equation_with_local_opt_gradient_at(
            self.training_data.x
        )

        if self._linear_correction:
            try:
                slope, intercept, _, _, _ = linregress(
                    f_of_x.flatten(), self.training_data.y.flatten()
                )
                f_of_x = intercept + slope * f_of_x
                df_dc *= slope
            except ValueError:
                pass

        error = f_of_x - self.training_data.y
        if not self._relative:
            return np.squeeze(error), df_dc
        return np.squeeze(error / self.training_data.y), df_dc / self.training_data.y


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
            # warnings.warn("Explicit training x should be 2 dim array, " +
            #               "reshaping array")
            x = x.reshape([-1, 1])
        if x.ndim > 2:
            raise TypeError("Explicit training x should be 2 dim array")

        if y.ndim == 1:
            # warnings.warn("Explicit training y should be 2 dim array, " +
            #               "reshaping array")
            y = y.reshape([-1, 1])
        if y.ndim > 2:
            raise TypeError("Explicit training y should be 2 dim array")

        self._x = x
        self._y = y

    @property
    def x(self):
        """independent x data"""
        return self._x

    @property
    def y(self):
        """dependent y data"""
        return self._y

    def __getitem__(self, items):
        """gets a subset of the `ExplicitTrainingData`

        Parameters
        ----------
        items : list or int
            index (or indices) of the subset

        Returns
        -------
        `ExplicitTrainingData` :
            a Subset
        """
        temp = ExplicitTrainingData(self._x[items, :], self._y[items, :])
        return temp

    def __len__(self):
        """gets the length of the first dimension of the data

        Returns
        -------
        int :
            index-able size
        """
        return self._x.shape[0]
