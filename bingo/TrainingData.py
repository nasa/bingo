"""
This module contains the definition of various data containers that store
training data for bingo.
"""

import abc
import warnings
import logging

from .Utils import calculate_partials

LOGGER = logging.getLogger(__name__)


class TrainingData(object, metaclass=abc.ABCMeta):
    """
    Training Data superclass which defines the methods needed for derived
    classes
    """

    @abc.abstractmethod
    def __getitem__(self, items):
        """
        This function allows for the sub-indexing of the training data
        :param items: list of indices for the subset
        :return: must return a TrainingData object
        """
        pass

    @abc.abstractmethod
    def size(self):
        """
        gets the number of indexable points in the training data
        :return: size of the training dataset
        """
        pass


class ExplicitTrainingData(TrainingData):
    """
    ExplicitTrainingData: Training data of this type contains an input array of
    data (x)  and an output array of data (y).  Both must be 2 dimensional
    numpy arrays
    """

    def __init__(self, x, y):
        """
        Initialization of explicit training data
        :param x: numpy array, dependent variable
        :param y: numpy array, independent variable
        """
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
        """
        gets a subset of the ExplicitTrainingData
        :param items: list or int, index (or indices) of the subset
        :return: an ExplicitTrainingData
        """
        temp = ExplicitTrainingData(self.x[items, :], self.y[items, :])
        return temp

    def size(self):
        """
        gets the length of the first dimension of the data
        :return: indexable size
        """
        return self.x.shape[0]


class ImplicitTrainingData(TrainingData):
    """
    ImplicitTrainingData: Training data of this type contains an input array of
    data (x)  and its time derivative (dx_dt).  Both must be 2 dimensional
    numpy arrays
    """

    def __init__(self, x, dx_dt=None):
        """
        Initialization of implicit training data
        :param x: numpy array, dependent variable
        :param dx_dt: numpy array,  time derivative of x
        """
        if x.ndim == 1:
            warnings.warn("Explicit training x should be 2 dim array, " +
                          "reshaping array")
            x = x.reshape([-1, 1])
        if x.ndim > 2:
            raise ValueError('Explicit training x should be 2 dim array')

        # dx_dt not provided
        if dx_dt is None:
            x, dx_dt, _ = calculate_partials(x)

        # dx_dt is provided
        else:
            if dx_dt.ndim != 2:
                raise ValueError('Implicit training dx_dt must be 2 dim array')

        self.x = x
        self.dx_dt = dx_dt

    def __getitem__(self, items):
        """
        gets a subset of the ExplicitTrainingData
        :param items: list or int, index (or indices) of the subset
        :return: an ExplicitTrainingData
        """
        temp = ImplicitTrainingData(self.x[items, :], self.dx_dt[items, :])
        return temp

    def size(self):
        """
        gets the length of the first dimension of the data
        :return: indexable size
        """
        return self.x.shape[0]
